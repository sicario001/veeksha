"""Tests for the preflight mock servers.

These servers are the *reference clock* the preflight scores the harness
against, so what is checked here is that they emit the right number of frames,
on the cadence they promise, and that they honestly report their own lateness
and connection ground truth. Everything is deliberately tiny (a handful of
chunks, single-digit-millisecond cadences) so the whole module runs in a couple
of seconds.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import List, Optional

import pytest
import websockets

from veeksha.client.realtime_tts import RealtimeTTSClient
from veeksha.config.client import RealtimeTTSClientConfig, TextPacingConfig
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.preflight.audio_server import (
    MockRealtimeAudioServer,
    MockSTTPreflightServer,
)
from veeksha.preflight.mock_engine import MockStreamingEngine
from veeksha.preflight.sharded_server import (
    PhaseSpreader,
    ShardedTelemetry,
    format_pfid,
    parse_pfid,
)
from veeksha.types import ChannelModality

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _sse_stream(
    port: int, max_tokens: Optional[int] = None, pfid: Optional[int] = None
) -> tuple[float, List[float]]:
    """Drive one SSE request; return (t_request_sent, per-chunk arrival stamps)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    payload: dict = {}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if pfid is not None:
        payload["messages"] = [{"role": "user", "content": format_pfid(pfid)}]
    body = json.dumps(payload).encode()
    writer.write(
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: mock\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    await writer.drain()
    t_sent = time.monotonic()
    await reader.readuntil(b"\r\n\r\n")

    stamps: List[float] = []
    while True:
        line = await reader.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        if line == b"data: [DONE]":
            break
        stamps.append(time.monotonic())
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    return t_sent, stamps


def _gaps_ms(stamps: List[float]) -> List[float]:
    return [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]


def _median(values: List[float]) -> float:
    xs = sorted(values)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def test_phase_spreader_spreads_across_one_cadence_window() -> None:
    spreader = PhaseSpreader()
    cadence = 0.05
    phases = [spreader.next_phase(cadence) for _ in range(4)]

    # First connection is unshifted, so low-concurrency runs are unaffected.
    assert phases[0] == 0.0
    assert phases[1] > 0.0
    # Evenly spaced, all inside one cadence window.
    step = phases[1]
    assert phases == pytest.approx([0.0, step, 2 * step, 3 * step])
    assert all(0.0 <= p < cadence for p in phases)


def test_pfid_text_marker_round_trips() -> None:
    assert parse_pfid(format_pfid(41)) == 41
    # Found anywhere in surrounding text; absent -> None.
    assert parse_pfid(f"please say hi {format_pfid(7)} now") == 7
    assert parse_pfid("no marker here") is None


def test_sharded_telemetry_merges_and_clears() -> None:
    telemetry = ShardedTelemetry()
    for value in range(100):
        telemetry.record(float(value))

    assert len(telemetry.merged()) == 100
    # Nearest-rank over 100 samples: index round(0.99 * 99) == 98.
    assert telemetry.p99() == pytest.approx(98.0)

    telemetry.clear()
    assert telemetry.merged() == []
    assert telemetry.p99() == 0.0


# ---------------------------------------------------------------------------
# text SSE mock
# ---------------------------------------------------------------------------


def test_sse_mock_emits_expected_count_and_cadence() -> None:
    engine = MockStreamingEngine(
        chunk_ms=8.0, prefill_ms=20.0, default_chunks=25, num_loops=1
    ).start()
    try:
        t_sent, stamps = asyncio.run(_sse_stream(engine.port))

        assert len(stamps) == 25
        # Prefill is respected (with slack for connect + accept on a loaded box).
        assert (stamps[0] - t_sent) * 1000.0 >= 15.0
        gaps = _gaps_ms(stamps)
        assert _median(gaps) == pytest.approx(8.0, abs=3.0)
        # The paired-timestamp record-keeping the checks depend on: one emit
        # stamp per chunk (this request carried no PFID, so it is keyed -1).
        assert len(engine.records()[-1].emitted_at) == 25
        assert engine.max_active_conns == 1
    finally:
        engine.stop()


def test_sse_mock_honours_requested_chunk_count_and_captures_bodies() -> None:
    engine = MockStreamingEngine(
        chunk_ms=2.0, prefill_ms=0.0, default_chunks=50, num_loops=1, store_bodies=True
    ).start()
    try:
        _, stamps = asyncio.run(_sse_stream(engine.port, max_tokens=7))
        assert len(stamps) == 7
        bodies = engine.captured_bodies()
        assert bodies and bodies[0] == {"max_tokens": 7}
    finally:
        engine.stop()


def test_sse_mock_spreads_connection_phases() -> None:
    # A cadence this wide makes one phase slot (cadence / 50) ~6 ms, i.e. far
    # above localhost noise, so the spread is observable with a single chunk.
    engine = MockStreamingEngine(
        chunk_ms=300.0, prefill_ms=0.0, default_chunks=1, num_loops=1
    ).start()
    try:

        async def _both() -> List[float]:
            results = await asyncio.gather(
                _sse_stream(engine.port), _sse_stream(engine.port)
            )
            return [stamps[0] - t_sent for t_sent, stamps in results]

        first_offsets = asyncio.run(_both())
        spread_ms = abs(first_offsets[0] - first_offsets[1]) * 1000.0
        # Two connections started together would otherwise emit in the same
        # instant; the spreader must move one of them by ~one phase slot.
        assert 2.0 < spread_ms < 30.0
    finally:
        engine.stop()


def test_sse_mock_serves_every_connection_with_multiple_loops() -> None:
    engine = MockStreamingEngine(
        chunk_ms=3.0, prefill_ms=0.0, default_chunks=10, num_loops=4
    ).start()
    try:

        async def _many() -> List[List[float]]:
            results = await asyncio.gather(
                *(_sse_stream(engine.port) for _ in range(24))
            )
            return [stamps for _, stamps in results]

        all_stamps = asyncio.run(_many())
        assert [len(s) for s in all_stamps] == [10] * 24
        assert engine.total_conns == 24
        # Every connection was served, and the peak-concurrency ground truth
        # reflects the real overlap (a completed-request COUNT is not
        # concurrency).
        assert engine.max_active_conns >= 2
        # Accept sharding actually distributes: one shared listening socket,
        # more than one loop wins accepts. (With SO_REUSEPORT on macOS all 24
        # would land on the last-bound listener.)
        assert 2 <= engine.shards_used() <= 4
    finally:
        engine.stop()


def test_sse_mock_records_keyed_by_pfid_with_receive_before_emit() -> None:
    engine = MockStreamingEngine(
        chunk_ms=3.0, prefill_ms=5.0, default_chunks=6, num_loops=1
    ).start()
    try:
        asyncio.run(_sse_stream(engine.port, pfid=314))

        records = engine.records()
        assert list(records.keys()) == [314]
        record = records[314]
        assert record.request_id == 314
        # One emit stamp per chunk, all after the receive stamp.
        assert len(record.emitted_at) == 6
        assert record.received_at < record.emitted_at[0]
        assert record.emitted_at == sorted(record.emitted_at)
        assert engine.unidentified_connections() == 0

        engine.reset_telemetry()
        assert engine.records() == {}
        assert engine.unidentified_connections() == 0
    finally:
        engine.stop()


def test_sse_mock_counts_unidentified_connections_without_a_pfid() -> None:
    engine = MockStreamingEngine(
        chunk_ms=2.0, prefill_ms=0.0, default_chunks=3, num_loops=1
    ).start()
    try:
        # No PFID embedded: still served, but recorded under a negative key and
        # counted as unidentified (a measurement error the scorer can see).
        asyncio.run(_sse_stream(engine.port))
        records = engine.records()
        assert list(records.keys()) == [-1]
        assert records[-1].request_id == -1
        assert len(records[-1].emitted_at) == 3
        assert engine.unidentified_connections() == 1
    finally:
        engine.stop()


def test_sse_mock_reset_telemetry_and_idempotent_stop() -> None:
    engine = MockStreamingEngine(
        chunk_ms=2.0, prefill_ms=0.0, default_chunks=5, num_loops=1, store_bodies=True
    ).start()
    try:
        asyncio.run(_sse_stream(engine.port))
        assert engine.records()
        assert engine.request_bodies

        engine.reset_telemetry()
        assert engine.records() == {}
        assert engine.request_bodies == []
        assert engine.max_active_conns == 0
        assert engine.total_conns == 0
    finally:
        engine.stop()
        engine.stop()  # idempotent


# ---------------------------------------------------------------------------
# realtime TTS mock, driven by the real client
# ---------------------------------------------------------------------------


def test_realtime_audio_mock_drives_the_real_realtime_tts_client() -> None:
    num_chunks = 6
    chunk_bytes = 320
    server = MockRealtimeAudioServer(
        num_chunks=num_chunks,
        chunk_bytes=chunk_bytes,
        first_delta_ms=5.0,
        audio_chunk_ms=5.0,
        sample_rate=24000,
        num_loops=2,
    ).start()
    try:
        config = RealtimeTTSClientConfig(
            model="mock-realtime-tts",
            api_base=f"http://127.0.0.1:{server.port}",
            sample_rate=24000,
            pacing=TextPacingConfig(tokens_per_second=500.0),
        )
        client = RealtimeTTSClient(config)
        request = Request(
            id=1,
            channels={
                ChannelModality.TEXT: TextChannelRequestContent(
                    input_text="hello mock realtime world"
                )
            },
        )

        result = asyncio.run(client.send_request(request, session_id=0))

        assert result.success, result.error_msg
        metrics = result.channels[ChannelModality.AUDIO].metrics
        assert metrics[AudioMetricKey.CHUNK_COUNT.value] == num_chunks
        assert len(metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]) == num_chunks
        assert all(
            row[1] == chunk_bytes
            for row in metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]
        )
        assert metrics[AudioMetricKey.SESSION_READY_OFFSET_MS.value] is not None
        assert metrics[AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value] is not None
        assert len(result.channels[ChannelModality.AUDIO].content) == (
            num_chunks * chunk_bytes
        )

        # One emit stamp per audio delta in the paired record (no PFID in this
        # request's text, so it is keyed -1).
        assert len(server.records()[-1].emitted_at) == num_chunks
        assert server.total_conns == 1

        server.reset_telemetry()
        assert server.emit_offsets_ms == []
        assert server.records() == {}
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# STT mock
# ---------------------------------------------------------------------------


async def _paced_stt_session(
    port: int, num_appends: int, append_dt: float, pfid: Optional[int] = None
) -> tuple[List[float], str]:
    """Speak the vllm_realtime dialect with an absolute-deadline send pacer."""
    frame = json.dumps(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(b"\x00" * 640).decode("ascii"),
        }
    )
    send_offsets: List[float] = []
    deltas: List[str] = []
    final_text = ""

    async with websockets.connect(f"ws://127.0.0.1:{port}/v1/realtime") as ws:
        created = json.loads(await ws.recv())
        assert created["type"] == "session.created"
        update: dict = {"type": "session.update", "model": "mock"}
        if pfid is not None:
            update["veeksha_request_id"] = pfid
        await ws.send(json.dumps(update))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        start = time.monotonic()
        for i in range(num_appends):
            scheduled = start + i * append_dt
            delay = scheduled - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            send_offsets.append((time.monotonic() - start) * 1000.0)
            await ws.send(frame)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "transcription.delta":
                deltas.append(msg["delta"])
            elif msg["type"] == "transcription.done":
                final_text = msg["text"]
                break

    assert deltas, "server emitted no transcript deltas"
    return send_offsets, final_text


def test_stt_mock_records_one_arrival_per_append_tracking_the_sender() -> None:
    num_appends = 12
    append_dt = 0.010
    transcript = "one two three four"
    server = MockSTTPreflightServer(
        transcript=transcript,
        first_delta_ms=5.0,
        transcript_delta_ms=3.0,
        num_loops=1,
    ).start()
    try:
        send_offsets, final_text = asyncio.run(
            _paced_stt_session(server.port, num_appends, append_dt)
        )
        assert final_text == transcript

        timelines = server.append_timelines()
        assert len(timelines) == 1
        arrivals = timelines[0]
        # One stamp per append frame, taken before any JSON/base64 parse.
        assert len(arrivals) == num_appends
        assert arrivals[0] == 0.0
        # The arrival timeline tracks the paced sender: same cadence, and each
        # arrival lands within a few ms of the client's own send stamp.
        arrival_gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
        assert _median(arrival_gaps) == pytest.approx(append_dt * 1000.0, abs=4.0)
        for sent, arrived in zip(send_offsets, arrivals):
            assert abs(arrived - sent) < 20.0

        # Partial hypotheses STREAM during the audio rather than arriving in a
        # burst after EOF: the mock cycles the transcript at the configured
        # cadence until the client's final commit, so the emit count tracks the
        # clip duration, not the word count. This is the regime in which word
        # interactivity (audio-sent vs transcript-seen) means anything, and the
        # only one that exercises paced-send and timestamped-receive together.
        # The transcript session carried no PFID, so its record is keyed -1.
        emits = len(server.records()[-1].emitted_at)
        audio_span_ms = num_appends * append_dt * 1000.0
        assert emits > len(transcript.split()), emits
        assert emits == pytest.approx(audio_span_ms / 3.0, rel=0.5), emits
        assert server.total_conns == 1

        server.reset_telemetry()
        assert server.append_timelines() == []
        assert server.records() == {}
    finally:
        server.stop()


def test_stt_mock_keys_records_by_session_update_pfid() -> None:
    server = MockSTTPreflightServer(
        transcript="one two", first_delta_ms=3.0, transcript_delta_ms=3.0, num_loops=1
    ).start()
    try:
        asyncio.run(_paced_stt_session(server.port, 6, 0.006, pfid=8080))

        records = server.records()
        # (a) the id, echoed on session.update, round-trips.
        assert list(records.keys()) == [8080]
        record = records[8080]
        assert record.request_id == 8080
        # received_at (first frame = session.update) precedes both the first
        # append arrival and the first transcript emit.
        assert record.first_append_at is not None
        assert record.received_at <= record.first_append_at
        assert record.emitted_at
        assert record.received_at < record.emitted_at[0]
        assert len(record.append_at) == 6
        assert server.unidentified_connections() == 0
    finally:
        server.stop()


def test_stt_mock_counts_unidentified_without_a_pfid() -> None:
    server = MockSTTPreflightServer(
        transcript="a b", first_delta_ms=2.0, transcript_delta_ms=2.0, num_loops=1
    ).start()
    try:
        # No veeksha_request_id on session.update: served, but unidentified.
        asyncio.run(_paced_stt_session(server.port, 4, 0.004))
        records = server.records()
        assert list(records.keys()) == [-1]
        assert server.unidentified_connections() == 1
    finally:
        server.stop()


def test_stt_mock_handles_concurrent_connections_across_loops() -> None:
    server = MockSTTPreflightServer(
        transcript="a b c", first_delta_ms=2.0, transcript_delta_ms=2.0, num_loops=4
    ).start()
    try:

        async def _many():
            return await asyncio.gather(
                *(_paced_stt_session(server.port, 5, 0.004) for _ in range(6))
            )

        results = asyncio.run(_many())
        assert [text for _, text in results] == ["a b c"] * 6

        timelines = server.append_timelines()
        assert len(timelines) == 6
        assert all(len(t) == 5 for t in timelines)
        assert server.total_conns == 6
    finally:
        server.stop()
