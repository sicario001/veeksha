"""End-to-end paired-timestamp plumbing: real client <-> mock server.

The preflight scores the harness by comparing, per event, a stamp the CLIENT
took against a stamp the mock SERVER took. Both run on one host reading
``time.monotonic()``, so the two are directly comparable. These tests drive each
mock with its REAL client and assert the plumbing that makes that comparison
possible:

* the preflight id embedded in the request round-trips into ``records()``;
* the server ``received_at`` precedes its first ``emitted_at`` (it cannot send a
  response before receiving the request);
* absolute client event times, reconstructed from the new anchors + the
  existing offsets, land on the correct side of the matching server stamp with
  a strictly positive delivery lag;
* on loopback that lag is small (a generous bound -- this pins plumbing
  correctness, not machine fidelity).
"""

from __future__ import annotations

import asyncio

import pytest

from veeksha.client.openai_chat import OpenAIChatCompletionsClient
from veeksha.client.realtime_tts import RealtimeTTSClient
from veeksha.client.stt import VllmRealtimeSTTClient, _ClipAssets
from veeksha.client.vajra_tts_stream import VajraTTSStreamClient
from veeksha.config.client import (
    OpenAIChatCompletionsClientConfig,
    RealtimeTTSClientConfig,
    STTClientConfig,
    TextPacingConfig,
    VajraTTSStreamClientConfig,
)
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import (
    AudioChannelRequestContent,
    TextChannelRequestContent,
)
from veeksha.core.tokenizer import TokenizerHandle, TokenizerProvider
from veeksha.preflight.audio_server import (
    MockRealtimeAudioServer,
    MockSTTPreflightServer,
)
from veeksha.preflight.mock_engine import MockStreamingEngine
from veeksha.preflight.sharded_server import format_pfid
from veeksha.preflight.vajra_server import MockVajraTTSStreamServer
from veeksha.types import ChannelModality

pytestmark = pytest.mark.unit

# Generous loopback lag bound. This test certifies plumbing (the two stamps
# refer to the same event and the arithmetic reconstructs correctly), not
# machine fidelity, so the bound only has to exclude a mis-paired stamp.
_MAX_LOOPBACK_LAG_MS = 50.0


def _assert_positive_small_lags(lags_ms, *, label: str) -> None:
    assert lags_ms, f"{label}: no paired events to compare"
    for lag in lags_ms:
        # Strictly positive: the client event genuinely happened after the
        # server stamp it is paired with (delivery lag is never negative).
        assert lag > 0.0, f"{label}: non-positive lag {lag:.3f} ms"
        assert lag < _MAX_LOOPBACK_LAG_MS, f"{label}: lag {lag:.3f} ms too large"


# ---------------------------------------------------------------------------
# text SSE  (receive direction: server emit -> client receive)
# ---------------------------------------------------------------------------


def _chat_client(api_base: str) -> OpenAIChatCompletionsClient:
    handle = TokenizerHandle(
        count_tokens=lambda text: len(str(text).split()),
        decode=lambda ids: "",
        encode=lambda text: [0] * len(str(text).split()),
    )
    config = OpenAIChatCompletionsClientConfig(
        api_base=api_base, api_key="preflight", model="mock"
    )
    return OpenAIChatCompletionsClient(
        config=config,
        tokenizer_provider=TokenizerProvider({ChannelModality.TEXT: handle}),
    )


def test_text_sse_pairs_client_receive_against_server_emit() -> None:
    request_id = 4242
    engine = MockStreamingEngine(
        chunk_ms=5.0, prefill_ms=10.0, default_chunks=12, num_loops=1
    ).start()
    try:
        client = _chat_client(f"http://127.0.0.1:{engine.port}/v1")
        request = Request(
            id=request_id,
            channels={
                ChannelModality.TEXT: TextChannelRequestContent(
                    input_text=f"{format_pfid(request_id)} say something"
                )
            },
        )
        result = asyncio.run(client.send_request(request, session_id=0))
        assert result.success, result.error_msg

        records = engine.records()
        # (a) the id round-trips.
        assert request_id in records, records.keys()
        assert engine.unidentified_connections() == 0
        record = records[request_id]

        # (b) the mock received the request before it emitted anything.
        assert record.emitted_at
        assert record.received_at < record.emitted_at[0]

        metrics = result.channels[ChannelModality.TEXT].metrics
        start = metrics["request_start_monotonic"]
        gaps = metrics["inter_chunk_times"]
        assert len(gaps) == len(record.emitted_at)

        # (c)+(d) reconstruct absolute client arrival of each chunk and pair it
        # with the server's send stamp for that chunk.
        running = 0.0
        lags = []
        for gap, emitted in zip(gaps, record.emitted_at):
            running += gap
            arrival = start + running
            lags.append((arrival - emitted) * 1000.0)
        _assert_positive_small_lags(lags, label="text")
    finally:
        engine.stop()


# ---------------------------------------------------------------------------
# realtime TTS  (receive direction: audio delta emit -> client receive)
# ---------------------------------------------------------------------------


def _pair_audio_receive(metrics, record, *, label: str) -> None:
    start = metrics["request_start_monotonic"]
    stamps = metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]
    assert record.emitted_at
    assert record.received_at < record.emitted_at[0]

    # C2 request delivery: the client's request-level send stamp is the first
    # application frame (session.update / session.config), which is exactly the
    # frame the mock stamps received_at against. The mock receives it AFTER the
    # client sent it -> positive delivery lag. (This is the defect the fix
    # targets: stamping the first paced frame instead made this negative.)
    request_sent = metrics["request_sent_monotonic"]
    assert isinstance(request_sent, float)
    c2_lag = (record.received_at - request_sent) * 1000.0
    _assert_positive_small_lags([c2_lag], label=f"{label} C2")

    # C3 response delivery: the client may stop on response.done before draining
    # the very last emit, so pair only the chunks both sides saw.
    pairs = min(len(stamps), len(record.emitted_at))
    assert pairs >= 2
    lags = [
        (start + stamps[i][0] / 1000.0 - record.emitted_at[i]) * 1000.0
        for i in range(pairs)
    ]
    _assert_positive_small_lags(lags, label=label)


def test_realtime_tts_pairs_client_receive_against_server_emit() -> None:
    request_id = 77
    server = MockRealtimeAudioServer(
        num_chunks=8,
        chunk_bytes=320,
        first_delta_ms=5.0,
        audio_chunk_ms=5.0,
        num_loops=1,
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
            id=request_id,
            channels={
                ChannelModality.TEXT: TextChannelRequestContent(
                    input_text=f"{format_pfid(request_id)} hello realtime"
                )
            },
        )
        result = asyncio.run(client.send_request(request, session_id=0))
        assert result.success, result.error_msg

        records = server.records()
        assert request_id in records, records.keys()
        assert server.unidentified_connections() == 0
        _pair_audio_receive(
            result.channels[ChannelModality.AUDIO].metrics,
            records[request_id],
            label="realtime_tts",
        )
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Vajra streaming TTS  (receive direction: binary PCM emit -> client receive)
# ---------------------------------------------------------------------------


def test_vajra_tts_pairs_client_receive_against_server_emit() -> None:
    request_id = 913
    server = MockVajraTTSStreamServer(
        num_chunks=8,
        chunk_bytes=320,
        first_delta_ms=5.0,
        audio_chunk_ms=5.0,
        num_loops=1,
    ).start()
    try:
        config = VajraTTSStreamClientConfig(
            model="mock-vajra-tts",
            api_base=f"http://127.0.0.1:{server.port}",
            sample_rate=24000,
            pacing=TextPacingConfig(tokens_per_second=500.0),
        )
        client = VajraTTSStreamClient(config)
        request = Request(
            id=request_id,
            channels={
                ChannelModality.TEXT: TextChannelRequestContent(
                    input_text=f"{format_pfid(request_id)} hello vajra"
                )
            },
        )
        result = asyncio.run(client.send_request(request, session_id=0))
        assert result.success, result.error_msg

        records = server.records()
        assert request_id in records, records.keys()
        assert server.unidentified_connections() == 0
        _pair_audio_receive(
            result.channels[ChannelModality.AUDIO].metrics,
            records[request_id],
            label="vajra_tts",
        )
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# STT  (send direction: client send -> server append arrival)
# ---------------------------------------------------------------------------


def test_stt_pairs_client_send_against_server_append_arrival(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    request_id = 5150
    sample_rate = 16000
    chunk_bytes = 1600  # 50 ms of PCM16 at 16 kHz.
    num_chunks = 6
    server = MockSTTPreflightServer(
        transcript="one two three",
        first_delta_ms=5.0,
        transcript_delta_ms=5.0,
        num_loops=1,
    ).start()
    try:
        config = STTClientConfig(
            provider="vllm_realtime",
            model="mock-stt",
            api_base=f"http://127.0.0.1:{server.port}",
            sample_rate=sample_rate,
            ws_chunk_size=chunk_bytes,
            ws_realtime_pacing=True,
            ws_ping_interval_s=None,
        )
        client = VllmRealtimeSTTClient(config)

        # Silence clip; only the decode source is faked, so pacing, encoding,
        # and the wire path are the real client's. Correlation rides in the
        # session.update field, not the audio.
        pcm = b"\x00" * (chunk_bytes * num_chunks)
        monkeypatch.setattr(
            client,
            "_clip_assets",
            lambda _path: _ClipAssets(
                pcm=pcm,
                wire_messages=[
                    client._encode_chunk(pcm[o : o + chunk_bytes])
                    for o in range(0, len(pcm), chunk_bytes)
                ],
            ),
        )

        # The client stats the path before decoding; _clip_assets (patched
        # above) supplies the PCM, so the file contents are irrelevant. The
        # preflight id is carried in metadata -- the driver's mechanism -- which
        # the client echoes into session.update as veeksha_request_id.
        audio_path = tmp_path / "clip.wav"
        audio_path.write_bytes(b"\x00" * 64)
        request = Request(
            id=request_id,
            channels={
                ChannelModality.AUDIO: AudioChannelRequestContent(
                    input_audio=str(audio_path)
                )
            },
            metadata={"dataset": "preflight", "preflight_pfid": request_id},
        )
        result = asyncio.run(client.send_request(request, session_id=0))
        assert result.success, result.error_msg

        records = server.records()
        # (a) the id, echoed on session.update, round-trips with 0% unidentified.
        assert request_id in records, records.keys()
        assert server.unidentified_connections() == 0
        record = records[request_id]

        # (b) the mock received the connection before it emitted any transcript,
        # and the first append arrival was stamped.
        assert record.first_append_at is not None
        assert record.emitted_at
        assert record.received_at <= record.first_append_at
        assert record.received_at < record.emitted_at[0]

        metrics = result.channels[ChannelModality.AUDIO].metrics

        # C2 request delivery: the session.update send stamp precedes the mock's
        # received_at (its first received frame) -> positive lag.
        request_sent = metrics["request_sent_monotonic"]
        assert isinstance(request_sent, float)
        _assert_positive_small_lags(
            [(record.received_at - request_sent) * 1000.0], label="stt C2 request"
        )

        anchor = metrics["audio_started_monotonic"]
        send_offsets = metrics["send_offsets_ms"]
        assert len(record.append_at) == len(send_offsets) == num_chunks

        # (c)+(d) per-append C2: client sent chunk i at anchor + send_offsets[i]
        # /1000; the server stamped its arrival at append_at[i]. The append lands
        # AFTER the send with a small positive delivery lag.
        lags = [
            (record.append_at[i] - (anchor + send_offsets[i] / 1000.0)) * 1000.0
            for i in range(num_chunks)
        ]
        _assert_positive_small_lags(lags, label="stt append")
    finally:
        server.stop()
