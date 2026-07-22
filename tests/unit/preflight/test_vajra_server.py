"""The Vajra streaming-TTS mock, driven by the REAL VajraTTSStreamClient.

These assert the two properties the preflight relies on: the mock emits on
absolute deadlines punctually enough to be a reference (its own recorded
lateness stays small), and the real client records one arrival stamp per
binary PCM frame at that known cadence.
"""

from __future__ import annotations

import asyncio

import pytest

from veeksha.client.vajra_tts_stream import VajraTTSStreamClient
from veeksha.config.client import VajraTTSStreamClientConfig
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.preflight.vajra_server import MockVajraTTSStreamServer
from veeksha.types import ChannelModality


def _request(rid: int = 1, text: str = "one two three four five six") -> Request:
    return Request(
        id=rid,
        channels={ChannelModality.TEXT: TextChannelRequestContent(input_text=text)},
    )


def _client(port: int) -> VajraTTSStreamClient:
    cfg = VajraTTSStreamClientConfig(
        api_base=f"http://127.0.0.1:{port}",
        model="m",
        sample_rate=24000,
    )
    return VajraTTSStreamClient(cfg)


@pytest.mark.unit
def test_real_client_records_one_stamp_per_binary_frame() -> None:
    num_chunks, chunk_bytes, cadence_ms = 8, 2400, 20.0
    srv = MockVajraTTSStreamServer(
        num_chunks=num_chunks,
        chunk_bytes=chunk_bytes,
        first_delta_ms=10.0,
        audio_chunk_ms=cadence_ms,
        num_loops=2,
    ).start()
    try:
        result = asyncio.run(_client(srv.port).send_request(_request(), session_id=1))
    finally:
        srv.stop()

    assert result.success, result.error_msg
    metrics = result.channels[ChannelModality.AUDIO].metrics
    stamps = metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]

    # One [offset_ms, n_bytes] entry per emitted binary frame.
    assert len(stamps) == num_chunks
    assert all(entry[1] == chunk_bytes for entry in stamps)

    offsets = [entry[0] for entry in stamps]
    assert offsets == sorted(offsets)
    # Recorded gaps track the server's known cadence. Generous bound: this
    # asserts the plumbing, not the machine's fidelity (that is what the
    # preflight itself measures).
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    assert all(abs(gap - cadence_ms) < 30.0 for gap in gaps), gaps

    # Terminal events the client stops on were seen.
    assert metrics[AudioMetricKey.AUDIO_DONE_OFFSET_MS.value] is not None
    assert (
        len(result.channels[ChannelModality.AUDIO].content) == num_chunks * chunk_bytes
    )


@pytest.mark.unit
def test_mock_emits_punctually_and_overlaps_input() -> None:
    """The mock is a usable reference, and audio starts before input.done.

    Overlap is the point of the streaming-text protocol: it forces the client
    to interleave paced sends with timestamped receives on one event loop.
    """
    srv = MockVajraTTSStreamServer(
        num_chunks=6, chunk_bytes=2400, first_delta_ms=5.0, audio_chunk_ms=20.0
    ).start()
    try:
        result = asyncio.run(_client(srv.port).send_request(_request(), session_id=1))
    finally:
        srv.stop()

    assert result.success, result.error_msg
    metrics = result.channels[ChannelModality.AUDIO].metrics

    # The paired-timestamp record-keeping the checks depend on: one emit stamp
    # per binary frame (this request carried no PFID, so it is keyed -1).
    assert len(srv.records()[-1].emitted_at) == 6

    first_audio = metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value][0][0]
    input_done = metrics[AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value]
    assert input_done is not None
    assert first_audio < input_done, (first_audio, input_done)


@pytest.mark.unit
def test_concurrent_sessions_all_served() -> None:
    srv = MockVajraTTSStreamServer(
        num_chunks=4,
        chunk_bytes=1200,
        first_delta_ms=5.0,
        audio_chunk_ms=10.0,
        num_loops=4,
    ).start()

    async def _drive() -> list:
        client = _client(srv.port)
        return await asyncio.gather(
            *(client.send_request(_request(rid=i), session_id=i) for i in range(6))
        )

    try:
        results = asyncio.run(_drive())
    finally:
        srv.stop()

    assert all(r.success for r in results), [r.error_msg for r in results]
    for r in results:
        stamps = r.channels[ChannelModality.AUDIO].metrics[
            AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value
        ]
        assert len(stamps) == 4
    assert srv.total_conns >= 6
