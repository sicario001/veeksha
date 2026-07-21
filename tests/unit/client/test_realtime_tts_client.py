"""Tests for the realtime (WebSocket) TTS client.

Protocol-shape units plus an end-to-end run against a mock realtime TTS server
that proves the client streams paced text deltas, collects audio deltas with a
per-chunk receive timeline, and emits the shared audio-contract metrics that the
AudioPerformanceEvaluator consumes. Needs `transformers` importable (client
package); runs in CI; locally it needs transformers importable (or a stub on
PYTHONPATH).
"""

from __future__ import annotations

import asyncio

from tests.helpers.mock_realtime_tts_server import MockRealtimeTTSServer
from veeksha.config.client import RealtimeTTSClientConfig
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.types import ChannelModality, ClientType


def _text_request(rid: int, text: str) -> Request:
    return Request(
        id=rid,
        channels={ChannelModality.TEXT: TextChannelRequestContent(input_text=text)},
    )


# ------------------------------------------------------------------ protocol units
def test_registered_under_client_type_realtime_tts():
    from veeksha.client.registry import ClientRegistry

    loader = ClientRegistry._registry[ClientType.REALTIME_TTS]
    assert loader is not None


def test_ws_url_and_classify():
    from veeksha.client.realtime_tts import RealtimeEventKind, _build_realtime_protocol

    cfg = RealtimeTTSClientConfig(api_base="http://host:9000", model="tts-1")
    proto = _build_realtime_protocol(cfg)
    url = proto.build_ws_url("http://host:9000")
    assert url.startswith("ws://host:9000/")
    assert "realtime" in url and "model=tts-1" in url
    # https -> wss
    assert proto.build_ws_url("https://host").startswith("wss://host/")
    # event classification maps the wire types
    assert (
        proto.classify("response.output_audio.delta") is RealtimeEventKind.AUDIO_DELTA
    )
    assert proto.classify("response.done") is RealtimeEventKind.RESPONSE_DONE
    assert proto.classify("unknown.type") is RealtimeEventKind.OTHER


def test_extract_audio_decodes_base64():
    import base64

    from veeksha.client.realtime_tts import _build_realtime_protocol

    cfg = RealtimeTTSClientConfig(api_base="http://h", model="m")
    proto = _build_realtime_protocol(cfg)
    payload = b"\x01\x02\x03\x04"
    event = {"delta": base64.b64encode(payload).decode()}
    assert proto.extract_audio(event) == payload
    assert proto.extract_audio({}) == b""


# ------------------------------------------------------------------ end to end
def test_realtime_tts_end_to_end_collects_audio_and_metrics():
    srv = MockRealtimeTTSServer(
        num_chunks=5,
        chunk_bytes=4800,
        first_delta_delay=0.03,
        delta_dt=0.02,
        sample_rate=24000,
    ).start()
    try:
        from veeksha.client.realtime_tts import RealtimeTTSClient

        cfg = RealtimeTTSClientConfig(
            api_base=f"http://127.0.0.1:{srv.port}",
            model="tts-1",
            sample_rate=24000,
        )
        client = RealtimeTTSClient(cfg)

        dispatched = []
        sent = []
        req = _text_request(1, "hello world this is a realtime speech test")
        result = asyncio.run(
            client.send_request(
                req,
                session_id=1,
                on_request_dispatched=lambda: dispatched.append(1),
                on_request_sent=lambda: sent.append(1),
            )
        )

        assert result.success, result.error_msg
        # dispatch (HTTP-200 analog) and sent callbacks each fired at least once
        assert dispatched and sent
        channel = result.channels[ChannelModality.AUDIO]
        metrics = channel.metrics
        # 5 audio deltas of 4800 bytes each => 24000 bytes of PCM
        assert metrics[AudioMetricKey.CHUNK_COUNT.value] == 5
        assert len(channel.content) == 5 * 4800
        # per-chunk receive timeline present (the streaming-RTF primitive)
        timeline = metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]
        assert len(timeline) == 5
        assert all(len(row) == 2 for row in timeline)  # [offset_ms, n_bytes]
        # TTFC is the first-chunk offset and positive
        assert metrics[AudioMetricKey.TTFC.value] > 0
        assert metrics[AudioMetricKey.SAMPLE_RATE.value] == 24000
        # text delta timeline recorded (paced input)
        assert len(metrics[AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value]) >= 1
    finally:
        srv.stop()


def test_realtime_tts_metrics_feed_audio_evaluator():
    """The realtime timeline drives the AudioPerformanceEvaluator's streaming RTF."""
    srv = MockRealtimeTTSServer(num_chunks=6, chunk_bytes=4800).start()
    try:
        from veeksha.client.realtime_tts import RealtimeTTSClient
        from veeksha.config.evaluator import PerformanceEvaluatorConfig
        from veeksha.evaluator.performance.audio import AudioPerformanceEvaluator

        cfg = RealtimeTTSClientConfig(
            api_base=f"http://127.0.0.1:{srv.port}", model="tts-1", sample_rate=24000
        )
        client = RealtimeTTSClient(cfg)
        result = asyncio.run(
            client.send_request(_text_request(1, "one two three four five"), 1)
        )
        assert result.success

        ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
        ev.register_request(1, 1, 0.0, None)
        ev.record_request_completed(1, 1, 1.0, result)
        m = ev.finalize().metrics
        assert m["num_completed_requests"] == 1
        assert "Time to First Audio (Mean)" in m
        assert "Streaming Real Time Factor (Mean)" in m
    finally:
        srv.stop()
