"""P3: native WebSocket receive + timer-wheel send pacing.

Drives the native WS engine against the Python mock realtime server (real
websockets handshake + framing), validating per-frame receive timing and
per-message paced-send precision.
"""

from __future__ import annotations

import json

import pytest

from veeksha.native.engine import NativeWsEngine, native_available

pytestmark = pytest.mark.skipif(
    not native_available(), reason="veeksha_native extension not built"
)


def _server(num_chunks=8, delta_dt=0.02):
    from tests.helpers.mock_realtime_tts_server import MockRealtimeTTSServer

    return MockRealtimeTTSServer(
        num_chunks=num_chunks,
        chunk_bytes=2400,
        first_delta_delay=0.02,
        delta_dt=delta_dt,
    ).start()


def test_native_ws_receives_frames_with_timeline():
    srv = _server(num_chunks=8)
    try:
        engine = NativeWsEngine("127.0.0.1", srv.port)
        init = [
            json.dumps({"type": "session.update"}),
            json.dumps({"type": "response.create"}),
        ]
        results = engine.stream("/v1/realtime", init, concurrency=4, timeout_s=3.0)
    finally:
        srv.stop()

    assert len(results) == 4
    ok = [r for r in results if r.success]
    assert len(ok) == 4
    r0 = ok[0]
    # session.updated + response.created + 8 audio deltas + audio.done + response.done
    assert len(r0.stream) >= 10
    assert "output_audio.delta" in r0.content
    # frames arrive in monotonic order with a positive first-frame offset
    offsets = [e.offset_s for e in r0.stream.events]
    assert offsets == sorted(offsets)
    assert r0.stream.time_to_first_event() > 0


def test_native_ws_timer_wheel_send_pacing_is_precise():
    srv = _server(num_chunks=4)
    try:
        engine = NativeWsEngine("127.0.0.1", srv.port)
        # session.update at t=0, 4 paced deltas every 30ms, response.create last
        msgs = [json.dumps({"type": "session.update"})]
        offsets = [0.0]
        for i in range(4):
            msgs.append(json.dumps({"type": "conversation.item.create", "i": i}))
            offsets.append(0.030 * (i + 1))
        msgs.append(json.dumps({"type": "response.create"}))
        offsets.append(0.030 * 5)
        results = engine.stream(
            "/v1/realtime", msgs, concurrency=2, send_offsets_s=offsets, timeout_s=3.0
        )
    finally:
        srv.stop()

    r = results[0]
    assert len(r.sent_offsets_s) == len(offsets)
    # each paced message dispatched close to its deadline (timer-wheel precision)
    drift = [abs(a - s) for a, s in zip(r.sent_offsets_s, offsets)]
    assert max(drift) < 0.020  # < 20ms per-dispatch send drift
