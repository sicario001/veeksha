"""Tests for the real per-request native receive engine.

Skipped unless the veeksha_native extension has been built.
"""

from __future__ import annotations

import pytest

from veeksha.native.engine import NativeReceiveEngine, NativeRequest, native_available

pytestmark = pytest.mark.skipif(
    not native_available(), reason="veeksha_native extension not built"
)


def _mock_engine(num_chunks=8):
    from veeksha.preflight.mock_engine import MockStreamingEngine

    return MockStreamingEngine(
        chunk_dt=0.02, prefill_s=0.03, default_chunks=num_chunks, num_loops=2
    ).start()


def _chat_request(num_chunks: int) -> NativeRequest:
    body = f'{{"model":"d","stream":true,"max_completion_tokens":{num_chunks}}}'
    return NativeRequest(path="/v1/chat/completions", body=body)


def test_native_engine_returns_content_and_timeline():
    engine = _mock_engine(num_chunks=8)
    try:
        native = NativeReceiveEngine("127.0.0.1", engine.port)
        results = native.run([_chat_request(8) for _ in range(10)], concurrency=5)
    finally:
        engine.stop()

    assert len(results) == 10
    ok = [r for r in results if r.success]
    assert len(ok) == 10
    r0 = ok[0]
    assert r0.status == 200
    # one TimedEventStream per request, 8 chunks, offsets in ascending seconds
    assert len(r0.stream) == 8
    offsets = [e.offset_s for e in r0.stream.events]
    assert offsets == sorted(offsets)
    assert r0.stream.time_to_first_event() > 0
    # content is the concatenated SSE data payloads
    assert "delta" in r0.content


def test_native_engine_timeline_matches_known_cadence():
    """The native per-chunk timeline reproduces the engine's 20ms cadence."""
    engine = _mock_engine(num_chunks=12)
    try:
        native = NativeReceiveEngine("127.0.0.1", engine.port)
        results = native.run([_chat_request(12) for _ in range(8)], concurrency=4)
    finally:
        engine.stop()

    drifts = []
    for r in results:
        if not r.success or len(r.stream) < 2:
            continue
        deltas = r.stream.inter_event_deltas()
        drifts.extend(abs(d - 0.02) for d in deltas)  # 20ms cadence
    assert drifts
    # localhost native receive: median inter-chunk error is small
    drifts.sort()
    p50 = drifts[len(drifts) // 2]
    assert p50 < 0.010  # < 10ms


def test_native_request_wire_format():
    req = NativeRequest(path="/v1/x", body='{"a":1}')
    wire = req.to_wire("example.com")
    assert wire.startswith("POST /v1/x HTTP/1.1\r\n")
    assert "Host: example.com\r\n" in wire
    assert "Content-Length: 7\r\n" in wire
    assert wire.endswith('\r\n\r\n{"a":1}')


def _preflight_config(**kw):
    from veeksha.config.preflight import PreflightCheckConfig

    base = dict(
        target_concurrency=100,
        num_client_threads=2,
        num_dispatcher_threads=1,
        num_completion_threads=1,
        num_chunks=20,
        budget_s=2.0,
        check_pacing=False,
    )
    base.update(kw)
    return PreflightCheckConfig(**base)


def test_native_batch_receive_drift_is_honest():
    """The real native engine records the cadence with low per-chunk drift —
    measured through the SAME preflight harness the Python path uses, so a
    passing native number cannot come from an easier measurement."""
    from veeksha.preflight.drivers import NATIVE, TextWorkload, measure_rung

    engine = _mock_engine(num_chunks=20)
    try:
        workload = TextWorkload(
            engine, _preflight_config(num_chunks=20), chunk_dt=0.02, prefill_s=0.02
        )
        point = measure_rung(workload, NATIVE, 100, 20.0, 1.05, n_requests=200)[
            workload.name
        ]
    finally:
        engine.stop()
    assert point.achieved >= 95  # native owns concurrency; ~all connect
    assert point.ivl_err_p99_ms < 20.0  # honest to the 20ms cadence
