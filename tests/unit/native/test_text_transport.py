"""P2 integration: veeksha text requests -> native transport -> text evaluator.

Proves the native engine drives real veeksha Request objects and produces
RequestResults the existing TextPerformanceEvaluator scores unchanged.
"""

from __future__ import annotations

import pytest

from veeksha.native.engine import native_available

pytestmark = pytest.mark.skipif(
    not native_available(), reason="veeksha_native extension not built"
)


def _mock_engine(num_chunks=10):
    from veeksha.preflight.mock_engine import MockStreamingEngine

    return MockStreamingEngine(
        chunk_dt=0.02, prefill_s=0.03, default_chunks=num_chunks, num_loops=4
    ).start()


def _text_request(rid: int, tokens: int):
    from veeksha.core.request import Request
    from veeksha.core.request_content import TextChannelRequestContent
    from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
    from veeksha.types import ChannelModality

    return Request(
        id=rid,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(input_text="hello world")
        },
        requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=tokens)),
    )


def test_text_requests_run_through_native_into_text_evaluator():
    from veeksha.config.evaluator import PerformanceEvaluatorConfig
    from veeksha.evaluator.performance.text import TextPerformanceEvaluator
    from veeksha.native.text_transport import run_text_requests
    from veeksha.types import ChannelModality

    engine = _mock_engine(num_chunks=10)
    try:
        requests = [_text_request(i, tokens=10) for i in range(12)]
        results = run_text_requests(
            requests, "127.0.0.1", engine.port, concurrency=6, model="d"
        )
    finally:
        engine.stop()

    assert len(results) == 12
    ok = [r for r in results if r.success]
    assert len(ok) == 12

    # the TextPerformanceEvaluator scores the native-produced results unchanged
    ev = TextPerformanceEvaluator(PerformanceEvaluatorConfig())
    for r in ok:
        ev.register_request(
            r.request_id, r.session_id, 0.0, r.channels[ChannelModality.TEXT]
        )
        ev.record_request_completed(r.request_id, r.session_id, 1.0, r)
    m = ev.finalize().metrics
    # TTFT + per-output-token timing surfaced from the native timeline
    assert any("Time to First" in k or "TTFT" in k for k in m)
    assert ok[0].channels[ChannelModality.TEXT].metrics["num_output_tokens"] == 10
