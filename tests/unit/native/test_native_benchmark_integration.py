"""N3: the benchmark-level native branch (_maybe_run_native) runs a bounded batch
over the native transport and returns the finalized EvaluationResult.

This is the wiring that makes `client.use_native_transport = True` take effect in
a real benchmark run (for plaintext, bounded, independent-session workloads).
"""

from __future__ import annotations

import pytest

from veeksha.native.engine import native_available

pytestmark = pytest.mark.skipif(
    not native_available(), reason="veeksha_native extension not built"
)


def _text_session(sid):
    from veeksha.core.request import Request
    from veeksha.core.request_content import TextChannelRequestContent
    from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
    from veeksha.core.session import Session
    from veeksha.core.session_graph import SessionGraph, SessionNode, add_node
    from veeksha.types import ChannelModality

    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    req = Request(
        id=sid,
        channels={ChannelModality.TEXT: TextChannelRequestContent(input_text="hi")},
        requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=10)),
    )
    return Session(id=sid, session_graph=graph, requests={0: req})


def test_maybe_run_native_text_bounded_batch():
    from veeksha.benchmark import _maybe_run_native
    from veeksha.config.benchmark import BenchmarkConfig
    from veeksha.config.client import OpenAIChatCompletionsClientConfig
    from veeksha.config.evaluator import PerformanceEvaluatorConfig
    from veeksha.config.runtime import RuntimeConfig
    from veeksha.config.traffic import ConcurrentTrafficConfig
    from veeksha.evaluator.performance.base import PerformanceEvaluator
    from veeksha.preflight.mock_engine import MockStreamingEngine

    engine = MockStreamingEngine(
        chunk_dt=0.02, prefill_s=0.03, default_chunks=10, num_loops=4
    ).start()
    try:
        cfg = BenchmarkConfig(
            client=OpenAIChatCompletionsClientConfig(
                api_base=f"http://127.0.0.1:{engine.port}/v1",
                model="d",
                use_native_transport=True,
            ),
            traffic_scheduler=ConcurrentTrafficConfig(target_concurrent_sessions=6),
            runtime=RuntimeConfig(max_sessions=12),
        )
        evaluator = PerformanceEvaluator(
            PerformanceEvaluatorConfig(target_channels=["text"], stream_metrics=False)
        )
        sessions = [_text_session(i) for i in range(12)]
        result = _maybe_run_native(cfg, evaluator, None, sessions)
    finally:
        engine.stop()

    assert result is not None  # native path handled the run
    assert any("Time to First" in k for k in result.metrics)


def test_maybe_run_native_returns_none_when_flag_off():
    from veeksha.benchmark import _maybe_run_native
    from veeksha.config.benchmark import BenchmarkConfig
    from veeksha.config.client import OpenAIChatCompletionsClientConfig
    from veeksha.config.runtime import RuntimeConfig

    cfg = BenchmarkConfig(
        client=OpenAIChatCompletionsClientConfig(
            api_base="http://127.0.0.1:9/v1", model="d", use_native_transport=False
        ),
        runtime=RuntimeConfig(max_sessions=4),
    )
    # flag off -> None (Python pipeline handles it)
    assert _maybe_run_native(cfg, object(), None, []) is None
