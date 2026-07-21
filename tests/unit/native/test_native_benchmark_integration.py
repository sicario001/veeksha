"""The benchmark-level native branch (_maybe_run_native) runs a bounded batch
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
        result, _ = _maybe_run_native(cfg, evaluator, None, sessions)
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
    # flag off -> None (Python pipeline handles it); sessions passed through
    result, sessions = _maybe_run_native(cfg, object(), None, [])
    assert result is None
    assert sessions == []


def _multiturn_session(sid, turns=3):
    from veeksha.core.request import Request
    from veeksha.core.request_content import TextChannelRequestContent
    from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
    from veeksha.core.session import Session
    from veeksha.core.session_graph import (
        SessionEdge,
        SessionGraph,
        SessionNode,
        add_edge,
        add_node,
    )
    from veeksha.types import ChannelModality

    graph = SessionGraph()
    requests = {}
    for k in range(turns):
        add_node(graph, SessionNode(id=k, wait_after_ready=0.0))
        if k:
            add_edge(graph, SessionEdge(src=k - 1, dst=k, is_history_parent=True))
        requests[k] = Request(
            id=sid * 100 + k,
            channels={
                ChannelModality.TEXT: TextChannelRequestContent(input_text=f"turn {k}")
            },
            requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=6)),
        )
    return Session(id=sid, session_graph=graph, requests=requests)


def test_maybe_run_native_linear_multiturn_runs_natively():
    """Linear text conversations route through the native chains engine, with
    history spliced natively between turns (verified via server body capture)."""
    import json

    from veeksha.benchmark import _maybe_run_native
    from veeksha.config.benchmark import BenchmarkConfig
    from veeksha.config.client import OpenAIChatCompletionsClientConfig
    from veeksha.config.evaluator import PerformanceEvaluatorConfig
    from veeksha.config.runtime import RuntimeConfig
    from veeksha.config.traffic import ConcurrentTrafficConfig
    from veeksha.evaluator.performance.base import PerformanceEvaluator
    from veeksha.preflight.mock_engine import MockStreamingEngine

    engine = MockStreamingEngine(
        chunk_dt=0.005,
        prefill_s=0.005,
        default_chunks=6,
        num_loops=2,
        store_bodies=True,
    ).start()
    try:
        cfg = BenchmarkConfig(
            client=OpenAIChatCompletionsClientConfig(
                api_base=f"http://127.0.0.1:{engine.port}/v1",
                model="d",
                use_native_transport=True,
            ),
            traffic_scheduler=ConcurrentTrafficConfig(target_concurrent_sessions=4),
            runtime=RuntimeConfig(max_sessions=6),
        )
        evaluator = PerformanceEvaluator(
            PerformanceEvaluatorConfig(target_channels=["text"], stream_metrics=False)
        )
        sessions = [_multiturn_session(i) for i in range(6)]
        result, _ = _maybe_run_native(cfg, evaluator, None, sessions)
    finally:
        engine.stop()

    assert result is not None  # chains path handled the run natively
    assert any("Time to First" in k for k in result.metrics)
    # history flowed: some captured request body carries an assistant message
    multi = [json.loads(b) for b in engine.request_bodies if b"assistant" in b]
    assert multi, "no request carried injected history"
    assert all(
        m["messages"][1]["content"]  # assistant slot filled with real output
        for m in multi
    )
