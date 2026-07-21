"""Rate-based (open-loop) native dispatch: native owns the arrival schedule.

The most important gap this closes — at high QPS, request *arrival* timing is
itself timing-critical and drifts in Python. With dispatch offsets, native
launches each request on its arrival deadline (kernel-time send on schedule).
"""

from __future__ import annotations

import pytest

from veeksha.native.engine import native_available

pytestmark = pytest.mark.skipif(
    not native_available(), reason="veeksha_native extension not built"
)


def _mock_engine():
    from veeksha.preflight.mock_engine import MockStreamingEngine

    return MockStreamingEngine(
        chunk_dt=0.02, prefill_s=0.02, default_chunks=5, num_loops=8
    ).start()


def _text_request(rid):
    from veeksha.core.request import Request
    from veeksha.core.request_content import TextChannelRequestContent
    from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
    from veeksha.types import ChannelModality

    return Request(
        id=rid,
        channels={ChannelModality.TEXT: TextChannelRequestContent(input_text="hi")},
        requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=5)),
    )


def test_native_open_loop_dispatch_follows_arrival_schedule():
    from veeksha.native.engine import NativeReceiveEngine, NativeRequest

    engine = _mock_engine()
    try:
        native = NativeReceiveEngine("127.0.0.1", engine.port)
        n = 16
        body = '{"model":"d","stream":true,"max_completion_tokens":5}'
        reqs = [NativeRequest(path="/v1/chat/completions", body=body) for _ in range(n)]
        offsets = [0.04 * i for i in range(n)]  # 40ms arrivals
        results = native.run(
            reqs, concurrency=64, timeout_s=30.0, dispatch_offsets_s=offsets
        )
    finally:
        engine.stop()

    ok = [r for r in results if r.success]
    assert len(ok) == n
    # native launched each request on its arrival deadline: actual dispatch
    # offset tracks the schedule within a few ms (native owns arrival timing).
    drift = [abs(r.dispatch_offset_s - offsets[r.index]) for r in results]
    drift.sort()
    assert drift[len(drift) // 2] < 0.010  # median < 10ms
    assert max(drift) < 0.050


def test_compute_dispatch_offsets_rate_schedule():
    from veeksha.config.generator.interval import FixedIntervalGeneratorConfig
    from veeksha.config.traffic import ConcurrentTrafficConfig, RateTrafficConfig
    from veeksha.core.seeding import SeedManager
    from veeksha.native.runner import compute_dispatch_offsets

    # build 4 single-request sessions
    sessions = [_Session(i) for i in range(4)]
    rate = RateTrafficConfig(
        interval_generator=FixedIntervalGeneratorConfig(interval=0.1)
    )
    offsets = compute_dispatch_offsets(sessions, rate, SeedManager(0))
    # fixed 0.1s interval => arrivals at 0, 0.1, 0.2, 0.3
    assert offsets == pytest.approx([0.0, 0.1, 0.2, 0.3])
    # concurrent traffic => closed-loop, no schedule
    assert (
        compute_dispatch_offsets(sessions, ConcurrentTrafficConfig(), SeedManager(0))
        is None
    )


class _Session:
    def __init__(self, sid):
        self.id = sid
        self.requests = {0: _text_request(sid)}
        self.session_graph = None


def test_maybe_run_native_open_loop_and_multiturn_fallback():
    from veeksha.benchmark import _maybe_run_native
    from veeksha.config.benchmark import BenchmarkConfig
    from veeksha.config.client import OpenAIChatCompletionsClientConfig
    from veeksha.config.evaluator import PerformanceEvaluatorConfig
    from veeksha.config.generator.interval import FixedIntervalGeneratorConfig
    from veeksha.config.runtime import RuntimeConfig
    from veeksha.config.traffic import RateTrafficConfig
    from veeksha.core.seeding import SeedManager
    from veeksha.evaluator.performance.base import PerformanceEvaluator

    engine = _mock_engine()
    try:
        cfg = BenchmarkConfig(
            client=OpenAIChatCompletionsClientConfig(
                api_base=f"http://127.0.0.1:{engine.port}/v1",
                model="d",
                use_native_transport=True,
            ),
            traffic_scheduler=RateTrafficConfig(
                interval_generator=FixedIntervalGeneratorConfig(interval=0.03)
            ),
            runtime=RuntimeConfig(max_sessions=12),
        )
        evaluator = PerformanceEvaluator(
            PerformanceEvaluatorConfig(target_channels=["text"], stream_metrics=False)
        )
        sessions = [_Session(i) for i in range(12)]
        result, _ = _maybe_run_native(
            cfg, evaluator, None, sessions, seed_manager=SeedManager(0)
        )
        assert result is not None  # rate-based single-turn handled natively
        assert any("Time to First" in k for k in result.metrics)

        # multi-turn session -> native declines (Python handles history), and
        # the drawn sessions are handed back for the Python pipeline to reuse.
        multiturn = [_MultiTurnSession(0)]
        ev2 = PerformanceEvaluator(
            PerformanceEvaluatorConfig(target_channels=["text"], stream_metrics=False)
        )
        fallback_result, fallback_sessions = _maybe_run_native(
            cfg, ev2, None, multiturn, seed_manager=SeedManager(0)
        )
        assert fallback_result is None
        assert fallback_sessions == multiturn
    finally:
        engine.stop()


class _MultiTurnSession:
    def __init__(self, sid):
        self.id = sid
        self.requests = {0: _text_request(sid), 1: _text_request(sid + 100)}
        self.session_graph = None
