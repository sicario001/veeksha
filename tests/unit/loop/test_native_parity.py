"""Cross-native loop parity: the same seed + config + workload run through the
Python and native main loops produce the same evaluator summary metric KEY
sets, the exact same request/session counts and prompt/output token counts,
and dispatch schedules that agree within tolerance (RATE offsets: seeded
draws identical by construction — see the wire-fidelity chain test — and
actual dispatch offsets within 50 ms).
"""

from __future__ import annotations

import dataclasses
import time

import pytest

pytest.importorskip(
    "veeksha.native.veeksha_native",
    reason="veeksha_native extension not built (run veeksha/native/build.sh)",
)

# Warm heavyweight imports up front: the first PythonMainLoop.start() would
# otherwise spend hundreds of ms importing while its RATE schedule is
# already running, collapsing the arrival spacing this test measures.
import veeksha.client.registry  # noqa: F401, E402
import veeksha.traffic.registry  # noqa: F401, E402
import veeksha.workers.client_runner  # noqa: F401, E402
import veeksha.workers.completion  # noqa: F401, E402
import veeksha.workers.dispatch  # noqa: F401, E402
import veeksha.workers.prefetch  # noqa: F401, E402

from tests.unit.loop.native_test_utils import (  # noqa: E402
    drain_loop_until_idle,
    make_linear_text_session,
    make_text_request,
    make_text_session,
    word_split_provider,
)
from tests.unit.native.mock_servers import SseChatServer  # noqa: E402
from veeksha.config.client import OpenAIChatCompletionsClientConfig  # noqa: E402
from veeksha.config.evaluator import PerformanceEvaluatorConfig  # noqa: E402
from veeksha.config.generator.interval import (  # noqa: E402
    FixedIntervalGeneratorConfig,
)
from veeksha.config.runtime import RuntimeConfig  # noqa: E402
from veeksha.config.traffic import RateTrafficConfig  # noqa: E402
from veeksha.core.seeding import SeedManager  # noqa: E402
from veeksha.loop import MainLoopConfig, create_main_loop  # noqa: E402
from veeksha.loop.interface import LoopEventKind  # noqa: E402
from veeksha.loop.source import PregeneratedSessionSource  # noqa: E402
from veeksha.evaluator.performance.base import PerformanceEvaluator  # noqa: E402
from veeksha.types import ChannelModality, ClientType  # noqa: E402

INTERVAL_S = 0.05
NUM_SINGLE_TURN = 4
MULTI_TURN_TEXTS = ["turn one text", "turn two text", "turn three text"]
TOTAL_REQUESTS = NUM_SINGLE_TURN + len(MULTI_TURN_TEXTS)
TOTAL_SESSIONS = NUM_SINGLE_TURN + 1


def _build_sessions():
    sessions = [
        make_text_session(
            i,
            make_text_request(i * 100, f"prompt {i} alpha beta", target_tokens=6),
        )
        for i in range(1, NUM_SINGLE_TURN + 1)
    ]
    sessions.append(make_linear_text_session(NUM_SINGLE_TURN + 1, MULTI_TURN_TEXTS))
    return sessions


def _run_loop(kind: str):
    """One full run: MainLoop + manual scoring replay + evaluator finalize.

    Returns (events, counters, evaluator, finalize_result).
    """
    srv = SseChatServer(num_chunks=4, chunk_gap_s=0.01, prefill_s=0.01).start()
    try:
        seed_manager = SeedManager(42)
        provider = word_split_provider("m")
        config = MainLoopConfig(
            runtime=RuntimeConfig(
                num_dispatcher_threads=2,
                num_completion_threads=2,
                num_client_threads=2,
            ),
            traffic=RateTrafficConfig(
                interval_generator=FixedIntervalGeneratorConfig(interval=INTERVAL_S)
            ),
            client=OpenAIChatCompletionsClientConfig(
                api_base=f"http://127.0.0.1:{srv.port}/v1", api_key="k", model="m"
            ),
            monotonic_anchor=time.monotonic(),
        )
        evaluator = PerformanceEvaluator(
            PerformanceEvaluatorConfig(stream_metrics=False),
            seed_manager=seed_manager,
            output_dir=None,
            benchmark_start_time=config.monotonic_anchor,
            client_type=ClientType.OPENAI_CHAT_COMPLETIONS,
        )
        # Re-anchor just before start so both loop implementations' arrival schedules run
        # from a comparable origin (the benchmark takes its anchor right
        # before the loop too).
        config = dataclasses.replace(config, monotonic_anchor=time.monotonic())
        loop = create_main_loop(
            kind, config, seed_manager=seed_manager, tokenizer_provider=provider
        )
        loop.start(PregeneratedSessionSource(_build_sessions()))
        events = drain_loop_until_idle(loop, expect_completed=TOTAL_REQUESTS)
        counters = loop.counters()

        # Replay events into the evaluator exactly like ResultDrain._replay
        # (sequentially, so this test is deterministic).
        for event in events:
            if event.kind == LoopEventKind.DISPATCHED:
                evaluator.register_request(
                    request_id=event.request_id,
                    session_id=event.session_id,
                    dispatched_at=event.dispatched_at,
                    channels=event.request.channels,
                    requested_output=event.request.requested_output,
                )
            else:
                result = event.result
                error = Exception(result.error_msg) if result.error_msg else None
                evaluator.record_request_completed(
                    request_id=result.request_id,
                    session_id=result.session_id,
                    completed_at=result.client_completed_at,
                    response=result,
                    error=error,
                )
        return events, counters, evaluator, evaluator.finalize()
    finally:
        srv.stop()


@pytest.fixture(scope="module")
def loop_runs():
    # Throwaway warmup run: pays the remaining one-time costs (thread pool
    # spin-up, httpx pools) before the measured runs.
    _run_loop("python")
    python_run = _run_loop("python")
    native_run = _run_loop("native")
    return python_run, native_run


@pytest.mark.unit
def test_summary_metric_key_sets_identical(loop_runs) -> None:
    (_, _, _, py_result), (_, _, _, nat_result) = loop_runs
    assert set(nat_result.metrics.keys()) == set(py_result.metrics.keys())


@pytest.mark.unit
def test_request_and_session_counts_exact(loop_runs) -> None:
    (py_events, py_counters, py_ev, _), (nat_events, nat_counters, nat_ev, _) = (
        loop_runs
    )
    for counters in (py_counters, nat_counters):
        assert counters.requests_dispatched == TOTAL_REQUESTS
        assert counters.requests_completed == TOTAL_REQUESTS
        assert counters.sessions_seen == TOTAL_SESSIONS
        assert counters.sessions_completed == TOTAL_SESSIONS
        assert counters.sessions_errored == 0
        assert counters.in_flight == 0
    for evaluator in (py_ev, nat_ev):
        assert evaluator.num_requests == TOTAL_REQUESTS
        assert evaluator.num_completed_requests == TOTAL_REQUESTS
        assert evaluator.num_errored_requests == 0
        completed, errored, in_progress = evaluator.get_session_counts()
        assert (completed, errored, in_progress) == (TOTAL_SESSIONS, 0, 0)

    def ids(events, kind):
        return sorted(e.request_id for e in events if e.kind == kind)

    assert ids(nat_events, LoopEventKind.DISPATCHED) == ids(
        py_events, LoopEventKind.DISPATCHED
    )
    assert ids(nat_events, LoopEventKind.COMPLETED) == ids(
        py_events, LoopEventKind.COMPLETED
    )


def _token_map(events):
    out = {}
    for e in events:
        if e.kind == LoopEventKind.COMPLETED and e.result and e.result.channels:
            metrics = e.result.channels[ChannelModality.TEXT].metrics
            out[e.request_id] = (
                metrics["num_total_prompt_tokens"],
                metrics["num_delta_prompt_tokens"],
                metrics["num_output_tokens"],
                e.result.channels[ChannelModality.TEXT].content,
            )
    return out


@pytest.mark.unit
def test_prompt_and_output_token_counts_exact(loop_runs) -> None:
    (py_events, _, _, _), (nat_events, _, _, _) = loop_runs
    py_tokens = _token_map(py_events)
    nat_tokens = _token_map(nat_events)
    assert len(py_tokens) == TOTAL_REQUESTS
    assert nat_tokens == py_tokens
    # The multi-turn accumulation is really exercised: turn 3's total prompt
    # includes both prior user turns AND both spliced assistant replies.
    turn3 = (NUM_SINGLE_TURN + 1) * 100 + 2
    words_per_reply = 4  # SseChatServer default: 4 chunks x "tok{i} "
    expected_total = 3 * 3 + 2 * words_per_reply  # 3 user turns + 2 assistants
    assert py_tokens[turn3][0] == expected_total


@pytest.mark.unit
def test_dispatch_schedules_agree_within_tolerance(loop_runs) -> None:
    (py_events, _, _, _), (nat_events, _, _, _) = loop_runs

    def offsets(events):
        dispatched = {
            e.request_id: e.dispatched_at
            for e in events
            if e.kind == LoopEventKind.DISPATCHED
        }
        base = min(dispatched.values())
        return {rid: at - base for rid, at in dispatched.items()}

    py_offsets = offsets(py_events)
    nat_offsets = offsets(nat_events)
    assert set(py_offsets) == set(nat_offsets)
    for rid in py_offsets:
        delta = abs(py_offsets[rid] - nat_offsets[rid])
        assert delta < 0.050, (
            f"request {rid}: python offset {py_offsets[rid] * 1000:.1f}ms vs "
            f"native {nat_offsets[rid] * 1000:.1f}ms"
        )
    # Root arrivals really follow the fixed 50 ms RATE schedule on both
    # engines (root request ids are i*100; consecutive roots ~INTERVAL_S
    # apart, so the spread spans the whole schedule).
    for engine_offsets in (py_offsets, nat_offsets):
        roots = sorted(o for rid, o in engine_offsets.items() if rid % 100 == 0)
        assert roots[-1] - roots[0] > (len(roots) - 2) * INTERVAL_S


@pytest.mark.unit
def test_lifecycle_stamps_ordered_on_native(loop_runs) -> None:
    """Native COMPLETED results carry the five converted lifecycle stamps in
    the same order relation the Python loop guarantees."""
    (_, _, _, _), (nat_events, _, _, _) = loop_runs
    completed = [e for e in nat_events if e.kind == LoopEventKind.COMPLETED]
    assert completed
    for e in completed:
        r = e.result
        assert r.scheduler_ready_at <= r.scheduler_dispatched_at
        assert r.scheduler_dispatched_at <= r.client_picked_up_at
        assert r.client_picked_up_at < r.client_completed_at
        assert r.client_completed_at <= r.result_processed_at
