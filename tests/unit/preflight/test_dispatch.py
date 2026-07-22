"""Tests for the dispatch-accuracy check.

The check's whole validity rests on one claim: that
``rate_schedule_offsets_ms`` reproduces the arrival times
``RateTrafficScheduler`` actually assigns. If the reproduction is subtly wrong
(one draw too many, the wrong seed path, an off-by-one in when
``_next_start_time`` advances) the check reports phantom drift for a harness
that was perfectly punctual, and no end-to-end assertion would catch it —
so the first test here schedules real sessions into a real scheduler and
compares against the scheduler's own bookkeeping.
"""

from __future__ import annotations

import math

import pytest

from veeksha.config.generator.interval import (
    FixedIntervalGeneratorConfig,
    GammaIntervalGeneratorConfig,
    PoissonIntervalGeneratorConfig,
)
from veeksha.config.preflight import PreflightCheckConfig
from veeksha.config.traffic import RateTrafficConfig
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import SessionGraph, SessionNode, add_node
from veeksha.preflight.drivers import PREFLIGHT_SEED, DispatchWorkload
from veeksha.preflight.scorer import rate_schedule_offsets_ms
from veeksha.traffic.rate import RateTrafficScheduler
from veeksha.types import ChannelModality

pytestmark = pytest.mark.unit


def _single_node_session(session_id: int) -> Session:
    """The session shape the preflight dispatches: one node, no think time."""
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    request = Request(
        id=session_id,
        channels={ChannelModality.TEXT: TextChannelRequestContent(input_text="x")},
    )
    return Session(id=session_id, session_graph=graph, requests={0: request})


@pytest.mark.parametrize(
    "interval_config",
    [
        PoissonIntervalGeneratorConfig(arrival_rate=250.0),
        PoissonIntervalGeneratorConfig(arrival_rate=7.5),
        GammaIntervalGeneratorConfig(arrival_rate=40.0, cv=0.8),
        FixedIntervalGeneratorConfig(interval=0.004),
    ],
)
def test_reproduced_schedule_equals_what_the_real_scheduler_assigned(
    interval_config,
) -> None:
    """The safety net: reproduction vs. a real ``RateTrafficScheduler``.

    Schedules N real single-node sessions into a scheduler built with the same
    seed and interval-generator config, then asserts the reproduced offsets
    equal the ``session_start_time`` values it assigned — exactly, not
    approximately, because both sides consume the identical seeded RNG stream.
    """
    n_sessions = 40
    seed = 12345
    scheduler = RateTrafficScheduler(
        config=RateTrafficConfig(interval_generator=interval_config),
        seed_manager=SeedManager(seed),
    )
    for session_id in range(n_sessions):
        scheduler.schedule_session(_single_node_session(session_id))

    assigned_ms = [
        scheduler._sessions[i].session_start_time * 1000.0 for i in range(n_sessions)
    ]
    reproduced_ms = rate_schedule_offsets_ms(seed, interval_config, n_sessions)

    assert reproduced_ms == assigned_ms
    # And the definition itself: the first session is due immediately, and
    # every later one is the running sum of the draws before it.
    assert reproduced_ms[0] == 0.0
    assert all(
        later >= earlier for earlier, later in zip(reproduced_ms, reproduced_ms[1:])
    )


def test_reproduction_uses_the_first_draw_of_the_interval_factory() -> None:
    """A different seed must give a different schedule; the same seed, the same.

    ``numpy_factory`` is a counter — calling it twice yields two different
    streams — so reproducing the schedule with a *second* draw would look
    plausible and be wrong. This pins the observable consequence.
    """
    config = PoissonIntervalGeneratorConfig(arrival_rate=50.0)
    same_a = rate_schedule_offsets_ms(7, config, 10)
    same_b = rate_schedule_offsets_ms(7, config, 10)
    different = rate_schedule_offsets_ms(8, config, 10)
    assert same_a == same_b
    assert same_a != different


def test_empty_schedule_for_no_sessions() -> None:
    assert rate_schedule_offsets_ms(1, PoissonIntervalGeneratorConfig(), 0) == []


# ------------------------------------------------------------------- sizing
def _tiny_config(**overrides) -> PreflightCheckConfig:
    kwargs = dict(
        target_concurrency=4,
        chunk_ms=5.0,
        prefill_ms=10.0,
        dispatch_response_chunks=4,
        budget_s=0.5,
        check_text=False,
        check_tts=False,
        check_vajra_tts=False,
        check_stt=False,
        check_dispatch=True,
        num_client_threads=2,
        num_dispatcher_threads=2,
        num_completion_threads=2,
        request_timeout_s=30,
        server_loops=2,
    )
    kwargs.update(overrides)
    return PreflightCheckConfig(**kwargs)


def test_arrival_rate_targets_the_requested_concurrency() -> None:
    # Little's law: offered load = rate x lifetime, so the check certifies the
    # same concurrency the other checks do.
    workload = DispatchWorkload(_tiny_config(target_concurrency=100))
    lifetime = workload.request_lifetime_s()
    assert workload.arrival_rate(100) == pytest.approx(100.0 / lifetime)


def test_measurement_is_sized_to_fit_inside_the_prefetch_burst_window() -> None:
    from veeksha.preflight.drivers import BURST_WINDOW_S

    workload = DispatchWorkload(_tiny_config(target_concurrency=100, budget_s=10.0))
    n_requests, rate, warning = workload._plan(100)
    duration_s = n_requests / rate + workload.request_lifetime_s()
    # Beyond the burst window PrefetchWorker throttles to ~20 sessions/s, and
    # the check would measure that throttle instead of dispatch accuracy.
    assert duration_s < BURST_WINDOW_S
    assert warning == ""


def test_impossible_sizing_warns_instead_of_silently_measuring_the_throttle() -> None:
    # A huge concurrency floor with a long lifetime cannot fit: the floor wins
    # (a measurement must be able to reach its concurrency), and the check says so
    # rather than reporting prefetch throttle as scheduler lateness.
    workload = DispatchWorkload(
        _tiny_config(
            target_concurrency=4000, dispatch_response_chunks=2000, chunk_ms=20.0
        )
    )
    _, _, warning = workload._plan(4000)
    assert "burst window" in warning
    assert "upper bound" in warning


# --------------------------------------------------------------- the real measurement
def test_dispatch_measurement_runs_the_real_pipeline_and_scores_finite_drift() -> None:
    config = _tiny_config()
    workload = DispatchWorkload(config)
    workload.start()
    try:
        n_requests = workload.request_count(4)
        measurement = workload.dispatch(concurrency=4, n_requests=n_requests)
        checks = workload.score(measurement)
    finally:
        workload.stop()

    assert len(measurement.results) == n_requests
    # The epoch is the scheduler's own zero, read back from the scheduler.
    assert not math.isnan(measurement.scheduler_epoch)
    assert measurement.extras["n_sessions"] == n_requests
    assert measurement.extras["arrival_rate"] > 0

    successes = [r for r in measurement.results if r.success]
    assert successes, "no request completed against the mock engine"
    # Lifecycle stamps prove the REAL workers ran: scheduler_* from
    # DispatchWorker, client_picked_up_at from ClientWorker,
    # result_processed_at from CompletionWorker.
    for result in successes:
        assert result.scheduler_ready_at is not None
        assert result.scheduler_dispatched_at is not None
        assert result.client_picked_up_at is not None
        assert result.result_processed_at is not None
        # Dispatch happened at or after the scheduler epoch, never before it.
        assert result.scheduler_dispatched_at >= measurement.scheduler_epoch

    (check,) = checks
    metrics = check.metrics
    assert metrics["completed"] == len(measurement.results)
    assert not math.isnan(metrics["dispatch_drift_p99_ms"])
    # Late-only, so never negative; a smoke bound, not the shipped threshold.
    assert 0.0 <= metrics["dispatch_drift_p99_ms"] < 1000.0
    # C1 is client-side only: it pairs no server stamp, so the delivery-lag and
    # correlation columns do not apply and stay NaN.
    assert math.isnan(metrics["c2_lag_p99_ms"])
    assert math.isnan(metrics["c3_lag_p99_ms"])
    assert math.isnan(metrics["served_fraction"])

    terms = check.gate(metrics, config)
    assert set(terms) == {"dispatch drift"}
