"""End-to-end tests for the preflight gate.

Deliberately tiny (a handful of streams, a few chunks each, millisecond
cadences) so the whole module runs in seconds — but the pipeline under test is
the REAL one: a measurement here starts the same prefetch/dispatch/client/completion
workers, the same scheduler and the same client that a benchmark does.
"""

from __future__ import annotations

import math

import pytest

from veeksha.config.preflight import PreflightCheckConfig
from veeksha.preflight.drivers import (
    MultiTurnTextWorkload,
    TextWorkload,
    VajraTtsWorkload,
)
from veeksha.preflight.report import (
    VERDICT_DISHONEST,
    VERDICT_ENGINE_LIMITED,
    VERDICT_HONEST,
)
from veeksha.preflight.validator import run_preflight_check, verdict_for

pytestmark = pytest.mark.unit


def _tiny_config(**overrides) -> PreflightCheckConfig:
    """A config whose checks finish in well under a second."""
    kwargs = dict(
        target_concurrency=4,
        num_chunks=12,
        chunk_ms=25.0,
        prefill_ms=10.0,
        budget_s=0.5,
        check_tts=False,
        check_vajra_tts=False,
        check_stt=False,
        check_dispatch=False,
        check_multiturn=False,
        num_client_threads=2,
        num_dispatcher_threads=2,
        num_completion_threads=2,
        request_timeout_s=30,
        server_loops=2,
    )
    kwargs.update(overrides)
    return PreflightCheckConfig(**kwargs)


# ------------------------------------------------------------- tri-state verdict
def _metrics(**overrides) -> dict:
    base = {
        "achieved": 4.0,
        "completed": 4.0,
        "served_fraction": 1.0,
        "unpaired_fraction": 0.0,
        "c2_lag_p99_ms": 1.0,
        "c2_lag_p50_ms": 0.5,
        "c3_lag_p99_ms": 1.0,
        "c3_lag_p50_ms": 0.5,
        "dispatch_drift_p99_ms": float("nan"),
        "min_lag_ms": 0.4,
        "think_time_drift_p99_ms": float("nan"),
        "think_time_drift_p50_ms": float("nan"),
        "think_time_arrival_p99_ms": float("nan"),
    }
    base.update(overrides)
    return base


def test_verdict_is_honest_when_every_term_holds() -> None:
    config = _tiny_config()
    verdict, failed, _ = verdict_for(_metrics(), {"lag": True}, 4, config=config)
    assert verdict == VERDICT_HONEST
    assert failed == []


def test_verdict_is_dishonest_when_a_gate_term_fails() -> None:
    config = _tiny_config()
    verdict, failed, _ = verdict_for(_metrics(), {"lag": False}, 4, config=config)
    assert verdict == VERDICT_DISHONEST
    assert failed == ["lag"]


def test_verdict_is_dishonest_when_achieved_concurrency_falls_short() -> None:
    config = _tiny_config()
    verdict, failed, _ = verdict_for(
        _metrics(achieved=1.0), {"lag": True}, 4, config=config
    )
    assert verdict == VERDICT_DISHONEST
    assert any("achieved concurrency" in term for term in failed)


def test_nan_fails_safe_as_dishonest_never_honest() -> None:
    config = _tiny_config()
    nan = float("nan")
    # A term computed from no samples is NaN; `NaN < threshold` is False, so
    # the term fails and the check can never read as honest. served_fraction is
    # healthy, so this is a client failure, not engine-limited.
    verdict, failed, note = verdict_for(
        _metrics(c3_lag_p99_ms=nan, completed=0.0, achieved=nan),
        {"lag": nan < config.delivery_lag_threshold_ms},
        4,
        config=config,
    )
    assert verdict == VERDICT_DISHONEST
    assert failed
    assert "no request produced scoreable timing data" in note


def test_mock_that_could_not_serve_is_engine_limited_not_a_client_failure() -> None:
    config = _tiny_config()
    # Even with every client term failing, a mock that served far less than the
    # offered load makes the check UNMEASURED: never a pass, never a client
    # failure. This is now the ONLY engine-limited case.
    verdict, failed, note = verdict_for(
        _metrics(c3_lag_p99_ms=999.0, served_fraction=0.10),
        {"lag": False},
        4,
        config=config,
    )
    assert verdict == VERDICT_ENGINE_LIMITED
    assert failed == []
    assert "UNMEASURED" in note


def test_a_late_but_serving_mock_no_longer_invalidates_the_measurement() -> None:
    config = _tiny_config()
    # The whole point of the rework: the reference is what the mock ACTUALLY
    # did, so a mock that served the load but ran late is not engine-limited.
    # If the client's lag held, the verdict is honest regardless of lateness.
    verdict, failed, _ = verdict_for(
        _metrics(served_fraction=1.0),
        {"lag": True},
        4,
        config=config,
    )
    assert verdict == VERDICT_HONEST
    assert failed == []


def test_unpaired_requests_below_the_bar_stay_honest() -> None:
    config = _tiny_config()
    # A tiny correlation loss (mock served the load) is tolerated; the gate
    # term itself is what fails when it grows, exercised by the driver gate.
    verdict, _, _ = verdict_for(
        _metrics(served_fraction=1.0, unpaired_fraction=0.0),
        {"correlation": True},
        4,
        config=config,
    )
    assert verdict == VERDICT_HONEST


# ------------------------------------------------------------- the real pipeline
def test_text_measurement_runs_the_real_pipeline_and_pairs_delivery_lag() -> None:
    config = _tiny_config()
    workload = TextWorkload(config)
    workload.start()
    try:
        measurement = workload.dispatch(concurrency=4, n_requests=8)
        checks = workload.score(measurement)
    finally:
        workload.stop()

    assert len(measurement.results) == 8
    successes = [r for r in measurement.results if r.success]
    assert successes, "no request completed against the mock engine"

    # Every lifecycle stamp is set, which is only true if the real workers ran:
    # scheduler_* comes from DispatchWorker, client_picked_up_at from
    # ClientWorker, result_processed_at from CompletionWorker.
    for result in successes:
        assert result.scheduler_ready_at is not None
        assert result.scheduler_dispatched_at is not None
        assert result.client_picked_up_at is not None
        assert result.client_completed_at is not None
        assert result.result_processed_at is not None
        assert (
            result.scheduler_dispatched_at
            <= result.client_picked_up_at
            <= result.client_completed_at
            <= result.result_processed_at
        )

    # The mock served the concurrency it was asked for, and the PFID marker
    # round-tripped: every successful request has a server record.
    assert measurement.achieved >= 2
    assert measurement.unidentified_conns == 0
    for result in successes:
        assert result.request_id in measurement.server_records

    (check,) = checks
    metrics = check.metrics
    assert metrics["completed"] == len(successes)
    # Delivery lag is finite, positive, and on the order of milliseconds: this
    # is a smoke bound, not the shipped threshold (a loaded CI box lags more).
    assert not math.isnan(metrics["c3_lag_p99_ms"])
    assert not math.isnan(metrics["c3_lag_p50_ms"])
    # The median (loopback floor) is positive and below the tail; a smoke
    # bound, not the shipped threshold.
    assert 0.0 <= metrics["c3_lag_p50_ms"] <= metrics["c3_lag_p99_ms"] < 200.0
    assert not math.isnan(metrics["c2_lag_p99_ms"])
    assert metrics["c2_lag_p99_ms"] >= 0.0
    # Full correlation on loopback.
    assert metrics["unpaired_fraction"] == pytest.approx(0.0)
    assert metrics["served_fraction"] == pytest.approx(1.0)


def test_vajra_measurement_pairs_response_delivery_while_input_is_sending() -> None:
    """The overlap case: audio arrives while the client is still paced-sending.

    This is the only check where the client's receive stamping competes with
    its own send loop on one event loop, which is where lag shows up first
    under concurrency — so it is worth asserting the overlap really happens
    rather than trusting the mock's trigger.
    """
    from veeksha.core.audio_contract import AudioMetricKey
    from veeksha.types import ChannelModality

    config = _tiny_config(
        check_text=False,
        check_vajra_tts=True,
        audio_num_chunks=10,
        audio_chunk_ms=20.0,
        audio_first_delta_ms=10.0,
        audio_chunk_bytes=960,
    )
    workload = VajraTtsWorkload(config)
    workload.start()
    try:
        measurement = workload.dispatch(concurrency=2, n_requests=4)
        checks = workload.score(measurement)
    finally:
        workload.stop()

    successes = [r for r in measurement.results if r.success]
    assert successes, "no request completed against the Vajra mock"

    for result in successes:
        metrics = result.channels[ChannelModality.AUDIO].metrics
        stamps = metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]
        assert len(stamps) == config.audio_num_chunks
        # Audio began before the input was finished: the interleaved path.
        first_audio_ms = stamps[0][0]
        input_done_ms = metrics[AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value]
        assert input_done_ms is not None
        assert first_audio_ms < input_done_ms, (first_audio_ms, input_done_ms)
        # Real workers ran (same stamp-chain proof as the text check).
        assert result.result_processed_at is not None

    (check,) = checks
    assert check.metrics["completed"] == len(successes)
    assert not math.isnan(check.metrics["c3_lag_p99_ms"])
    assert check.metrics["c3_lag_p99_ms"] < 200.0
    # The median delivery floor is non-negative (a rare concurrent frame
    # misalignment can push the raw min negative — the positive-lag invariant
    # gate is what surfaces that; the median is the stable smoke signal).
    assert check.metrics["c3_lag_p50_ms"] >= 0.0


def test_multiturn_measurement_runs_both_turns_and_scores_think_time() -> None:
    """Both turns of a session run through the REAL pipeline; the session
    correlates and think-time drift is finite and sane."""
    from collections import defaultdict

    config = _tiny_config(
        check_text=False,
        check_multiturn=True,
        dispatch_response_chunks=3,
        think_time_s=0.05,
    )
    workload = MultiTurnTextWorkload(config)
    workload.start()
    try:
        # 2 sessions -> 4 turns.
        measurement = workload.dispatch(concurrency=2, n_requests=2)
        checks = workload.score(measurement)
    finally:
        workload.stop()

    successes = [r for r in measurement.results if r.success]
    assert successes, "no turn completed against the mock engine"

    # Group by session: both turns of a session actually ran, each carrying the
    # full lifecycle stamp chain (only true if the real workers ran).
    by_session = defaultdict(list)
    for result in successes:
        by_session[result.session_id].append(result)
    complete_sessions = [turns for turns in by_session.values() if len(turns) == 2]
    assert complete_sessions, "no session produced both turns"

    for turns in complete_sessions:
        for result in turns:
            assert result.scheduler_dispatched_at is not None
            assert result.client_picked_up_at is not None
            assert result.client_completed_at is not None
            assert result.result_processed_at is not None
        turn1, turn2 = sorted(turns, key=lambda r: r.scheduler_dispatched_at)
        # Turn 2 was released only AFTER turn 1 completed (the child dependency
        # plus think time), which is the whole point of the check.
        assert turn2.scheduler_dispatched_at >= turn1.client_completed_at
        # Both turns' PFIDs round-tripped to a server record.
        assert turn1.request_id in measurement.server_records
        assert turn2.request_id in measurement.server_records

    (check,) = checks
    metrics = check.metrics
    assert metrics["completed"] >= 1
    # Think-time drift is finite and late-only (>= 0); a smoke bound, not the
    # shipped threshold (a loaded CI box releases later).
    assert not math.isnan(metrics["think_time_drift_p99_ms"])
    assert not math.isnan(metrics["think_time_drift_p50_ms"])
    assert (
        0.0
        <= metrics["think_time_drift_p50_ms"]
        <= metrics["think_time_drift_p99_ms"]
        < 500.0
    )
    # The server-arrival variant is reported and finite too.
    assert not math.isnan(metrics["think_time_arrival_p99_ms"])
    assert metrics["think_time_arrival_p99_ms"] >= 0.0
    # Full correlation on loopback.
    assert metrics["unpaired_fraction"] == pytest.approx(0.0)
    assert metrics["served_fraction"] == pytest.approx(1.0)


def test_run_preflight_check_measures_the_target_once_with_a_legend() -> None:
    config = _tiny_config(target_concurrency=3)
    report = run_preflight_check(config)

    assert len(report.checks) == 1
    check = report.checks[0]
    assert not check.skipped
    # Exactly one measurement, and it is the target concurrency.
    assert check.point is not None
    assert check.target_concurrency == 3
    assert check.verdict_at_target == check.point.verdict
    assert check.verdict_at_target in (
        VERDICT_HONEST,
        VERDICT_DISHONEST,
        VERDICT_ENGINE_LIMITED,
    )
    # A run that is only inconclusive must not fail the gate.
    if check.verdict_at_target == VERDICT_ENGINE_LIMITED:
        assert report.passed

    text = report.format_text()
    assert "COLUMN LEGEND" in text
    # Every column is explained, and think-time drift is explained as what it
    # is rather than left as a cryptic abbreviation.
    for column in (
        "ach",
        "cmpl",
        "c2P99",
        "c2P50",
        "c3P99",
        "c3P50",
        "dispP99",
        "unpr",
        "ttP99",
        "ttP50",
        "ttSrvP99",
    ):
        assert column in text
    assert "THINK-TIME drift" in text
    # The removed columns are gone.
    assert "cqP99" not in text
    assert "srvLt" not in text
    assert "WHAT IS AND IS NOT IN THESE NUMBERS" in text
    # The check's own note disambiguates its columns.
    assert "RESPONSE-DELIVERY lag" in text
    # The concurrency lives in the header, not in a per-row column.
    assert "target concurrency: 3" in text
    assert "max honest concurrency" not in text
