"""The gates and the tri-state verdict.

Every enabled check is measured exactly once, at ``config.target_concurrency``:
a preflight answers "can this box measure faithfully at MY concurrency", which
is one measurement. Its verdict is deliberately tri-state:

``honest``
    every gate term held, the check actually reached the concurrency it
    claims, and the mock served the offered load.
``dishonest``
    the mock served the load and the client's delivery lag (or correlation)
    still failed a gate. This is the only verdict that fails the run.
``engine-limited``
    the MOCK could not serve the offered connections at all, so the load under
    measurement was never applied. Client fidelity is *unmeasured* — reported
    inconclusive, never a pass, never a client failure. This is the only
    surviving engine-limited case: it fires solely on ``served_fraction <
    min_served_fraction``. Because every check pairs against what the mock
    ACTUALLY did (its real receive/send stamps), a mock that ran late but kept
    serving no longer invalidates the client measurement, so mock lateness is
    neither gated nor reported.

Gate terms are evaluated as ``value < threshold``, so NaN — the value the
scorer produces when there is nothing to score — is False in every term. An
unmeasured property fails safe rather than reading as perfect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from veeksha.logger import init_logger
from veeksha.preflight.drivers import CheckMetrics, PreflightWorkload, build_workloads
from veeksha.preflight.report import (
    VERDICT_DISHONEST,
    VERDICT_ENGINE_LIMITED,
    VERDICT_HONEST,
    CheckResult,
    ConcurrencyPoint,
    PreflightReport,
)

if TYPE_CHECKING:
    from veeksha.config.preflight import PreflightCheckConfig

logger = init_logger(__name__)

__all__ = ["run_preflight_check", "verdict_for"]


def verdict_for(
    metrics: Dict[str, float],
    gate_terms: Dict[str, bool],
    concurrency: int,
    config: "PreflightCheckConfig",
) -> tuple[str, List[str], str]:
    """Return ``(verdict, failed_term_names, note)`` for one measurement.

    Order matters: whether the mock could serve the offered load is checked
    FIRST. If it could not, the load under measurement was never applied, so
    the client's delivery lag describes nothing and must be reported
    inconclusive rather than blamed on the client. (Client-side-only checks —
    C1 dispatch — leave ``served_fraction`` NaN and skip this branch.)
    """
    served = metrics.get("served_fraction", float("nan"))
    if served == served and served < config.min_served_fraction:  # not NaN
        return (
            VERDICT_ENGINE_LIMITED,
            [],
            (
                f"the mock served only {served * 100:.0f}% of the offered "
                f"requests (< {config.min_served_fraction * 100:.0f}%): it "
                "could not sustain the load, so client fidelity here is "
                "UNMEASURED, not failed"
            ),
        )

    failed = [name for name, ok in gate_terms.items() if not ok]

    # The multi-turn check applies its concurrency as active SESSIONS; during a
    # session's think time it holds a scheduler slot with no open connection, so
    # the mock's peak-connection count legitimately undershoots the session
    # concurrency and must not be gated against it (it signals this with
    # gates_achieved_concurrency=0). Its load application is verified by
    # served_fraction + correlation instead.
    if metrics.get("gates_achieved_concurrency", 1.0):
        achieved = metrics.get("achieved", float("nan"))
        required = config.achieved_concurrency_fraction * concurrency
        if not achieved >= required:  # NaN-safe: NaN >= x is False
            failed.append(f"achieved concurrency ({achieved:.0f} < {required:.0f})")

    note = ""
    if not metrics.get("completed", 0.0) > 0:
        note = (
            "no request produced scoreable timing data; the gates fail safe "
            "(NaN is never honest)"
        )

    return (
        (VERDICT_HONEST if not failed else VERDICT_DISHONEST),
        failed,
        note,
    )


def _point(
    check: CheckMetrics,
    concurrency: int,
    config: "PreflightCheckConfig",
) -> ConcurrencyPoint:
    gate_terms = check.gate(check.metrics, config)
    verdict, failed, note = verdict_for(check.metrics, gate_terms, concurrency, config)
    nan = float("nan")
    m = check.metrics
    return ConcurrencyPoint(
        achieved=int(m.get("achieved", 0) or 0),
        completed=int(m.get("completed", 0) or 0),
        c2_lag_p99_ms=m.get("c2_lag_p99_ms", nan),
        c2_lag_p50_ms=m.get("c2_lag_p50_ms", nan),
        c3_lag_p99_ms=m.get("c3_lag_p99_ms", nan),
        c3_lag_p50_ms=m.get("c3_lag_p50_ms", nan),
        dispatch_drift_p99_ms=m.get("dispatch_drift_p99_ms", nan),
        unpaired_fraction=m.get("unpaired_fraction", nan),
        think_time_drift_p99_ms=m.get("think_time_drift_p99_ms", nan),
        think_time_drift_p50_ms=m.get("think_time_drift_p50_ms", nan),
        think_time_arrival_p99_ms=m.get("think_time_arrival_p99_ms", nan),
        verdict=verdict,
        failed_terms=failed,
        note=note,
    )


def _measure_workload(
    workload: PreflightWorkload,
    config: "PreflightCheckConfig",
) -> List[CheckResult]:
    """Run one workload once at the target and summarize every check it yields.

    One dispatch can answer more than one question (ASR scores the send and
    receive halves of word interactivity from the same traffic), so a workload
    may yield several checks from that single run.
    """
    concurrency = config.target_concurrency
    logger.info("preflight: %s at concurrency %d", type(workload).__name__, concurrency)
    checks = workload.measure(concurrency)
    return [
        CheckResult(
            name=check.name,
            notes=[check.notes],
            point=_point(check, concurrency, config),
            target_concurrency=concurrency,
        )
        for check in checks
    ]


def run_preflight_check(config: "PreflightCheckConfig") -> PreflightReport:
    """Measure every enabled check at the target concurrency and report."""
    report = PreflightReport(
        target_concurrency=config.target_concurrency,
        num_client_threads=(
            config.num_client_threads
            if config.num_client_threads is not None
            else "auto"
        ),
        delivery_lag_threshold_ms=config.delivery_lag_threshold_ms,
        max_unpaired_fraction=config.max_unpaired_fraction,
        min_served_fraction=config.min_served_fraction,
        dispatch_drift_threshold_ms=config.dispatch_drift_threshold_ms,
        pacing_drift_threshold_ms=config.pacing_drift_threshold_ms,
        think_time_drift_threshold_ms=config.think_time_drift_threshold_ms,
    )

    for workload in build_workloads(config):
        try:
            workload.start()
        except Exception as exc:  # pragma: no cover - optional deps / ports
            logger.exception("preflight: %s failed to start", type(workload).__name__)
            report.checks.append(
                CheckResult(
                    name=workload.name,
                    target_concurrency=config.target_concurrency,
                    skipped=True,
                    skip_reason=f"could not start the mock server: {exc}",
                )
            )
            continue
        try:
            report.checks.extend(_measure_workload(workload, config))
        finally:
            workload.stop()

    _add_recommendations(report, config)
    return report


def _add_recommendations(
    report: PreflightReport, config: "PreflightCheckConfig"
) -> None:
    target = config.target_concurrency
    recs = report.recommendations

    for check in report.checks:
        if check.skipped:
            continue
        verdict = check.verdict_at_target
        if verdict == VERDICT_ENGINE_LIMITED:
            recs.append(
                f"{check.name}: INCONCLUSIVE at {target} — the mock server "
                "could not serve the offered connections, so the load was "
                "never applied and nothing was proven about the harness. Raise "
                "server_loops, or run the mock on a dedicated host, and "
                "re-measure."
            )
        elif verdict == VERDICT_DISHONEST:
            recs.append(
                f"{check.name}: dishonest at {target}. Benchmark at a lower "
                "target concurrency, add client threads (num_client_threads) "
                "— one asyncio loop stays honest for only a few hundred "
                "streams — or move the mock to a dedicated host so the box is "
                "not serving both sides of the measurement."
            )
            failed_terms = check.point.failed_terms if check.point else []
            if "request correlation" in failed_terms:
                recs.append(
                    "request correlation exceeded its threshold: too many "
                    "requests had no matching server record, so the harness "
                    "lost the client<->server pairing. This is a MEASUREMENT "
                    "error (the delivery-lag samples are missing, not merely "
                    "smaller); check the PFID marker plumbing before trusting "
                    "any number from this check."
                )
            if "audio pacing adherence" in failed_terms:
                recs.append(
                    "audio pacing adherence exceeded its threshold: the client "
                    "sent audio chunks later than the 1x-realtime schedule, so "
                    "it is quietly benchmarking slower-than-realtime audio. "
                    "Add client threads (num_client_threads) or benchmark at a "
                    "lower concurrency so the pacer keeps up."
                )
            if "text pacing adherence" in failed_terms:
                recs.append(
                    "text pacing adherence exceeded its threshold: the client "
                    "streamed its input text deltas later than the pacer's "
                    "emulated decode cadence, so the TTS server saw input at the "
                    "wrong rate. Add client threads (num_client_threads) or "
                    "benchmark at a lower concurrency so the pacer keeps up."
                )
            if "think-time drift" in failed_terms:
                recs.append(
                    "think-time drift exceeded its threshold: the harness "
                    "released multi-turn follow-up turns later than "
                    "completion+think, so the server saw the conversation's "
                    "turns at the wrong cadence. This is fed by the "
                    "completion-notify path — raise num_completion_threads — and "
                    "by scheduler contention; benchmark at a lower concurrency "
                    "if it persists."
                )
            if "dispatch drift" in failed_terms:
                recs.append(
                    "dispatch drift exceeded its threshold: requests started "
                    "later than the arrival schedule said, so the run applied "
                    "a different arrival process than the one you configured. "
                    "This is scheduler-side, not transport-side — every "
                    "scheduler serializes each dispatch and each completion on "
                    "one threading.Condition, so more dispatcher or client "
                    "threads will not move it. Benchmark at a lower arrival "
                    "rate, or use the native engine."
                )

    if report.passed and not report.inconclusive:
        recs.append(
            f"system validated: veeksha timings are faithful at concurrency "
            f"{target} with these settings."
        )
