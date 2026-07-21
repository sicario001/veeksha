"""Preflight validator — finds the max concurrency this system can measure honestly."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

from veeksha.preflight.mock_engine import MockStreamingEngine
from veeksha.preflight.drivers import (
    NATIVE,
    PYTHON,
    ChainDispatchWorkload,
    DispatchWorkload,
    SttWorkload,
    TextWorkload,
    TtsWorkload,
    measure_rung,
)
from veeksha.preflight.probe import probe_pacing
from veeksha.preflight.report import CheckResult, ConcurrencyPoint, PreflightReport
from veeksha.logger import init_logger

if TYPE_CHECKING:
    from veeksha.config.preflight import PreflightCheckConfig

logger = init_logger(__name__)

# Empirically ~300 streams saturate one Python asyncio loop (measured on the
# real pipeline against a known-cadence mock server); only sizes recommendations.
_STREAMS_PER_LOOP = 300


def _ladder(target: int) -> List[int]:
    # Rungs from small up to (and including) the target — we only care whether
    # this system stays honest up to the concurrency you actually benchmark at.
    base = [10, 25, 50, 100, 200, 300, 400, 600, 800, 1200, 1600, 2400]
    pts = sorted({c for c in base if c < target} | {target})
    return [c for c in pts if c >= 1]


def _summarize(
    name: str, target: int, points: List[ConcurrencyPoint], notes: List[str]
) -> CheckResult:
    max_honest = max((p.concurrency for p in points if p.honest), default=0)
    passed = max_honest >= target
    if not passed:
        failed = [p for p in points if p.concurrency > max_honest and not p.honest]
        if failed and all(p.engine_limited for p in failed):
            notes = notes + [
                f"INCONCLUSIVE above {max_honest}: the mock server saturated "
                "before the client did, so client fidelity there is unmeasured "
                "(not a client failure). Re-run with the server on a dedicated "
                "host to certify higher."
            ]
    return CheckResult(
        name=name,
        max_honest_concurrency=max_honest,
        target_concurrency=target,
        passed=passed,
        points=points,
        notes=notes,
    )


def _run_check(
    name: str,
    target: int,
    ladder: List[int],
    measure,  # callable(concurrency) -> ConcurrencyPoint
) -> CheckResult:
    return _summarize(name, target, [measure(c) for c in ladder], [])


def _run_workload(
    workload, path: str, target: int, ladder: List[int], config
) -> List[CheckResult]:
    """Walk the ladder once for one (workload, dispatch path) and summarize
    every view it reports."""
    by_view: dict = {v.name: [] for v in workload.views}
    for c in ladder:
        points = measure_rung(
            workload, path, c, config.drift_threshold_ms, config.stretch_threshold
        )
        for view_name, point in points.items():
            by_view[view_name].append(point)

    suffix = " (native)" if path == NATIVE else ""
    out = []
    for view in workload.views:
        notes = [view.notes] if view.notes else []
        out.append(_summarize(view.name + suffix, target, by_view[view.name], notes))
    return out


def run_preflight_check(config: "PreflightCheckConfig") -> PreflightReport:
    target = config.target_concurrency
    ladder = _ladder(target)
    chunk_dt = config.chunk_ms / 1000.0
    prefill_s = config.prefill_ms / 1000.0

    engine_loops = config.engine_loops or min(24, max(4, math.ceil(target / 150)))
    engine = MockStreamingEngine(
        chunk_dt=chunk_dt,
        prefill_s=prefill_s,
        default_chunks=config.num_chunks,
        num_loops=engine_loops,
    ).start()
    logger.info(
        "preflight: engine on port %d (%d loops), target concurrency %d",
        engine.port,
        engine_loops,
        target,
    )

    report = PreflightReport(
        target_concurrency=target,
        num_client_threads=config.num_client_threads,
        drift_threshold_ms=config.drift_threshold_ms,
        stretch_threshold=config.stretch_threshold,
    )

    # Which dispatch paths to certify. Both walk the same ladder against the
    # same servers through the same harness; only the transport differs.
    paths = [PYTHON]
    if config.compare_native:
        try:
            from veeksha import native

            if native.is_available():
                paths.append(NATIVE)
            else:
                report.checks.append(
                    CheckResult(
                        name="native transport",
                        max_honest_concurrency=0,
                        target_concurrency=target,
                        passed=True,
                        points=[],
                        notes=[
                            "SKIPPED: veeksha_native extension not built; run "
                            "veeksha/native/build.sh <python> to enable."
                        ],
                    )
                )
        except Exception:  # pragma: no cover - extension optional
            pass

    workloads = [
        TextWorkload(engine, config, chunk_dt, prefill_s),
        DispatchWorkload(engine, config, chunk_dt, prefill_s),
        ChainDispatchWorkload(engine, config, chunk_dt, prefill_s),
    ]
    if config.check_audio_transport:
        workloads += [TtsWorkload(config), SttWorkload(config)]

    try:
        for workload in workloads:
            workload.start()
            try:
                for path in paths:
                    if path == NATIVE and not workload.supports_native:
                        continue
                    # The first batch pays cold-start costs (thread spin-up,
                    # httpx pool fill, engine loops warming) that would falsely
                    # fail the lowest rung. Both paths get the same discarded
                    # warmup, so neither is penalized for being measured first.
                    try:
                        measure_rung(
                            workload,
                            path,
                            min(50, max(10, target // 2)),
                            config.drift_threshold_ms,
                            config.stretch_threshold,
                        )
                    except Exception:  # pragma: no cover - warmup is best-effort
                        pass
                    report.checks.extend(
                        _run_workload(workload, path, target, ladder, config)
                    )
            finally:
                workload.stop()

        # engine saturation is a property of the box, not of either path
        text_checks = [c for c in report.checks if c.name.startswith("receive-drift")]
        if any(
            p.server_jitter_p99_ms >= config.drift_threshold_ms
            for c in text_checks
            for p in c.points
        ):
            text_checks[0].notes.append(
                "engine jitter approached the drift threshold at high concurrency — "
                "this box may be generating+consuming load near its own limit; "
                "re-run on a dedicated host (or raise engine_loops) to measure higher."
            )

        # ---- send-pacing check (scheduler precision; no transport involved) ----
        if config.check_pacing:

            def measure_pace(c: int) -> ConcurrencyPoint:
                r = probe_pacing(c, config.pacing_clip_s, config.chunk_ms)
                # honest requires BOTH aggregate real-time pacing AND per-chunk
                # dispatch precision (the granularity interactivity cares about).
                honest = (
                    r["stretch_p99"] < config.stretch_threshold
                    and r["send_drift_p99_ms"] < config.drift_threshold_ms
                )
                return ConcurrencyPoint(
                    concurrency=c,
                    achieved=c,
                    ivl_err_p99_ms=r["send_drift_p99_ms"],  # per-dispatch send drift
                    stretch_p99=r["stretch_p99"],
                    ttfc_p99_ms=float("nan"),
                    throughput=float("nan"),
                    server_jitter_p99_ms=0.0,
                    honest=honest,
                )

            report.checks.append(
                _run_check(
                    "pacing accuracy (realtime audio)", target, ladder, measure_pace
                )
            )
    finally:
        engine.stop()

    _add_recommendations(report, config)
    return report


def _add_recommendations(
    report: PreflightReport, config: "PreflightCheckConfig"
) -> None:
    recs = report.recommendations
    target = config.target_concurrency

    recv = next((c for c in report.checks if c.name.startswith("receive")), None)
    if recv is not None:
        # was concurrency capped below target by too few connections/loops?
        capped = [
            p
            for p in recv.points
            if p.concurrency <= target and p.achieved < 0.95 * p.concurrency
        ]
        if capped:
            need_threads = math.ceil(target / _STREAMS_PER_LOOP)
            if config.max_connections is not None and config.max_connections < target:
                recs.append(
                    f"httpx pool capped achieved concurrency (max_connections="
                    f"{config.max_connections}); set client.max_connections to None "
                    f"(unlimited) or >= {target}."
                )
            if config.num_client_threads < need_threads:
                recs.append(
                    f"increase num_client_threads to >= {need_threads} "
                    f"(~{_STREAMS_PER_LOOP} streams/loop) to reach concurrency {target}."
                )
        if not recv.passed and recv.max_honest_concurrency > 0:
            rec = (
                f"receive-drift is honest only up to ~{recv.max_honest_concurrency}; "
                f"either lower the benchmark concurrency to that or add client threads"
            )
            # Only point at the native path if this run actually measured it
            # reaching further — otherwise it is advice we have not evidenced.
            native_check = next(
                (c for c in report.checks if c.name.endswith("(native)")), None
            )
            if (
                native_check is not None
                and native_check.max_honest_concurrency > recv.max_honest_concurrency
            ):
                rec += (
                    f", or set client.use_native_transport (measured honest to "
                    f"~{native_check.max_honest_concurrency} above)"
                )
            recs.append(rec + ".")

    if report.passed:
        recs.append(
            f"system validated: Veeksha timings are faithful up to concurrency "
            f"{target} with the current settings."
        )
