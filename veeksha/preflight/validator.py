"""Preflight validator — finds the max concurrency this system can measure honestly."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

from veeksha.preflight.dummy_engine import DummyStreamingEngine
from veeksha.preflight.probe import probe_pacing, probe_receive_drift
from veeksha.preflight.report import CheckResult, ConcurrencyPoint, PreflightReport
from veeksha.logger import init_logger

if TYPE_CHECKING:
    from veeksha.config.preflight import PreflightCheckConfig

logger = init_logger(__name__)

# Empirically ~300 streams saturate one Python asyncio loop (see
# analysis/bench/RESULTS.md); used only to size recommendations.
_STREAMS_PER_LOOP = 300


def _ladder(target: int) -> List[int]:
    # Rungs from small up to (and including) the target — we only care whether
    # this system stays honest up to the concurrency you actually benchmark at.
    base = [10, 25, 50, 100, 200, 300, 400, 600, 800, 1200, 1600, 2400]
    pts = sorted({c for c in base if c < target} | {target})
    return [c for c in pts if c >= 1]


def _run_check(
    name: str,
    target: int,
    ladder: List[int],
    measure,  # callable(concurrency) -> ConcurrencyPoint
) -> CheckResult:
    points: List[ConcurrencyPoint] = []
    notes: List[str] = []
    max_honest = 0
    dishonest_seen = False
    for c in ladder:
        p = measure(c)
        points.append(p)
        if p.honest:
            max_honest = max(max_honest, c)
        else:
            dishonest_seen = True
            # one point past the knee is enough to locate it
            if c > target:
                break
    passed = max_honest >= target
    return CheckResult(
        name=name,
        max_honest_concurrency=max_honest,
        target_concurrency=target,
        passed=passed,
        points=points,
        notes=notes,
    )


def run_preflight_check(config: "PreflightCheckConfig") -> PreflightReport:
    target = config.target_concurrency
    ladder = _ladder(target)
    chunk_dt = config.chunk_ms / 1000.0
    prefill_s = config.prefill_ms / 1000.0

    engine_loops = config.engine_loops or min(24, max(4, math.ceil(target / 150)))
    engine = DummyStreamingEngine(
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

    # Warmup: the first batch pays cold-start costs (thread spin-up, httpx pool
    # fill, engine loops warming) that would falsely fail the lowest rung. Run a
    # small discarded probe first so the measured sweep is representative.
    warm_c = min(50, max(10, target // 2))
    try:
        probe_receive_drift(
            engine,
            warm_c,
            config.num_client_threads,
            config.num_dispatcher_threads,
            config.num_completion_threads,
            config.num_chunks,
            chunk_dt,
            prefill_s,
            n_requests=max(60, warm_c * 2),
            max_connections=config.max_connections,
        )
    except Exception:  # pragma: no cover - warmup is best-effort
        pass

    try:
        # ---- receive-drift check ----
        def measure_recv(c: int) -> ConcurrencyPoint:
            n_req = max(
                60,
                min(
                    2000,
                    int(
                        config.budget_s * c / (prefill_s + config.num_chunks * chunk_dt)
                    ),
                ),
            )
            m = probe_receive_drift(
                engine,
                c,
                config.num_client_threads,
                config.num_dispatcher_threads,
                config.num_completion_threads,
                config.num_chunks,
                chunk_dt,
                prefill_s,
                n_req,
                max_connections=config.max_connections,
            )
            # engine must not be the bottleneck for the point to count.
            engine_ok = m["server_jitter_p99_ms"] < max(
                config.drift_threshold_ms, m["ivl_err_p99_ms"] * 0.5
            )
            honest = (
                m["achieved"] >= 0.95 * c
                and m["ivl_err_p99_ms"] < config.drift_threshold_ms
                and m["stretch_p99"] < config.stretch_threshold
                and engine_ok
            )
            return ConcurrencyPoint(
                concurrency=c,
                achieved=int(m["achieved"]),
                ivl_err_p99_ms=m["ivl_err_p99_ms"],
                stretch_p99=m["stretch_p99"],
                ttfc_p99_ms=m["ttfc_p99_ms"],
                throughput=m["throughput"],
                server_jitter_p99_ms=m["server_jitter_p99_ms"],
                honest=honest,
            )

        recv = _run_check("receive-drift accuracy", target, ladder, measure_recv)
        # detect engine-bottleneck points and annotate
        if any(
            p.server_jitter_p99_ms >= config.drift_threshold_ms for p in recv.points
        ):
            recv.notes.append(
                "engine jitter approached the drift threshold at high concurrency — "
                "this box may be generating+consuming load near its own limit; "
                "re-run on a dedicated host (or raise engine_loops) to measure higher."
            )
        report.checks.append(recv)

        # ---- pacing check (only if requested) ----
        if config.check_pacing:

            def measure_pace(c: int) -> ConcurrencyPoint:
                stretch = probe_pacing(c, config.pacing_clip_s, config.chunk_ms)
                honest = stretch < config.stretch_threshold
                return ConcurrencyPoint(
                    concurrency=c,
                    achieved=c,
                    ivl_err_p99_ms=float("nan"),
                    stretch_p99=stretch,
                    ttfc_p99_ms=float("nan"),
                    throughput=float("nan"),
                    server_jitter_p99_ms=0.0,
                    honest=honest,
                )

            pace = _run_check(
                "pacing accuracy (realtime audio)", target, ladder, measure_pace
            )
            report.checks.append(pace)

        # ---- optional native (C++) receive-path comparison ----
        if config.compare_native:
            from veeksha import native

            if not native.is_available():
                report.checks.append(
                    CheckResult(
                        "receive-drift accuracy (native)",
                        0,
                        target,
                        False,
                        notes=[
                            "veeksha_native extension not built; run "
                            "veeksha/native/build.sh <python> to enable."
                        ],
                    )
                )
            else:

                def measure_native(c: int) -> ConcurrencyPoint:
                    n_req = max(
                        60,
                        min(
                            2000,
                            int(
                                config.budget_s
                                * c
                                / (prefill_s + config.num_chunks * chunk_dt)
                            ),
                        ),
                    )
                    engine.reset_telemetry()
                    m = native.receive_drift(
                        "127.0.0.1",
                        engine.port,
                        c,
                        config.num_chunks,
                        config.chunk_ms,
                        config.prefill_ms,
                        n_req,
                    )
                    achieved = engine.max_active_conns  # true concurrency reached
                    srv = engine.server_jitter_p99_ms()
                    engine_ok = srv < max(
                        config.drift_threshold_ms, m["ivl_err_p99_ms"] * 0.5
                    )
                    honest = (
                        achieved >= 0.95 * c
                        and m["ivl_err_p99_ms"] < config.drift_threshold_ms
                        and m["stretch_p99"] < config.stretch_threshold
                        and engine_ok
                    )
                    return ConcurrencyPoint(
                        concurrency=c,
                        achieved=int(achieved),
                        ivl_err_p99_ms=m["ivl_err_p99_ms"],
                        stretch_p99=m["stretch_p99"],
                        ttfc_p99_ms=float("nan"),
                        throughput=float("nan"),
                        server_jitter_p99_ms=srv,
                        honest=honest,
                    )

                # warmup (discarded) so the lowest native rung isn't cold-penalized
                try:
                    native.receive_drift(
                        "127.0.0.1",
                        engine.port,
                        warm_c,
                        config.num_chunks,
                        config.chunk_ms,
                        config.prefill_ms,
                        max(60, warm_c * 2),
                    )
                except Exception:  # pragma: no cover
                    pass

                nat = _run_check(
                    "receive-drift accuracy (native)", target, ladder, measure_native
                )
                nat.notes.append(
                    "single native OS thread vs the Python pipeline above; "
                    "same engine + same metric definitions."
                )
                report.checks.append(nat)
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
            recs.append(
                f"receive-drift is honest only up to ~{recv.max_honest_concurrency}; "
                f"either lower the benchmark concurrency to that, add client threads, "
                f"or use the native dispatcher path once available."
            )

    if report.passed:
        recs.append(
            f"system validated: Veeksha timings are faithful up to concurrency "
            f"{target} with the current settings."
        )
