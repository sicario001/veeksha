"""Preflight report data structures + formatting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


def _f(v: float, w: int, prec: int, suffix: str = "") -> str:
    """Format a value at width ``w`` (+ suffix); NaN renders as a dash."""
    if v != v:  # NaN
        return f"{'-':>{w + len(suffix)}}"
    return f"{v:>{w}.{prec}f}{suffix}"


@dataclass
class ConcurrencyPoint:
    """Measured preflight at one concurrency level.

    Verdict is tri-state: ``honest`` (client stayed faithful), dishonest
    (client drifted while the mock server held its schedule), or
    ``engine_limited`` (the mock server itself saturated — client fidelity is
    UNMEASURABLE at this rung, which must not be reported as a client failure).
    """

    concurrency: int
    achieved: int  # max simultaneous connections actually reached
    ivl_err_p99_ms: float  # p99 |recorded inter-chunk - true cadence|
    stretch_p99: float  # p99 recorded-stream-duration / ideal
    ttfc_p99_ms: float
    throughput: float
    server_jitter_p99_ms: float
    honest: bool  # passed the thresholds AND engine was not the bottleneck
    engine_limited: bool = False  # mock server saturated; rung is inconclusive
    completion_drift_p99_ms: float = float("nan")  # client-done -> processed lag

    @property
    def verdict(self) -> str:
        if self.honest:
            return "yes"
        return "srv-lim" if self.engine_limited else "NO"


@dataclass
class CheckResult:
    """Outcome of one preflight check (receive-drift or pacing)."""

    name: str
    max_honest_concurrency: int
    target_concurrency: int
    passed: bool
    points: List[ConcurrencyPoint] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    skipped: bool = False  # check could not run (e.g. optional dep missing)


@dataclass
class PreflightReport:
    """Top-level preflight report."""

    target_concurrency: int
    num_client_threads: int
    drift_threshold_ms: float
    stretch_threshold: float
    checks: List[CheckResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if not c.skipped)

    def format_text(self) -> str:
        lines: List[str] = []
        verdict = "PASS" if self.passed else "FAIL"
        lines.append("=" * 72)
        lines.append(f"VEEKSHA MEASUREMENT PREFLIGHT — {verdict}")
        lines.append("=" * 72)
        lines.append(
            f"target concurrency: {self.target_concurrency}   "
            f"num_client_threads: {self.num_client_threads}   "
            f"thresholds: ivl_p99<{self.drift_threshold_ms:.0f}ms, "
            f"stretch<{self.stretch_threshold:.2f}"
        )
        lines.append("")
        for chk in self.checks:
            if chk.skipped:
                status = "SKIP"
            else:
                status = "PASS" if chk.passed else "FAIL"
            lines.append(f"[{status}] {chk.name}")
            if not chk.skipped:
                lines.append(
                    f"       max honest concurrency on this system: "
                    f"{chk.max_honest_concurrency}"
                    f"   (you asked for {chk.target_concurrency})"
                )
            if chk.points:
                lines.append(
                    f"       {'conc':>6} {'ach':>6} {'ivlP99':>8} {'strP99':>7} "
                    f"{'ttfcP99':>8} {'tput/s':>8} {'srvJit':>7} {'d2P99':>7}  honest"
                )
                for p in chk.points:
                    lines.append(
                        f"       {p.concurrency:>6} {p.achieved:>6} "
                        f"{_f(p.ivl_err_p99_ms, 7, 1, 'm')} "
                        f"{_f(p.stretch_p99, 7, 3)} "
                        f"{_f(p.ttfc_p99_ms, 7, 0, 'm')} {_f(p.throughput, 8, 0)} "
                        f"{_f(p.server_jitter_p99_ms, 6, 1, 'm')} "
                        f"{_f(p.completion_drift_p99_ms, 6, 1, 'm')}   {p.verdict}"
                    )
                if any(p.engine_limited for p in chk.points):
                    lines.append(
                        "       (srv-lim = the MOCK SERVER saturated at that rung; "
                        "client fidelity there is unmeasured, not failed)"
                    )
            for n in chk.notes:
                lines.append(f"       note: {n}")
            lines.append("")
        if self.recommendations:
            lines.append("recommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
            lines.append("")
        return "\n".join(lines)
