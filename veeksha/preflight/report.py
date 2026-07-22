"""Preflight report: result structures, the rendered table, and its legend.

A number nobody can interpret is worse than no number, so every column printed
here is explained twice: once globally (what the column *is*) and once per
check (what THIS check's numbers include and exclude). The per-check note
exists because the same column carries a slightly different pairing in
different checks — ``c2Lag`` is one lag per request for text/TTS but one lag
per audio chunk for ASR — and a reader who assumes otherwise draws the wrong
conclusion from a correct table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = [
    "CheckResult",
    "ConcurrencyPoint",
    "PreflightReport",
    "VERDICT_DISHONEST",
    "VERDICT_ENGINE_LIMITED",
    "VERDICT_HONEST",
    "VERDICT_NOT_MEASURED",
]

#: The verdict is tri-state, plus a "never ran" state for skipped checks.
VERDICT_HONEST = "honest"
#: The client drifted while the mock served the offered load.
VERDICT_DISHONEST = "dishonest"
#: The MOCK could not serve the offered connections: client fidelity here is
#: UNMEASURED. Never a pass, never a client failure — inconclusive.
VERDICT_ENGINE_LIMITED = "engine-limited"
VERDICT_NOT_MEASURED = "not-measured"

_VERDICT_CELL = {
    VERDICT_HONEST: "yes",
    VERDICT_DISHONEST: "NO",
    VERDICT_ENGINE_LIMITED: "srv-lim",
    VERDICT_NOT_MEASURED: "-",
}


def _f(value: float, width: int, precision: int, suffix: str = "") -> str:
    """Format ``value`` right-aligned; NaN (i.e. "unmeasured") renders as a dash."""
    if value != value:  # NaN
        return f"{'-':>{width + len(suffix)}}"
    return f"{value:>{width}.{precision}f}{suffix}"


@dataclass
class ConcurrencyPoint:
    """One check measured once, at the target concurrency."""

    #: Peak SIMULTANEOUS connections the mock observed. A count of completed
    #: requests is not concurrency and must never be reported as it.
    achieved: int
    #: Requests that produced scoreable timing data.
    completed: int
    #: C2 request-delivery lag, p99/p50 (ms). NaN for the client-side-only C1.
    c2_lag_p99_ms: float
    c2_lag_p50_ms: float
    #: C3 response-delivery lag, p99/p50 (ms). NaN for the client-side-only C1.
    c3_lag_p99_ms: float
    c3_lag_p50_ms: float
    #: C1 dispatch lateness, p99 (ms). NaN for the paired C2/C3 checks. For the
    #: STT/TTS checks this column carries the client-side pacing adherence.
    dispatch_drift_p99_ms: float
    #: Fraction of requests that could not be paired with a server record.
    unpaired_fraction: float
    #: Multi-turn think-time drift, client-side (gated) p99/p50 (ms), and the
    #: server-arrival (reported) p99. NaN for every non-multi-turn check.
    think_time_drift_p99_ms: float
    think_time_drift_p50_ms: float
    think_time_arrival_p99_ms: float
    verdict: str
    #: Named gate terms that did not hold, for the "why did it fail" line.
    failed_terms: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def cell(self) -> str:
        return _VERDICT_CELL.get(self.verdict, self.verdict)


@dataclass
class CheckResult:
    """One preflight check, measured once at the target concurrency."""

    name: str
    #: Column disambiguation + include/exclude statement for THIS check.
    notes: List[str] = field(default_factory=list)
    #: The single measurement. ``None`` when the check never ran.
    point: Optional[ConcurrencyPoint] = None
    target_concurrency: int = 0
    #: Whether this check's verdict can fail the run (all shipped checks do).
    gated: bool = True
    skipped: bool = False
    skip_reason: str = ""

    @property
    def verdict_at_target(self) -> str:
        if self.point is None:
            return VERDICT_NOT_MEASURED
        return self.point.verdict

    @property
    def passed(self) -> bool:
        """True when the check was measured honest at the target."""
        return self.verdict_at_target == VERDICT_HONEST

    @property
    def failed(self) -> bool:
        """True only when the client was measured DISHONEST.

        Inconclusive (engine-limited) is deliberately not a failure: the mock
        saturated before the client did, so nothing was proven either way.
        """
        return self.gated and self.verdict_at_target == VERDICT_DISHONEST

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        if self.passed:
            return "PASS"
        if self.verdict_at_target == VERDICT_ENGINE_LIMITED:
            return "INCONCLUSIVE"
        return "FAIL"


@dataclass
class PreflightReport:
    """Top-level preflight outcome, renderable as text."""

    target_concurrency: int
    num_client_threads: object
    delivery_lag_threshold_ms: float
    max_unpaired_fraction: float
    #: Gate for the ASR/TTS C1 pacing terms (scheduling, not transport).
    pacing_drift_threshold_ms: float = float("nan")
    #: Gate for the multi-turn think-time C1 check (scheduling quantity).
    think_time_drift_threshold_ms: float = float("nan")
    #: Below this served fraction the mock could not sustain the load and the
    #: check is engine-limited (inconclusive), not a client failure.
    min_served_fraction: float = float("nan")
    #: Gate for the C1 dispatch check, looser than ``delivery_lag_threshold_ms``
    #: on purpose (a late request still measures its own latency correctly; a
    #: late chunk stamp corrupts TPOT).
    dispatch_drift_threshold_ms: float = float("nan")
    checks: List[CheckResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """False only if some gated check was DISHONEST at the target."""
        return not any(check.failed for check in self.checks)

    @property
    def inconclusive(self) -> bool:
        return any(
            not check.skipped and check.verdict_at_target == VERDICT_ENGINE_LIMITED
            for check in self.checks
        )

    # ------------------------------------------------------------------ render
    def format_text(self) -> str:
        lines: List[str] = []
        headline = "PASS" if self.passed else "FAIL"
        if self.passed and self.inconclusive:
            headline = "PASS (with inconclusive checks)"
        lines.append("=" * 78)
        lines.append(f"VEEKSHA MEASUREMENT PREFLIGHT - {headline}")
        lines.append("=" * 78)
        lines.append(
            f"target concurrency: {self.target_concurrency}   "
            f"num_client_threads: {self.num_client_threads}"
        )
        lines.append(
            f"thresholds: delivery lag p99 < {self.delivery_lag_threshold_ms:.1f}ms, "
            f"unpaired < {self.max_unpaired_fraction * 100:.0f}%, "
            f"dispatch drift p99 < {self.dispatch_drift_threshold_ms:.1f}ms, "
            f"pacing p99 < {self.pacing_drift_threshold_ms:.1f}ms, "
            f"think-time drift p99 < {self.think_time_drift_threshold_ms:.1f}ms, "
            f"served >= {self.min_served_fraction * 100:.0f}%"
        )
        lines.append("")
        lines.extend(self._legend())
        for check in self.checks:
            lines.extend(self._check_block(check))
        if self.recommendations:
            lines.append("recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
            lines.append("")
        return "\n".join(lines)

    def _legend(self) -> List[str]:
        return [
            "COLUMN LEGEND (a dash = not measured / does not apply to this "
            "check; read each check's note):",
            "  ach      peak SIMULTANEOUS connections the mock server observed. "
            "Ground truth for offered load;",
            "           a count of completed requests is NOT concurrency.",
            "  cmpl     requests that produced scoreable timing data",
            "  c2P99    C2 REQUEST-delivery lag p99 (ms), GATED: client "
            "request_sent stamp -> mock receive stamp,",
            "           one per request (both taken at the first application "
            "frame, so the pairing is exact).",
            "  c2P50    C2 delivery lag p50 (ms): the loopback FLOOR, so a large "
            "constant transit cost is visible.",
            "  c3P99    C3 RESPONSE-delivery lag p99 (ms), GATED: mock send "
            "stamp -> client arrival stamp.",
            "  c3P50    C3 delivery lag p50 (ms): the loopback floor for the "
            "response direction.",
            "  dispP99  C1 schedule-adherence p99 (ms), GATED, client-side only "
            "(also holds in production):",
            "           dispatch check = how late the scheduler STARTED each "
            "request vs its seeded arrival time;",
            "           ASR = how late each audio chunk was SENT vs the 1x "
            "schedule; TTS/Vajra = how late each",
            "           input text delta was SENT vs the pacer's emulated "
            "decode cadence.",
            "  unpr     fraction of requests with NO matching server record "
            "(correlation loss, a measurement",
            "           error). GATED: a non-trivial fraction fails the check "
            "rather than shrinking the sample.",
            "  ttP99    C1 multi-turn THINK-TIME drift p99 (ms), GATED, "
            "client-side only: how late the harness",
            "           RELEASED turn 2 vs client_completed_at[turn1] + "
            "think_time. Late-only. Folds in the",
            "           completion-notify path (the analog the old "
            "completion-queue column was a proxy for),",
            "           measured directly.",
            "  ttP50    think-time drift p50 (ms): the turnaround floor.",
            "  ttSrvP99 think-time drift p99 measured at the SERVER (mock "
            "received_at[turn2] vs the same intended),",
            "           so it also carries the C2 delivery lag. REPORTED, not "
            "gated.",
            "  honest   yes = client faithful | NO = the mock served the load "
            "and the client still failed a gate |",
            "           srv-lim = mock could not serve the load, client fidelity "
            "UNMEASURED (never pass, never fail)",
            "",
            "WHAT IS AND IS NOT IN THESE NUMBERS:",
            "  delivery lag (C2/C3)  L = later stamp - earlier stamp, one host, "
            "one clock. INCLUDES the loopback",
            "                 path (write syscall, kernel, read wake) + the "
            "receiving side's scheduling delay.",
            "                 EXCLUDES parse cost on BOTH sides: each stamp sits "
            "next to its syscall, before any",
            "                 JSON/base64 work. p50 is the floor; p99 is the "
            "gated tail. Never a difference of Ls.",
            "  dispatch drift (C1)  INCLUDES scheduler wake, ready-heap pop, "
            "scheduler-Condition contention and",
            "                 DispatchWorker scheduling; EXCLUDES everything "
            "client-side (connect, send, network,",
            "                 server). Late-only: the ready-heap cannot pop "
            "early. The one check that needs no mock.",
            "  NaN gates fail safe: an unmeasured term can never read as honest.",
            "",
        ]

    def _check_block(self, check: CheckResult) -> List[str]:
        lines = [f"[{check.status}] {check.name}"]
        if check.skipped:
            if check.skip_reason:
                lines.append(f"       skipped: {check.skip_reason}")
            lines.append("")
            return lines
        point = check.point
        if point is not None:
            lines.append(
                f"       {'ach':>5} {'cmpl':>5} {'c2P99':>7} {'c2P50':>7} "
                f"{'c3P99':>7} {'c3P50':>7} {'dispP99':>8} {'unpr':>6} "
                f"{'ttP99':>7} {'ttP50':>7} {'ttSrvP99':>8}  honest"
            )
            lines.append(
                f"       {point.achieved:>5} {point.completed:>5} "
                f"{_f(point.c2_lag_p99_ms, 6, 2, 'm')} "
                f"{_f(point.c2_lag_p50_ms, 6, 2, 'm')} "
                f"{_f(point.c3_lag_p99_ms, 6, 2, 'm')} "
                f"{_f(point.c3_lag_p50_ms, 6, 2, 'm')} "
                f"{_f(point.dispatch_drift_p99_ms, 7, 2, 'm')} "
                f"{_f(point.unpaired_fraction * 100.0, 5, 1, '%')} "
                f"{_f(point.think_time_drift_p99_ms, 6, 2, 'm')} "
                f"{_f(point.think_time_drift_p50_ms, 6, 2, 'm')} "
                f"{_f(point.think_time_arrival_p99_ms, 7, 2, 'm')}  "
                f"{point.cell}"
            )
            if point.failed_terms:
                lines.append(f"         ^ failed: {', '.join(point.failed_terms)}")
            if point.note:
                lines.append(f"         ^ {point.note}")
        for note in check.notes:
            lines.append(f"       note: {note}")
        lines.append("")
        return lines
