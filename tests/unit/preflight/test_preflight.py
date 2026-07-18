"""Tests for the preflight validator."""

from __future__ import annotations

from veeksha.config.preflight import PreflightCheckConfig
from veeksha.preflight.dummy_engine import DummyStreamingEngine
from veeksha.preflight.probe import probe_pacing, probe_receive_drift
from veeksha.preflight.report import (
    CheckResult,
    ConcurrencyPoint,
    PreflightReport,
)
from veeksha.preflight.validator import _ladder, run_preflight_check


# ------------------------------------------------------------------ pure logic
def test_ladder_includes_target_and_is_capped_at_target():
    ladder = _ladder(100)
    assert 100 in ladder
    assert ladder == sorted(ladder)
    assert max(ladder) == 100  # capped at the target, no rungs beyond it
    assert all(c <= 100 for c in ladder)


def test_report_passed_and_formatting():
    p = ConcurrencyPoint(
        concurrency=50,
        achieved=50,
        ivl_err_p99_ms=1.0,
        stretch_p99=1.01,
        ttfc_p99_ms=60,
        throughput=100,
        server_jitter_p99_ms=0.2,
        honest=True,
    )
    chk = CheckResult("receive-drift accuracy", 50, 50, True, [p])
    report = PreflightReport(50, 3, 5.0, 1.05, [chk])
    assert report.passed
    text = report.format_text()
    assert "PASS" in text
    assert "receive-drift accuracy" in text


def test_report_fails_when_any_check_fails():
    chk = CheckResult("x", 20, 100, False, [])
    report = PreflightReport(100, 3, 5.0, 1.05, [chk])
    assert not report.passed
    assert "FAIL" in report.format_text()


# ------------------------------------------------------------------ pacing probe
def test_probe_pacing_is_realtime_at_low_concurrency():
    # a 0.5s clip paced at low concurrency should take ~0.5s (ratio ~1.0)
    ratio = probe_pacing(concurrency=4, clip_s=0.5, chunk_ms=20.0)
    assert ratio < 1.15


# ------------------------------------------------------------------ engine + probe
def test_dummy_engine_serves_and_probe_measures():
    engine = DummyStreamingEngine(
        chunk_dt=0.02, prefill_s=0.03, default_chunks=10, num_loops=2
    ).start()
    try:
        m = probe_receive_drift(
            engine,
            concurrency=5,
            num_client_threads=2,
            num_dispatcher_threads=1,
            num_completion_threads=1,
            num_chunks=10,
            chunk_dt=0.02,
            prefill_s=0.03,
            n_requests=30,
            max_connections=None,
        )
        assert m["achieved"] >= 1
        # on any sane machine, low-concurrency inter-chunk error is small
        assert m["ivl_err_p99_ms"] < 50.0
        assert m["server_jitter_p99_ms"] < 50.0
    finally:
        engine.stop()


# ------------------------------------------------------------------ end to end
def test_run_preflight_check_end_to_end_small():
    cfg = PreflightCheckConfig(
        target_concurrency=10,
        num_client_threads=2,
        num_dispatcher_threads=1,
        num_completion_threads=1,
        num_chunks=10,
        budget_s=1.5,
        pacing_clip_s=0.5,
    )
    report = run_preflight_check(cfg)
    # structural guarantees (not timing-strict, to avoid CI flakiness)
    assert len(report.checks) == 2
    names = {c.name for c in report.checks}
    assert any("receive" in n for n in names)
    assert any("pacing" in n for n in names)
    for chk in report.checks:
        assert chk.points  # measured at least one concurrency
    assert isinstance(report.format_text(), str)
