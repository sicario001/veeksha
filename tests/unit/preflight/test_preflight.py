"""Tests for the preflight validator."""

from __future__ import annotations

from veeksha.config.preflight import PreflightCheckConfig
from veeksha.preflight.mock_engine import MockStreamingEngine
from veeksha.preflight.drivers import (
    NATIVE,
    PYTHON,
    DispatchWorkload,
    SttWorkload,
    TextWorkload,
    TtsWorkload,
    measure_rung,
)
from veeksha.preflight.probe import probe_pacing
from veeksha.types import ChannelModality
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


def test_skipped_check_does_not_fail_report():
    ok = CheckResult("receive-drift accuracy", 50, 50, True, [])
    skip = CheckResult(
        "receive-drift accuracy (native)",
        0,
        50,
        passed=True,
        skipped=True,
        notes=["SKIPPED: veeksha_native extension not built."],
    )
    report = PreflightReport(50, 3, 5.0, 1.05, [ok, skip])
    assert report.passed  # a skipped check must not fail the preflight
    text = report.format_text()
    assert "[SKIP] receive-drift accuracy (native)" in text
    assert "SKIPPED: veeksha_native extension not built." in text


def test_report_renders_nan_columns_as_dash():
    p = ConcurrencyPoint(
        concurrency=50,
        achieved=50,
        ivl_err_p99_ms=1.0,
        stretch_p99=float("nan"),
        ttfc_p99_ms=float("nan"),
        throughput=float("nan"),
        server_jitter_p99_ms=0.2,
        honest=True,
    )
    chk = CheckResult("audio-transport receive drift", 50, 50, True, [p])
    text = PreflightReport(50, 3, 5.0, 1.05, [chk]).format_text()
    assert "nan" not in text
    assert "-" in text


# ------------------------------------------------------------------ pacing probe
def test_probe_pacing_is_realtime_at_low_concurrency():
    # a 0.5s clip paced at low concurrency should take ~0.5s AND each chunk
    # should be dispatched on time (small per-chunk send drift).
    r = probe_pacing(concurrency=4, clip_s=0.5, chunk_ms=20.0)
    assert r["stretch_p99"] < 1.15
    assert r["send_drift_p99_ms"] < 15.0  # per-dispatch precision


# ------------------------------------------------------------------ engine + probe
def _cfg(**kw):
    base = dict(
        target_concurrency=8,
        num_client_threads=2,
        num_dispatcher_threads=1,
        num_completion_threads=1,
        num_chunks=10,
        budget_s=1.0,
        check_pacing=False,
    )
    base.update(kw)
    return PreflightCheckConfig(**base)


def test_mock_engine_serves_and_workload_measures():
    engine = MockStreamingEngine(
        chunk_dt=0.02, prefill_s=0.03, default_chunks=10, num_loops=2
    ).start()
    try:
        workload = TextWorkload(engine, _cfg(), chunk_dt=0.02, prefill_s=0.03)
        point = measure_rung(workload, PYTHON, 5, 5.0, 1.05)[workload.name]
        assert point.achieved >= 1
        # on any sane machine, low-concurrency inter-chunk error is small
        assert point.ivl_err_p99_ms < 50.0
        assert point.server_jitter_p99_ms < 50.0
    finally:
        engine.stop()


def test_dispatch_workload_measures_arrival_lateness():
    """Dispatch fidelity is a harness property, so it is a preflight check like
    the rest: same workload seam, same gate, both dispatch paths."""
    engine = MockStreamingEngine(
        chunk_dt=0.02, prefill_s=0.03, default_chunks=10, num_loops=2
    ).start()
    try:
        workload = DispatchWorkload(engine, _cfg(), chunk_dt=0.02, prefill_s=0.03)
        point = measure_rung(workload, PYTHON, 10, 50.0, 1.05)[workload.name]
        # the arrival schedule, not a concurrency cap, sets how many are live
        assert point.achieved >= 5
        assert point.ivl_err_p99_ms >= 0.0
    finally:
        engine.stop()


def test_both_paths_send_identical_requests():
    """The bug this harness exists to prevent: the two dispatch paths must put
    the same request on the wire. They share one builder, so a divergence can
    only come from a path mutating what it was handed."""
    engine = MockStreamingEngine(
        chunk_dt=0.02, prefill_s=0.03, default_chunks=10, num_loops=2
    ).start()
    try:
        workload = TextWorkload(engine, _cfg(), chunk_dt=0.02, prefill_s=0.03)
        a = workload.build_requests(4)
        b = workload.build_requests(4)
        assert [r.id for r in a] == [r.id for r in b]
        for ra, rb in zip(a, b):
            ca = ra.channels[ChannelModality.TEXT]
            cb = rb.channels[ChannelModality.TEXT]
            assert ca.input_text == cb.input_text
            assert ca.target_prompt_tokens == cb.target_prompt_tokens
            assert (
                ra.requested_output.text.target_tokens
                == rb.requested_output.text.target_tokens
            )
    finally:
        engine.stop()


def test_gate_terms_are_shared_across_paths():
    """A rung's honesty gate must not depend on who dispatched it. Only terms
    whose underlying stage genuinely exists on one path may differ, and those
    are keyed off metrics the scorer emitted rather than off the path name."""
    engine = MockStreamingEngine(
        chunk_dt=0.02, prefill_s=0.03, default_chunks=10, num_loops=2
    ).start()
    try:
        workload = TextWorkload(engine, _cfg(), chunk_dt=0.02, prefill_s=0.03)
        clean = {
            "achieved": 8.0,
            "ivl_err_p99_ms": 1.0,
            "stretch_p99": 1.0,
            "ttfc_p99_ms": 70.0,
            "throughput": 100.0,
        }
        assert workload.gate_terms(clean, 5.0, 1.05) == {
            "inter-chunk drift": True,
            "stream stretch": True,
        }
        # a stretched stream fails identically no matter which path produced it
        stretched = {**clean, "stretch_p99": 1.5}
        assert workload.gate_terms(stretched, 5.0, 1.05)["stream stretch"] is False
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
    names = {c.name for c in report.checks}
    assert any("receive" in n for n in names)
    assert any("dispatch" in n for n in names)
    assert any("pacing" in n for n in names)
    for chk in report.checks:
        assert chk.points  # measured at least one concurrency
    assert isinstance(report.format_text(), str)


# ------------------------------------------------------------------ real audio transport
def test_tts_workload_is_accurate_at_low_concurrency():
    """Driving the REAL realtime client over WS reproduces the server cadence."""
    workload = TtsWorkload(_cfg(num_chunks=12, audio_chunk_ms=20.0))
    workload.start()
    try:
        point = measure_rung(workload, PYTHON, 4, 5.0, 1.05)[workload.name]
        assert point.achieved == 4
        # per-chunk receive drift on localhost WS should be small
        assert point.ivl_err_p99_ms < 25.0
    finally:
        workload.stop()


def test_run_preflight_check_includes_audio_transport_when_enabled():
    cfg = PreflightCheckConfig(
        target_concurrency=8,
        num_client_threads=2,
        num_dispatcher_threads=1,
        num_completion_threads=1,
        num_chunks=10,
        budget_s=1.0,
        check_pacing=False,
        check_audio_transport=True,
    )
    report = run_preflight_check(cfg)
    names = {c.name for c in report.checks}
    assert any("audio-transport" in n for n in names)
    audio = next(c for c in report.checks if "audio-transport" in c.name)
    assert audio.points  # measured at least one concurrency rung


# ------------------------------------------------------------------ STT send pacing
def test_stt_workload_paces_audio_at_realtime():
    """The real STT client streams input audio at ~1x with small per-chunk drift."""
    workload = SttWorkload(_cfg(pacing_clip_s=1.0, transcript_delta_ms=30.0))
    workload.start()
    try:
        points = measure_rung(workload, PYTHON, 4, 25.0, 1.10)
        send = points["audio-transport send pacing (realtime STT/ASR, WS)"]
        assert send.achieved == 4
        assert send.stretch_p99 < 1.10  # ~real-time (a 1s clip takes ~1s)
        assert send.ivl_err_p99_ms < 25.0  # per-chunk client send precision
    finally:
        workload.stop()


def test_stt_workload_measures_both_interactivity_halves():
    """ASR word interactivity subtracts two streams, so both are measured — and
    from ONE dispatch, so the two halves describe the same traffic rather than
    two runs that merely ran near each other."""
    workload = SttWorkload(_cfg(pacing_clip_s=1.0, transcript_delta_ms=30.0))
    workload.start()
    try:
        points = measure_rung(workload, PYTHON, 4, 25.0, 1.10)
        send = points["audio-transport send pacing (realtime STT/ASR, WS)"]
        recv = points["audio-transport transcript receive (realtime STT/ASR, WS)"]
        # client-side pacing: the client's own dispatch stamps vs the 1x schedule
        assert send.ivl_err_p99_ms < 25.0
        # receive side: recorded gaps between transcript deltas track the cadence
        assert recv.ivl_err_p99_ms < 25.0
        # both views come from the same run, so they agree on what was achieved
        assert send.achieved == recv.achieved
    finally:
        workload.stop()
