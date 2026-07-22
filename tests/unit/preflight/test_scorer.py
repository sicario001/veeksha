"""Tests for the preflight scoring math.

Every expected value here is computed by hand, because these functions ARE the
definitions of the three checks — the reconstruction of absolute client times
and the delivery lag between two paired stamps. If they drift, every verdict
the preflight prints is wrong in a way no integration test would notice.
"""

from __future__ import annotations

import math

import pytest

from veeksha.preflight.scorer import (
    absolute_times_from_offsets_ms,
    asr_send_schedule_ms,
    audio_arrival_times,
    delivery_lag_ms,
    delivery_lags_ms,
    dispatch_drift_ms,
    p50,
    p99,
    percentile_or_nan,
    rate_schedule_offsets_ms,
    send_drift,
    stt_send_times,
    text_arrival_times,
    text_pacing_schedule_ms,
    think_time_drift_ms,
)
from veeksha.preflight.sharded_server import percentile

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------ delivery lag
def test_delivery_lag_is_later_minus_earlier_in_ms() -> None:
    # 0.5 ms on the wire: the receiver stamped 0.0005 s after the sender.
    assert delivery_lag_ms(100.000, 100.0005) == pytest.approx(0.5)


def test_delivery_lag_is_not_clamped_so_broken_pairing_is_visible() -> None:
    # A correctly paired event is always positive; a negative is EVIDENCE the
    # pairing (or its anchor) is wrong, and must surface rather than read as 0.
    assert delivery_lag_ms(100.001, 100.000) == pytest.approx(-1.0)


def test_delivery_lags_pair_positionally_and_stop_at_the_shorter() -> None:
    earlier = [10.0, 10.010, 10.020]
    later = [10.001, 10.012, 10.021]
    assert delivery_lags_ms(earlier, later) == pytest.approx([1.0, 2.0, 1.0])
    # Only the events both sides saw are scored.
    assert delivery_lags_ms([10.0, 10.010], [10.001]) == pytest.approx([1.0])
    assert delivery_lags_ms([], [1.0, 2.0]) == []


# ------------------------------------------------------ absolute reconstruction
def test_text_arrival_is_the_running_sum_of_inter_chunk_times() -> None:
    # arrival[i] = start + sum(inter_chunk_times[0..i]); element 0 (TTFC) is a
    # real arrival here — it has a server send stamp to be paired with.
    start = 1000.0
    gaps_s = [0.010, 0.020, 0.021]
    assert text_arrival_times(start, gaps_s) == pytest.approx(
        [1000.010, 1000.030, 1000.051]
    )


def test_audio_arrival_adds_the_offset_to_the_anchor() -> None:
    start = 500.0
    # offsets are in MS; arrival is anchor + offset/1000.
    assert audio_arrival_times(start, [20.0, 40.0, 61.0]) == pytest.approx(
        [500.020, 500.040, 500.061]
    )


def test_stt_send_is_the_anchor_plus_send_offsets() -> None:
    anchor = 42.0
    # send_offsets_ms[0] is 0.0 by construction: the first send IS the anchor.
    assert stt_send_times(anchor, [0.0, 128.0, 256.0]) == pytest.approx(
        [42.0, 42.128, 42.256]
    )


def test_reconstruction_of_a_missing_or_nan_anchor_is_empty_not_wrong_origin() -> None:
    # An unanchored series cannot be paired; it must vanish rather than be
    # measured from the wrong origin (0.0 would pair everything against epoch).
    assert text_arrival_times(None, [0.01, 0.02]) == []
    assert audio_arrival_times(float("nan"), [10.0, 20.0]) == []
    assert stt_send_times(None, [0.0, 128.0]) == []
    assert absolute_times_from_offsets_ms(None, [1.0]) == []


def test_reconstruction_of_no_events_is_empty() -> None:
    assert text_arrival_times(1000.0, []) == []
    assert audio_arrival_times(1000.0, []) == []


# ------------------------------------------ end-to-end: paired lag over a stream
def test_reconstructed_arrival_pairs_into_a_positive_delivery_lag() -> None:
    # The mock sent chunk i at emitted[i]; the client's start anchor + running
    # gap sum lands just after each, a positive lag per chunk.
    start = 2000.0
    emitted = [2000.008, 2000.028, 2000.049]
    gaps_s = [0.010, 0.020, 0.021]  # arrivals: 2000.010, .030, .051
    arrivals = text_arrival_times(start, gaps_s)
    lags = delivery_lags_ms(emitted, arrivals)
    assert lags == pytest.approx([2.0, 2.0, 2.0])
    assert all(lag > 0.0 for lag in lags)


# ------------------------------------------------- ASR chunk-pacing (C1, client)
def test_asr_schedule_matches_the_clients_pacing_formula() -> None:
    # 4096 bytes of PCM16 at 16 kHz is 128 ms of audio.
    assert asr_send_schedule_ms(4, 4096, 16000) == pytest.approx(
        [0.0, 128.0, 256.0, 384.0]
    )
    assert asr_send_schedule_ms(0, 4096, 16000) == []
    assert asr_send_schedule_ms(4, 0, 16000) == []


def test_pacing_drift_is_late_only_against_the_intended_schedule() -> None:
    schedule = [0.0, 128.0, 256.0, 384.0]
    # chunk 1 is 2 ms late; chunk 2 is 6 ms EARLY (scored 0, not -6); chunk 3
    # is 10 ms late. Averaging the earliness in would cancel real lateness.
    actual = [0.0, 130.0, 250.0, 394.0]
    assert send_drift(actual, schedule) == pytest.approx([0.0, 2.0, 0.0, 10.0])
    assert all(v >= 0.0 for v in send_drift(actual, schedule))


def test_pacing_drift_stops_at_the_shorter_sequence() -> None:
    assert send_drift([0.0, 130.0], [0.0, 128.0, 256.0]) == pytest.approx([0.0, 2.0])
    assert send_drift([], [0.0, 128.0]) == []


# ---------------------------------------------------------------- dispatch (C1)
def test_dispatch_drift_is_late_only() -> None:
    scheduled = [0.0, 4.0, 8.0, 12.0]
    # Request 1 is 1.5 ms late; request 2 dispatched "early" — impossible for a
    # deadline heap, so it scores 0.0 rather than a negative that would cancel
    # the lateness of request 3.
    actual = [0.0, 5.5, 7.0, 20.0]
    assert dispatch_drift_ms(actual, scheduled) == pytest.approx([0.0, 1.5, 0.0, 8.0])
    assert all(value >= 0.0 for value in dispatch_drift_ms(actual, scheduled))


def test_dispatch_drift_of_a_punctual_harness_is_all_zeros() -> None:
    scheduled = [0.0, 4.0, 8.0]
    assert dispatch_drift_ms(list(scheduled), scheduled) == pytest.approx([0.0] * 3)


def test_dispatch_drift_pairs_positionally_and_stops_at_the_shorter_list() -> None:
    assert dispatch_drift_ms([0.0, 9.0], [0.0, 4.0, 8.0]) == pytest.approx([0.0, 5.0])


def test_dispatch_drift_of_nothing_is_empty_and_scores_nan() -> None:
    # No dispatch data must never read as "perfectly punctual": the gate is
    # `value < threshold`, and NaN fails every comparison.
    assert dispatch_drift_ms([], []) == []
    assert math.isnan(p99(dispatch_drift_ms([], [])))


# --------------------------------------------------- multi-turn think time (C1)
def test_think_time_drift_is_late_only_ms_against_completion_plus_think() -> None:
    # intended = completed[turn1] + think; drift = max(0, (sent2 - intended))*1000.
    # turn 1 finished at 100.0, think 0.3 -> intended release 100.3. Turn 2 went
    # out 5 ms late at 100.305 -> 5.0 ms drift.
    assert think_time_drift_ms(100.0, 100.305, 0.3) == pytest.approx(5.0)


def test_think_time_drift_of_an_on_time_or_early_release_is_zero() -> None:
    # Exactly on the intended release: 0. The scheduler cannot release turn 2
    # before completion+think, so an "early" send (a broken measurement) scores
    # 0 rather than a negative that would cancel real lateness elsewhere.
    assert think_time_drift_ms(100.0, 100.3, 0.3) == pytest.approx(0.0)
    assert think_time_drift_ms(100.0, 100.29, 0.3) == pytest.approx(0.0)


def test_think_time_drift_is_nan_when_a_stamp_is_missing() -> None:
    # A session whose turn did not produce a usable stamp is unmeasured, so its
    # drift is NaN (fails the gate safe), never a flattering 0.
    assert math.isnan(think_time_drift_ms(None, 100.3, 0.3))
    assert math.isnan(think_time_drift_ms(100.0, None, 0.3))
    assert math.isnan(think_time_drift_ms(float("nan"), 100.3, 0.3))
    # p99 of an all-NaN / empty batch is NaN too.
    assert math.isnan(p99([]))


def test_think_time_drift_server_arrival_variant_adds_the_delivery_lag() -> None:
    # Passing turn 2's SERVER received_at instead of the client send stamp gives
    # the server-arrival variant: the same intended baseline, but the value now
    # additionally carries the send->receive delivery lag. Client send at
    # 100.305 (5 ms drift); server saw it 100.307 (7 ms past intended).
    assert think_time_drift_ms(100.0, 100.307, 0.3) == pytest.approx(7.0)


# ---------------------------------------------- TTS/Vajra text-delta pacing (C1)
def test_text_pacing_schedule_is_the_fixed_cadence_anchored_on_first_delta() -> None:
    from veeksha.config.client import TextPacingConfig

    # Default fixed pacing: gap = tokens_per_delta / tokens_per_second = 1/20 s
    # = 50 ms. Offsets are relative to the first delta (offset 0 = 0.0), so
    # delta i is due i*50 ms after it.
    pacing = TextPacingConfig(tokens_per_second=20.0)
    assert text_pacing_schedule_ms(pacing, request_id=0, n_segments=4) == pytest.approx(
        [0.0, 50.0, 100.0, 150.0]
    )
    assert text_pacing_schedule_ms(pacing, request_id=7, n_segments=0) == []


def test_text_pacing_drift_pairs_against_send_drift_late_only() -> None:
    from veeksha.config.client import TextPacingConfig

    pacing = TextPacingConfig(tokens_per_second=20.0)  # 50 ms fixed gaps
    schedule = text_pacing_schedule_ms(pacing, request_id=0, n_segments=4)
    # Actual first-delta-relative offsets: delta 2 is 4 ms late, delta 3 is 3 ms
    # early (scored 0, not -3). Same late-only rule as the STT/dispatch checks.
    actual_rel = [0.0, 50.0, 104.0, 147.0]
    assert send_drift(actual_rel, schedule) == pytest.approx([0.0, 0.0, 4.0, 0.0])


# ------------------------------------------------------------------ percentiles
def test_p99_is_nearest_rank_and_p50_is_the_median() -> None:
    values = [float(v) for v in range(100)]
    # Nearest-rank over 100 samples: index round(0.99 * 99) == 98.
    assert p99(values) == pytest.approx(98.0)
    assert p50(values) == pytest.approx(50.0)
    assert percentile_or_nan(values, 0.5) == pytest.approx(50.0)


def test_p50_reports_the_loopback_floor() -> None:
    # A constant transit cost lands in the median even when the tail is clean:
    # p50 exists precisely so that a large floor is visible, not hidden.
    lags = [3.0, 3.1, 2.9, 3.0, 3.2]
    assert p50(lags) == pytest.approx(3.0)


def test_p99_and_p50_of_nothing_are_nan_not_zero() -> None:
    # The mock-server percentile returns 0.0 for "never late", which is right
    # there and catastrophic in a gate: no samples would read as perfect.
    assert percentile([], 0.99) == 0.0
    assert math.isnan(p99([]))
    assert math.isnan(p50([]))


def test_nan_fails_every_gate_comparison() -> None:
    # The property every gate term relies on: `value < threshold` is False for
    # NaN, so an unmeasured measurement fails safe instead of passing silently.
    unmeasured = p99([])
    assert not (unmeasured < 5.0)
    assert not (unmeasured >= 5.0)


# ------------------------------------------------------------------ dispatch schedule
def test_empty_schedule_for_no_sessions() -> None:
    from veeksha.config.generator.interval import PoissonIntervalGeneratorConfig

    assert rate_schedule_offsets_ms(1, PoissonIntervalGeneratorConfig(), 0) == []
