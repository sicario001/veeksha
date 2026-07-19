"""Cross-stream metrics: the general input->output timing operators.

Proves interactivity (ASR) and voice latency (voice-to-voice) are the SAME two
primitives — response_latency and visibility_latencies — not per-domain code.
"""

from __future__ import annotations

import math

from veeksha.core.cross_stream import (
    INPUT,
    OUTPUT,
    VisibilityStats,
    events_of_kind,
    response_latency,
    visibility_latencies,
)
from veeksha.core.timed_event_stream import StreamEvent, TimedEventStream
from veeksha.types import ChannelModality


def _stream(offsets_kinds):
    return TimedEventStream(
        ChannelModality.AUDIO,
        [StreamEvent(offset_s=o, size=1, kind=k) for o, k in offsets_kinds],
    )


# ------------------------------------------------------------------ split by kind
def test_events_of_kind_splits_input_and_output():
    mixed = _stream([(0.0, INPUT), (0.5, INPUT), (0.7, OUTPUT), (1.2, OUTPUT)])
    assert len(events_of_kind(mixed, INPUT)) == 2
    assert len(events_of_kind(mixed, OUTPUT)) == 2


# ------------------------------------------------------------------ response latency
def test_response_latency_last_input_to_first_output():
    inp = _stream([(0.0, INPUT), (1.5, INPUT)])  # user speaks until t=1.5s
    out = _stream([(1.9, OUTPUT), (2.4, OUTPUT)])  # agent starts at t=1.9s
    # voice latency = first_output(1.9) - last_input(1.5) = 0.4s
    assert math.isclose(response_latency(inp, out), 0.4, rel_tol=1e-9)


def test_response_latency_none_when_a_stream_is_empty():
    inp = _stream([(0.0, INPUT)])
    out = _stream([])
    assert response_latency(inp, out) is None
    assert response_latency(out, inp) is None


def test_response_latency_can_be_negative_under_barge_in():
    inp = _stream([(0.0, INPUT), (2.0, INPUT)])  # user still speaking at 2.0
    out = _stream([(1.5, OUTPUT)])  # agent interrupts at 1.5
    assert response_latency(inp, out) < 0


# ------------------------------------------------------------------ visibility latencies
def test_visibility_latencies_mean_and_clamp():
    # input completes at (100, 200); visible at (150, 500) => latencies 50, 300
    stats = visibility_latencies([(100.0, 150.0), (200.0, 500.0)])
    assert isinstance(stats, VisibilityStats)
    assert stats.count == 2
    assert math.isclose(stats.mean_latency, (50.0 + 300.0) / 2)
    # an output that precedes its input clamps to 0
    clamped = visibility_latencies([(300.0, 100.0)])
    assert clamped.latencies == [0.0]


def test_visibility_latencies_none_when_empty():
    assert visibility_latencies([]) is None


# ------------------------------------------------------------------ interactivity == the operator
def test_asr_interactivity_uses_the_shared_operator():
    """compute_interactivity_stats produces the same numbers as the primitive."""
    from veeksha.evaluator.performance.asr_interactivity import (
        compute_interactivity_stats,
    )

    ref = [
        {"word": "hello", "start_ms": 0.0, "end_ms": 200.0},
        {"word": "world", "start_ms": 220.0, "end_ms": 500.0},
    ]
    snaps = [
        {"elapsed_ms": 300.0, "transcript": "hello"},
        {"elapsed_ms": 700.0, "transcript": "hello world"},
    ]
    stats = compute_interactivity_stats(
        {"reference_word_timestamps": ref, "transcript_snapshots": snaps}
    )
    assert stats is not None
    # hello: seen 300 - end 200 = 100ms; world: seen 700 - end 500 = 200ms
    expected = visibility_latencies([(200.0, 300.0), (500.0, 700.0)])
    assert math.isclose(stats.mean_latency_ms, expected.mean_latency)
    assert stats.word_count == expected.count
