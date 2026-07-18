"""TimedEventStream derivations must match the existing text metric formulas."""

from __future__ import annotations

import math

from veeksha.core.timed_event_stream import StreamEvent, TimedEventStream
from veeksha.types import ChannelModality


def _text_formulas(ict, num_output_tokens):
    e2e = sum(ict)
    ttfc = ict[0]
    tbc = sum(ict[1:]) / len(ict[1:]) if len(ict) > 1 else None
    tpot = (e2e - ttfc) / (num_output_tokens - 1) if num_output_tokens > 1 else None
    return e2e, ttfc, tbc, tpot


def test_matches_text_metric_formulas():
    ict = [0.5, 0.1, 0.12, 0.09, 0.11, 0.1]  # ttfc + 5 deltas
    n = len(ict)
    stream = TimedEventStream.from_inter_chunk_times(ict)

    e2e, ttfc, tbc, tpot = _text_formulas(ict, n)
    assert math.isclose(stream.end_to_end(), e2e, rel_tol=1e-9)
    assert math.isclose(stream.time_to_first_event(), ttfc, rel_tol=1e-9)
    assert math.isclose(stream.mean_inter_event(), tbc, rel_tol=1e-9)
    assert math.isclose(stream.time_per_unit(n), tpot, rel_tol=1e-9)
    # inter-event deltas equal the text delta list minus the first (TTFC) entry
    for got, want in zip(stream.inter_event_deltas(), ict[1:]):
        assert math.isclose(got, want, rel_tol=1e-9)


def test_offsets_are_cumulative():
    ict = [0.2, 0.3, 0.5]
    stream = TimedEventStream.from_inter_chunk_times(ict)
    offsets = [e.offset_s for e in stream.events]
    assert offsets == [0.2, 0.5, 1.0]


def test_rtf_undefined_for_text_defined_for_audio():
    stream = TimedEventStream.from_inter_chunk_times([0.5, 0.1, 0.1])
    assert stream.real_time_factor() is None  # text tokens have no wall duration

    # audio: size = PCM bytes; 16kHz mono 16-bit -> 32000 bytes/sec
    def bytes_to_seconds(n_bytes: int) -> float:
        return n_bytes / 32000.0

    events = [StreamEvent(offset_s=0.1 * i, size=3200) for i in range(1, 11)]
    audio = TimedEventStream(ChannelModality.AUDIO, events, bytes_to_seconds)
    # 10 events * 3200 bytes = 32000 bytes = 1.0s of audio; last offset = 1.0s
    assert math.isclose(audio.produced_content_duration_s(), 1.0, rel_tol=1e-9)
    assert math.isclose(audio.real_time_factor(), 1.0, rel_tol=1e-9)


def test_empty_stream_is_safe():
    s = TimedEventStream(ChannelModality.TEXT, [])
    assert s.time_to_first_event() is None
    assert s.end_to_end() is None
    assert s.inter_event_deltas() == []
    assert s.mean_inter_event() is None
    assert s.time_per_unit() is None
    assert s.streaming_real_time_factor() is None


def test_streaming_rtf():
    # 10 chunks, 4800 bytes each (0.1s @24kHz), arriving every 50ms.
    def bytes_to_seconds(n_bytes: int) -> float:
        return n_bytes / 48000.0

    events = [StreamEvent(offset_s=0.05 * i, size=4800) for i in range(1, 11)]
    s = TimedEventStream(ChannelModality.AUDIO, events, bytes_to_seconds)
    # wall span first->last = 0.45s; delivered after first = 1.0 - 0.1 = 0.9s
    # streaming_rtf = 0.45 / 0.9 = 0.5
    assert math.isclose(s.streaming_real_time_factor(), 0.5, rel_tol=1e-9)
