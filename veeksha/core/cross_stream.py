"""Cross-stream timing metrics — derivations over TWO TimedEventStreams.

Per-stream metrics (TTF / inter-event / RTF) live on ``TimedEventStream``. The
other half of the abstraction is *cross-stream*: metrics that align an INPUT timeline against an OUTPUT
timeline. Two general operators cover every case we have and the ones we're
heading toward (ASR interactivity, voice-to-voice latency):

  * ``response_latency`` — last input event -> first output event. The
    voice-to-voice headline (user stops speaking -> agent starts speaking) and
    any turn/response latency.
  * ``visibility_latencies`` — for aligned (input, output) event pairs, how long
    after an input unit landed did its effect become visible in the output.
    ASR word interactivity is exactly this, with word-content alignment.

Offsets are unit-agnostic: the operators do arithmetic only, so callers may work
in seconds (TimedEventStream) or milliseconds (ASR snapshots) as long as both
sides of a pair share the unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from veeksha.core.timed_event_stream import TimedEventStream


def response_latency(
    input_stream: TimedEventStream, output_stream: TimedEventStream
) -> Optional[float]:
    """Time from the last INPUT event to the first OUTPUT event.

    This is the general turn-latency operator: for voice-to-voice it is the
    end-of-user-speech -> start-of-agent-speech latency; for a single request it
    reduces to the output stream's time-to-first-event when the input is empty.
    Returns None when either stream has no events. May be negative under overlap
    (barge-in): the caller decides whether to clamp.
    """
    if not input_stream.events or not output_stream.events:
        return None
    last_input = max(e.offset_s for e in input_stream.events)
    first_output = min(e.offset_s for e in output_stream.events)
    return first_output - last_input


@dataclass(frozen=True)
class VisibilityStats:
    """Aggregate of per-unit visibility latencies (shared, unit-agnostic)."""

    mean_latency: float
    count: int
    latencies: List[float]


def visibility_latencies(
    pairs: Sequence[Tuple[float, float]],
) -> Optional[VisibilityStats]:
    """Latency between each aligned (input_offset, output_offset) pair.

    ``pairs`` are (when the input unit was complete, when it became visible in the
    output). Each latency is clamped at 0 (an output can't precede its cause).
    Returns None when there are no pairs. Both offsets in a pair must share a unit.
    """
    latencies = [max(0.0, out_offset - in_offset) for in_offset, out_offset in pairs]
    if not latencies:
        return None
    return VisibilityStats(
        mean_latency=sum(latencies) / len(latencies),
        count=len(latencies),
        latencies=latencies,
    )
