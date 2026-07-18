"""TimedEventStream — the modality-agnostic timing primitive.

Every streaming response (text tokens, audio chunks, video frames) is a timeline
of ``(offset, size)`` events. Making that the canonical primitive lets the
headline metrics be *one* set of derivations across modalities (doc
``analysis/05_proposed_abstraction.md`` §2):

  * first-event latency   -> TTFT / TTFA / TTFC
  * inter-event deltas     -> TPOT / inter-chunk / inter-frame
  * total                  -> end-to-end latency
  * real-time factor       -> wall / produced-content-duration (audio/video)

Absolute offsets (from the request send-anchor) are strictly richer than the
delta list text currently stores — deltas are just ``diff(offsets)`` — and they
preserve alignment for interactivity. ``from_inter_chunk_times`` bridges the two
so existing text data flows in unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from veeksha.types import ChannelModality


@dataclass
class StreamEvent:
    """One delivered unit on a channel.

    Attributes:
        offset_s: seconds from the request send-anchor (absolute, monotonic).
        size: tokens for text, PCM bytes for audio, 1 for a frame.
        kind: "output" (produced by the engine) or "input" (sent by us, e.g. the
            paced audio for ASR).
    """

    offset_s: float
    size: int = 1
    kind: str = "output"


def _tokens_have_no_wall_duration(_size: int) -> Optional[float]:
    """Default content-duration fn: undefined (text tokens have no wall length)."""
    return None


@dataclass
class TimedEventStream:
    """A modality-tagged timeline of delivered events."""

    modality: ChannelModality
    events: List[StreamEvent] = field(default_factory=list)
    # size (in this channel's units) -> produced content duration in seconds, or
    # None when undefined (text). For audio: bytes -> audio seconds.
    unit_duration_fn: Callable[[int], Optional[float]] = _tokens_have_no_wall_duration

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_inter_chunk_times(
        cls,
        inter_chunk_times: List[float],
        modality: ChannelModality = ChannelModality.TEXT,
        sizes: Optional[List[int]] = None,
        unit_duration_fn: Optional[Callable[[int], Optional[float]]] = None,
    ) -> "TimedEventStream":
        """Build a stream from text's delta list (cumulative sum -> offsets).

        ``inter_chunk_times[0]`` is the send-anchor->first-chunk time (TTFC);
        subsequent entries are chunk-to-chunk deltas. Offsets are the running sum.
        """
        offsets: List[float] = []
        acc = 0.0
        for dt in inter_chunk_times:
            acc += dt
            offsets.append(acc)
        events = [
            StreamEvent(offset_s=o, size=(sizes[i] if sizes else 1))
            for i, o in enumerate(offsets)
        ]
        return cls(
            modality=modality,
            events=events,
            unit_duration_fn=unit_duration_fn or _tokens_have_no_wall_duration,
        )

    # ---- generic derivations ------------------------------------------------
    def __len__(self) -> int:
        return len(self.events)

    def time_to_first_event(self) -> Optional[float]:
        """TTFT / TTFA / TTFC — first event offset."""
        return self.events[0].offset_s if self.events else None

    def inter_event_deltas(self) -> List[float]:
        """Gaps between consecutive events (TPOT / inter-chunk / inter-frame)."""
        return [b.offset_s - a.offset_s for a, b in zip(self.events, self.events[1:])]

    def end_to_end(self) -> Optional[float]:
        """Last event offset (from send-anchor)."""
        return self.events[-1].offset_s if self.events else None

    def total_size(self) -> int:
        return sum(e.size for e in self.events)

    def produced_content_duration_s(self) -> Optional[float]:
        return self.unit_duration_fn(self.total_size())

    def real_time_factor(self) -> Optional[float]:
        """RTF = wall time / produced content duration (None when undefined)."""
        e2e = self.end_to_end()
        produced = self.produced_content_duration_s()
        if e2e is None or not produced:
            return None
        return e2e / produced

    def mean_inter_event(self) -> Optional[float]:
        """TBC — mean of inter-event deltas (excludes the first-event latency)."""
        deltas = self.inter_event_deltas()
        return sum(deltas) / len(deltas) if deltas else None

    def time_per_unit(self, num_units: Optional[int] = None) -> Optional[float]:
        """TPOT — (E2E - TTF) / (units - 1). ``num_units`` defaults to event count."""
        n = num_units if num_units is not None else len(self.events)
        ttf = self.time_to_first_event()
        e2e = self.end_to_end()
        if ttf is None or e2e is None or n is None or n <= 1:
            return None
        return (e2e - ttf) / (n - 1)
