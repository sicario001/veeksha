"""Interactivity scoring for realtime speech-to-text transcripts."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Optional

from veeksha.evaluator.performance.asr_normalizer import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass(frozen=True)
class ReferenceWord:
    word: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class TranscriptSnapshot:
    elapsed_ms: float
    transcript: str


@dataclass(frozen=True)
class InteractivityStats:
    mean_latency_ms: float
    word_count: int
    latencies_ms: list[float]


def compute_interactivity_stats(
    channel_metrics: dict[str, Any],
) -> Optional[InteractivityStats]:
    """Compute word visibility latency for one STT request.

    Missing ``reference_word_timestamps`` means the trace does not support this
    metric, so this returns ``None``. Missing or empty snapshots mean the metric
    is unavailable for this request.
    """
    if channel_metrics.get("reference_word_timestamps") is None:
        return None

    reference_words = _expand_reference_words(
        _parse_reference_words(channel_metrics["reference_word_timestamps"])
    )
    snapshot_rows = channel_metrics.get("transcript_snapshots")
    if snapshot_rows is None:
        return None
    snapshots = _parse_transcript_snapshots(snapshot_rows)
    if not snapshots:
        return None

    if not reference_words:
        return None

    reference_tokens = [word.word for word in reference_words]
    first_seen_ms: list[Optional[float]] = [None] * len(reference_tokens)

    # Intern tokens as small ints: SequenceMatcher only relies on element
    # equality/hashing, so a bijective token->id mapping shared between the
    # reference and every hypothesis yields identical matching blocks while
    # hashing and comparing much faster than strings.
    token_ids: dict[str, int] = {}
    reference_token_ids = [
        token_ids.setdefault(token, len(token_ids)) for token in reference_tokens
    ]

    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq1(reference_token_ids)

    previous_transcript: Optional[str] = None
    previous_hypothesis_ids: Optional[list[int]] = None
    previous_blocks: Optional[list[difflib.Match]] = None

    for snapshot in sorted(snapshots, key=lambda item: item.elapsed_ms):
        # Streaming partials often repeat the previous transcript (or reduce
        # to the same normalized token sequence); the matching blocks depend
        # only on the token sequences, so reuse them.
        if snapshot.transcript == previous_transcript:
            hypothesis_ids = previous_hypothesis_ids
        else:
            hypothesis_ids = [
                token_ids.setdefault(token, len(token_ids))
                for token in _normalize_words(snapshot.transcript)
            ]
            previous_transcript = snapshot.transcript
        if not hypothesis_ids:
            previous_hypothesis_ids = hypothesis_ids
            previous_blocks = []
            continue
        if hypothesis_ids == previous_hypothesis_ids:
            blocks = previous_blocks
        else:
            matcher.set_seq2(hypothesis_ids)
            # Equal-opcode ranges are exactly the matching blocks (the final
            # sentinel block has size 0 and marks nothing).
            blocks = matcher.get_matching_blocks()
            previous_hypothesis_ids = hypothesis_ids
            previous_blocks = blocks

        elapsed_ms = snapshot.elapsed_ms
        for ref_start, _hyp_start, block_size in blocks:
            for ref_index in range(ref_start, ref_start + block_size):
                if (
                    first_seen_ms[ref_index] is None
                    and elapsed_ms >= reference_words[ref_index].start_ms
                ):
                    first_seen_ms[ref_index] = elapsed_ms

    latencies = [
        max(0.0, seen_ms - word.end_ms)
        for seen_ms, word in zip(first_seen_ms, reference_words)
        if seen_ms is not None
    ]
    if not latencies:
        return None
    return InteractivityStats(
        mean_latency_ms=sum(latencies) / len(latencies),
        word_count=len(latencies),
        latencies_ms=latencies,
    )


def _parse_reference_words(value: Any) -> list[ReferenceWord]:
    rows = _expect_list(value, "reference_word_timestamps")
    if not rows:
        raise ValueError("reference_word_timestamps must not be empty.")

    words: list[ReferenceWord] = []
    for index, row in enumerate(rows):
        item = _expect_dict(row, f"reference_word_timestamps[{index}]")
        words.append(
            ReferenceWord(
                word=_expect_str(item, "word", index),
                start_ms=_expect_float(item, "start_ms", index),
                end_ms=_expect_float(item, "end_ms", index),
            )
        )
    return words


def _parse_transcript_snapshots(value: Any) -> list[TranscriptSnapshot]:
    rows = _expect_list(value, "transcript_snapshots")
    snapshots: list[TranscriptSnapshot] = []
    for index, row in enumerate(rows):
        item = _expect_dict(row, f"transcript_snapshots[{index}]")
        snapshots.append(
            TranscriptSnapshot(
                elapsed_ms=_expect_float(item, "elapsed_ms", index),
                transcript=_expect_str(item, "transcript", index),
            )
        )
    return snapshots


def _expect_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list, got {type(value).__name__}.")
    return value


def _expect_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object, got {type(value).__name__}.")
    return value


def _expect_str(item: dict[str, Any], field: str, index: int) -> str:
    value = item[field]
    if not isinstance(value, str):
        raise TypeError(f"{field} at index {index} must be str.")
    return value


def _expect_float(item: dict[str, Any], field: str, index: int) -> float:
    value = item[field]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} at index {index} must be numeric.") from exc


def _expand_reference_words(words: list[ReferenceWord]) -> list[ReferenceWord]:
    expanded: list[ReferenceWord] = []
    for word in words:
        expanded.extend(
            ReferenceWord(
                word=token,
                start_ms=word.start_ms,
                end_ms=word.end_ms,
            )
            for token in _normalize_words(word.word)
        )
    return expanded


def _normalize_words(text: str) -> list[str]:
    return _WORD_RE.findall(_normalizer(text))
