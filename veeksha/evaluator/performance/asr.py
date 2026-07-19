"""ASR-specific scoring helpers for realtime speech-to-text benchmarks."""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional

import jiwer

from veeksha.evaluator.performance.asr_interactivity import compute_interactivity_stats
from veeksha.evaluator.performance.asr_normalizer import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


@dataclass(frozen=True)
class WERStats:
    """Edit counts and WER percentage for one normalized transcript comparison."""

    errors: int
    reference_words: int
    wer: float


def compute_wer_stats(reference: str, hypothesis: str) -> WERStats:
    """WER counts and percentage using the leaderboard normalizer + jiwer."""
    ref = _normalizer(reference)
    hyp = _normalizer(hypothesis)
    output = jiwer.process_words(ref, hyp)
    errors = output.substitutions + output.deletions + output.insertions
    reference_words = output.hits + output.substitutions + output.deletions
    if reference_words == 0:
        wer = 0.0 if errors == 0 else 100.0
    else:
        wer = (errors / reference_words) * 100
    return WERStats(errors=errors, reference_words=reference_words, wer=wer)


@dataclass
class WERAggregate:
    """Accumulates multiple WER aggregation modes for comparable ASR samples."""

    sample_count: int = 0
    wer_sum: float = 0.0
    duration_weighted_wer_sum: float = 0.0
    duration_s_sum: float = 0.0
    errors: int = 0
    reference_words: int = 0

    def add(self, stats: WERStats, duration_s: float) -> None:
        self.sample_count += 1
        self.wer_sum += stats.wer
        self.errors += stats.errors
        self.reference_words += stats.reference_words
        if duration_s > 0:
            self.duration_weighted_wer_sum += stats.wer * duration_s
            self.duration_s_sum += duration_s

    def summary(self, prefix: str) -> Dict[str, Optional[float]]:
        sample_mean = (
            self.wer_sum / self.sample_count if self.sample_count > 0 else None
        )
        corpus = (
            (self.errors / self.reference_words) * 100
            if self.reference_words > 0
            else None
        )
        duration_weighted = (
            self.duration_weighted_wer_sum / self.duration_s_sum
            if self.duration_s_sum > 0
            else None
        )
        return {
            f"{prefix}_sample_count": float(self.sample_count),
            f"{prefix}_sample_mean_wer": sample_mean,
            f"{prefix}_corpus_wer": corpus,
            f"{prefix}_duration_weighted_wer": duration_weighted,
        }


@dataclass
class ASRScoredSample:
    dataset: str
    duration_s: float
    final_stats: WERStats
    partial_stats: Optional[WERStats]


@dataclass
class ASRDatasetAggregates:
    final: WERAggregate = field(default_factory=WERAggregate)
    partial: WERAggregate = field(default_factory=WERAggregate)


class ASRMetricAccumulator:
    """Builds ASR WER summaries across request-scoped ASR samples.

    Thread-safe: scoring runs concurrently on completion worker threads
    outside the evaluator locks, so sample collection guards its own state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: List[ASRScoredSample] = []

    def add_clip_sample(
        self,
        *,
        dataset: str,
        duration_s: float,
        final_stats: WERStats,
        partial_stats: Optional[WERStats],
    ) -> None:
        sample = ASRScoredSample(
            dataset=dataset,
            duration_s=duration_s,
            final_stats=final_stats,
            partial_stats=partial_stats,
        )
        with self._lock:
            self._samples.append(sample)

    def get_summary(self) -> Dict[str, Optional[float]]:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return {}

        overall = ASRDatasetAggregates()
        by_dataset: DefaultDict[str, ASRDatasetAggregates] = defaultdict(
            ASRDatasetAggregates
        )

        for sample in samples:
            overall.final.add(sample.final_stats, sample.duration_s)
            by_dataset[sample.dataset].final.add(sample.final_stats, sample.duration_s)
            if sample.partial_stats is not None:
                overall.partial.add(sample.partial_stats, sample.duration_s)
                by_dataset[sample.dataset].partial.add(
                    sample.partial_stats, sample.duration_s
                )

        summary = {}
        summary.update(overall.final.summary("asr_final"))
        summary.update(overall.partial.summary("asr_partial"))
        for dataset, aggregates in sorted(by_dataset.items()):
            slug = _metric_slug(dataset)
            summary.update(aggregates.final.summary(f"asr_dataset_{slug}_final"))
            summary.update(aggregates.partial.summary(f"asr_dataset_{slug}_partial"))
        return summary


def _metric_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


@dataclass
class ASRRequestMetrics:
    """Request-level ASR fields persisted into request_level_metrics.jsonl."""

    audio_file: Optional[str]
    final_transcript: str
    expected_transcript: str
    partial_transcript: Optional[str]
    reference_word_timestamps: Optional[List[Dict[str, Any]]]
    transcript_snapshots: List[Dict[str, Any]]
    final_wer: Optional[float]
    partial_wer: Optional[float]
    dataset: str
    source_id: Optional[str]
    sample_id: Optional[str]
    time_to_first_visible_text: Optional[float]
    time_to_first_partial: Optional[float]
    time_to_final_transcript: Optional[float]
    interactivity: Optional[float]
    interactivity_word_count: int

    def to_request_row(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_id": self.source_id,
            "sample_id": self.sample_id,
            "audio_file": self.audio_file,
            "time_to_first_visible_text": (
                round(self.time_to_first_visible_text, 3)
                if self.time_to_first_visible_text is not None
                else None
            ),
            "time_to_first_partial": (
                round(self.time_to_first_partial, 3)
                if self.time_to_first_partial is not None
                else None
            ),
            "time_to_final_transcript": (
                round(self.time_to_final_transcript, 3)
                if self.time_to_final_transcript is not None
                else None
            ),
            "interactivity": (
                round(self.interactivity, 3) if self.interactivity is not None else None
            ),
            "interactivity_word_count": self.interactivity_word_count,
            "partial_transcript": self.partial_transcript,
            "final_transcript": self.final_transcript,
            "expected_transcript": self.expected_transcript,
            "reference_word_timestamps": self.reference_word_timestamps,
            "transcript_snapshots": self.transcript_snapshots,
            "partial_wer": (
                round(self.partial_wer, 3) if self.partial_wer is not None else None
            ),
            "final_wer": (
                round(self.final_wer, 3) if self.final_wer is not None else None
            ),
        }


def score_asr_request(
    *,
    request_id: int,
    channel_metrics: Dict[str, Any],
    duration_s: float,
    accumulator: ASRMetricAccumulator,
) -> ASRRequestMetrics:
    """Validate one STT response, score it, and add it to ASR aggregates."""
    final_transcript = channel_metrics.get("final_transcript")
    expected_transcript = channel_metrics.get("expected_transcript")
    if final_transcript is None or expected_transcript is None:
        raise ValueError(
            f"STT response for request {request_id} missing "
            f"final_transcript={final_transcript!r} / "
            f"expected_transcript={expected_transcript!r}."
        )

    partial_transcript = channel_metrics.get("partial_transcript")
    dataset = str(channel_metrics.get("dataset") or "unknown")
    audio_file = _optional_str(channel_metrics.get("audio_file"))
    source_id = _optional_str(channel_metrics.get("source_id"))
    sample_id = _optional_str(channel_metrics.get("sample_id"))
    reference_word_timestamps = _optional_list_of_dicts(
        channel_metrics.get("reference_word_timestamps")
    )
    transcript_snapshots = _list_of_dicts_or_empty(
        channel_metrics.get("transcript_snapshots")
    )
    time_to_first_visible_text = _optional_float(
        channel_metrics.get("time_to_first_visible_text")
    )
    time_to_first_partial = _optional_float(
        channel_metrics.get("time_to_first_partial")
    )
    time_to_final_transcript = _optional_float(
        channel_metrics.get("time_to_final_transcript")
    )
    interactivity_stats = compute_interactivity_stats(channel_metrics)
    interactivity = (
        interactivity_stats.mean_latency_ms if interactivity_stats is not None else None
    )
    interactivity_word_count = (
        interactivity_stats.word_count if interactivity_stats is not None else 0
    )

    final_stats = compute_wer_stats(str(expected_transcript), str(final_transcript))
    final_wer = final_stats.wer
    partial_stats = None
    partial_wer = None
    if partial_transcript:
        partial_stats = compute_wer_stats(
            str(expected_transcript), str(partial_transcript)
        )
        partial_wer = partial_stats.wer
    accumulator.add_clip_sample(
        dataset=dataset,
        duration_s=duration_s,
        final_stats=final_stats,
        partial_stats=partial_stats,
    )

    return ASRRequestMetrics(
        audio_file=audio_file,
        final_transcript=str(final_transcript),
        expected_transcript=str(expected_transcript),
        partial_transcript=partial_transcript,
        reference_word_timestamps=reference_word_timestamps,
        transcript_snapshots=transcript_snapshots,
        final_wer=final_wer,
        partial_wer=partial_wer,
        dataset=dataset,
        source_id=source_id,
        sample_id=sample_id,
        time_to_first_visible_text=time_to_first_visible_text,
        time_to_first_partial=time_to_first_partial,
        time_to_final_transcript=time_to_final_transcript,
        interactivity=interactivity,
        interactivity_word_count=interactivity_word_count,
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_list_of_dicts(value: Any) -> Optional[List[Dict[str, Any]]]:
    if value is None:
        return None
    return _list_of_dicts_or_empty(value)


def _list_of_dicts_or_empty(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
