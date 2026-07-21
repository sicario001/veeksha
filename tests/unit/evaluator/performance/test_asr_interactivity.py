import random
from typing import Any, Dict

import pytest

from veeksha.evaluator.performance.asr import (
    ASRMetricAccumulator,
    compute_interactivity_stats,
    score_asr_request,
)


def _streaming_fixture() -> Dict[str, Any]:
    """Deterministic synthetic streaming-STT request.

    Exercises growing partial transcripts, unstable tail rewrites, duplicated
    snapshots, filler/number/contraction normalization, and imperfect
    hypotheses. The expected metric values in the regression test below were
    produced by the original (pre-optimization) scoring implementation.
    """
    rng = random.Random(42)
    vocab = (
        "the quick brown fox jumps over lazy dog and a half twenty one "
        "dollars mr smith won't say uh hundred and three point five st "
        "colour center o'clock nineteen sixty"
    ).split()
    ref_words = [vocab[rng.randrange(len(vocab))] for _ in range(120)]
    reference_word_timestamps = [
        {"word": word, "start_ms": 250.0 * i, "end_ms": 250.0 * i + 200.0}
        for i, word in enumerate(ref_words)
    ]
    hyp_words = [
        word if rng.random() > 0.1 else vocab[rng.randrange(len(vocab))]
        for word in ref_words
    ]
    snapshots = []
    shown_count = 0
    elapsed = 0.0
    while shown_count < len(hyp_words):
        shown_count = min(len(hyp_words), shown_count + rng.randrange(1, 4))
        elapsed += rng.uniform(300.0, 700.0)
        shown = list(hyp_words[:shown_count])
        if rng.random() < 0.3 and shown_count > 3:
            shown[-1] = vocab[rng.randrange(len(vocab))]  # unstable tail
        snapshots.append(
            {"elapsed_ms": round(elapsed, 3), "transcript": " ".join(shown)}
        )
        if rng.random() < 0.15:  # repeated snapshot at a later timestamp
            elapsed += 120.0
            snapshots.append(
                {"elapsed_ms": round(elapsed, 3), "transcript": " ".join(shown)}
            )
    return {
        "final_transcript": " ".join(hyp_words),
        "expected_transcript": " ".join(ref_words),
        "partial_transcript": " ".join(hyp_words[:80]),
        "dataset": "fixture",
        "reference_word_timestamps": reference_word_timestamps,
        "transcript_snapshots": snapshots,
    }


@pytest.mark.unit
def test_compute_interactivity_stats_from_snapshots() -> None:
    stats = compute_interactivity_stats(
        {
            "reference_word_timestamps": [
                {"word": "hello", "start_ms": 0, "end_ms": 100},
                {"word": "world", "start_ms": 200, "end_ms": 300},
            ],
            "transcript_snapshots": [
                {"elapsed_ms": 150, "transcript": "hello"},
                {"elapsed_ms": 320, "transcript": "hello world"},
            ],
        }
    )

    assert stats is not None
    assert stats.word_count == 2
    assert stats.latencies_ms == [50, 20]
    assert stats.mean_latency_ms == 35


@pytest.mark.unit
def test_compute_interactivity_expands_and_skips_normalized_reference_words() -> None:
    stats = compute_interactivity_stats(
        {
            "reference_word_timestamps": [
                {"word": "don't", "start_ms": 0, "end_ms": 100},
                {"word": "uh", "start_ms": 120, "end_ms": 180},
                {"word": "stop", "start_ms": 200, "end_ms": 300},
            ],
            "transcript_snapshots": [
                {"elapsed_ms": 150, "transcript": "do not"},
                {"elapsed_ms": 340, "transcript": "do not stop"},
            ],
        }
    )

    assert stats is not None
    assert stats.word_count == 3
    assert stats.latencies_ms == [50, 50, 40]
    assert stats.mean_latency_ms == pytest.approx(46.6666667)


@pytest.mark.unit
def test_compute_interactivity_stats_empty_snapshots_returns_none() -> None:
    stats = compute_interactivity_stats(
        {
            "reference_word_timestamps": [
                {"word": "hello", "start_ms": 0, "end_ms": 100},
            ],
            "transcript_snapshots": [],
        }
    )

    assert stats is None


@pytest.mark.unit
def test_score_asr_request_writes_interactivity_row_fields() -> None:
    metrics = score_asr_request(
        request_id=0,
        channel_metrics={
            "final_transcript": "hello world",
            "expected_transcript": "hello world",
            "dataset": "toy",
            "reference_word_timestamps": [
                {"word": "hello", "start_ms": 0, "end_ms": 100},
                {"word": "world", "start_ms": 200, "end_ms": 300},
            ],
            "transcript_snapshots": [
                {"elapsed_ms": 150, "transcript": "hello"},
                {"elapsed_ms": 320, "transcript": "hello world"},
            ],
        },
        duration_s=0.3,
        accumulator=ASRMetricAccumulator(),
    )

    row = metrics.to_request_row()
    assert row["interactivity"] == 35
    assert row["interactivity_word_count"] == 2


@pytest.mark.unit
def test_streaming_fixture_matches_pre_optimization_values() -> None:
    """Pin exact metric values produced by the original scoring code.

    The interactivity/WER implementations were optimized for speed; this
    fixture asserts bit-identical numerical results against values captured
    from the pre-optimization implementation on the same input.
    """
    channel_metrics = _streaming_fixture()
    assert len(channel_metrics["transcript_snapshots"]) == 70

    stats = compute_interactivity_stats(channel_metrics)
    assert stats is not None
    assert stats.word_count == 96
    assert stats.mean_latency_ms == 2057.4468854166666
    assert sum(stats.latencies_ms) == 197514.90099999998

    accumulator = ASRMetricAccumulator()
    metrics = score_asr_request(
        request_id=0,
        channel_metrics=channel_metrics,
        duration_s=30.0,
        accumulator=accumulator,
    )
    assert metrics.final_wer == 13.513513513513514
    assert metrics.partial_wer == 42.34234234234234

    summary = accumulator.get_summary()
    assert summary["asr_final_corpus_wer"] == 13.513513513513514
    assert summary["asr_final_sample_mean_wer"] == 13.513513513513514
    assert summary["asr_final_duration_weighted_wer"] == 13.513513513513514
    assert summary["asr_partial_corpus_wer"] == 42.34234234234234
    assert summary["asr_final_sample_count"] == 1.0
