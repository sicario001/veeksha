"""AudioPerformanceEvaluator derives audio metrics from the shared primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.core import audio_contract as ac
from veeksha.evaluator.performance.audio import AudioPerformanceEvaluator
from veeksha.evaluator.performance.channel_base import BaseChannelPerformanceEvaluator
from veeksha.types import AudioTask, ChannelModality


@dataclass
class _Chan:
    metrics: Dict[str, Any]
    content: Any = None


@dataclass
class _Resp:
    channels: Dict[ChannelModality, _Chan]


def _tts_aggregate_response(ttfc_ms, e2e_ms, audio_bytes, sample_rate, raw_pcm=True):
    """Mimics the HTTP TTSClient output: aggregate metrics + raw audio content."""
    return _Resp(
        channels={
            ChannelModality.AUDIO: _Chan(
                content=audio_bytes,
                metrics={
                    ac.AUDIO_TASK: "tts",
                    ac.AudioMetricKey.TTFC: ttfc_ms,
                    ac.END_TO_END_LATENCY: e2e_ms,
                    ac.SAMPLE_RATE: sample_rate,
                    ac.AudioMetricKey.RAW_PCM: raw_pcm,
                },
            )
        }
    )


def _audio_response(timeline, sample_rate=ac.DEFAULT_AUDIO_SAMPLE_RATE):
    return _Resp(
        channels={
            ChannelModality.AUDIO: _Chan(
                metrics={
                    ac.AUDIO_CHUNK_TIMESTAMPS: timeline,
                    ac.SAMPLE_RATE: sample_rate,
                    ac.AUDIO_TASK: "tts",
                }
            )
        }
    )


def test_audio_evaluator_conforms_to_base():
    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    assert isinstance(ev, BaseChannelPerformanceEvaluator)


def test_ttfa_rtf_duration_from_timeline():
    sr = ac.DEFAULT_AUDIO_SAMPLE_RATE  # 24000; 48000 bytes/sec
    # 10 chunks, each 4800 bytes = 0.1s of audio, arriving every 50ms.
    # total 48000 bytes = 1.0s audio; last chunk at offset 500ms => e2e 0.5s.
    bytes_per_chunk = 4800
    timeline = [[50.0 * i, bytes_per_chunk] for i in range(1, 11)]  # first at 50ms
    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    ev.register_request(1, 1, 0.0, None)
    ev.record_request_completed(1, 1, 1.0, _audio_response(timeline, sr))

    result = ev.finalize()
    assert result.channel == ChannelModality.AUDIO
    assert result.metrics["num_completed_requests"] == 1

    # TTFA = first chunk offset = 50ms = 0.05s
    assert math.isclose(
        result.metrics["Time to First Audio (Mean)"], 0.05, rel_tol=1e-6
    )
    # generated audio duration = 48000 bytes / 48000 = 1.0s
    assert math.isclose(
        result.metrics["Generated Audio Duration (Mean)"], 1.0, rel_tol=1e-6
    )
    # e2e (last offset) = 500ms = 0.5s
    assert math.isclose(result.metrics["End to End Latency (Mean)"], 0.5, rel_tol=1e-6)
    # RTF = e2e / audio_duration = 0.5 / 1.0 = 0.5  (faster than real time)
    assert math.isclose(result.metrics["Real Time Factor (Mean)"], 0.5, rel_tol=1e-6)
    # streaming RTF = wall(0.45) / delivered-after-first(0.9) = 0.5
    assert math.isclose(
        result.metrics["Streaming Real Time Factor (Mean)"], 0.5, rel_tol=1e-6
    )


def test_empty_audio_is_safe():
    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    ev.record_request_completed(1, 1, 1.0, _audio_response([]))
    result = ev.finalize()
    assert result.metrics["num_completed_requests"] == 0


def test_tts_aggregate_dialect_from_http_client():
    # The HTTP TTSClient emits TTFC + end_to_end_latency (ms) + raw audio bytes,
    # no per-chunk timeline. 24kHz -> 48000 bytes/sec; 24000 bytes = 0.5s audio.
    sr = ac.DEFAULT_AUDIO_SAMPLE_RATE
    audio = b"\x00" * 24000  # 0.5s of PCM
    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    ev.register_request(1, 1, 0.0, None)
    ev.record_request_completed(
        1,
        1,
        1.0,
        _tts_aggregate_response(
            ttfc_ms=80.0, e2e_ms=600.0, audio_bytes=audio, sample_rate=sr
        ),
    )
    m = ev.finalize().metrics
    assert m["num_completed_requests"] == 1
    assert math.isclose(m["Time to First Audio (Mean)"], 0.08, rel_tol=1e-6)  # 80ms
    assert math.isclose(m["End to End Latency (Mean)"], 0.6, rel_tol=1e-6)  # 600ms
    assert math.isclose(m["Generated Audio Duration (Mean)"], 0.5, rel_tol=1e-6)
    # RTF = 0.6 / 0.5 = 1.2 (slower than real time)
    assert math.isclose(m["Real Time Factor (Mean)"], 1.2, rel_tol=1e-6)


# ---------------------------------------------------------------- STT (ASR) scoring
def _stt_response(
    *,
    expected_transcript,
    final_transcript,
    reference_word_timestamps=None,
    transcript_snapshots=None,
    time_to_first_visible_text=None,
    time_to_final_transcript=None,
    pcm_byte_count=32000,
    sample_rate=16000,
    dataset="unit",
):
    """Mimic the realtime STTClient output: transcripts + ground truth + timing."""
    metrics = {
        ac.AUDIO_TASK: AudioTask.STT,
        ac.SAMPLE_RATE: sample_rate,
        ac.PCM_BYTE_COUNT: pcm_byte_count,
        "expected_transcript": expected_transcript,
        "final_transcript": final_transcript,
        "dataset": dataset,
        "reference_word_timestamps": reference_word_timestamps,
        "transcript_snapshots": transcript_snapshots or [],
        "time_to_first_visible_text": time_to_first_visible_text,
        "time_to_final_transcript": time_to_final_transcript,
    }
    return _Resp(channels={ChannelModality.AUDIO: _Chan(metrics=metrics)})


def test_stt_perfect_transcript_scores_zero_wer():
    from veeksha.types import AudioTask  # noqa: F401 (used via module-level import)

    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    ev.register_request(1, 1, 0.0, None)
    ev.record_request_completed(
        1,
        1,
        1.0,
        _stt_response(
            expected_transcript="the quick brown fox",
            final_transcript="the quick brown fox",
        ),
    )
    result = ev.finalize()
    m = result.metrics
    assert m["num_asr_scored_requests"] == 1
    # corpus WER aggregate present and zero for a perfect transcript
    assert m["asr_final_corpus_wer"] == 0.0
    assert m["asr_final_sample_count"] == 1.0


def test_stt_one_quarter_word_errors_scores_25_percent_wer():
    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    ev.register_request(1, 1, 0.0, None)
    # 1 of 4 reference words wrong => 25% WER
    ev.record_request_completed(
        1,
        1,
        1.0,
        _stt_response(
            expected_transcript="the quick brown fox",
            final_transcript="the quick brown cat",
        ),
    )
    m = ev.finalize().metrics
    assert m["asr_final_corpus_wer"] == 25.0


def test_stt_interactivity_and_first_text_latency_aggregated():
    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    # reference words with end timestamps; snapshots reveal each word over time
    ref = [
        {"word": "hello", "start_ms": 0.0, "end_ms": 200.0},
        {"word": "world", "start_ms": 220.0, "end_ms": 500.0},
    ]
    snaps = [
        {"elapsed_ms": 300.0, "transcript": "hello"},
        {"elapsed_ms": 700.0, "transcript": "hello world"},
    ]
    ev.register_request(1, 1, 0.0, None)
    ev.record_request_completed(
        1,
        1,
        1.0,
        _stt_response(
            expected_transcript="hello world",
            final_transcript="hello world",
            reference_word_timestamps=ref,
            transcript_snapshots=snaps,
            time_to_first_visible_text=300.0,
            time_to_final_transcript=700.0,
        ),
    )
    m = ev.finalize().metrics
    # interactivity + first-text latency distributions surfaced
    assert any(k.startswith("Word Interactivity Latency") for k in m)
    assert any(k.startswith("Time to First Visible Text") for k in m)


def test_stt_missing_transcript_is_skipped_with_warning():
    """An STT response without ground truth is skipped — never scored as TTS."""
    import logging

    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    # Metrics that WOULD satisfy the TTS aggregate dialect if the STT branch
    # incorrectly fell through (e2e + byte count present).
    metrics = {
        ac.AUDIO_TASK: AudioTask.STT,
        ac.SAMPLE_RATE: 16000,
        ac.PCM_BYTE_COUNT: 32000,
        ac.END_TO_END_LATENCY: 500.0,
        "final_transcript": "hello",
        # expected_transcript missing
    }
    resp = _Resp(channels={ChannelModality.AUDIO: _Chan(metrics=metrics)})

    audio_logger = logging.getLogger("veeksha.evaluator.performance.audio")
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    audio_logger.addHandler(handler)
    try:
        ev.record_request_completed(1, 1, 1.0, resp)
    finally:
        audio_logger.removeHandler(handler)

    assert any("missing" in r.getMessage() for r in records)
    m = ev.finalize().metrics
    assert m["num_completed_requests"] == 0
    assert "num_asr_scored_requests" not in m
    # no TTS timing sketches polluted by the malformed STT row
    assert not any(k.startswith("End to End Latency") for k in m)
    assert not any(k.startswith("Generated Audio Duration") for k in m)


def test_stt_malformed_reference_timestamps_does_not_raise():
    """One malformed row must not raise out of record_request_completed."""
    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    ev.register_request(1, 1, 0.0, None)
    ev.record_request_completed(
        1,
        1,
        1.0,
        _stt_response(
            expected_transcript="hello world",
            final_transcript="hello world",
            # entries missing start_ms/end_ms -> KeyError inside interactivity
            reference_word_timestamps=[{"word": "hello"}],
            transcript_snapshots=[{"elapsed_ms": 100.0, "transcript": "hello"}],
        ),
    )  # must not raise
    m = ev.finalize().metrics
    assert m.get("num_asr_scored_requests", 0) == 0  # row skipped, not scored


def test_stt_and_tts_do_not_cross_contaminate():
    """A TTS response in the same evaluator must not enter the ASR path."""
    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    sr = ac.DEFAULT_AUDIO_SAMPLE_RATE
    timeline = [[50.0 * i, 4800] for i in range(1, 11)]
    ev.register_request(1, 1, 0.0, None)
    ev.record_request_completed(1, 1, 1.0, _audio_response(timeline, sr))
    ev.register_request(2, 2, 0.0, None)
    ev.record_request_completed(
        2,
        2,
        1.0,
        _stt_response(expected_transcript="a b c d", final_transcript="a b c d"),
    )
    m = ev.finalize().metrics
    assert m["num_asr_scored_requests"] == 1  # only the STT request scored
    assert m["asr_final_sample_count"] == 1.0
    assert m["num_completed_requests"] == 2  # both counted as completed


def test_tts_interactivity_from_text_and_audio_timelines():
    """Realtime TTS: the paced input-text timeline aligned against the audio
    timeline yields first-input->first-audio and the shared response-latency
    operator (last input -> first audio)."""
    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    resp = _Resp(
        channels={
            ChannelModality.AUDIO: _Chan(
                metrics={
                    ac.AUDIO_TASK: "tts",
                    ac.SAMPLE_RATE: ac.DEFAULT_AUDIO_SAMPLE_RATE,
                    # text deltas at 50ms and 150ms; audio at 400ms and 500ms
                    ac.AudioMetricKey.TEXT_DELTA_TIMESTAMPS: [
                        [50.0, 5],
                        [150.0, 5],
                    ],
                    ac.AUDIO_CHUNK_TIMESTAMPS: [[400.0, 4800], [500.0, 4800]],
                }
            )
        }
    )
    ev.record_request_completed(1, 1, 0.0, resp)
    metrics = ev.finalize().metrics
    fifa = [k for k in metrics if "First Input to First Audio" in k]
    vrl = [k for k in metrics if "Voice Response Latency" in k]
    assert fifa and vrl
    # first input (0.05s) -> first audio (0.4s) = 0.35s
    assert math.isclose(metrics["First Input to First Audio (Mean)"], 0.35)
    # last input (0.15s) -> first audio (0.4s) = 0.25s
    assert math.isclose(metrics["Voice Response Latency (Mean)"], 0.25)
