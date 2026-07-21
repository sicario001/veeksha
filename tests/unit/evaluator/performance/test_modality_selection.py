"""I12: the top-level PerformanceEvaluator selects the right channel evaluator
per modality from config, so text, TTS and ASR all run from one branch.

This is the wiring-validation for the single-source-of-truth goal: given
``target_channels``, the PerformanceEvaluator instantiates the matching channel
evaluator via the registry and routes responses to it — no per-modality code in
the dispatcher.
"""

from __future__ import annotations

from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.core import audio_contract as ac
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.evaluator.performance.audio import AudioPerformanceEvaluator
from veeksha.evaluator.performance.base import PerformanceEvaluator
from veeksha.types import AudioTask, ChannelModality


def _audio_result(request_id: int, metrics: dict, content=b"") -> RequestResult:
    return RequestResult(
        request_id=request_id,
        session_id=request_id,
        channels={
            ChannelModality.AUDIO: ChannelResponse(
                modality=ChannelModality.AUDIO, content=content, metrics=metrics
            )
        },
    )


def _tts_metrics():
    return {
        ac.AUDIO_TASK: AudioTask.TTS,
        ac.AudioMetricKey.TTFC: 80.0,
        ac.END_TO_END_LATENCY: 600.0,
        ac.SAMPLE_RATE: ac.DEFAULT_AUDIO_SAMPLE_RATE,
        ac.AudioMetricKey.RAW_PCM: True,
    }


def _stt_metrics():
    return {
        ac.AUDIO_TASK: AudioTask.STT,
        ac.SAMPLE_RATE: 16000,
        ac.PCM_BYTE_COUNT: 32000,
        "expected_transcript": "the quick brown fox",
        "final_transcript": "the quick brown fox",
        "dataset": "unit",
        "transcript_snapshots": [],
    }


def test_audio_channel_selected_and_scores_tts_and_stt():
    """target_channels=[audio] wires AudioPerformanceEvaluator for both audio tasks."""
    config = PerformanceEvaluatorConfig(target_channels=["audio"], stream_metrics=False)
    ev = PerformanceEvaluator(config)

    # the AUDIO channel resolved to the audio evaluator (registry selection)
    audio_eval = ev._get_channel_evaluator(ChannelModality.AUDIO)
    assert isinstance(audio_eval, AudioPerformanceEvaluator)

    # a TTS response (aggregate dialect: TTFC + e2e + audio bytes)
    tts = _audio_result(1, _tts_metrics(), content=b"\x00" * 24000)
    ev.register_request(1, 1, 0.0, tts.channels)
    ev.record_request_completed(1, 1, 1.0, tts)

    # an STT response (transcript + ground truth => WER)
    stt = _audio_result(2, _stt_metrics())
    ev.register_request(2, 2, 0.0, stt.channels)
    ev.record_request_completed(2, 2, 1.0, stt)

    m = audio_eval.finalize().metrics
    assert m["num_completed_requests"] == 2
    assert m["num_asr_scored_requests"] == 1  # only the STT one scored WER
    assert m["asr_final_corpus_wer"] == 0.0  # perfect transcript
    assert "Time to First Audio (Mean)" in m  # TTS timing surfaced


def test_text_channel_still_selected_for_text_target():
    """target_channels=[text] resolves the text evaluator (no audio regression)."""
    from veeksha.evaluator.performance.text import TextPerformanceEvaluator

    config = PerformanceEvaluatorConfig(target_channels=["text"], stream_metrics=False)
    ev = PerformanceEvaluator(config)
    text_eval = ev._get_channel_evaluator(ChannelModality.TEXT)
    assert isinstance(text_eval, TextPerformanceEvaluator)
    # audio channel is NOT instantiated when not targeted
    assert ChannelModality.AUDIO not in ev._channel_evaluators


def test_audio_channel_config_defaults_are_present():
    """audio_channel is populated by default so [audio] works without extra config."""
    config = PerformanceEvaluatorConfig(target_channels=["audio"], stream_metrics=False)
    assert config.get_channel_config(ChannelModality.AUDIO) is not None
