"""AudioPerformanceEvaluator derives audio metrics from the shared primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.core import audio_contract as ac
from veeksha.evaluator.performance.audio import AudioPerformanceEvaluator
from veeksha.evaluator.performance.channel_base import BaseChannelPerformanceEvaluator
from veeksha.types import ChannelModality


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
