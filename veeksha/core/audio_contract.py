"""Audio measurement contract — PCM constants + metric-key vocabulary.

Pure measurement facts shared by audio clients and the audio evaluator (ported
from the voice branches so the eventual merge lines up). 16-bit mono PCM is the
baseline; durations derive from byte counts.
"""

from __future__ import annotations

DEFAULT_AUDIO_SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2  # 16-bit mono
WAV_HEADER_BYTES = 44


def pcm_bytes_to_duration_s(
    n_bytes: float, sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
) -> float:
    """Duration in seconds of ``n_bytes`` of 16-bit mono PCM at ``sample_rate``."""
    denom = sample_rate * BYTES_PER_SAMPLE
    return (n_bytes / denom) if denom else 0.0


def pcm_bytes_to_duration_ms(
    n_bytes: float, sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
) -> float:
    return pcm_bytes_to_duration_s(n_bytes, sample_rate) * 1000.0


# Keys the audio channel puts in ChannelResponse.metrics (subset of the branch
# AudioMetricKey vocabulary, enough for the timing metrics).
AUDIO_CHUNK_TIMESTAMPS = "audio_chunk_timestamps"  # list[[offset_ms, n_bytes]]
SAMPLE_RATE = "sample_rate"
PCM_BYTE_COUNT = "pcm_byte_count"
END_TO_END_LATENCY = "end_to_end_latency"  # seconds
AUDIO_TASK = "audio_task"  # "tts" | "stt" (informational)
