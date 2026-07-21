"""Audio measurement contract: PCM constants and the metric-key vocabulary.

Pure measurement facts shared by the audio (TTS / realtime / STT) clients and the
audio evaluators. Transport-specific protocol handling lives in the clients.
Ported from the voice branches; `AudioMetricKey` is the canonical vocabulary.
"""

from __future__ import annotations

from enum import StrEnum

# 16-bit mono PCM at 24 kHz is the shared baseline across TTS dialects.
DEFAULT_AUDIO_SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
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
    """Duration in ms of ``n_bytes`` of 16-bit mono PCM at ``sample_rate``."""
    return pcm_bytes_to_duration_s(n_bytes, sample_rate) * 1000.0


class AudioMetricKey(StrEnum):
    TTFC = "ttfc"
    END_TO_END_LATENCY = "end_to_end_latency"
    GENERATED_AUDIO_DURATION = "generated_audio_duration"
    RTF = "rtf"
    CHUNK_COUNT = "chunk_count"
    RAW_PCM = "raw_pcm"
    SAMPLE_RATE = "sample_rate"
    PCM_BYTE_COUNT = "pcm_byte_count"
    INPUT_CHARS = "input_chars"
    INPUT_TOKENS = "input_tokens"
    INPUT_TEXT = "input_text"
    SESSION_SIZE = "session_size"
    SESSION_DURATION = "session_duration"

    # ----- Realtime input-streaming interactivity keys -----
    # Time convention for all realtime event-time values below: every *_offset_ms
    # / timestamp value is a float millisecond offset relative to request start
    # (the client's WS-connect initiation), measured with time.monotonic().
    #
    # Raw-contract keys are emitted by the websocket client:
    TEXT_DELTA_TIMESTAMPS = "text_delta_timestamps"  # list[[offset_ms, n_chars]]
    AUDIO_CHUNK_TIMESTAMPS = (
        "audio_chunk_timestamps"  # list[[offset_ms, n_bytes_decoded_pcm]]
    )
    WS_CONNECT_LATENCY_MS = "ws_connect_latency_ms"
    SESSION_READY_OFFSET_MS = "session_ready_offset_ms"  # nullable
    RESPONSE_CREATED_OFFSET_MS = "response_created_offset_ms"  # nullable
    INPUT_COMMIT_OFFSET_MS = "input_commit_offset_ms"
    AUDIO_DONE_OFFSET_MS = "audio_done_offset_ms"  # nullable
    RESPONSE_DONE_OFFSET_MS = "response_done_offset_ms"  # nullable

    # Diagnostic delivery metrics.
    STREAMING_RTF = "streaming_rtf"


# Convenience module-level aliases (the keys the timing evaluator reads). These
# are the StrEnum members, usable interchangeably with their string values as
# dict keys.
AUDIO_CHUNK_TIMESTAMPS = AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS
SAMPLE_RATE = AudioMetricKey.SAMPLE_RATE
PCM_BYTE_COUNT = AudioMetricKey.PCM_BYTE_COUNT
END_TO_END_LATENCY = AudioMetricKey.END_TO_END_LATENCY
# Task tag the audio clients set in ChannelResponse.metrics (not a metric value).
AUDIO_TASK = "audio_task"
