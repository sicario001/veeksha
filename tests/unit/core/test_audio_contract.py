"""Tests for the audio measurement contract."""

from __future__ import annotations

import math

from veeksha.core import audio_contract as ac


def test_pcm_duration_helpers():
    # 24kHz mono 16-bit -> 48000 bytes/sec
    assert math.isclose(ac.pcm_bytes_to_duration_s(48000), 1.0, rel_tol=1e-9)
    assert math.isclose(ac.pcm_bytes_to_duration_ms(48000), 1000.0, rel_tol=1e-9)
    assert math.isclose(ac.pcm_bytes_to_duration_s(24000, sample_rate=12000), 1.0)
    assert ac.pcm_bytes_to_duration_s(0) == 0.0


def test_metric_key_values_and_str_behaviour():
    assert str(ac.AudioMetricKey.RTF) == "rtf"
    assert str(ac.AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS) == "audio_chunk_timestamps"
    assert str(ac.AudioMetricKey.STREAMING_RTF) == "streaming_rtf"
    # StrEnum members compare equal to their string value (usable as dict keys)
    d = {ac.AudioMetricKey.RTF: 0.5}
    assert d["rtf"] == 0.5


def test_module_aliases_match_enum():
    assert ac.AUDIO_CHUNK_TIMESTAMPS == "audio_chunk_timestamps"
    assert ac.SAMPLE_RATE == "sample_rate"
    assert ac.PCM_BYTE_COUNT == "pcm_byte_count"
    assert ac.END_TO_END_LATENCY == "end_to_end_latency"
    assert ac.AUDIO_TASK == "audio_task"
