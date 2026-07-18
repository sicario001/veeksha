"""Lock in the audio enum values (must match the voice branches for a clean merge)."""

from __future__ import annotations

from veeksha.types import (
    AudioTask,
    ClientType,
    EvaluationType,
    ServerType,
    TraceFlavorType,
    VerificationType,
)


def test_audio_task_values():
    assert (AudioTask.TTS, AudioTask.STT, AudioTask.LLM_AUDIO) == (1, 2, 3)


def test_client_type_audio_values():
    assert (ClientType.TTS, ClientType.REALTIME_TTS, ClientType.STT) == (4, 5, 6)
    # existing text clients unchanged
    assert ClientType.OPENAI_CHAT_COMPLETIONS == 1


def test_trace_flavor_audio_values():
    assert (
        TraceFlavorType.SHAREGPT,
        TraceFlavorType.SEED_TTS_TEXT,
        TraceFlavorType.AUDIO,
    ) == (6, 7, 8)


def test_eval_server_verification_values():
    assert EvaluationType.AUDIO_QUALITY == 3
    assert ServerType.VAJRA == 3
    assert VerificationType.AUDIO == 1


def test_from_str_roundtrip():
    # name-based lookup is what config parsing uses
    assert ClientType.from_str("stt") == ClientType.STT
    assert AudioTask.from_str("tts") == AudioTask.TTS
