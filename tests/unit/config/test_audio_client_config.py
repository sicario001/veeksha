"""Tests for the ported TTS/STT/realtime client configs."""

from __future__ import annotations

import pytest

from veeksha.config.client import (
    RealtimeTTSClientConfig,
    STTClientConfig,
    TextPacingConfig,
    TTSClientConfig,
)
from veeksha.types import ClientType


def test_get_type():
    assert TTSClientConfig().get_type() == ClientType.TTS
    assert STTClientConfig().get_type() == ClientType.STT
    assert RealtimeTTSClientConfig().get_type() == ClientType.REALTIME_TTS


def test_default_poly_children_do_not_raise():
    # vidhi instantiates default children for non-selected poly configs.
    TTSClientConfig()
    STTClientConfig()
    RealtimeTTSClientConfig()


def test_pool_fix_is_inherited():
    # max_connections (the httpx pool fix) must survive on the audio configs.
    assert TTSClientConfig().max_connections is None
    assert STTClientConfig().max_connections is None


def test_tts_validation():
    TTSClientConfig(api_base="http://x/v1", model="m", voice_id="v")  # ok
    with pytest.raises(ValueError):
        TTSClientConfig(api_base="http://x/v1", model="m")  # missing voice_id
    with pytest.raises(ValueError):
        TTSClientConfig(api_base="http://x/v1", voice_id="v")  # missing model


def test_stt_validation():
    STTClientConfig(api_base="ws://x", model="m", provider="vllm_realtime")  # ok
    with pytest.raises(ValueError):
        STTClientConfig(api_base="ws://x", model="m", provider="nope")  # bad provider
    with pytest.raises(ValueError):
        STTClientConfig(
            api_base="ws://x", model="m", provider="vllm_realtime", ws_chunk_size=0
        )


def test_text_pacing_validation():
    TextPacingConfig()  # defaults ok
    with pytest.raises(ValueError):
        TextPacingConfig(tokens_per_second=0)
    with pytest.raises(ValueError):
        TextPacingConfig(gap_distribution="bogus")


def test_audio_configs_expose_build_tokenizer_provider():
    # benchmark.py branches on this: audio clients bring their own tokenizer,
    # text clients fall back to the HF tokenizer for the model.
    from veeksha.config.client import OpenAIChatCompletionsClientConfig

    assert hasattr(TTSClientConfig(), "build_tokenizer_provider")
    assert hasattr(STTClientConfig(), "build_tokenizer_provider")
    assert hasattr(RealtimeTTSClientConfig(), "build_tokenizer_provider")
    assert not hasattr(OpenAIChatCompletionsClientConfig(), "build_tokenizer_provider")
