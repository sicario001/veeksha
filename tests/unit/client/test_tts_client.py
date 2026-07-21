"""Unit tests for the ported HTTP TTSClient (no server needed)."""

from __future__ import annotations

from veeksha.client.registry import ClientRegistry
from veeksha.client.tts import TTSClient, _build_audio_speech_url
from veeksha.config.client import TTSClientConfig
from veeksha.types import ClientType


def test_audio_speech_url_building():
    assert _build_audio_speech_url("http://h:8000") == "http://h:8000/v1/audio/speech"
    assert (
        _build_audio_speech_url("http://h:8000/v1") == "http://h:8000/v1/audio/speech"
    )
    assert (
        _build_audio_speech_url("http://h:8000/v1/") == "http://h:8000/v1/audio/speech"
    )


def test_build_request_payload():
    cfg = TTSClientConfig(
        api_base="http://h/v1", model="tts-1", voice_id="alloy", raw_pcm=True
    )
    client = TTSClient(cfg)
    req = client._build_request("hello world")
    assert req.url.endswith("/v1/audio/speech")
    assert req.payload["model"] == "tts-1"
    assert req.payload["input"] == "hello world"
    assert req.payload["voice"] == "alloy"
    assert req.payload["response_format"] == "pcm"  # raw_pcm=True
    assert req.payload["stream"] is True
    assert req.payload["stream_format"] == "audio"


def test_wav_format_when_not_raw_pcm():
    cfg = TTSClientConfig(
        api_base="http://h/v1", model="tts-1", voice_id="v", raw_pcm=False
    )
    assert TTSClient(cfg)._build_request("x").payload["response_format"] == "wav"


def test_registered_under_client_type_tts():
    # get() resolves the lazy loader and instantiates with the given config
    client = ClientRegistry.get(
        ClientType.TTS,
        TTSClientConfig(api_base="http://h/v1", model="m", voice_id="v"),
    )
    assert isinstance(client, TTSClient)
