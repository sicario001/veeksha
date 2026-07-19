import pytest

from veeksha.client.stt import VajraOpenAIRealtimeSTTClient, _slice_pcm16_bytes
from veeksha.config.client import STTClientConfig


@pytest.mark.unit
def test_slice_pcm16_bytes_uses_millisecond_offsets() -> None:
    pcm = bytes(range(20))

    sliced = _slice_pcm16_bytes(pcm, 1000, start_ms=2.0, end_ms=6.0)

    assert sliced == pcm[4:12]


@pytest.mark.unit
def test_slice_pcm16_bytes_passthrough_without_offsets() -> None:
    pcm = bytes(range(20))

    assert _slice_pcm16_bytes(pcm, 1000, start_ms=None, end_ms=None) is pcm


@pytest.mark.unit
def test_slice_pcm16_bytes_open_ended_slices() -> None:
    pcm = bytes(range(20))

    assert _slice_pcm16_bytes(pcm, 1000, start_ms=None, end_ms=6.0) == pcm[:12]
    assert _slice_pcm16_bytes(pcm, 1000, start_ms=2.0, end_ms=None) == pcm[4:]


@pytest.mark.unit
def test_slice_pcm16_bytes_rejects_negative_start() -> None:
    pcm = bytes(range(20))

    with pytest.raises(ValueError, match="must be non-negative"):
        _slice_pcm16_bytes(pcm, 1000, start_ms=-1.0, end_ms=6.0)


@pytest.mark.unit
def test_slice_pcm16_bytes_rejects_end_not_after_start() -> None:
    pcm = bytes(range(20))

    with pytest.raises(ValueError, match="must be greater than"):
        _slice_pcm16_bytes(pcm, 1000, start_ms=6.0, end_ms=6.0)


@pytest.mark.unit
def test_slice_pcm16_bytes_rejects_end_past_clip() -> None:
    pcm = bytes(range(20))

    with pytest.raises(ValueError, match="exceeds decoded clip length"):
        _slice_pcm16_bytes(pcm, 1000, start_ms=0.0, end_ms=11.0)


@pytest.mark.unit
@pytest.mark.parametrize("field_name", ["ws_ping_interval_s", "ws_ping_timeout_s"])
def test_stt_client_config_rejects_nonpositive_ws_ping(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be > 0 or None"):
        STTClientConfig(
            provider="vajra_openai_realtime",
            model="mistralai/Voxtral-Mini-4B-Realtime-2602",
            api_base="http://localhost:8003",
            **{field_name: 0},
        )


def _vajra_realtime_client() -> VajraOpenAIRealtimeSTTClient:
    config = STTClientConfig(
        provider="vajra_openai_realtime",
        model="mistralai/Voxtral-Mini-4B-Realtime-2602",
        api_base="http://localhost:8003",
    )
    return VajraOpenAIRealtimeSTTClient(config)


@pytest.mark.unit
def test_vajra_openai_realtime_parses_transcription_events() -> None:
    client = _vajra_realtime_client()

    assert client._parse_message(
        {"type": "conversation.item.input_audio_transcription.delta", "delta": "hi"}
    ) == ("delta", "hi")
    assert client._parse_message(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "hi there",
        }
    ) == ("done", "hi there")
    assert client._parse_message({"type": "input_audio_buffer.committed"}) == ("", "")


@pytest.mark.unit
def test_vajra_openai_realtime_maps_failed_item_to_error() -> None:
    client = _vajra_realtime_client()

    kind, text = client._parse_message(
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "error": {"type": "server_error", "message": "boom"},
        }
    )

    assert (kind, text) == ("error", "boom")


@pytest.mark.unit
def test_vajra_openai_realtime_maps_session_error() -> None:
    client = _vajra_realtime_client()

    assert client._parse_message(
        {"type": "error", "error": {"message": "bad request"}}
    ) == ("error", "bad request")
