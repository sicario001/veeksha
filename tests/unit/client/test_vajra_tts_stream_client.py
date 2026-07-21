import asyncio
import json

import pytest

from veeksha.client.vajra_tts_stream import (
    VajraTTSStreamClient,
    VajraTTSStreamProtocol,
)
from veeksha.config.client import TextPacingConfig, VajraTTSStreamClientConfig
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.types import ChannelModality


def _config(**overrides) -> VajraTTSStreamClientConfig:
    kwargs = dict(
        model="qwen-tts",
        api_base="http://localhost:8081",
        voice_id="vivian",
        pacing=TextPacingConfig(tokens_per_second=10000.0),
    )
    kwargs.update(overrides)
    return VajraTTSStreamClientConfig(**kwargs)


def _client(**overrides) -> VajraTTSStreamClient:
    return VajraTTSStreamClient(_config(**overrides))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_config_requires_model() -> None:
    with pytest.raises(ValueError, match="model is required"):
        VajraTTSStreamClientConfig(api_base="http://localhost:8081")


@pytest.mark.unit
def test_config_requires_api_base() -> None:
    with pytest.raises(ValueError, match="api_base is required"):
        VajraTTSStreamClientConfig(model="qwen-tts")


@pytest.mark.unit
def test_config_rejects_nonpositive_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate must be > 0"):
        _config(sample_rate=0)


@pytest.mark.unit
def test_config_defaults_skip_validation_for_polymorphic_instantiation() -> None:
    # The flat_dataclass framework instantiates non-selected polymorphic
    # children with defaults; that must not raise.
    config = VajraTTSStreamClientConfig()
    assert config.model == ""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "api_base,expected",
    [
        (
            "http://localhost:8081",
            "ws://localhost:8081/v1/audio/speech/stream",
        ),
        (
            "http://localhost:8081/openai",
            "ws://localhost:8081/openai/v1/audio/speech/stream",
        ),
        (
            "http://localhost:8081/openai/",
            "ws://localhost:8081/openai/v1/audio/speech/stream",
        ),
        (
            "https://tts.example.com/v1",
            "wss://tts.example.com/v1/audio/speech/stream",
        ),
    ],
)
def test_build_ws_url(api_base: str, expected: str) -> None:
    protocol = VajraTTSStreamProtocol(_config(), api_key=None)

    assert protocol.build_ws_url(api_base) == expected


@pytest.mark.unit
def test_session_config_json_minimal() -> None:
    protocol = VajraTTSStreamProtocol(_config(voice_id=""), api_key=None)

    message = json.loads(protocol.session_config_json())

    assert message == {
        "type": "session.config",
        "response_format": "pcm",
        "stream_audio": True,
    }


@pytest.mark.unit
def test_session_config_json_includes_optional_fields() -> None:
    protocol = VajraTTSStreamProtocol(
        _config(language="en", instructions="calm", task_type="CustomVoice"),
        api_key=None,
    )

    message = json.loads(protocol.session_config_json())

    assert message["voice"] == "vivian"
    assert message["language"] == "en"
    assert message["instructions"] == "calm"
    assert message["task_type"] == "CustomVoice"


@pytest.mark.unit
def test_headers_include_bearer_token_only_when_set() -> None:
    assert VajraTTSStreamProtocol(_config(), api_key=None).headers() == {}
    assert VajraTTSStreamProtocol(_config(), api_key="sk-x").headers() == {
        "Authorization": "Bearer sk-x"
    }


# ---------------------------------------------------------------------------
# send_request against a fake websocket
# ---------------------------------------------------------------------------


class _FakeVajraWebSocket:
    """Scripted server: audio.start after first input.text, audio after done."""

    def __init__(self, *, audio_done_error: bool = False) -> None:
        self.sent_messages: list[dict] = []
        self._first_text = asyncio.Event()
        self._input_done = asyncio.Event()
        self._audio_done_error = audio_done_error
        self._recv_index = 0

    async def send(self, message: str) -> None:
        payload = json.loads(message)
        self.sent_messages.append(payload)
        if payload["type"] == "input.text":
            self._first_text.set()
        elif payload["type"] == "input.done":
            self._input_done.set()

    async def recv(self) -> str | bytes:
        index = self._recv_index
        self._recv_index += 1
        if index == 0:
            await self._first_text.wait()
            return json.dumps(
                {
                    "type": "audio.start",
                    "session_id": "sess-1",
                    "format": "pcm",
                    "sample_rate": 24000,
                }
            )
        await self._input_done.wait()
        if index == 1:
            return b"\x01\x00" * 1200
        if index == 2:
            return b"\x02\x00" * 1200
        if index == 3:
            return json.dumps(
                {
                    "type": "audio.done",
                    "session_id": "sess-1",
                    "total_bytes": 4800,
                    "error": self._audio_done_error,
                }
            )
        return json.dumps({"type": "session.done", "session_id": "sess-1"})


class _FakeConnection:
    def __init__(self, websocket: _FakeVajraWebSocket) -> None:
        self._websocket = websocket

    async def __aenter__(self) -> _FakeVajraWebSocket:
        return self._websocket

    async def __aexit__(self, *_args) -> None:
        return None


def _request(text: str = "hello world from vajra") -> Request:
    return Request(
        id=1,
        channels={ChannelModality.TEXT: TextChannelRequestContent(input_text=text)},
    )


def _run_request(
    client: VajraTTSStreamClient,
    websocket: _FakeVajraWebSocket,
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
):
    monkeypatch.setattr(client, "_connect", lambda: _FakeConnection(websocket))
    return asyncio.run(
        client.send_request(
            _request(),
            session_id=7,
            on_request_sent=lambda: events.append("sent"),
            on_request_dispatched=lambda: events.append("dispatched"),
        )
    )


@pytest.mark.unit
def test_send_request_collects_audio_and_timeline_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    websocket = _FakeVajraWebSocket()
    events: list[str] = []

    result = _run_request(client, websocket, monkeypatch, events)

    assert result.success is True
    assert events == ["dispatched", "sent"]

    # Wire protocol: session.config, one input.text per whitespace token,
    # then input.done; deltas reconstruct the input exactly.
    assert websocket.sent_messages[0]["type"] == "session.config"
    assert websocket.sent_messages[0]["voice"] == "vivian"
    text_messages = [
        message
        for message in websocket.sent_messages
        if message["type"] == "input.text"
    ]
    assert "".join(message["text"] for message in text_messages) == (
        "hello world from vajra"
    )
    assert websocket.sent_messages[-1]["type"] == "input.done"

    channel = result.channels[ChannelModality.AUDIO]
    assert channel.content == b"\x01\x00" * 1200 + b"\x02\x00" * 1200

    metrics = channel.metrics
    assert metrics[AudioMetricKey.CHUNK_COUNT.value] == 2
    assert metrics[AudioMetricKey.RAW_PCM.value] is True
    assert metrics[AudioMetricKey.SAMPLE_RATE.value] == 24000
    assert metrics[AudioMetricKey.TTFC.value] > 0.0
    assert len(metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]) == 2
    assert all(
        n_bytes == 2400
        for _, n_bytes in metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]
    )
    assert len(metrics[AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value]) == len(
        text_messages
    )
    assert metrics[AudioMetricKey.WS_CONNECT_LATENCY_MS.value] is not None
    assert metrics[AudioMetricKey.SESSION_READY_OFFSET_MS.value] is not None
    assert metrics[AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value] is not None
    assert metrics[AudioMetricKey.AUDIO_DONE_OFFSET_MS.value] is not None
    assert metrics[AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value] is not None
    assert metrics[AudioMetricKey.INPUT_TEXT.value] == "hello world from vajra"

    # Absolute anchors (additive): request_sent stamps the first paced frame,
    # so it is >= t_start, and each audio offset reconstructs to an absolute
    # arrival at or after that send stamp.
    start = metrics["request_start_monotonic"]
    sent = metrics["request_sent_monotonic"]
    assert isinstance(start, float)
    assert isinstance(sent, float)
    assert sent >= start
    for offset, _ in metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]:
        assert start + offset / 1000.0 >= sent


@pytest.mark.unit
def test_send_request_maps_audio_done_error_to_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    websocket = _FakeVajraWebSocket(audio_done_error=True)
    events: list[str] = []

    result = _run_request(client, websocket, monkeypatch, events)

    assert result.success is False
    assert result.error_code == 500
    assert "audio.done" in result.error_msg
    # Partial audio was received, so the AUDIO channel is still exported for
    # partial-result accounting, and the sent callback fired exactly once.
    assert ChannelModality.AUDIO in result.channels
    assert events == ["dispatched", "sent"]


class _ErrorEventWebSocket(_FakeVajraWebSocket):
    async def recv(self) -> str | bytes:
        await self._first_text.wait()
        return json.dumps({"type": "error", "message": "no such voice"})


@pytest.mark.unit
def test_send_request_maps_error_event_to_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    websocket = _ErrorEventWebSocket()
    events: list[str] = []

    result = _run_request(client, websocket, monkeypatch, events)

    assert result.success is False
    assert result.error_code == 500
    assert result.error_msg == "no such voice"
    assert events == ["dispatched", "sent"]


@pytest.mark.unit
def test_send_request_without_text_channel_returns_400() -> None:
    client = _client()

    result = asyncio.run(client.send_request(Request(id=5, channels={}), session_id=1))

    assert result.success is False
    assert result.error_code == 400
