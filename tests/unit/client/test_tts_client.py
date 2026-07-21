"""Per-chunk arrival records emitted by the HTTP streaming TTS client.

``audio_chunk_timestamps`` must carry the same shape the realtime TTS client
emits -- ``list[[offset_ms_from_t_start, n_bytes]]`` -- so downstream consumers
read both dialects uniformly.
"""

from __future__ import annotations

import asyncio

import pytest

from veeksha.client.tts import TTSClient
from veeksha.config.client import TTSClientConfig
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.types import ChannelModality

_CHUNKS = [b"\x00\x01" * 64, b"\x02\x03" * 32, b"\x04\x05" * 96]


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes], delay_s: float) -> None:
        self._chunks = chunks
        self._delay_s = delay_s
        self.headers = {"Content-Type": "audio/pcm"}

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self, chunk_size: int | None = None):
        for chunk in self._chunks:
            await asyncio.sleep(self._delay_s)
            yield chunk


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *_args) -> None:
        return None


class _FakeHTTPClient:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    def stream(self, *_args, **_kwargs) -> _FakeStreamContext:
        return _FakeStreamContext(self._response)


def _tts_client() -> TTSClient:
    config = TTSClientConfig(
        api_base="http://127.0.0.1:1/v1",
        api_key="test-key",
        model="dummy-tts",
        voice_id="alloy",
    )
    return TTSClient(config)


@pytest.mark.unit
def test_audio_chunk_timestamps_match_the_realtime_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _tts_client()
    delay_s = 0.01
    monkeypatch.setattr(
        client,
        "_get_client",
        lambda: _FakeHTTPClient(_FakeStreamResponse(_CHUNKS, delay_s)),
    )

    request = Request(
        id=5,
        channels={ChannelModality.TEXT: TextChannelRequestContent(input_text="hello")},
    )
    result = asyncio.run(client.send_request(request, session_id=2))

    assert result.success is True
    metrics = result.channels[ChannelModality.AUDIO].metrics
    stamps = metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]

    # One [offset_ms, n_bytes] pair per received chunk.
    assert len(stamps) == metrics[AudioMetricKey.CHUNK_COUNT.value] == len(_CHUNKS)
    for (offset_ms, n_bytes), chunk in zip(stamps, _CHUNKS):
        assert isinstance(offset_ms, float)
        assert isinstance(n_bytes, int)
        assert n_bytes == len(chunk)

    offsets = [offset for offset, _ in stamps]
    assert offsets == sorted(offsets)
    assert offsets[0] > 0.0
    # The first arrival offset is exactly the recorded TTFC.
    assert offsets[0] == pytest.approx(metrics[AudioMetricKey.TTFC.value], abs=1e-3)
    assert offsets[-1] >= delay_s * len(_CHUNKS) * 1000 * 0.5


@pytest.mark.unit
def test_absolute_anchors_reconstruct_the_relative_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The added anchors are floats and reconstruct the audio offsets exactly.

    ``request_start_monotonic`` is the instant every ``audio_chunk_timestamps``
    offset is measured from, so absolute arrival = anchor + offset/1000 must
    land between the send stamp and now, and the offsets themselves are
    untouched by the additive anchor recording.
    """
    client = _tts_client()
    monkeypatch.setattr(
        client,
        "_get_client",
        lambda: _FakeHTTPClient(_FakeStreamResponse(_CHUNKS, 0.01)),
    )
    request = Request(
        id=7,
        channels={ChannelModality.TEXT: TextChannelRequestContent(input_text="hello")},
    )
    result = asyncio.run(client.send_request(request, session_id=2))
    metrics = result.channels[ChannelModality.AUDIO].metrics

    start = metrics["request_start_monotonic"]
    sent = metrics["request_sent_monotonic"]
    assert isinstance(start, float)
    assert isinstance(sent, float)
    # The POST leaves after t_start and before the first chunk arrives.
    assert sent >= start
    stamps = metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]
    first_abs = start + stamps[0][0] / 1000.0
    assert first_abs >= sent
    # Absolute reconstruction is monotonic and matches the recorded offsets.
    abs_times = [start + offset / 1000.0 for offset, _ in stamps]
    assert abs_times == sorted(abs_times)
    for (offset, _), absolute in zip(stamps, abs_times):
        assert (absolute - start) * 1000.0 == pytest.approx(offset, abs=1e-6)


@pytest.mark.unit
def test_empty_chunks_are_not_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _tts_client()
    monkeypatch.setattr(
        client,
        "_get_client",
        lambda: _FakeHTTPClient(_FakeStreamResponse([b"", _CHUNKS[0], b""], 0.0)),
    )

    request = Request(
        id=6,
        channels={ChannelModality.TEXT: TextChannelRequestContent(input_text="hello")},
    )
    result = asyncio.run(client.send_request(request, session_id=2))

    metrics = result.channels[ChannelModality.AUDIO].metrics
    assert len(metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value]) == 1
    assert metrics[AudioMetricKey.CHUNK_COUNT.value] == 1
