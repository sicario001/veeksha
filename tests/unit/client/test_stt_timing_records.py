"""Raw per-event timing records emitted by the STT client.

Two recordings back the preflight's send-pacing and receive-cadence checks:

- ``send_offsets_ms``: one entry per audio chunk, stamped immediately before
  the socket send, anchored at the first send (entry 0 is 0.0).
- ``transcript_delta_offsets_ms``: one entry per transcript delta received,
  stamped on ``recv()`` return before parsing. Unlike ``transcript_snapshots``
  (deduplicated by content) it counts every wire event.

The client only records; all drift math lives outside the timing window.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from veeksha.client.stt import STTStreamResult, VllmRealtimeSTTClient, _ClipAssets
from veeksha.config.client import STTClientConfig
from veeksha.core.request import Request
from veeksha.core.request_content import AudioChannelRequestContent
from veeksha.types import ChannelModality

# 1600 bytes of PCM16 at 16 kHz == 50 ms of audio per chunk.
_SAMPLE_RATE = 16000
_CHUNK_BYTES = 1600
_CHUNK_MS = 50.0
_NUM_CHUNKS = 5


def _client(pacing: bool) -> VllmRealtimeSTTClient:
    config = STTClientConfig(
        provider="vllm_realtime",
        model="dummy-stt",
        api_base="http://localhost:8025",
        sample_rate=_SAMPLE_RATE,
        ws_chunk_size=_CHUNK_BYTES,
        ws_realtime_pacing=pacing,
    )
    return VllmRealtimeSTTClient(config)


class _ScriptedWebSocket:
    """Fake WS: records appends, replays a scripted delta sequence after them."""

    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas
        self._recv_index = 0
        self._audio_sent = asyncio.Event()
        self._eof_received = asyncio.Event()
        self.appends: list[str] = []

    async def recv(self) -> str:
        if self._recv_index == 0:
            self._recv_index += 1
            return json.dumps({"type": "session.created"})

        await self._audio_sent.wait()
        delta_index = self._recv_index - 1
        self._recv_index += 1
        if delta_index < len(self._deltas):
            return json.dumps(
                {"type": "transcription.delta", "delta": self._deltas[delta_index]}
            )
        # Real servers only complete after end-of-audio; waiting here keeps the
        # send loop alive for the whole paced upload.
        await self._eof_received.wait()
        return json.dumps({"type": "transcription.done", "text": "hello"})

    async def send(self, message: str | bytes) -> None:
        if isinstance(message, str):
            payload = json.loads(message)
            if payload.get("type") == "input_audio_buffer.append":
                self.appends.append(message)
                self._audio_sent.set()
            elif payload.get("type") == "input_audio_buffer.commit" and payload.get(
                "final"
            ):
                self._eof_received.set()


class _FakeConnection:
    def __init__(self, websocket: _ScriptedWebSocket) -> None:
        self._websocket = websocket

    async def __aenter__(self) -> _ScriptedWebSocket:
        return self._websocket

    async def __aexit__(self, *_args) -> None:
        return None


def _run_stream(
    client: VllmRealtimeSTTClient,
    websocket: _ScriptedWebSocket,
    monkeypatch: pytest.MonkeyPatch,
) -> STTStreamResult:
    monkeypatch.setattr(
        websockets,
        "connect",
        lambda *_args, **_kwargs: _FakeConnection(websocket),
    )
    pcm = b"\x00" * (_CHUNK_BYTES * _NUM_CHUNKS)
    return asyncio.run(client._stream(pcm))


@pytest.mark.unit
def test_send_offsets_track_the_realtime_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = _ScriptedWebSocket(["hello"])

    result = _run_stream(_client(pacing=True), websocket, monkeypatch)

    offsets = result.send_offsets_ms
    # One stamp per audio chunk actually put on the wire.
    assert len(offsets) == _NUM_CHUNKS == len(websocket.appends)
    # The first send is the anchor.
    assert offsets[0] == 0.0
    assert offsets == sorted(offsets)
    # 1x pacing: chunk i is due at i * 50 ms; loose bound for scheduler jitter.
    for index, offset in enumerate(offsets):
        assert offset >= index * _CHUNK_MS - 1.0
        assert offset < index * _CHUNK_MS + 40.0

    # The absolute anchor is the first-send instant, so send offsets reconstruct
    # to absolute send times exactly (the anchor is additive, offsets unchanged).
    anchor = result.audio_started_monotonic
    assert isinstance(anchor, float)
    for offset in offsets:
        assert ((anchor + offset / 1000.0) - anchor) * 1000.0 == pytest.approx(
            offset, abs=1e-6
        )


@pytest.mark.unit
def test_send_offsets_recorded_without_pacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = _ScriptedWebSocket(["hello"])

    result = _run_stream(_client(pacing=False), websocket, monkeypatch)

    offsets = result.send_offsets_ms
    assert len(offsets) == _NUM_CHUNKS
    assert offsets[0] == 0.0
    assert offsets == sorted(offsets)
    # Unpaced sends finish far inside the 1x schedule.
    assert offsets[-1] < (_NUM_CHUNKS - 1) * _CHUNK_MS


@pytest.mark.unit
def test_transcript_delta_offsets_count_every_wire_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deltas 2 and 3 add no new visible text, so the snapshot recorder dedupes
    # them away; the raw offsets must still count them.
    websocket = _ScriptedWebSocket(["hello", "", ""])

    result = _run_stream(_client(pacing=False), websocket, monkeypatch)

    assert result.final_transcript == "hello"
    assert result.chunk_count == 3
    assert len(result.transcript_delta_offsets_ms) == 3
    assert len(result.transcript_snapshots) < 3
    assert all(offset >= 0.0 for offset in result.transcript_delta_offsets_ms)
    assert result.transcript_delta_offsets_ms == sorted(
        result.transcript_delta_offsets_ms
    )


@pytest.mark.unit
def test_send_request_emits_raw_timing_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    client = _client(pacing=False)
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"\x00" * 64)

    pcm = b"\x00" * (_CHUNK_BYTES * 2)
    monkeypatch.setattr(
        client,
        "_clip_assets",
        lambda _path: _ClipAssets(
            pcm=pcm,
            wire_messages=[
                client._encode_chunk(pcm[offset : offset + _CHUNK_BYTES])
                for offset in range(0, len(pcm), _CHUNK_BYTES)
            ],
        ),
    )

    websocket = _ScriptedWebSocket(["hello", ""])
    monkeypatch.setattr(
        websockets,
        "connect",
        lambda *_args, **_kwargs: _FakeConnection(websocket),
    )

    request = Request(
        id=3,
        channels={
            ChannelModality.AUDIO: AudioChannelRequestContent(
                input_audio=str(audio_path)
            )
        },
    )
    result = asyncio.run(client.send_request(request, session_id=1))

    assert result.success is True
    metrics = result.channels[ChannelModality.AUDIO].metrics
    assert metrics["send_offsets_ms"][0] == 0.0
    assert len(metrics["send_offsets_ms"]) == 2
    assert len(metrics["transcript_delta_offsets_ms"]) == 2
    assert all(
        isinstance(offset, float) for offset in metrics["transcript_delta_offsets_ms"]
    )
    # Absolute anchors, all floats. request_start (t_start) <= request_sent (the
    # session.update handshake, C2 request stamp) <= audio_started (first append,
    # the separate per-append send anchor).
    assert isinstance(metrics["request_start_monotonic"], float)
    assert isinstance(metrics["request_sent_monotonic"], float)
    assert isinstance(metrics["audio_started_monotonic"], float)
    assert metrics["request_start_monotonic"] <= metrics["request_sent_monotonic"]
    assert metrics["request_sent_monotonic"] <= metrics["audio_started_monotonic"]
    # Pre-existing keys are untouched.
    assert metrics["final_transcript"] == "hello"
    assert metrics["chunk_count"] == 2
