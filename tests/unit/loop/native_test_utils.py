"""Shared helpers for the native-engine adapter tests: capture servers that
record the exact wire bytes/frames a client sends, session builders, and a
tiny WAV generator for STT tests.

The HTTP capture server records the FULL request bytes (head + body) so
wire-fidelity tests can compare the native main loop's output byte-for-byte
against what the Python clients (httpx) send. The WS capture servers record
every received text frame payload in order.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import threading
import time
import wave
from typing import Dict, List, Optional

from tests.unit.native.mock_servers import (
    AsyncServerThread,
    _write_http_chunk,
    _ws_handshake,
    _ws_read_frame,
    _ws_send_text,
)
from veeksha.core.request import Request
from veeksha.core.request_content import (
    AudioChannelRequestContent,
    ImageChannelRequestContent,
    TextChannelRequestContent,
)
from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
from veeksha.core.session import Session
from veeksha.core.session_graph import (
    SessionEdge,
    SessionGraph,
    SessionNode,
    add_edge,
    add_node,
)
from veeksha.core.tokenizer import build_word_split_tokenizer_provider
from veeksha.types import ChannelModality

__all__ = [
    "CaptureHttpServer",
    "RealtimeTtsCaptureServer",
    "SttCaptureServer",
    "make_text_request",
    "make_text_session",
    "make_linear_text_session",
    "write_test_wav",
    "word_split_provider",
    "wait_until",
    "drain_loop_until_idle",
]


def word_split_provider(model: str = "test-model"):
    return build_word_split_tokenizer_provider(model)


def wait_until(predicate, timeout_s: float = 20.0, interval_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def drain_loop_until_idle(loop, expect_completed: int, timeout_s: float = 30.0):
    """Drain a MainLoop until `expect_completed` COMPLETED events arrived and
    the loop reports idle; stop, join, and return all events in order."""
    from veeksha.loop.interface import LoopEventKind

    events = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        events.extend(loop.drain_events(max_items=512, timeout_s=0.1))
        completed = sum(1 for e in events if e.kind == LoopEventKind.COMPLETED)
        counters = loop.counters()
        if (
            completed >= expect_completed
            and counters.idle
            and counters.intake_exhausted
        ):
            break
    else:
        raise AssertionError(
            f"loop did not finish in {timeout_s}s "
            f"(events={len(events)}, counters={loop.counters()})"
        )
    loop.request_stop(grace_s=-1.0)
    assert loop.join(timeout_s=10.0), "loop threads did not join"
    while True:
        late = loop.drain_events(max_items=1024, timeout_s=0.05)
        if not late:
            break
        events.extend(late)
    return events


# ---------------------------------------------------------------------------
# session builders
# ---------------------------------------------------------------------------


def make_text_request(
    request_id: int,
    text: str,
    target_tokens: int = 8,
    metadata: Optional[dict] = None,
) -> Request:
    return Request(
        id=request_id,
        channels={ChannelModality.TEXT: TextChannelRequestContent(input_text=text)},
        metadata=metadata or {},
        requested_output=RequestedOutputSpec(
            text=TextOutputSpec(target_tokens=target_tokens)
        ),
    )


def make_text_session(session_id: int, request: Request) -> Session:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    return Session(id=session_id, session_graph=graph, requests={0: request})


def make_linear_text_session(
    session_id: int,
    texts: List[str],
    *,
    is_history_parent: bool = True,
    base_request_id: Optional[int] = None,
    target_tokens: int = 8,
) -> Session:
    """Linear multi-turn text session; request ids default to sid*100+turn."""
    base = base_request_id if base_request_id is not None else session_id * 100
    graph = SessionGraph()
    requests: Dict[int, Request] = {}
    for i, text in enumerate(texts):
        add_node(graph, SessionNode(id=i, wait_after_ready=0.0))
        requests[i] = make_text_request(base + i, text, target_tokens=target_tokens)
    for i in range(len(texts) - 1):
        add_edge(
            graph,
            SessionEdge(src=i, dst=i + 1, is_history_parent=is_history_parent),
        )
    return Session(id=session_id, session_graph=graph, requests=requests)


def make_image_history_session(session_id: int) -> Session:
    """A 2-turn session whose history parent carries an IMAGE channel —
    natively ineligible (non-text dynamic history)."""
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    add_node(graph, SessionNode(id=1, wait_after_ready=0.0))
    add_edge(graph, SessionEdge(src=0, dst=1, is_history_parent=True))
    parent = Request(
        id=session_id * 100,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(input_text="look"),
            ChannelModality.IMAGE: ImageChannelRequestContent(input_image="img://x"),
        },
    )
    child = make_text_request(session_id * 100 + 1, "and now?")
    return Session(id=session_id, session_graph=graph, requests={0: parent, 1: child})


def make_stt_session(session_id: int, audio_path: str) -> Session:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    request = Request(
        id=session_id * 100,
        channels={
            ChannelModality.AUDIO: AudioChannelRequestContent(input_audio=audio_path)
        },
        metadata={"ground_truth": "hello world"},
    )
    return Session(id=session_id, session_graph=graph, requests={0: request})


def write_test_wav(path: str, duration_s: float = 0.2, sample_rate: int = 16000) -> str:
    """A tiny deterministic 16-bit mono WAV (440 Hz sine)."""
    n = int(duration_s * sample_rate)
    frames = b"".join(
        struct.pack(
            "<h", int(12000 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
        )
        for i in range(n)
    )
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)
    return path


# ---------------------------------------------------------------------------
# capture servers
# ---------------------------------------------------------------------------


class CaptureHttpServer(AsyncServerThread):
    """Records the FULL raw request bytes (head + body), then streams a
    deterministic response.

    mode="sse": chunked text/event-stream with `num_chunks` chat deltas and
    a [DONE] terminator. mode="audio": chunked audio/pcm byte stream.
    """

    def __init__(
        self, mode: str = "sse", num_chunks: int = 3, token_text: str = "tok{i} "
    ):
        super().__init__()
        self.mode = mode
        self.num_chunks = num_chunks
        self.token_text = token_text
        self.requests: List[bytes] = []
        self._lock = threading.Lock()

    def full_reply(self) -> str:
        return "".join(self.token_text.format(i=i) for i in range(self.num_chunks))

    async def _handle(self, reader, writer):
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = await reader.read(65536)
            if not chunk:
                return
            data += chunk
        head, _, body = data.partition(b"\r\n\r\n")
        clen = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                clen = int(line.split(b":", 1)[1])
        while len(body) < clen:
            chunk = await reader.read(65536)
            if not chunk:
                break
            body += chunk
        with self._lock:
            self.requests.append(head + b"\r\n\r\n" + body)

        if self.mode == "sse":
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            await writer.drain()
            for i in range(self.num_chunks):
                await asyncio.sleep(0.005)
                payload = json.dumps(
                    {"choices": [{"delta": {"content": self.token_text.format(i=i)}}]}
                )
                _write_http_chunk(writer, f"data: {payload}\n\n")
                await writer.drain()
            _write_http_chunk(writer, "data: [DONE]\n\n")
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        else:
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: audio/pcm\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            await writer.drain()
            for i in range(self.num_chunks):
                await asyncio.sleep(0.005)
                payload = bytes([i % 251] * 64)
                writer.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        # wait briefly for the client to hang up (httpx keeps the pooled
        # connection alive past the response)
        try:
            await asyncio.wait_for(reader.read(65536), timeout=1)
        except Exception:
            pass


class RealtimeTtsCaptureServer(AsyncServerThread):
    """Realtime-TTS WS endpoint recording every received frame payload.

    Sends session.updated on connect; on response.create emits audio deltas
    on a fixed cadence, then a COMPLETED response.done (so the Python
    client's terminal validation passes).
    """

    def __init__(self, num_chunks: int = 3, chunk_gap_s: float = 0.02):
        super().__init__()
        self.num_chunks = num_chunks
        self.chunk_gap_s = chunk_gap_s
        self.frames: List[List[str]] = []  # per connection, in order
        self._lock = threading.Lock()

    async def _handle(self, reader, writer):
        if not await _ws_handshake(reader, writer):
            return
        received: List[str] = []
        with self._lock:
            self.frames.append(received)
        _ws_send_text(writer, json.dumps({"type": "session.updated"}))
        await writer.drain()
        while True:
            opcode, payload = await _ws_read_frame(reader)
            if opcode == 0x8:
                return
            if opcode not in (0x1, 0x2):
                continue
            text = payload.decode("utf-8", errors="replace")
            received.append(text)
            if "response.create" in text:
                break
        for _ in range(self.num_chunks):
            await asyncio.sleep(self.chunk_gap_s)
            _ws_send_text(
                writer,
                json.dumps(
                    {"type": "response.output_audio.delta", "delta": "QUJDRA=="}
                ),
            )
            await writer.drain()
        _ws_send_text(writer, json.dumps({"type": "response.output_audio.done"}))
        _ws_send_text(
            writer,
            json.dumps({"type": "response.done", "response": {"status": "completed"}}),
        )
        await writer.drain()
        try:
            await reader.read(65536)
        except Exception:
            pass


class SttCaptureServer(AsyncServerThread):
    """vllm_realtime STT WS endpoint recording every received frame payload.

    Sends session.created on connect (the Python client waits for it); on
    the FINAL commit emits transcript deltas then transcription.done.
    """

    def __init__(
        self, num_deltas: int = 3, delta_gap_s: float = 0.02, word: str = "w{i} "
    ):
        super().__init__()
        self.num_deltas = num_deltas
        self.delta_gap_s = delta_gap_s
        self.word = word
        self.frames: List[List[str]] = []  # per connection, in order
        self._lock = threading.Lock()

    def transcript(self) -> str:
        return "".join(self.word.format(i=i) for i in range(self.num_deltas))

    async def _handle(self, reader, writer):
        if not await _ws_handshake(reader, writer):
            return
        received: List[str] = []
        with self._lock:
            self.frames.append(received)
        _ws_send_text(writer, json.dumps({"type": "session.created"}))
        await writer.drain()
        while True:
            opcode, payload = await _ws_read_frame(reader)
            if opcode == 0x8:
                return
            if opcode not in (0x1, 0x2):
                continue
            text = payload.decode("utf-8", errors="replace")
            received.append(text)
            if "input_audio_buffer.commit" in text and "final" in text:
                break
        for i in range(self.num_deltas):
            await asyncio.sleep(self.delta_gap_s)
            _ws_send_text(
                writer,
                json.dumps(
                    {"type": "transcription.delta", "delta": self.word.format(i=i)}
                ),
            )
            await writer.drain()
        _ws_send_text(
            writer,
            json.dumps({"type": "transcription.done", "text": self.transcript()}),
        )
        await writer.drain()
        try:
            await reader.read(65536)
        except Exception:
            pass
