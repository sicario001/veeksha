"""Built-in streaming mock engine with a known emit schedule.

An OpenAI-compatible SSE endpoint (``POST /v1/chat/completions``) that emits
chunks on an ABSOLUTE schedule (a prefill delay, then a fixed inter-chunk
cadence). Because the schedule is known, any deviation in Veeksha's *recorded*
timings is harness drift, not model behaviour. The paired checks score the
client against the stamp the mock took when it ACTUALLY sent each chunk
(``emitted_at``), so the mock's own adherence to its schedule never enters a
reported number and is not tracked.

Sharded across N accept loops on one shared listening socket (see
``sharded_server``) so it can sustain thousands of connections on Linux and
Darwin alike.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Dict, List, Optional

from veeksha.logger import init_logger
from veeksha.preflight.sharded_server import (
    ACCEPT_BACKLOG,
    PhaseSpreader,
    ServerRecord,
    ServerRecordBook,
    ShardedLoopServer,
    parse_pfid,
)

logger = init_logger(__name__)

__all__ = ["MockStreamingEngine"]


class MockStreamingEngine(ShardedLoopServer):
    """Localhost streaming SSE engine with a deterministic emit schedule.

    Per stream: sleep ``prefill_ms``, then emit ``num_chunks`` chat-delta
    events on the absolute deadlines ``T_conn_phase + prefill + i*chunk_ms``,
    then ``data: [DONE]``.

    Defaults model a real model rather than a toy: ``chunk_ms=20``
    (~50 tok/s) and ``prefill_ms=200``. A mock that answers in 0.4 s reports a
    concurrency ceiling no real 5 s-response benchmark ever reaches, because
    harness load scales with requests/second = N / lifetime.
    """

    def __init__(
        self,
        chunk_ms: float = 20.0,
        prefill_ms: float = 200.0,
        default_chunks: int = 240,
        host: str = "127.0.0.1",
        num_loops: int = 8,
        chunk_content: str = "x",
        store_bodies: bool = False,
    ):
        super().__init__(host=host, num_loops=num_loops)
        self.chunk_ms = chunk_ms
        self.prefill_ms = prefill_ms
        self.default_chunks = default_chunks
        self.chunk_content = chunk_content
        self.chunk_dt = chunk_ms / 1000.0
        self.prefill_s = prefill_ms / 1000.0
        # Pre-serialize the (identical) SSE data line once, so the emit loop does
        # no per-chunk json.dumps — keeps the server's emit lateness low.
        payload = json.dumps(
            {"choices": [{"index": 0, "delta": {"content": chunk_content}}]}
        )
        self._data_line = f"data: {payload}\n".encode()
        self._phase = PhaseSpreader()
        # Paired-timestamp ground truth: when this mock received each request
        # and when it sent each chunk (see ServerRecordBook).
        self._records = ServerRecordBook()
        # Optional request-body capture (tests: verify multi-turn history
        # injection actually put prior turns' outputs into subsequent requests).
        self.store_bodies = store_bodies
        self.request_bodies: List[bytes] = []
        self._body_lock = threading.Lock()

    async def _start_server(self, sock: socket.socket):
        return await asyncio.start_server(
            self._handle, sock=sock, backlog=ACCEPT_BACKLOG
        )

    # --------------------------------------------------------------- serving
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # Connection counting lives in the ShardedLoopServer base — one
        # definition of "achieved concurrency" for every mock server.
        # Coalesced writes would land on the client as chunk lateness this mock
        # did not intend, so never let Nagle batch an SSE chunk. (asyncio's
        # selector transport already does this for TCP; being explicit means the
        # reference clock does not depend on that staying true.)
        sock = writer.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        with self.track_connection():
            await self._serve(reader, writer)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
        except (
            asyncio.IncompleteReadError,
            asyncio.TimeoutError,
            asyncio.LimitOverrunError,
        ):
            writer.close()
            return
        # Ground truth first: the read that completed the request head has just
        # returned, so stamp before parsing a single header.
        record = self._records.open(time.monotonic())

        content_length = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                try:
                    content_length = int(line.split(b":", 1)[1].strip())
                except ValueError:
                    content_length = 0

        body = b""
        if content_length:
            try:
                body = await asyncio.wait_for(
                    reader.readexactly(content_length), timeout=10.0
                )
            except (asyncio.IncompleteReadError, asyncio.TimeoutError):
                body = b""

        if self.store_bodies and body:
            with self._body_lock:
                self.request_bodies.append(body)

        # Correlation: the preflight embeds ``PFID:<n>`` in the prompt, so the
        # id is anywhere in the messages content. Parsed AFTER the receive
        # stamp — this cost belongs to neither side's delivery lag.
        record.request_id = parse_pfid(body.decode("utf-8", "ignore")) if body else None

        num_chunks = self._chunks_from_body(body)

        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: close\r\n\r\n"
        )
        try:
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            return

        # Spread this connection's schedule so deadlines don't all land at once.
        start = time.monotonic() + self._phase.next_phase(self.chunk_dt)
        for i in range(num_chunks):
            scheduled = start + self.prefill_s + i * self.chunk_dt
            now = time.monotonic()
            if scheduled > now:
                await asyncio.sleep(scheduled - now)
            # The absolute send stamp the client's arrival of this chunk is
            # paired against, taken immediately before the write.
            sent_at = time.monotonic()
            record.emitted_at.append(sent_at)
            writer.write(self._data_line)  # pre-serialized once (see __init__)
            try:
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                return
        writer.write(b"data: [DONE]\n")
        try:
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    def _chunks_from_body(self, body: bytes) -> int:
        if not body:
            return self.default_chunks
        try:
            obj = json.loads(body.decode("utf-8", "ignore"))
        except json.JSONDecodeError:
            return self.default_chunks
        if not isinstance(obj, dict):
            return self.default_chunks
        for key in ("max_completion_tokens", "max_tokens", "min_tokens"):
            v = obj.get(key)
            if isinstance(v, int) and v > 0:
                return v
        return self.default_chunks

    # --------------------------------------------------------------- telemetry
    def reset_telemetry(self) -> None:
        self._phase.reset()
        self._records.clear()
        self.reset_connection_peak()
        with self._body_lock:
            self.request_bodies = []

    def records(self) -> Dict[int, ServerRecord]:
        """Per-request receive/emit stamps, keyed by the request's PFID."""
        return self._records.records()

    def unidentified_connections(self) -> int:
        """Connections served without a usable preflight id (see ServerRecordBook)."""
        return self._records.unidentified_connections()

    def captured_bodies(self) -> List[Optional[dict]]:
        """Decoded captured request bodies (requires ``store_bodies=True``)."""
        with self._body_lock:
            raw = list(self.request_bodies)
        out: List[Optional[dict]] = []
        for body in raw:
            try:
                out.append(json.loads(body.decode("utf-8", "ignore")))
            except json.JSONDecodeError:
                out.append(None)
        return out
