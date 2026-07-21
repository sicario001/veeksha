"""Built-in streaming mock engine with a known emit schedule.

An OpenAI-compatible SSE endpoint that emits chunks on an absolute schedule
(prefill delay, then a fixed inter-chunk cadence). Because the schedule is known,
any deviation in Veeksha's *recorded* timings is harness drift, not model
behaviour. The engine records its own emit lateness so the validator can confirm
the engine itself is not the bottleneck. Sharded across N accept loops on one
shared listening socket (see ``sharded_server``) so it can sustain thousands of
connections on Linux and Darwin alike.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import List

from veeksha.logger import init_logger
from veeksha.preflight.sharded_server import (
    PhaseSpreader,
    ShardedLoopServer,
    ShardedTelemetry,
)

logger = init_logger(__name__)


class MockStreamingEngine(ShardedLoopServer):
    """Localhost streaming SSE engine with a deterministic emit schedule."""

    def __init__(
        self,
        chunk_dt: float = 0.020,
        prefill_s: float = 0.050,
        default_chunks: int = 40,
        host: str = "127.0.0.1",
        num_loops: int = 8,
        chunk_content: str = "x",
        store_bodies: bool = False,
    ):
        super().__init__(host=host, num_loops=num_loops)
        self.chunk_dt = chunk_dt
        self.prefill_s = prefill_s
        self.default_chunks = default_chunks
        # Pre-serialize the (identical) SSE data line once, so the emit loop does
        # no per-chunk json.dumps — keeps the server's emit lateness low.
        _payload = json.dumps(
            {"choices": [{"index": 0, "delta": {"content": chunk_content}}]}
        )
        self._data_line = f"data: {_payload}\n".encode()
        self._lateness = ShardedTelemetry()
        self._phase = PhaseSpreader()
        self.active_conns = 0
        self.max_active_conns = 0
        self._conn_lock = threading.Lock()
        # Optional request-body capture (tests: verify native history injection
        # actually put prior turns' outputs into subsequent requests).
        self.store_bodies = store_bodies
        self.request_bodies: List[bytes] = []
        self._body_lock = threading.Lock()

    async def _start_server(self, sock: socket.socket):
        return await asyncio.start_server(self._handle, sock=sock)

    # --------------------------------------------------------------- serving
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        with self._conn_lock:
            self.active_conns += 1
            self.max_active_conns = max(self.max_active_conns, self.active_conns)
        try:
            await self._serve(reader, writer)
        finally:
            with self._conn_lock:
                self.active_conns -= 1

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

        # spread this connection's schedule so deadlines don't all land at once
        start = time.monotonic() + self._phase.next_phase(self.chunk_dt)
        for i in range(num_chunks):
            scheduled = start + self.prefill_s + i * self.chunk_dt
            now = time.monotonic()
            if scheduled > now:
                await asyncio.sleep(scheduled - now)
            # lock-free per-thread telemetry; lateness computed before recording
            self._lateness.record((time.monotonic() - scheduled) * 1000.0)
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
        for key in ("max_completion_tokens", "max_tokens", "min_tokens"):
            v = obj.get(key)
            if isinstance(v, int) and v > 0:
                return v
        return self.default_chunks

    # --------------------------------------------------------------- telemetry
    def reset_telemetry(self) -> None:
        self._lateness.clear()
        with self._conn_lock:
            self.max_active_conns = 0

    def server_jitter_p99_ms(self) -> float:
        return self._lateness.p99()
