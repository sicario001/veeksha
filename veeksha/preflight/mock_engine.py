"""Built-in streaming mock engine with a known emit schedule.

An OpenAI-compatible SSE endpoint that emits chunks on an absolute schedule
(prefill delay, then a fixed inter-chunk cadence). Because the schedule is known,
any deviation in Veeksha's *recorded* timings is harness drift, not model
behaviour. The engine records its own emit lateness so the validator can confirm
the engine itself is not the bottleneck. Sharded across N accept loops via
SO_REUSEPORT so it can sustain thousands of connections.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import List, Optional, Tuple

from veeksha.logger import init_logger

logger = init_logger(__name__)


class MockStreamingEngine:
    """Localhost streaming SSE engine with a deterministic emit schedule."""

    def __init__(
        self,
        chunk_dt: float = 0.020,
        prefill_s: float = 0.050,
        default_chunks: int = 40,
        host: str = "127.0.0.1",
        num_loops: int = 8,
    ):
        self.chunk_dt = chunk_dt
        self.prefill_s = prefill_s
        self.default_chunks = default_chunks
        self.host = host
        self.num_loops = num_loops
        self.port: int = 0
        self._threads: List[threading.Thread] = []
        self._stoppers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._emit_lateness_ms: List[float] = []
        self._lat_lock = threading.Lock()
        self.active_conns = 0
        self.max_active_conns = 0
        self._conn_lock = threading.Lock()

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
            except asyncio.IncompleteReadError, asyncio.TimeoutError:
                body = b""

        num_chunks = self._chunks_from_body(body)

        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: close\r\n\r\n"
        )
        try:
            await writer.drain()
        except ConnectionResetError, BrokenPipeError:
            return

        start = time.monotonic()
        for i in range(num_chunks):
            scheduled = start + self.prefill_s + i * self.chunk_dt
            now = time.monotonic()
            if scheduled > now:
                await asyncio.sleep(scheduled - now)
            with self._lat_lock:
                self._emit_lateness_ms.append((time.monotonic() - scheduled) * 1000.0)
            payload = json.dumps({"choices": [{"index": 0, "delta": {"content": "x"}}]})
            writer.write(f"data: {payload}\n".encode())
            try:
                await writer.drain()
            except ConnectionResetError, BrokenPipeError:
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

    # --------------------------------------------------------------- lifecycle
    def start(self) -> "MockStreamingEngine":
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.host, 0))
        self.port = s.getsockname()[1]
        s.close()

        readies = [threading.Event() for _ in range(self.num_loops)]

        def _run(idx: int):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            stop_ev = asyncio.Event()
            self._stoppers.append((loop, stop_ev))

            async def _main():
                server = await asyncio.start_server(
                    self._handle, self.host, self.port, reuse_port=True
                )
                readies[idx].set()
                await stop_ev.wait()  # graceful shutdown signal
                server.close()
                try:
                    await server.wait_closed()
                except Exception:
                    pass

            try:
                loop.run_until_complete(_main())
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        for i in range(self.num_loops):
            t = threading.Thread(
                target=_run, args=(i,), daemon=True, name=f"preflight-engine-{i}"
            )
            t.start()
            self._threads.append(t)
        for r in readies:
            r.wait(timeout=5.0)
        return self

    def stop(self) -> None:
        for loop, ev in list(self._stoppers):
            try:
                loop.call_soon_threadsafe(ev.set)
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)

    def reset_telemetry(self) -> None:
        with self._lat_lock:
            self._emit_lateness_ms.clear()
        with self._conn_lock:
            self.max_active_conns = 0

    def server_jitter_p99_ms(self) -> float:
        with self._lat_lock:
            xs = sorted(self._emit_lateness_ms)
        if not xs:
            return 0.0
        return xs[min(len(xs) - 1, int(0.99 * (len(xs) - 1)))]
