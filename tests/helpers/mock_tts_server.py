"""A mock HTTP TTS server for end-to-end tests.

Responds to ``POST /v1/audio/speech`` by streaming raw PCM audio bytes on a known
schedule (prefill delay, then fixed-size chunks at a fixed cadence). Because the
schedule and byte counts are known, the audio metrics the client/evaluator
produce are deterministic and assertable.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import List, Optional, Tuple


class MockTTSServer:
    def __init__(
        self,
        chunk_bytes: int = 4800,
        num_chunks: int = 10,
        chunk_dt: float = 0.05,
        prefill_s: float = 0.05,
        host: str = "127.0.0.1",
        num_loops: int = 8,
    ):
        self.chunk_bytes = chunk_bytes
        self.num_chunks = num_chunks
        self.chunk_dt = chunk_dt
        self.prefill_s = prefill_s
        self.host = host
        self.num_loops = num_loops
        self.port: int = 0
        self._payload = b"\x00" * chunk_bytes  # precomputed once
        self._emit_lateness_ms: List[float] = []
        self._lat_lock = threading.Lock()
        self._stoppers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._threads: List[threading.Thread] = []

    @property
    def total_bytes(self) -> int:
        return self.chunk_bytes * self.num_chunks

    def server_jitter_p99_ms(self) -> float:
        with self._lat_lock:
            xs = sorted(self._emit_lateness_ms)
        if not xs:
            return 0.0
        return xs[min(len(xs) - 1, int(round(0.99 * (len(xs) - 1))))]

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
        except Exception:
            writer.close()
            return
        content_length = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                try:
                    content_length = int(line.split(b":", 1)[1].strip())
                except ValueError:
                    content_length = 0
        if content_length:
            try:
                await asyncio.wait_for(reader.readexactly(content_length), timeout=10.0)
            except Exception:
                pass

        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/octet-stream\r\n"
            b"Connection: close\r\n\r\n"
        )
        try:
            await writer.drain()
        except ConnectionResetError, BrokenPipeError:
            return

        start = time.monotonic()
        for i in range(self.num_chunks):
            scheduled = start + self.prefill_s + i * self.chunk_dt
            now = time.monotonic()
            if scheduled > now:
                await asyncio.sleep(scheduled - now)
            with self._lat_lock:
                self._emit_lateness_ms.append((time.monotonic() - scheduled) * 1000.0)
            writer.write(self._payload)  # precomputed
            try:
                await writer.drain()
            except ConnectionResetError, BrokenPipeError:
                return
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    def start(self) -> "MockTTSServer":
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
                    self._serve, self.host, self.port, reuse_port=True
                )
                readies[idx].set()
                await stop_ev.wait()
                server.close()
                try:
                    await server.wait_closed()
                except Exception:
                    pass

            try:
                loop.run_until_complete(_main())
            finally:
                loop.close()

        for i in range(self.num_loops):
            t = threading.Thread(target=_run, args=(i,), daemon=True, name="mock-tts")
            t.start()
            self._threads.append(t)
        for r in readies:
            r.wait(timeout=5.0)
        return self

    def stop(self) -> None:
        for loop, ev in self._stoppers:
            try:
                loop.call_soon_threadsafe(ev.set)
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
