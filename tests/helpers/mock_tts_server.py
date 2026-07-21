"""A mock HTTP TTS server for end-to-end tests.

Responds to ``POST /v1/audio/speech`` by streaming raw PCM audio bytes on a known
schedule (prefill delay, then fixed-size chunks at a fixed cadence). Because the
schedule and byte counts are known, the audio metrics the client/evaluator
produce are deterministic and assertable. Sharded via ``ShardedLoopServer`` (one
shared listening socket, ``num_loops`` accept loops).
"""

from __future__ import annotations

import asyncio
import socket
import time

from veeksha.preflight.sharded_server import ShardedLoopServer, ShardedTelemetry


class MockTTSServer(ShardedLoopServer):
    def __init__(
        self,
        chunk_bytes: int = 4800,
        num_chunks: int = 10,
        chunk_dt: float = 0.05,
        prefill_s: float = 0.05,
        host: str = "127.0.0.1",
        num_loops: int = 8,
    ):
        super().__init__(host=host, num_loops=num_loops)
        self.chunk_bytes = chunk_bytes
        self.num_chunks = num_chunks
        self.chunk_dt = chunk_dt
        self.prefill_s = prefill_s
        self._payload = b"\x00" * chunk_bytes  # precomputed once
        self._lateness = ShardedTelemetry()

    @property
    def total_bytes(self) -> int:
        return self.chunk_bytes * self.num_chunks

    async def _start_server(self, sock: socket.socket):
        return await asyncio.start_server(self._serve, sock=sock)

    def server_jitter_p99_ms(self) -> float:
        return self._lateness.p99()

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
        except (ConnectionResetError, BrokenPipeError):
            return

        start = time.monotonic()
        for i in range(self.num_chunks):
            scheduled = start + self.prefill_s + i * self.chunk_dt
            now = time.monotonic()
            if scheduled > now:
                await asyncio.sleep(scheduled - now)
            # lock-free per-thread telemetry; lateness computed before recording
            self._lateness.record((time.monotonic() - scheduled) * 1000.0)
            writer.write(self._payload)  # precomputed
            try:
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                return
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
