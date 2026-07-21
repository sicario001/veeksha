"""A mock OpenAI-Realtime TTS WebSocket server for tests + preflight.

Speaks the minimal realtime contract the RealtimeTTSClient expects: on
``session.update`` it replies ``session.updated``, and on ``response.create`` it
emits ``response.created``, ``num_chunks`` ``response.output_audio.delta`` frames
(base64 PCM) on an ABSOLUTE schedule (so send backpressure never accumulates into
the cadence), then ``response.output_audio.done`` / ``response.done``.

Like the text ``MockStreamingEngine``, it is sharded across ``num_loops`` accept
loops on one shared listening socket (``ShardedLoopServer`` — SO_REUSEPORT does
not distribute TCP accepts on macOS) so it can sustain many connections without
one loop becoming the bottleneck, and it records its own emit lateness
(``server_jitter_p99_ms``) so a benchmark can confirm the *server* is not the
limiter at a given concurrency.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import time
from typing import Optional

import websockets

from veeksha.preflight.sharded_server import (
    PhaseSpreader,
    ShardedLoopServer,
    ShardedTelemetry,
)


class MockRealtimeTTSServer(ShardedLoopServer):
    def __init__(
        self,
        num_chunks: int = 5,
        chunk_bytes: int = 4800,
        first_delta_delay: float = 0.04,
        delta_dt: float = 0.02,
        sample_rate: int = 24000,
        host: str = "127.0.0.1",
        num_loops: int = 8,
    ):
        super().__init__(host=host, num_loops=num_loops)
        self.num_chunks = num_chunks
        self.chunk_bytes = chunk_bytes
        self.first_delta_delay = first_delta_delay
        self.delta_dt = delta_dt
        self.sample_rate = sample_rate
        self._lateness = ShardedTelemetry()
        self._phase = PhaseSpreader()
        # Pre-serialize every repeated message ONCE: the audio delta is identical
        # for every chunk of every connection, so json.dumps-per-chunk is pure
        # wasted CPU on the emit loop (and shows up as server jitter). Do it here.
        encoded = base64.b64encode(b"\x00" * chunk_bytes).decode("ascii")
        self._delta_msg = json.dumps(
            {"type": "response.output_audio.delta", "delta": encoded}
        )
        self._audio_done_msg = json.dumps({"type": "response.output_audio.done"})
        self._response_done_msg = json.dumps(
            {"type": "response.done", "response": {"status": "completed"}}
        )
        self._session_updated_msg = json.dumps(
            {
                "type": "session.updated",
                "session": {
                    "audio": {
                        "output": {"format": {"type": "audio/pcm", "rate": sample_rate}}
                    }
                },
            }
        )

    async def _start_server(self, sock: socket.socket):
        return await websockets.serve(self._handler, sock=sock)

    def server_jitter_p99_ms(self) -> float:
        return self._lateness.p99()

    def reset_telemetry(self) -> None:
        self._lateness.clear()

    async def _emit_audio(self, ws) -> None:
        await ws.send(json.dumps({"type": "response.created"}))
        start = time.monotonic() + self._phase.next_phase(self.delta_dt)
        for i in range(self.num_chunks):
            scheduled = start + self.first_delta_delay + i * self.delta_dt
            now = time.monotonic()
            if scheduled > now:
                await asyncio.sleep(scheduled - now)
            # lock-free per-thread telemetry; lateness computed before recording
            self._lateness.record((time.monotonic() - scheduled) * 1000.0)
            try:
                await ws.send(self._delta_msg)  # pre-serialized once (see __init__)
            except Exception:
                return
        try:
            await ws.send(self._audio_done_msg)
            await ws.send(self._response_done_msg)
            await ws.close()  # terminal: let clients stop promptly
        except Exception:
            pass

    async def _handler(self, ws) -> None:
        audio_task: Optional[asyncio.Future] = None
        try:
            async for raw in ws:
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                etype = event.get("type") if isinstance(event, dict) else None
                if etype == "session.update":
                    await ws.send(self._session_updated_msg)
                elif etype == "response.create" and audio_task is None:
                    audio_task = asyncio.ensure_future(self._emit_audio(ws))
                # conversation.item.create: text deltas, just drained.
        except Exception:
            pass
        finally:
            if audio_task is not None:
                audio_task.cancel()
