"""A mock realtime STT WebSocket server (vllm_realtime protocol) for tests.

Speaks the minimal vLLM realtime contract the STTClient expects: on connect it
sends ``session.created``, then streams a few ``transcription.delta`` messages
and a final ``transcription.done`` on a known schedule while draining the audio
the client sends. Because the transcript and timing are fixed, the client's
metrics are deterministic. Sharded via ``ShardedLoopServer`` (one shared
listening socket, ``num_loops`` accept loops).
"""

from __future__ import annotations

import asyncio
import json
import socket

import websockets

from veeksha.preflight.sharded_server import ShardedLoopServer


class MockSTTServer(ShardedLoopServer):
    def __init__(
        self,
        transcript: str = "the quick brown fox",
        first_delta_delay: float = 0.05,
        delta_dt: float = 0.03,
        host: str = "127.0.0.1",
        num_loops: int = 8,
    ):
        super().__init__(host=host, num_loops=num_loops)
        self.transcript = transcript
        self.first_delta_delay = first_delta_delay
        self.delta_dt = delta_dt
        # Pre-serialize the fixed transcript messages once so the emit loop stays
        # cheap under many connections (no per-word json.dumps).
        words = transcript.split()
        self._created_msg = json.dumps({"type": "session.created"})
        self._delta_msgs = [
            json.dumps({"type": "transcription.delta", "delta": (" " + w if i else w)})
            for i, w in enumerate(words)
        ]
        self._done_msg = json.dumps({"type": "transcription.done", "text": transcript})

    async def _start_server(self, sock: socket.socket):
        return await websockets.serve(self._handler, sock=sock)

    async def _handler(self, ws) -> None:
        await ws.send(self._created_msg)

        async def _sender() -> None:
            await asyncio.sleep(self.first_delta_delay)
            for delta_msg in self._delta_msgs:  # pre-serialized
                try:
                    await ws.send(delta_msg)
                except Exception:
                    return
                await asyncio.sleep(self.delta_dt)
            try:
                await ws.send(self._done_msg)
                await ws.close()  # terminal: let clients stop promptly
            except Exception:
                pass

        task = asyncio.ensure_future(_sender())
        try:
            async for _ in ws:  # drain client audio/control messages
                pass
        except Exception:
            pass
        finally:
            task.cancel()
