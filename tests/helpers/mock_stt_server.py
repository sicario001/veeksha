"""A mock realtime STT WebSocket server (vllm_realtime protocol) for tests.

Speaks the minimal vLLM realtime contract the STTClient expects: on connect it
sends ``session.created``, then streams a few ``transcription.delta`` messages
and a final ``transcription.done`` on a known schedule while draining the audio
the client sends. Because the transcript and timing are fixed, the client's
metrics are deterministic.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from typing import List, Optional, Tuple

import websockets


class MockSTTServer:
    def __init__(
        self,
        transcript: str = "the quick brown fox",
        first_delta_delay: float = 0.05,
        delta_dt: float = 0.03,
        host: str = "127.0.0.1",
        num_loops: int = 8,
    ):
        self.transcript = transcript
        self.first_delta_delay = first_delta_delay
        self.delta_dt = delta_dt
        self.host = host
        self.num_loops = num_loops
        self.port: int = 0
        # Pre-serialize the fixed transcript messages once so the emit loop stays
        # cheap under many connections (no per-word json.dumps).
        words = transcript.split()
        self._created_msg = json.dumps({"type": "session.created"})
        self._delta_msgs = [
            json.dumps({"type": "transcription.delta", "delta": (" " + w if i else w)})
            for i, w in enumerate(words)
        ]
        self._done_msg = json.dumps({"type": "transcription.done", "text": transcript})
        self._stoppers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._threads: List[threading.Thread] = []

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

    def start(self) -> "MockSTTServer":
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
                async with websockets.serve(
                    self._handler, self.host, self.port, reuse_port=True
                ):
                    readies[idx].set()
                    await stop_ev.wait()

            try:
                loop.run_until_complete(_main())
            finally:
                loop.close()

        for i in range(self.num_loops):
            t = threading.Thread(target=_run, args=(i,), daemon=True, name="mock-stt")
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
