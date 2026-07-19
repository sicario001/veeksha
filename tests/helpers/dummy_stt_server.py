"""A dummy realtime STT WebSocket server (vllm_realtime protocol) for tests.

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


class DummySTTServer:
    def __init__(
        self,
        transcript: str = "the quick brown fox",
        first_delta_delay: float = 0.05,
        delta_dt: float = 0.03,
        host: str = "127.0.0.1",
    ):
        self.transcript = transcript
        self.first_delta_delay = first_delta_delay
        self.delta_dt = delta_dt
        self.host = host
        self.port: int = 0
        self._stoppers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._threads: List[threading.Thread] = []

    async def _handler(self, ws) -> None:
        await ws.send(json.dumps({"type": "session.created"}))
        words = self.transcript.split()

        async def _sender() -> None:
            await asyncio.sleep(self.first_delta_delay)
            for i, w in enumerate(words):
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "transcription.delta",
                                "delta": (" " + w if i else w),
                            }
                        )
                    )
                except Exception:
                    return
                await asyncio.sleep(self.delta_dt)
            try:
                await ws.send(
                    json.dumps({"type": "transcription.done", "text": self.transcript})
                )
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

    def start(self) -> "DummySTTServer":
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.host, 0))
        self.port = s.getsockname()[1]
        s.close()
        ready = threading.Event()

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            stop_ev = asyncio.Event()
            self._stoppers.append((loop, stop_ev))

            async def _main():
                async with websockets.serve(self._handler, self.host, self.port):
                    ready.set()
                    await stop_ev.wait()

            try:
                loop.run_until_complete(_main())
            finally:
                loop.close()

        t = threading.Thread(target=_run, daemon=True, name="dummy-stt")
        t.start()
        self._threads.append(t)
        ready.wait(timeout=5.0)
        return self

    def stop(self) -> None:
        for loop, ev in self._stoppers:
            try:
                loop.call_soon_threadsafe(ev.set)
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
