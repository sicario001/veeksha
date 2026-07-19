"""A mock OpenAI-Realtime TTS WebSocket server for tests.

Speaks the minimal realtime contract the RealtimeTTSClient expects: on
``session.update`` it replies ``session.updated`` (echoing the output sample
rate), and on ``response.create`` it emits ``response.created``, a fixed number
of ``response.output_audio.delta`` frames (base64 PCM) on a known schedule, then
``response.output_audio.done`` and ``response.done``. Deterministic transcript
timing makes the client's audio metrics reproducible.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import threading
from typing import List, Optional, Tuple

import websockets


class MockRealtimeTTSServer:
    def __init__(
        self,
        num_chunks: int = 5,
        chunk_bytes: int = 4800,
        first_delta_delay: float = 0.04,
        delta_dt: float = 0.02,
        sample_rate: int = 24000,
        host: str = "127.0.0.1",
    ):
        self.num_chunks = num_chunks
        self.chunk_bytes = chunk_bytes
        self.first_delta_delay = first_delta_delay
        self.delta_dt = delta_dt
        self.sample_rate = sample_rate
        self.host = host
        self.port: int = 0
        self._stoppers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._threads: List[threading.Thread] = []

    async def _emit_audio(self, ws) -> None:
        await ws.send(json.dumps({"type": "response.created"}))
        await asyncio.sleep(self.first_delta_delay)
        pcm = b"\x00" * self.chunk_bytes
        encoded = base64.b64encode(pcm).decode("ascii")
        for _ in range(self.num_chunks):
            try:
                await ws.send(
                    json.dumps(
                        {"type": "response.output_audio.delta", "delta": encoded}
                    )
                )
            except Exception:
                return
            await asyncio.sleep(self.delta_dt)
        try:
            await ws.send(json.dumps({"type": "response.output_audio.done"}))
            await ws.send(
                json.dumps(
                    {"type": "response.done", "response": {"status": "completed"}}
                )
            )
            await ws.close()  # terminal: let clients stop promptly
        except Exception:
            pass

    async def _handler(self, ws) -> None:
        audio_task: Optional[asyncio.Future] = None
        try:
            async for raw in ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError, TypeError, ValueError:
                    continue
                etype = event.get("type") if isinstance(event, dict) else None
                if etype == "session.update":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "session.updated",
                                "session": {
                                    "audio": {
                                        "output": {
                                            "format": {
                                                "type": "audio/pcm",
                                                "rate": self.sample_rate,
                                            }
                                        }
                                    }
                                },
                            }
                        )
                    )
                elif etype == "response.create":
                    audio_task = asyncio.ensure_future(self._emit_audio(ws))
                # conversation.item.create: text deltas, just drained.
        except Exception:
            pass
        finally:
            if audio_task is not None:
                audio_task.cancel()

    def start(self) -> "MockRealtimeTTSServer":
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

        t = threading.Thread(target=_run, daemon=True, name="mock-realtime-tts")
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
