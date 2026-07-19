"""A mock OpenAI-Realtime TTS WebSocket server for tests + preflight.

Speaks the minimal realtime contract the RealtimeTTSClient expects: on
``session.update`` it replies ``session.updated``, and on ``response.create`` it
emits ``response.created``, ``num_chunks`` ``response.output_audio.delta`` frames
(base64 PCM) on an ABSOLUTE schedule (so send backpressure never accumulates into
the cadence), then ``response.output_audio.done`` / ``response.done``.

Like the text ``MockStreamingEngine``, it is sharded across ``num_loops`` accept
loops via SO_REUSEPORT so it can sustain many connections without one loop
becoming the bottleneck, and it records its own emit lateness
(``server_jitter_p99_ms``) so a benchmark can confirm the *server* is not the
limiter at a given concurrency.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import threading
import time
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
        num_loops: int = 8,
    ):
        self.num_chunks = num_chunks
        self.chunk_bytes = chunk_bytes
        self.first_delta_delay = first_delta_delay
        self.delta_dt = delta_dt
        self.sample_rate = sample_rate
        self.host = host
        self.num_loops = num_loops
        self.port: int = 0
        self._emit_lateness_ms: List[float] = []
        self._lat_lock = threading.Lock()
        self._stoppers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._threads: List[threading.Thread] = []

    def server_jitter_p99_ms(self) -> float:
        with self._lat_lock:
            xs = sorted(self._emit_lateness_ms)
        if not xs:
            return 0.0
        return xs[min(len(xs) - 1, int(round(0.99 * (len(xs) - 1))))]

    def reset_telemetry(self) -> None:
        with self._lat_lock:
            self._emit_lateness_ms.clear()

    async def _emit_audio(self, ws) -> None:
        await ws.send(json.dumps({"type": "response.created"}))
        pcm = b"\x00" * self.chunk_bytes
        encoded = base64.b64encode(pcm).decode("ascii")
        start = time.monotonic()
        for i in range(self.num_chunks):
            scheduled = start + self.first_delta_delay + i * self.delta_dt
            now = time.monotonic()
            if scheduled > now:
                await asyncio.sleep(scheduled - now)
            with self._lat_lock:
                self._emit_lateness_ms.append((time.monotonic() - scheduled) * 1000.0)
            try:
                await ws.send(
                    json.dumps(
                        {"type": "response.output_audio.delta", "delta": encoded}
                    )
                )
            except Exception:
                return
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
            t = threading.Thread(
                target=_run, args=(i,), daemon=True, name="mock-rt-tts"
            )
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
