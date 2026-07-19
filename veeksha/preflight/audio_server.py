"""A minimal fixed-cadence realtime-TTS WebSocket server for the audio preflight.

The audio analogue of ``MockStreamingEngine``: it emits audio deltas on a known
schedule so the preflight can measure how faithfully the REAL realtime client
records per-chunk arrival timing over an actual WebSocket transport. Speaks just
enough of the OpenAI-realtime contract that ``RealtimeTTSClient`` accepts it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import threading
from typing import List, Optional, Tuple

import websockets


class MockRealtimeAudioServer:
    """Emits ``num_chunks`` audio deltas ``chunk_dt`` apart on response.create."""

    def __init__(
        self,
        num_chunks: int = 20,
        chunk_bytes: int = 4800,
        first_delta_delay: float = 0.02,
        chunk_dt: float = 0.02,
        sample_rate: int = 24000,
        host: str = "127.0.0.1",
    ):
        self.num_chunks = num_chunks
        self.chunk_bytes = chunk_bytes
        self.first_delta_delay = first_delta_delay
        self.chunk_dt = chunk_dt
        self.sample_rate = sample_rate
        self.host = host
        self.port: int = 0
        self._stoppers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._threads: List[threading.Thread] = []

    async def _emit_audio(self, ws) -> None:
        await ws.send(json.dumps({"type": "response.created"}))
        pcm = b"\x00" * self.chunk_bytes
        encoded = base64.b64encode(pcm).decode("ascii")
        # Absolute-deadline schedule so send backpressure never accumulates drift
        # into the emitted cadence (the server must be the honest reference).
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.first_delta_delay
        for _ in range(self.num_chunks):
            sleep_s = deadline - loop.time()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            try:
                await ws.send(
                    json.dumps(
                        {"type": "response.output_audio.delta", "delta": encoded}
                    )
                )
            except Exception:
                return
            deadline += self.chunk_dt
        try:
            await ws.send(json.dumps({"type": "response.output_audio.done"}))
            await ws.send(
                json.dumps(
                    {"type": "response.done", "response": {"status": "completed"}}
                )
            )
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
        except Exception:
            pass
        finally:
            if audio_task is not None:
                audio_task.cancel()

    def start(self) -> "MockRealtimeAudioServer":
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

        t = threading.Thread(target=_run, daemon=True, name="preflight-audio")
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


class MockSTTPreflightServer:
    """STT WebSocket server that RECORDS where the client's audio actually lands.

    For ASR the interactivity-critical drift is on the *send* side: veeksha must
    stream the input audio at 1x real time. This server measures that at the
    ground-truth point — it timestamps each ``input_audio_buffer.append`` on
    arrival (per connection, relative to that connection's first append) — while
    emitting transcript deltas on a fixed schedule so receive timing is
    deterministic too. Speaks the minimal vllm_realtime contract STTClient expects.
    """

    def __init__(
        self,
        transcript: str = "the quick brown fox jumps over the lazy dog",
        first_delta_delay: float = 0.05,
        delta_dt: float = 0.03,
        host: str = "127.0.0.1",
    ):
        self.transcript = transcript
        self.first_delta_delay = first_delta_delay
        self.delta_dt = delta_dt
        self.host = host
        self.port: int = 0
        # per-connection append-arrival offsets (ms from that conn's first append)
        self.append_arrivals: List[List[float]] = []
        self._stoppers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._threads: List[threading.Thread] = []

    async def _handler(self, ws) -> None:
        import time

        await ws.send(json.dumps({"type": "session.created"}))
        words = self.transcript.split()

        async def _emit_transcript() -> None:
            # Only after the client's EOF, per the real-server contract ("done"
            # comes after all audio) — so the full paced send is measured, not
            # truncated when the transcript finishes early.
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
                await ws.close()
            except Exception:
                pass

        arrivals: List[float] = []
        first_append: Optional[float] = None
        transcript_task: Optional[asyncio.Future] = None
        recorded = False
        try:
            async for raw in ws:  # drain client audio; timestamp each append
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError, TypeError, ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                etype = event.get("type")
                if etype == "input_audio_buffer.append":
                    now = time.monotonic()
                    if first_append is None:
                        first_append = now
                    arrivals.append((now - first_append) * 1000.0)
                elif etype == "input_audio_buffer.commit" and event.get("final"):
                    # client EOF: full audio received. Record the arrival timeline
                    # HERE (deterministic — before the connection closes) so the
                    # measurement never races the close, then emit the transcript.
                    if not recorded and arrivals:
                        self.append_arrivals.append(list(arrivals))
                        recorded = True
                    if transcript_task is None:
                        transcript_task = asyncio.ensure_future(_emit_transcript())
        except Exception:
            pass
        finally:
            if transcript_task is not None:
                transcript_task.cancel()
            if not recorded and arrivals:  # client that closed without a final commit
                self.append_arrivals.append(arrivals)

    def start(self) -> "MockSTTPreflightServer":
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

        t = threading.Thread(target=_run, daemon=True, name="preflight-stt")
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
