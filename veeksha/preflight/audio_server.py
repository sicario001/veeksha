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
import time
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
        num_loops: int = 8,
    ):
        self.num_chunks = num_chunks
        self.chunk_bytes = chunk_bytes
        self.first_delta_delay = first_delta_delay
        self.chunk_dt = chunk_dt
        self.sample_rate = sample_rate
        self.host = host
        self.num_loops = num_loops
        self.port: int = 0
        self._emit_lateness_ms: List[float] = []
        self._lat_lock = threading.Lock()
        # Pre-serialize repeated messages once (identical for every chunk/conn).
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
        start = time.monotonic()
        for i in range(self.num_chunks):
            scheduled = start + self.first_delta_delay + i * self.chunk_dt
            now = time.monotonic()
            if scheduled > now:
                await asyncio.sleep(scheduled - now)
            with self._lat_lock:
                self._emit_lateness_ms.append((time.monotonic() - scheduled) * 1000.0)
            try:
                await ws.send(self._delta_msg)  # pre-serialized
            except Exception:
                return
        try:
            await ws.send(self._audio_done_msg)
            await ws.send(self._response_done_msg)
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
                    await ws.send(self._session_updated_msg)
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
                target=_run, args=(i,), daemon=True, name="preflight-audio"
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
        num_loops: int = 8,
    ):
        self.transcript = transcript
        self.first_delta_delay = first_delta_delay
        self.delta_dt = delta_dt
        self.host = host
        self.num_loops = num_loops
        self.port: int = 0
        # per-connection append-arrival offsets (ms from that conn's first append)
        self.append_arrivals: List[List[float]] = []
        self._arr_lock = threading.Lock()
        # Pre-serialize the (fixed) transcript messages once so the emit loop is
        # cheap and doesn't steal cycles from receiving/timestamping appends.
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
        import time

        await ws.send(self._created_msg)

        async def _emit_transcript() -> None:
            # Only after the client's EOF, per the real-server contract ("done"
            # comes after all audio) — so the full paced send is measured, not
            # truncated when the transcript finishes early.
            await asyncio.sleep(self.first_delta_delay)
            for delta_msg in self._delta_msgs:  # pre-serialized
                try:
                    await ws.send(delta_msg)
                except Exception:
                    return
                await asyncio.sleep(self.delta_dt)
            try:
                await ws.send(self._done_msg)  # pre-serialized
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
                        with self._arr_lock:
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
                with self._arr_lock:
                    self.append_arrivals.append(arrivals)

    def start(self) -> "MockSTTPreflightServer":
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
                # Sharded across num_loops loops via SO_REUSEPORT so the server's
                # own receive-processing lag stays small at high concurrency —
                # otherwise the recorded append-arrival time would reflect server
                # load, not the client's send pacing.
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
                target=_run, args=(i,), daemon=True, name="preflight-stt"
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
