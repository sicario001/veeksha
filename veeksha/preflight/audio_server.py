"""Fixed-cadence realtime audio WebSocket servers for the audio preflight.

The audio analogues of ``MockStreamingEngine``:

* ``MockRealtimeAudioServer`` — emits audio deltas on a known schedule so the
  preflight can measure how faithfully the REAL realtime client records
  per-chunk arrival timing over an actual WebSocket transport. Speaks just
  enough of the OpenAI-realtime contract that ``RealtimeTTSClient`` accepts it.
* ``MockSTTPreflightServer`` — records where the client's paced audio actually
  lands (the ground-truth vantage for ASR send drift) and emits transcript
  deltas on a fixed schedule, with its own emit-lateness telemetry so a
  server-limited rung is never misattributed to the client.

Both shard accept loops on one shared listening socket (``sharded_server``) so
the servers themselves stay punctual at high concurrency on Linux and Darwin.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import threading
import time
from typing import List, Optional

import websockets

from veeksha.preflight.sharded_server import (
    PhaseSpreader,
    ShardedLoopServer,
    ShardedTelemetry,
)


class MockRealtimeAudioServer(ShardedLoopServer):
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
        super().__init__(host=host, num_loops=num_loops)
        self.num_chunks = num_chunks
        self.chunk_bytes = chunk_bytes
        self.first_delta_delay = first_delta_delay
        self.chunk_dt = chunk_dt
        self.sample_rate = sample_rate
        self._lateness = ShardedTelemetry()
        self._phase = PhaseSpreader()
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

    async def _start_server(self, sock: socket.socket):
        return await websockets.serve(self._handler, sock=sock)

    def server_jitter_p99_ms(self) -> float:
        return self._lateness.p99()

    def reset_telemetry(self) -> None:
        self._lateness.clear()
        self.reset_connection_peak()

    async def _emit_audio(self, ws) -> None:
        await ws.send(json.dumps({"type": "response.created"}))
        # spread this connection's schedule so deadlines don't all land at once
        start = time.monotonic() + self._phase.next_phase(self.chunk_dt)
        for i in range(self.num_chunks):
            scheduled = start + self.first_delta_delay + i * self.chunk_dt
            now = time.monotonic()
            if scheduled > now:
                await asyncio.sleep(scheduled - now)
            # lock-free per-thread telemetry; lateness computed before recording
            self._lateness.record((time.monotonic() - scheduled) * 1000.0)
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
        with self.track_connection():
            try:
                async for raw in ws:
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError, TypeError, ValueError:
                        continue
                    etype = event.get("type") if isinstance(event, dict) else None
                    if etype == "session.update":
                        await ws.send(self._session_updated_msg)
                    elif etype == "response.create" and audio_task is None:
                        audio_task = asyncio.ensure_future(self._emit_audio(ws))
            except Exception:
                pass
            finally:
                if audio_task is not None:
                    audio_task.cancel()


class MockSTTPreflightServer(ShardedLoopServer):
    """STT WebSocket server that RECORDS where the client's audio actually lands.

    For ASR the interactivity-critical drift is on the *send* side: veeksha must
    stream the input audio at 1x real time. This server measures that at the
    ground-truth point — it timestamps each ``input_audio_buffer.append`` the
    moment the frame is received, BEFORE any JSON parsing of the base64-heavy
    payload, so the recorded arrival is not inflated by the server's own decode
    cost — while emitting transcript deltas on a fixed schedule (with emit
    lateness recorded, so the validator can tell a saturated server apart from a
    drifting client). Speaks the minimal vllm_realtime contract STTClient expects.
    """

    def __init__(
        self,
        transcript: str = "the quick brown fox jumps over the lazy dog",
        first_delta_delay: float = 0.05,
        delta_dt: float = 0.03,
        host: str = "127.0.0.1",
        num_loops: int = 8,
    ):
        super().__init__(host=host, num_loops=num_loops)
        self.transcript = transcript
        self.first_delta_delay = first_delta_delay
        self.delta_dt = delta_dt
        # per-connection append-arrival offsets (ms from that conn's first append)
        self.append_arrivals: List[List[float]] = []
        self._arr_lock = threading.Lock()
        self._lateness = ShardedTelemetry()
        self._phase = PhaseSpreader()
        # Pre-serialize the (fixed) transcript messages once so the emit loop is
        # cheap and doesn't steal cycles from receiving/timestamping appends.
        words = transcript.split()
        self._created_msg = json.dumps({"type": "session.created"})
        self._delta_msgs = [
            json.dumps({"type": "transcription.delta", "delta": (" " + w if i else w)})
            for i, w in enumerate(words)
        ]
        self._done_msg = json.dumps({"type": "transcription.done", "text": transcript})

    async def _start_server(self, sock: socket.socket):
        return await websockets.serve(self._handler, sock=sock)

    def server_jitter_p99_ms(self) -> float:
        """p99 lateness of the server's own scheduled transcript emits (ms).

        The control the send-drift check gates on: if the server can't even
        keep its own emit schedule, its append-arrival timestamps are equally
        late and the rung is server-limited, not client-dishonest.
        """
        return self._lateness.p99()

    def reset_telemetry(self) -> None:
        self._lateness.clear()
        self.append_arrivals = []
        self.reset_connection_peak()

    async def _handler(self, ws) -> None:
        with self.track_connection():
            await self._handle_conn(ws)

    async def _handle_conn(self, ws) -> None:
        await ws.send(self._created_msg)

        async def _emit_transcript() -> None:
            # Only after the client's EOF, per the real-server contract ("done"
            # comes after all audio) — so the full paced send is measured, not
            # truncated when the transcript finishes early. Emits on ABSOLUTE
            # deadlines (no accumulating relative-sleep drift), like every other
            # mock server, so the recorded lateness is true scheduler jitter.
            start = (
                time.monotonic()
                + self.first_delta_delay
                + self._phase.next_phase(self.delta_dt)
            )
            for i, delta_msg in enumerate(self._delta_msgs):  # pre-serialized
                scheduled = start + i * self.delta_dt
                delay = scheduled - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                self._lateness.record((time.monotonic() - scheduled) * 1000.0)
                try:
                    await ws.send(delta_msg)
                except Exception:
                    return
            try:
                await ws.send(self._done_msg)  # pre-serialized
                await ws.close()
            except Exception:
                pass

        arrivals: List[float] = []
        first_append: Optional[float] = None
        transcript_task: Optional[asyncio.Future] = None
        recorded = False

        def _record() -> None:
            nonlocal recorded
            if not recorded and arrivals:
                with self._arr_lock:
                    self.append_arrivals.append(list(arrivals))
                recorded = True

        try:
            async for raw in ws:  # drain client audio; timestamp each append
                # Ground truth first: stamp arrival BEFORE any parsing. Appends
                # are recognized by a cheap substring probe on the (str) frame —
                # json.loads of a base64 payload would land ~ms later under load
                # and be misattributed to the client's send pacing.
                now = time.monotonic()
                text = raw if isinstance(raw, str) else raw.decode("utf-8", "ignore")
                if '"input_audio_buffer.append"' in text[:64]:
                    if first_append is None:
                        first_append = now
                    arrivals.append((now - first_append) * 1000.0)
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError, TypeError, ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "input_audio_buffer.commit" and event.get(
                    "final"
                ):
                    # client EOF: full audio received. Record the arrival timeline
                    # HERE (deterministic — before the connection closes) so the
                    # measurement never races the close, then emit the transcript.
                    _record()
                    if transcript_task is None:
                        transcript_task = asyncio.ensure_future(_emit_transcript())
        except Exception:
            pass
        finally:
            if transcript_task is not None:
                transcript_task.cancel()
            _record()  # client that closed without a final commit
