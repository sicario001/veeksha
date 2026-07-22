"""Fixed-cadence mock for Vajra's native streaming-text TTS protocol.

The third audio server in the preflight family (alongside
:class:`~veeksha.preflight.audio_server.MockRealtimeAudioServer` and
:class:`~veeksha.preflight.audio_server.MockSTTPreflightServer`), speaking just
enough of the ``/v1/audio/speech/stream`` contract that the REAL
:class:`~veeksha.client.vajra_tts_stream.VajraTTSStreamClient` accepts it.

Two things differ from the OpenAI-realtime mock, and both matter for what the
preflight measures:

* **Binary audio frames.** Vajra sends raw int16 PCM as WebSocket binary
  frames rather than base64 inside JSON. The client stamps arrival before it
  touches the payload either way, so receive-drift scoring is identical — but
  the transport path being measured is genuinely different (no base64 decode
  on the client's receive path).
* **Audio starts while text is still being sent.** Emission is triggered by
  the FIRST ``input.text`` delta, not by the terminal ``input.done``. That is
  what a real streaming-text TTS server does, and it deliberately exercises
  the harder client path: paced sends and timestamped receives interleaved on
  one event loop. The emit cadence stays absolute-deadline deterministic, so
  "when we are supposed to receive a packet" is as well-defined as in the
  serialized case.

Like its siblings the client is scored against the stamp the mock took when it
ACTUALLY sent each frame (``emitted_at``), so the mock's own adherence to those
absolute deadlines never enters a reported number and is not tracked.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Dict, List, Optional

import websockets

from veeksha.preflight.sharded_server import (
    ACCEPT_BACKLOG,
    PhaseSpreader,
    ServerRecord,
    ServerRecordBook,
    ShardedLoopServer,
    parse_pfid,
)

__all__ = ["MockVajraTTSStreamServer"]


class MockVajraTTSStreamServer(ShardedLoopServer):
    """Emits ``num_chunks`` binary PCM frames ``audio_chunk_ms`` apart.

    Wire contract (the subset ``VajraTTSStreamClient`` drives):

    * in: ``session.config`` -> (no reply; the client does not wait for one)
    * in: ``input.text`` (paced text deltas) -> on the FIRST one: out
      ``audio.start``, then ``num_chunks`` binary PCM frames of
      ``chunk_bytes`` each on the ABSOLUTE deadlines
      ``T_conn_phase + first_delta_ms + i*audio_chunk_ms``
    * in: ``input.done`` -> ignored (emission is already under way)
    * out (terminal): ``audio.done`` then ``session.done`` -- the client stops
      on ``session.done``, so it must always be sent last.
    """

    def __init__(
        self,
        num_chunks: int = 20,
        chunk_bytes: int = 4800,
        first_delta_ms: float = 20.0,
        audio_chunk_ms: float = 20.0,
        sample_rate: int = 24000,
        host: str = "127.0.0.1",
        num_loops: int = 8,
        close_after_done: bool = False,
    ):
        super().__init__(host=host, num_loops=num_loops)
        self.num_chunks = num_chunks
        self.chunk_bytes = chunk_bytes
        self.first_delta_ms = first_delta_ms
        self.audio_chunk_ms = audio_chunk_ms
        self.first_delta_delay = first_delta_ms / 1000.0
        self.chunk_dt = audio_chunk_ms / 1000.0
        self.sample_rate = sample_rate
        self.close_after_done = close_after_done
        self._phase = PhaseSpreader()
        # Per-connection offsets (ms) of this server's own audio sends,
        # relative to that connection's first send: the mock's own view of the
        # cadence the client is about to score.
        self.emit_offsets_ms: List[List[float]] = []
        self._emit_lock = threading.Lock()
        # Paired-timestamp ground truth (see ServerRecordBook).
        self._records = ServerRecordBook()
        # Pre-build the repeated payloads once (identical for every chunk and
        # connection) so serialization cost never lands inside a deadline.
        self._pcm_frame = b"\x00" * chunk_bytes
        self._audio_start_msg = json.dumps(
            {"type": "audio.start", "sample_rate": sample_rate}
        )
        self._audio_done_msg = json.dumps({"type": "audio.done"})
        self._session_done_msg = json.dumps({"type": "session.done"})

    async def _start_server(self, sock: socket.socket):
        return await websockets.serve(self._handler, sock=sock, backlog=ACCEPT_BACKLOG)

    # --------------------------------------------------------------- telemetry
    def reset_telemetry(self) -> None:
        self._phase.reset()
        self._records.clear()
        with self._emit_lock:
            self.emit_offsets_ms = []
        self.reset_connection_peak()

    def records(self) -> Dict[int, ServerRecord]:
        """Per-request receive/emit stamps, keyed by the request's PFID."""
        return self._records.records()

    def unidentified_connections(self) -> int:
        """Connections served without a usable preflight id (see ServerRecordBook)."""
        return self._records.unidentified_connections()

    # ----------------------------------------------------------------- serving
    async def _emit_audio(self, ws, record) -> None:
        await ws.send(self._audio_start_msg)
        # Spread this connection's schedule so deadlines don't all land at once
        # (identical schedules put every connection's deadline in the same
        # instant; one loop then walks the burst and the client records the
        # walk as jitter).
        start = time.monotonic() + self._phase.next_phase(self.chunk_dt)
        offsets: List[float] = []
        first: Optional[float] = None
        try:
            for i in range(self.num_chunks):
                scheduled = start + self.first_delta_delay + i * self.chunk_dt
                now = time.monotonic()
                if scheduled > now:
                    await asyncio.sleep(scheduled - now)
                sent_at = time.monotonic()
                # The absolute send time this frame's client arrival is paired
                # against, taken immediately before the send.
                record.emitted_at.append(sent_at)
                if first is None:
                    first = sent_at
                offsets.append((sent_at - first) * 1000.0)
                try:
                    await ws.send(self._pcm_frame)  # binary frame, pre-built
                except Exception:
                    return
            try:
                await ws.send(self._audio_done_msg)
                # Terminal for the client: must be last.
                await ws.send(self._session_done_msg)
                if self.close_after_done:
                    await ws.close()
            except Exception:
                pass
        finally:
            if offsets:
                with self._emit_lock:
                    self.emit_offsets_ms.append(offsets)

    async def _handler(self, ws) -> None:
        with self.track_connection():
            await self._handle_conn(ws)

    async def _handle_conn(self, ws) -> None:
        audio_task: Optional[asyncio.Future] = None
        record = None
        try:
            async for raw in ws:
                if record is None:
                    # Ground truth first: stamp the connection's first frame
                    # before it is parsed. One connection carries one request.
                    record = self._records.open(time.monotonic())
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    continue  # the client never sends binary
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                etype = event.get("type") if isinstance(event, dict) else None
                if etype != "input.text":
                    continue
                # Correlation: the preflight embeds ``PFID:<n>`` in the input
                # text, so it rides in a text delta — normally the first, but
                # keep looking until it is found in case the pacer's
                # segmentation pushed it into a later one.
                if record.request_id is None:
                    record.request_id = parse_pfid(str(raw))
                # Emission starts with the first text delta (see module
                # docstring): real streaming-text TTS overlaps audio with
                # input, and that interleaving is the client path worth gating.
                if audio_task is None:
                    audio_task = asyncio.ensure_future(self._emit_audio(ws, record))
        except Exception:
            pass
        finally:
            # The client closes once it has seen session.done, so the emit task
            # is normally finished by here; cancelling covers early
            # disconnects, and its ``finally`` still records what it emitted.
            if audio_task is not None:
                audio_task.cancel()
