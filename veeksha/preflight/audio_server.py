"""Fixed-cadence realtime audio WebSocket servers for the audio preflight.

The audio analogues of :class:`~veeksha.preflight.mock_engine.MockStreamingEngine`:

* :class:`MockRealtimeAudioServer` — emits audio deltas on a known absolute
  schedule so the preflight can measure how faithfully the REAL
  ``RealtimeTTSClient`` records per-chunk arrival timing over an actual
  WebSocket transport. Speaks just enough of the OpenAI-realtime contract that
  the client accepts it.
* :class:`MockSTTPreflightServer` — records where the client's paced audio
  actually lands (the ground-truth vantage for ASR send drift) and emits
  transcript deltas on a fixed absolute schedule. The paired checks score the
  client against the stamps the mock actually took (``append_at`` /
  ``emitted_at``), so the mock's own schedule adherence never enters a reported
  number. Speaks the minimal ``vllm_realtime`` dialect ``STTClient`` expects.

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

__all__ = [
    "MockRealtimeAudioServer",
    "MockSTTPreflightServer",
]


class MockRealtimeAudioServer(ShardedLoopServer):
    """Emits ``num_chunks`` audio deltas ``audio_chunk_ms`` apart, from the first
    text delta.

    Wire contract (the subset ``RealtimeTTSClient`` drives):

    * in: ``session.update`` -> out: ``session.updated``
    * in: the FIRST ``conversation.item.create`` (paced text delta) -> out:
      ``response.created``, then ``num_chunks``
      ``response.output_audio.delta`` frames (base64 PCM16, ``chunk_bytes``
      each) on the ABSOLUTE deadlines ``T_conn_phase + first_delta_ms +
      i*audio_chunk_ms``, then ``response.output_audio.done`` and
      ``response.done`` with ``status: completed``.
    * in: ``response.create`` (sent by the client after all text) -> ignored
      once emission is under way; it only triggers emission for the degenerate
      no-text case.

    Audio therefore OVERLAPS the client's paced text input, which is what a
    streaming TTS server does and what veeksha's interactivity metrics assume
    (``audio_before_commit_ratio`` measures precisely the audio delivered
    before input completion).
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
        # Close the WS after response.done (test servers want clients to stop
        # promptly; the preflight leaves closing to the client under test).
        self.close_after_done = close_after_done
        self._phase = PhaseSpreader()
        # Per-connection offsets (ms) of this server's own audio-delta sends,
        # relative to that connection's first send: the mock's own view of the
        # cadence the client is about to score.
        self.emit_offsets_ms: List[List[float]] = []
        self._emit_lock = threading.Lock()
        # Paired-timestamp ground truth (see ServerRecordBook).
        self._records = ServerRecordBook()
        # Pre-serialize repeated messages once (identical for every chunk/conn).
        encoded = base64.b64encode(b"\x00" * chunk_bytes).decode("ascii")
        self._delta_msg = json.dumps(
            {"type": "response.output_audio.delta", "delta": encoded}
        )
        self._response_created_msg = json.dumps({"type": "response.created"})
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
        await ws.send(self._response_created_msg)
        # Spread this connection's schedule so deadlines don't all land at once.
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
                # The absolute send time the client's arrival of this very delta
                # is paired against, taken immediately before the send.
                record.emitted_at.append(sent_at)
                if first is None:
                    first = sent_at
                offsets.append((sent_at - first) * 1000.0)
                try:
                    await ws.send(self._delta_msg)  # pre-serialized
                except Exception:
                    return
            try:
                await ws.send(self._audio_done_msg)
                await ws.send(self._response_done_msg)
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
                text = raw if isinstance(raw, str) else str(raw)
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                etype = event.get("type") if isinstance(event, dict) else None
                if etype == "session.update":
                    await ws.send(self._session_updated_msg)
                    continue
                if etype not in ("conversation.item.create", "response.create"):
                    continue
                # Correlation: the preflight embeds ``PFID:<n>`` in the input
                # text, so it rides in a text delta — normally the first, but
                # keep looking until it is found in case the pacer's
                # segmentation pushed it into a later one.
                if record.request_id is None:
                    record.request_id = parse_pfid(text)
                if audio_task is None:
                    # Emission starts with the FIRST text delta, not with
                    # response.create: a streaming TTS server begins synthesis
                    # eagerly as input arrives, so audio overlaps the client's
                    # paced text sends. veeksha's own metrics assume this —
                    # audio_before_commit_ratio and post_commit_audio_delivery_ms
                    # (evaluator/performance/audio_interactivity.py) measure
                    # exactly the audio delivered before/after input completion,
                    # and would be degenerate if the server waited. Overlap is
                    # also what exercises the client's paced-send and
                    # timestamped-receive paths concurrently on one loop.
                    # response.create is still accepted as a trigger for the
                    # degenerate case of a request with no text segments.
                    audio_task = asyncio.ensure_future(self._emit_audio(ws, record))
        except Exception:
            pass
        finally:
            # The client closes once it has seen response.done, so by here the
            # emit task is normally finished; cancelling covers early
            # disconnects. Its ``finally`` still records whatever it emitted.
            if audio_task is not None:
                audio_task.cancel()


class MockSTTPreflightServer(ShardedLoopServer):
    """STT WebSocket server that RECORDS where the client's paced audio lands.

    For ASR the interactivity-critical drift is on the *send* side: veeksha must
    stream the input audio at 1x real time. This server measures that at the
    ground-truth point — it timestamps each ``input_audio_buffer.append`` the
    moment the frame is received, BEFORE any JSON parsing of the base64-heavy
    payload, so the recorded arrival is not inflated by the server's own decode
    cost — while emitting transcript deltas on a fixed absolute schedule. The
    client is scored against these actual receive/send stamps, so the mock's own
    schedule adherence never enters a reported number and is not tracked.

    Wire contract (``vllm_realtime`` dialect):

    * out on connect: ``session.created``
    * in: ``session.update``, then ``input_audio_buffer.commit`` (non-final,
      the client's handshake), then ``input_audio_buffer.append`` frames
    * in: ``input_audio_buffer.commit`` with ``final: true`` (client EOF) ->
      out: ``transcription.delta`` per word on absolute deadlines
      ``T + first_delta_ms + i*transcript_delta_ms``, then
      ``transcription.done``.

    Preflight correlation rides in a ``veeksha_request_id`` field on the
    ``session.update`` frame (the audio itself cannot carry a marker — the STT
    client's clip decode round-trips through float, which corrupts any exact
    PCM bytes). The id is read after the receive stamp, off the first frame.
    """

    def __init__(
        self,
        transcript: str = "the quick brown fox jumps over the lazy dog",
        first_delta_ms: float = 50.0,
        transcript_delta_ms: float = 30.0,
        host: str = "127.0.0.1",
        num_loops: int = 8,
    ):
        super().__init__(host=host, num_loops=num_loops)
        self.transcript = transcript
        self.first_delta_ms = first_delta_ms
        self.transcript_delta_ms = transcript_delta_ms
        self.first_delta_delay = first_delta_ms / 1000.0
        self.delta_dt = transcript_delta_ms / 1000.0
        # Per-connection append-arrival offsets (ms from that conn's first
        # append) — the server-side ground truth for ASR send pacing.
        self.append_arrivals: List[List[float]] = []
        self._arr_lock = threading.Lock()
        self._phase = PhaseSpreader()
        # Paired-timestamp ground truth (see ServerRecordBook).
        self._records = ServerRecordBook()
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
        return await websockets.serve(self._handler, sock=sock, backlog=ACCEPT_BACKLOG)

    # --------------------------------------------------------------- telemetry
    def append_timelines(self) -> List[List[float]]:
        """Per-connection append-arrival offsets in ms (snapshot copy)."""
        with self._arr_lock:
            return [list(t) for t in self.append_arrivals]

    def reset_telemetry(self) -> None:
        self._phase.reset()
        self._records.clear()
        with self._arr_lock:
            self.append_arrivals = []
        self.reset_connection_peak()

    def records(self) -> Dict[int, ServerRecord]:
        """Per-request stamps, keyed by the session.update ``veeksha_request_id``."""
        return self._records.records()

    def unidentified_connections(self) -> int:
        """Connections served without a usable preflight id (see ServerRecordBook)."""
        return self._records.unidentified_connections()

    # ----------------------------------------------------------------- serving
    async def _handler(self, ws) -> None:
        with self.track_connection():
            await self._handle_conn(ws)

    async def _handle_conn(self, ws) -> None:
        await ws.send(self._created_msg)

        eof_seen = asyncio.Event()
        record = None

        async def _emit_transcript() -> None:
            # Partial hypotheses stream WHILE the client is still pacing audio
            # in — that is what a real streaming ASR server does, and it is the
            # only regime in which word-level interactivity means anything
            # (visibility latency is "when did this word's transcript appear
            # relative to when its audio went out"; if every delta landed after
            # EOF the metric would collapse to a constant). It is also the only
            # regime that exercises the client's paced-send and timestamped-
            # receive paths concurrently on one event loop.
            #
            # Deltas cycle the transcript so the cadence spans the whole clip;
            # the terminal ``done`` carries the full transcript and is sent
            # only after the client's final commit, so the accumulated partials
            # never determine the final text.
            start = (
                time.monotonic()
                + self.first_delta_delay
                + self._phase.next_phase(self.delta_dt)
            )
            i = 0
            while not eof_seen.is_set():
                scheduled = start + i * self.delta_dt
                delay = scheduled - time.monotonic()
                if delay > 0:
                    try:
                        await asyncio.wait_for(eof_seen.wait(), timeout=delay)
                        break  # EOF arrived while waiting for this deadline
                    except asyncio.TimeoutError:
                        pass
                # The absolute send time this delta's client arrival is paired
                # against, taken immediately before the send.
                sent_at = time.monotonic()
                if record is not None:
                    record.emitted_at.append(sent_at)
                try:
                    await ws.send(self._delta_msgs[i % len(self._delta_msgs)])
                except Exception:
                    return
                i += 1
            try:
                await ws.send(self._done_msg)  # pre-serialized, full transcript
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
                if record is None:
                    # First frame on this connection: stamp before parsing.
                    record = self._records.open(now)
                text = raw if isinstance(raw, str) else raw.decode("utf-8", "ignore")
                if '"input_audio_buffer.append"' in text[:64]:
                    record.append_at.append(now)
                    if first_append is None:
                        first_append = now
                        record.first_append_at = now
                        # Partials start with the audio, not after it.
                        if transcript_task is None:
                            transcript_task = asyncio.ensure_future(_emit_transcript())
                    arrivals.append((now - first_append) * 1000.0)
                    continue
                try:
                    event = json.loads(text)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "session.update":
                    # Correlation: the client echoes the preflight id here (the
                    # audio itself cannot carry a marker). Parsed after the
                    # receive stamp.
                    rid = event.get("veeksha_request_id")
                    if isinstance(rid, int):
                        record.request_id = rid
                    continue
                if event.get("type") == "input_audio_buffer.commit" and event.get(
                    "final"
                ):
                    # Client EOF: full audio received. Record the arrival
                    # timeline HERE (deterministic — before the connection
                    # closes) so the measurement never races the close, then
                    # let the streaming partials finish with the terminal
                    # ``done``.
                    _record()
                    eof_seen.set()
                    if transcript_task is None:
                        # No append was ever seen (empty clip): still answer.
                        transcript_task = asyncio.ensure_future(_emit_transcript())
        except Exception:
            pass
        finally:
            if transcript_task is not None:
                transcript_task.cancel()
            _record()  # client that closed without a final commit
