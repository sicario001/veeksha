"""NativeMainLoop: MainLoop adapter over the ``veeksha_native`` C++ main loop.

Wraps ``veeksha_native.NativeBenchmarkLoop`` behind the exact ``MainLoop``
protocol the Python main loop implements (interface.py):

- ``start(source)`` spawns a Python *feeder* thread that pulls sessions from
  the ``SessionSource``, compiles them with :class:`PlanCompiler`, streams
  RATE interval draws, and pushes plans into the native intake ring
  (backpressure comes from the ring, mirroring PrefetchWorker's throttle).
- ``drain_events()`` pulls native ``NativeLoopEvent``s and translates them into
  ``LoopEvent``s: DISPATCHED events re-attach the Python-side request
  context from the compiler sidecar; COMPLETED events are translated into
  full ``RequestResult`` objects with the EXACT per-modality metric keys the
  evaluators consume (see ``_translate_result``). Sidecar entries are
  dropped after COMPLETED.
- Engine-epoch milliseconds are converted to Python ``time.monotonic``
  floats via the construction-time clock handshake
  (``loop_epoch_py_monotonic_s``), so scoring sees the same clock domain
  as the Python main loop.

Known fidelity deviations vs the Python clients (the native result structs
are plain timing/count data; they do not retain response payloads):

- Audio byte payloads are not retained natively: TTS results carry
  ``content=b""``; realtime-TTS ``audio_chunk_timestamps`` sizes are the WS
  event payload byte counts, not decoded-PCM byte counts.
- Realtime-TTS ``text_delta_timestamps`` / ``input_commit_offset_ms`` are
  anchored at the post-handshake pacing anchor rather than request start
  (differs by the WS connect latency); ``ws_connect_latency_ms`` is None.
- STT ``time_to_*`` values are anchored at request start (the native result
  does not export the audio-start <-> request-start bridge; differs by
  connect+handshake latency), audio-end is estimated as the paced clip
  duration, ``transcript_snapshots`` holds only the final transcript row
  (per-delta transcript text is not retained), and the final transcript is
  the accumulated delta text (a ``transcription.done`` full-text payload is
  not consulted).
- Error message strings are the native reasons ("timeout", "http status
  500", ...) rather than the Python clients' exception texts; codes are
  mapped onto the same HTTP-ish numbers.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.loop.interface import (
    LoopCounters,
    LoopEvent,
    LoopEventKind,
    MainLoopConfig,
    SessionSource,
)
from veeksha.loop.plan_compiler import (
    KIND_STT_WS,
    KIND_TEXT_SSE,
    KIND_TTS_HTTP,
    KIND_TTS_REALTIME_WS,
    NativeIneligible,
    PlanCompiler,
    RequestSidecar,
    build_interval_generator,
    to_native_endpoint,
    to_native_runtime,
    to_native_traffic,
)
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

logger = init_logger(__name__)


def _round_ms(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


def _map_error(error: str) -> Tuple[Optional[int], Optional[str]]:
    """Map a native error reason onto (error_code, error_msg).

    Codes follow the Python clients' conventions: HTTP status pass-through,
    408 timeout, 503 unreachable, 520 unclassified.
    """
    if not error:
        return None, None
    if error.startswith("http status "):
        try:
            return int(error.rsplit(" ", 1)[1]), error
        except ValueError:
            return 520, error
    if error == "timeout":
        return 408, error
    if "connect" in error or error.startswith("resolve failed"):
        return 503, error
    return 520, error


class _SessionContentStore:
    """Extracted parent-turn contents for history token accounting.

    Keyed ``session_id -> {node_id: content}``; entries are dropped once all
    of a session's requests have been translated. Touched only by the drain
    caller thread.
    """

    def __init__(self) -> None:
        self._contents: Dict[int, Dict[int, str]] = {}
        self._seen: Dict[int, int] = {}

    def record(
        self, session_id: int, node_id: int, content: str, session_size: int
    ) -> None:
        self._contents.setdefault(session_id, {})[node_id] = content
        self._bump(session_id, session_size)

    def _bump(self, session_id: int, session_size: int) -> None:
        seen = self._seen.get(session_id, 0) + 1
        if seen >= session_size:
            self._contents.pop(session_id, None)
            self._seen.pop(session_id, None)
        else:
            self._seen[session_id] = seen

    def get(self, session_id: int, node_id: int) -> str:
        return self._contents.get(session_id, {}).get(node_id, "")


class NativeMainLoop:
    """MainLoop implementation backed by ``veeksha_native.NativeBenchmarkLoop``."""

    def __init__(
        self,
        config: MainLoopConfig,
        *,
        seed_manager: Any,
        tokenizer_provider: Any = None,
        client: Any = None,
    ):
        """Initialize the loop.

        Args:
            config: Minimal plain-data loop configuration.
            seed_manager: SeedManager (RATE interval chain).
            tokenizer_provider: Tokenizer provider for translation-time
                prompt/output token accounting (text transport).
            client: Accepted for construction parity with PythonMainLoop
                (the benchmark passes its warmup client); only its tokenizer
                handle is consulted as a fallback — the native main loop never
                calls the client.

        Raises:
            NativeIneligible: the client config is not natively serviceable
                (callers — ``create_main_loop`` — catch this and fall back).
        """
        from veeksha.native import get_module

        self._vn = get_module()
        self._config = config
        self._seed_manager = seed_manager

        self._text_tokenizer = None
        if tokenizer_provider is not None:
            try:
                self._text_tokenizer = tokenizer_provider.for_modality(
                    ChannelModality.TEXT
                )
            except KeyError:
                self._text_tokenizer = None
        if self._text_tokenizer is None and client is not None:
            self._text_tokenizer = getattr(client, "text_tokenizer_handle", None)

        self._compiler = PlanCompiler(config.client, seed_manager, vn=self._vn)

        self._loop: Any = None
        self._anchor_s: float = 0.0
        self._feeder: Optional[threading.Thread] = None
        self._feeder_stop = threading.Event()
        self._sessions_skipped = 0

        self._sidecar_lock = threading.Lock()
        self._sidecars: Dict[int, RequestSidecar] = {}
        self._contents = _SessionContentStore()

        self._started = False
        self._joined = False

    # ------------------------------------------------------------------
    # MainLoop protocol
    # ------------------------------------------------------------------

    def start(self, source: SessionSource) -> None:
        """Construct the native loop and start the feeder thread."""
        if self._started:
            raise RuntimeError("NativeMainLoop.start() called twice")
        self._started = True

        vn = self._vn
        self._loop = vn.NativeBenchmarkLoop(
            to_native_runtime(vn, self._config.runtime, self._config.traffic),
            to_native_traffic(vn, self._config.traffic),
            to_native_endpoint(vn, self._config.client),
            time.monotonic(),
        )
        self._anchor_s = self._loop.loop_epoch_py_monotonic_s

        self._feeder = threading.Thread(
            target=self._feed, args=(source,), name="native-feeder", daemon=True
        )
        self._feeder.start()
        logger.info(
            "Started native benchmark loop (reactor backend: %s)",
            getattr(vn, "reactor_backend", "unknown"),
        )

    def drain_events(
        self, max_items: int = 256, timeout_s: float = 0.1
    ) -> List[LoopEvent]:
        """Pull native events and translate them (order-preserving)."""
        assert self._loop is not None, "drain_events before start()"
        events = self._loop.drain_events(max_items=max_items, timeout_s=timeout_s)
        return [self._translate_event(ev) for ev in events]

    def counters(self) -> LoopCounters:
        if self._loop is None:
            return LoopCounters(idle=not self._started)
        c = self._loop.counters()
        return LoopCounters(
            sessions_completed=c.sessions_completed,
            sessions_errored=c.sessions_errored,
            sessions_seen=c.sessions_seen,
            requests_dispatched=c.requests_dispatched,
            requests_completed=c.requests_completed,
            in_flight=c.in_flight,
            intake_exhausted=bool(c.intake_exhausted),
            idle=bool(c.idle),
        )

    def in_flight_request_ids(self) -> Set[int]:
        if self._loop is None:
            return set()
        return {int(rid) for rid in self._loop.in_flight_request_ids()}

    def dispatched_request_ids(self) -> Set[int]:
        if self._loop is None:
            return set()
        return {int(rid) for rid in self._loop.dispatched_request_ids()}

    def request_stop(self, grace_s: float) -> None:
        """Stop dispatching new work.

        The Python loop's ``grace_s < 0`` ("wait for in-flight during join")
        is only issued by the monitor when nothing is pending, so it maps to
        an immediate native stop; ``grace_s >= 0`` means the budget was
        already spent — native fails stragglers now, keeping partials.
        """
        self._feeder_stop.set()
        if self._loop is not None:
            self._loop.request_stop(max(grace_s, 0.0))

    def join(self, timeout_s: float = 10.0) -> bool:
        if not self._started or self._joined:
            return True
        self._joined = True
        ok = True
        if self._feeder is not None:
            self._feeder.join(timeout_s)
            ok = ok and not self._feeder.is_alive()
        if self._loop is not None:
            ok = self._loop.join(timeout_s) and ok
        return ok

    # ------------------------------------------------------------------
    # feeder
    # ------------------------------------------------------------------

    def _feed(self, source: SessionSource) -> None:
        """generate -> compile -> feed_intervals/feed_sessions -> close_intake."""
        from veeksha.config.traffic import RateTrafficConfig

        is_rate = isinstance(self._config.traffic, RateTrafficConfig)
        interval_gen = (
            build_interval_generator(self._config.traffic, self._seed_manager)
            if is_rate
            else None
        )
        # The Python scheduler anchors the arrival schedule at
        # config.monotonic_anchor (benchmark start); the native main loop's
        # epoch is its construction time. Shift the first fed interval by
        # the startup delta so absolute arrival times match across loop implementations
        # (past-due sessions dispatch immediately on both, exactly like the
        # Python scheduler's behavior for schedule times already in the
        # past; native accumulates negative starts correctly).
        anchor_shift_s = self._anchor_s - self._config.monotonic_anchor
        try:
            while not self._feeder_stop.is_set():
                session = source.next_session()
                if session is None:
                    break
                # One seeded draw per generated session, fed regardless of
                # compile outcome so a skipped session cannot shift the
                # arrival offsets of the sessions after it.
                if interval_gen is not None:
                    gap = interval_gen.get_next_interval() - anchor_shift_s
                    anchor_shift_s = 0.0
                    self._loop.feed_intervals([gap])
                try:
                    plan, sidecars = self._compiler.compile(
                        session, self._loop.register_blob
                    )
                except NativeIneligible as exc:
                    self._sessions_skipped += 1
                    logger.warning(
                        "Native main loop skipped session %s (not natively "
                        "serviceable): %s",
                        session.id,
                        exc.reason,
                    )
                    continue
                with self._sidecar_lock:
                    self._sidecars.update(sidecars)
                self._loop.feed_sessions([plan])
        except Exception:
            logger.exception("Native feeder thread failed; closing intake")
        finally:
            self._loop.close_intake()

    # ------------------------------------------------------------------
    # event translation
    # ------------------------------------------------------------------

    def _to_monotonic(self, ms: float) -> float:
        return self._anchor_s + ms / 1000.0

    def _translate_event(self, ev: Any) -> LoopEvent:
        vn = self._vn
        request_id = int(ev.request_id)
        if ev.kind == vn.EventKind.DISPATCHED:
            with self._sidecar_lock:
                sidecar = self._sidecars.get(request_id)
            return LoopEvent(
                kind=LoopEventKind.DISPATCHED,
                request_id=request_id,
                session_id=int(ev.session_id),
                session_total_requests=int(ev.session_total_requests),
                ready_at=self._to_monotonic(ev.ready_ms),
                dispatched_at=self._to_monotonic(ev.dispatched_ms),
                request=sidecar.request if sidecar else None,
            )

        with self._sidecar_lock:
            sidecar = self._sidecars.pop(request_id, None)
        result = self._translate_result(ev.result, sidecar)
        return LoopEvent(
            kind=LoopEventKind.COMPLETED,
            request_id=request_id,
            session_id=int(ev.session_id),
            session_total_requests=int(ev.session_total_requests),
            result=result,
        )

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._text_tokenizer is None:
            return len(text.split())
        return len(self._text_tokenizer.encode(text))

    def _translate_result(
        self, res: Any, sidecar: Optional[RequestSidecar]
    ) -> RequestResult:
        """RawRequestResult -> RequestResult with per-modality metric keys."""
        error_code, error_msg = _map_error(res.error)
        success = error_msg is None

        channels: Dict[ChannelModality, ChannelResponse] = {}
        session_id = int(res.session_id)
        session_size = int(res.session_total_requests)

        if sidecar is None:
            logger.warning(
                "Native COMPLETED event for unknown request %s", res.request_id
            )
        elif sidecar.kind == KIND_TEXT_SSE:
            channels = self._text_channels(res, sidecar, success)
        elif sidecar.kind == KIND_TTS_HTTP:
            channels = self._tts_http_channels(res, sidecar, success)
        elif sidecar.kind == KIND_TTS_REALTIME_WS:
            channels = self._tts_realtime_channels(res, sidecar, success)
        elif sidecar.kind == KIND_STT_WS:
            channels = self._stt_channels(res, sidecar, success)

        return RequestResult(
            request_id=int(res.request_id),
            session_id=session_id,
            session_total_requests=session_size,
            channels=channels,
            success=success,
            error_code=error_code,
            error_msg=error_msg,
            scheduler_ready_at=self._to_monotonic(res.scheduler_ready_ms),
            scheduler_dispatched_at=self._to_monotonic(res.scheduler_dispatched_ms),
            client_picked_up_at=self._to_monotonic(res.client_picked_up_ms),
            client_completed_at=self._to_monotonic(res.client_completed_ms),
            result_processed_at=self._to_monotonic(res.result_processed_ms),
        )

    # ---- TEXT_SSE ------------------------------------------------------

    def _text_channels(
        self, res: Any, sidecar: RequestSidecar, success: bool
    ) -> Dict[ChannelModality, ChannelResponse]:
        content: str = res.content
        session_id = int(res.session_id)
        session_size = int(res.session_total_requests)

        # Token accounting exactly like openai_chat.send_request: total =
        # per-message token counts summed over the body's messages (spliced
        # assistant turns included), delta = this turn's user text, output =
        # tokenizer count of the extracted content. Computed HERE, at
        # translation time — never on the native hot path. (Read the content
        # store BEFORE recording this turn: recording the session's last
        # turn releases the store entry.)
        num_total_prompt_tokens = 0
        for part_kind, payload in sidecar.message_parts:
            if part_kind == "text":
                num_total_prompt_tokens += self._count_tokens(payload)
            else:
                num_total_prompt_tokens += self._count_tokens(
                    self._contents.get(session_id, payload)
                )
        num_delta_prompt_tokens = self._count_tokens(sidecar.delta_text)
        num_output_tokens = self._count_tokens(content)

        # Record the extracted content for children's history token
        # accounting (parents' COMPLETED events precede children's in the
        # ordered drain).
        self._contents.record(
            session_id, sidecar.node_id, content if success else "", session_size
        )

        if not (success and content):
            return {}

        offsets = [s.offset_ms for s in res.recv_stamps]
        inter_chunk_times: List[float] = []
        if offsets:
            inter_chunk_times.append(offsets[0] / 1000.0)
            inter_chunk_times.extend(
                (b - a) / 1000.0 for a, b in zip(offsets, offsets[1:])
            )

        return {
            ChannelModality.TEXT: ChannelResponse(
                modality=ChannelModality.TEXT,
                content=content,
                metrics={
                    "is_stream": True,
                    "inter_chunk_times": inter_chunk_times,
                    "num_delta_prompt_tokens": num_delta_prompt_tokens,
                    "num_total_prompt_tokens": num_total_prompt_tokens,
                    "num_output_tokens": num_output_tokens,
                },
            )
        }

    # ---- TTS_HTTP ------------------------------------------------------

    def _tts_http_channels(
        self, res: Any, sidecar: RequestSidecar, success: bool
    ) -> Dict[ChannelModality, ChannelResponse]:
        from veeksha.core.audio_contract import AudioMetricKey

        if not success:
            return {}
        offsets = [s.offset_ms for s in res.recv_stamps]
        ttfc = offsets[0] if offsets else 0.0
        end_to_end_ms = res.client_completed_ms - res.client_picked_up_ms
        return {
            ChannelModality.AUDIO: ChannelResponse(
                modality=ChannelModality.AUDIO,
                # Native counts audio bytes but does not retain them.
                content=b"",
                metrics={
                    "audio_task": AudioTask.TTS,
                    AudioMetricKey.TTFC.value: round(ttfc or 0.0, 3),
                    AudioMetricKey.END_TO_END_LATENCY.value: round(end_to_end_ms, 3),
                    AudioMetricKey.CHUNK_COUNT.value: len(offsets),
                    AudioMetricKey.RAW_PCM.value: self._config.client.raw_pcm,
                    AudioMetricKey.SAMPLE_RATE.value: self._config.client.sample_rate,
                    AudioMetricKey.INPUT_CHARS.value: len(sidecar.input_text),
                    AudioMetricKey.INPUT_TOKENS.value: sidecar.input_tokens,
                    AudioMetricKey.INPUT_TEXT.value: sidecar.input_text,
                },
            )
        }

    # ---- TTS_REALTIME_WS ------------------------------------------------

    def _tts_realtime_channels(
        self, res: Any, sidecar: RequestSidecar, success: bool
    ) -> Dict[ChannelModality, ChannelResponse]:
        from veeksha.core.audio_contract import AudioMetricKey

        audio_chunk_ts = [[s.offset_ms, s.size] for s in res.recv_stamps]
        text_delta_ts = [
            [offset, n_chars]
            for offset, n_chars in zip(res.send_offsets_ms, sidecar.segment_chars)
        ]

        first_event_offset: Dict[str, float] = {}
        for event_type, offset in res.event_offsets_ms:
            first_event_offset.setdefault(event_type, offset)

        ttfc = audio_chunk_ts[0][0] if audio_chunk_ts else None
        end_to_end_ms = res.client_completed_ms - res.client_picked_up_ms
        send_offsets = list(res.send_offsets_ms)

        metrics = {
            "audio_task": AudioTask.TTS,
            AudioMetricKey.TTFC.value: round(ttfc or 0.0, 3),
            AudioMetricKey.END_TO_END_LATENCY.value: round(end_to_end_ms, 3),
            AudioMetricKey.CHUNK_COUNT.value: len(audio_chunk_ts),
            AudioMetricKey.RAW_PCM.value: True,
            AudioMetricKey.SAMPLE_RATE.value: self._config.client.sample_rate,
            AudioMetricKey.INPUT_CHARS.value: len(sidecar.input_text),
            AudioMetricKey.INPUT_TOKENS.value: sidecar.input_tokens,
            AudioMetricKey.INPUT_TEXT.value: sidecar.input_text,
            AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value: text_delta_ts,
            AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: audio_chunk_ts,
            # Not exported by the native result (see module deviations note).
            AudioMetricKey.WS_CONNECT_LATENCY_MS.value: None,
            AudioMetricKey.SESSION_READY_OFFSET_MS.value: _round_ms(
                first_event_offset.get("session.updated")
            ),
            AudioMetricKey.RESPONSE_CREATED_OFFSET_MS.value: _round_ms(
                first_event_offset.get("response.created")
            ),
            AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: _round_ms(
                send_offsets[-1] if send_offsets else None
            ),
            AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: _round_ms(
                first_event_offset.get("response.output_audio.done")
            ),
            AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: _round_ms(
                first_event_offset.get("response.done")
            ),
        }

        has_partial = bool(audio_chunk_ts or text_delta_ts)
        if not (success or has_partial):
            return {}
        return {
            ChannelModality.AUDIO: ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=b"",  # audio payloads are not retained natively
                metrics=metrics,
            )
        }

    # ---- STT_WS ---------------------------------------------------------

    def _stt_channels(
        self, res: Any, sidecar: RequestSidecar, success: bool
    ) -> Dict[ChannelModality, ChannelResponse]:
        from veeksha.client.stt import _clean_transcript

        if not success:
            return {}

        final_transcript = _clean_transcript(res.content)
        delta_offsets = [s.offset_ms for s in res.recv_stamps]
        chunk_count = len(delta_offsets)
        if final_transcript and chunk_count == 0:
            chunk_count = 1

        first_event_offset: Dict[str, float] = {}
        for event_type, offset in res.event_offsets_ms:
            first_event_offset.setdefault(event_type, offset)
        done_offset = first_event_offset.get("transcription.done")

        # Anchors (see module deviations note): time-to-* use request start
        # as the audio-start anchor; end-of-audio is estimated as the paced
        # clip duration from that anchor.
        ttfc = delta_offsets[0] if delta_offsets else None
        if ttfc is None and final_transcript and done_offset is not None:
            ttfc = done_offset
        time_to_first_visible_text = ttfc
        audio_end_est = sidecar.input_audio_duration_ms

        time_to_first_partial = None
        for offset in delta_offsets:
            if offset >= audio_end_est:
                time_to_first_partial = offset - audio_end_est
                break
        time_to_final_transcript = (
            done_offset - audio_end_est if done_offset is not None else None
        )
        partial_transcript = (
            final_transcript if time_to_first_partial is not None else None
        )

        snapshots = []
        if final_transcript:
            snapshot_at = done_offset if done_offset is not None else (ttfc or 0.0)
            snapshots.append(
                {"elapsed_ms": round(snapshot_at, 3), "transcript": final_transcript}
            )

        end_to_end_ms = res.client_completed_ms - res.client_picked_up_ms
        metrics: Dict[str, Any] = {
            "audio_task": AudioTask.STT,
            "ttfc": round(ttfc or 0.0, 3),
            "end_to_end_latency": round(end_to_end_ms, 3),
            "time_to_first_visible_text": _round_ms(time_to_first_visible_text),
            "time_to_first_partial": _round_ms(time_to_first_partial),
            "time_to_final_transcript": _round_ms(time_to_final_transcript),
            "chunk_count": chunk_count,
            "pcm_byte_count": sidecar.pcm_byte_count,
            "raw_pcm": True,
            "input_tokens": len(final_transcript.split()),
            "sample_rate": self._config.client.sample_rate,
            "input_audio_duration_ms": round(sidecar.input_audio_duration_ms, 3),
            "partial_transcript": partial_transcript,
            "final_transcript": final_transcript,
            "transcript_snapshots": snapshots,
            # The native loop's own paced-send stamps (anchored at the first
            # audio frame on the wire, stt.py's audio_started_at).
            "send_offsets_ms": [round(v, 3) for v in res.send_offsets_ms],
        }
        # Ground truth and dataset metadata flow from the request, exactly
        # like stt.py.
        for key, value in sidecar.metadata.items():
            metrics.setdefault(key, value)

        return {
            ChannelModality.AUDIO: ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=final_transcript,
                metrics=metrics,
            )
        }
