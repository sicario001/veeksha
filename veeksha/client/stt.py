"""STT clients for realtime streaming speech-to-text (vajra_openai_realtime, vllm_realtime).

Both providers stream PCM16 over a WebSocket and report transcription metrics
(ttfc, end-to-end latency, RTF; see ``STTStreamResult`` for the timing metric
definitions). They share one lifecycle in ``_STTClientBase``:
audio is paced at 1x playback when ``ws_realtime_pacing`` is on, and the send
and receive loops run concurrently so ``ttfc`` reflects when the first partial
actually arrives rather than when the upload finishes. Each provider only
supplies four small protocol hooks (open session, encode a chunk, EOF sentinel,
parse a message).

``STTClient(config)`` is a factory returning the client for ``config.provider``
(see ``_PROVIDERS``), so the registry needs only one entry.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import threading
import time
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import librosa
import numpy as np

from veeksha.client.base import BaseLLMClient
from veeksha.core.request import Request
from veeksha.core.request_content import AudioChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import STTClientConfig

logger = init_logger(__name__)

BYTES_PER_SAMPLE = 2
TranscriptSnapshotRow = dict[str, float | str]

# Voxtral streaming tokens injected by the model that should be stripped.
_STREAMING_TOKEN_RE = re.compile(r"\[STREAMING_(?:PAD|WORD)\]")


def _pcm_duration_ms(pcm_bytes: int, sample_rate: int) -> float:
    """Compute PCM16 mono audio duration in ms."""
    return (pcm_bytes / BYTES_PER_SAMPLE / sample_rate) * 1000


def _clean_transcript(text: str) -> str:
    """Strip Voxtral streaming control tokens and collapse whitespace."""
    text = _STREAMING_TOKEN_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _audio_to_pcm16_bytes(audio_path: str, target_sr: int) -> bytes:
    """Load an audio file and convert to raw PCM16 bytes at target sample rate."""
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


def _metadata_ms(metadata: dict, key: str) -> Optional[float]:
    """Read an optional millisecond offset from request metadata."""
    value = metadata.get(key)
    if value is None:
        return None
    return float(value)


def _slice_pcm16_bytes(
    pcm_bytes: bytes | memoryview,
    sample_rate: int,
    *,
    start_ms: Optional[float],
    end_ms: Optional[float],
) -> bytes | memoryview:
    """Slice raw PCM16 mono bytes by millisecond offsets.

    Accepts a ``memoryview`` for zero-copy slicing of cached clip PCM (the
    slice then aliases the cached buffer instead of duplicating ~minutes of
    audio per session).
    """
    if start_ms is None and end_ms is None:
        return pcm_bytes

    total_samples = len(pcm_bytes) // BYTES_PER_SAMPLE
    start_sample = 0 if start_ms is None else int(round(start_ms * sample_rate / 1000))
    end_sample = (
        total_samples if end_ms is None else int(round(end_ms * sample_rate / 1000))
    )

    if start_sample < 0:
        raise ValueError(f"input_audio_start_ms must be non-negative; got {start_ms}")
    if end_sample <= start_sample:
        raise ValueError(
            "input_audio_end_ms must be greater than input_audio_start_ms; "
            f"got start_ms={start_ms}, end_ms={end_ms}"
        )
    if end_sample > total_samples:
        raise ValueError(
            "Requested audio slice exceeds decoded clip length: "
            f"end_ms={end_ms}, clip_ms={_pcm_duration_ms(len(pcm_bytes), sample_rate):.3f}"
        )

    start_byte = start_sample * BYTES_PER_SAMPLE
    end_byte = end_sample * BYTES_PER_SAMPLE
    return pcm_bytes[start_byte:end_byte]


# Decoded clips (and their pre-encoded wire messages) are cached per client;
# traces replay a finite clip set, so this is bounded in practice. The FIFO
# cap guards pathological manifests with thousands of distinct clips.
_CLIP_CACHE_MAX_CLIPS = 64


@dataclass(frozen=True)
class _ClipAssets:
    """Deterministic per-clip artifacts shared by every session replaying it.

    ``wire_messages[i]`` is the provider append-message for the full chunk
    ``pcm[i * chunk_size : (i + 1) * chunk_size]``; the trailing partial chunk
    (from the clip end or a per-session slice) is encoded on the fly.
    """

    pcm: bytes
    wire_messages: list[str]


@dataclass
class STTStreamResult:
    """Provider-neutral transcript and timing output from one streaming request.

    All timings are in milliseconds, measured with ``time.monotonic`` deltas:

    - ``ttfc`` (time to first content): from the first audio byte on the wire
      to the first delta whose own payload still contains text after
      control-token cleaning. Empty progress/keepalive deltas and pure
      control-token pads do not count. Falls back to the completion message
      when the stream produces a final transcript without any content-bearing
      delta.
    - ``time_to_first_visible_text``: from the first audio byte on the wire to
      the first moment the assembled (concatenated then cleaned) transcript is
      non-empty, i.e. when a user watching the live transcript would first see
      text. Matches ``ttfc`` for well-formed streams; the two differ when
      cleaning a delta in isolation disagrees with cleaning the assembled
      transcript (e.g. a control token split across deltas cleans non-empty on
      its own but vanishes once joined). Same completion fallback as ``ttfc``.
    - ``time_to_first_partial``: from end-of-audio (EOF sentinel sent) to the
      first delta after EOF with a non-empty assembled transcript; ``None``
      when no such delta arrives before completion.
    - ``time_to_final_transcript``: from end-of-audio to the completion
      message.
    """

    ttfc: Optional[float]
    time_to_first_visible_text: Optional[float]
    time_to_first_partial: Optional[float]
    time_to_final_transcript: Optional[float]
    partial_transcript: Optional[str]
    final_transcript: str
    transcript_snapshots: list[TranscriptSnapshotRow]
    chunk_count: int
    pcm_byte_count: int


class TranscriptSnapshotRecorder:
    """Records evolving transcripts relative to first sent audio byte."""

    def __init__(self) -> None:
        self._audio_started_at: Optional[float] = None
        self._snapshots: list[TranscriptSnapshotRow] = []

    @property
    def snapshots(self) -> list[TranscriptSnapshotRow]:
        return self._snapshots

    def mark_audio_started(self, now: float) -> None:
        if self._audio_started_at is None:
            self._audio_started_at = now

    def add(self, now: float, transcript: str) -> None:
        if not transcript:
            return
        if self._snapshots and self._snapshots[-1]["transcript"] == transcript:
            return
        self._snapshots.append(
            {
                "elapsed_ms": round(self._elapsed_ms(now), 3),
                "transcript": transcript,
            }
        )

    def _elapsed_ms(self, now: float) -> float:
        assert (
            self._audio_started_at is not None
        ), "snapshot recorded before any audio was sent"
        return (now - self._audio_started_at) * 1000


class _STTClientBase(BaseLLMClient):
    """Shared streaming lifecycle for realtime STT providers.

    Subclasses set ``ws_path`` and implement the four protocol hooks; the
    request lifecycle, 1x pacing, concurrent stream, error mapping, and metrics
    assembly live here.
    """

    #: WebSocket path appended to ``api_base`` (set by subclasses).
    ws_path: str = ""

    def __init__(self, config: "STTClientConfig", **kwargs) -> None:
        super().__init__(config)
        self._provider = config.provider
        self._model = config.model
        self._sample_rate = config.sample_rate
        self._ws_chunk_size = config.ws_chunk_size
        self._pacing = config.ws_realtime_pacing
        self._request_timeout = config.request_timeout
        self._ws_ping_interval_s = config.ws_ping_interval_s
        self._ws_ping_timeout_s = config.ws_ping_timeout_s
        self._ws_compression = "deflate" if config.ws_permessage_deflate else None
        self._ws_url = self._http_to_ws(self.ws_path)

        # Per-clip decode + wire-message cache, shared across the worker
        # threads that all hold this one client instance.
        self._clip_cache: OrderedDict[str, _ClipAssets] = OrderedDict()
        self._clip_cache_lock = threading.Lock()
        self._clip_locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------
    # Provider protocol hooks
    # ------------------------------------------------------------------

    @abstractmethod
    async def _open_session(self, ws) -> None:
        """Complete the provider handshake before audio is sent."""

    @abstractmethod
    def _encode_chunk(self, chunk: bytes | memoryview) -> str | bytes:
        """Frame a PCM16 chunk into the message the provider expects."""
        raise NotImplementedError

    @abstractmethod
    def _eof(self) -> str | bytes:
        """Return the end-of-audio sentinel message."""
        raise NotImplementedError

    @abstractmethod
    def _parse_message(self, msg: dict) -> tuple[str, str]:
        """Map a server message to ``(kind, text)``.

        ``kind`` is ``"delta"``, ``"done"``, ``"error"``, or ``""`` (ignore).
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _http_to_ws(self, path: str) -> str:
        """Convert the http(s) api_base to a ws(s) URL with ``path`` appended."""
        base = self.api_base or ""
        if base.startswith("https://"):
            base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://") :]
        return f"{base}{path}"

    async def _maybe_pace_until(self, target_at: float) -> None:
        """Sleep until an absolute audio schedule time when pacing is on."""
        if not self._pacing:
            return
        delay_s = target_at - time.monotonic()
        if delay_s > 0:
            await asyncio.sleep(delay_s)

    def _clip_lock(self, audio_path: str) -> threading.Lock:
        """Return the per-clip lock guarding decode/encode for one path."""
        with self._clip_cache_lock:
            lock = self._clip_locks.get(audio_path)
            if lock is None:
                lock = self._clip_locks[audio_path] = threading.Lock()
            return lock

    def _clip_assets(self, audio_path: str) -> _ClipAssets:
        """Decode a clip and pre-encode its append messages, cached per clip.

        Blocking (audio decode + base64/JSON encode); callers on an event
        loop must run this in an executor. The per-clip lock makes the
        thundering herd at benchmark start decode each clip exactly once.
        """
        with self._clip_lock(audio_path):
            with self._clip_cache_lock:
                assets = self._clip_cache.get(audio_path)
            if assets is not None:
                return assets

            pcm = _audio_to_pcm16_bytes(audio_path, self._sample_rate)
            view = memoryview(pcm)
            full_end = len(pcm) - len(pcm) % self._ws_chunk_size
            wire_messages = [
                self._encode_chunk(view[offset : offset + self._ws_chunk_size])
                for offset in range(0, full_end, self._ws_chunk_size)
            ]
            assets = _ClipAssets(pcm=pcm, wire_messages=wire_messages)

            with self._clip_cache_lock:
                self._clip_cache[audio_path] = assets
                while len(self._clip_cache) > _CLIP_CACHE_MAX_CLIPS:
                    self._clip_cache.popitem(last=False)
            return assets

    async def _stream(
        self,
        pcm_bytes: bytes | memoryview,
        wire_messages: Optional[list[str]] = None,
    ) -> STTStreamResult:
        """Stream pre-decoded PCM16 to the provider and collect the transcript.

        The sender and receiver run concurrently so partial transcripts are
        timed when they arrive, not after the (paced) upload completes. ``ttfc``
        is anchored to ``audio_started_at`` (the first audio byte on the wire),
        so clip decode, connect, and handshake latency are excluded.

        ``wire_messages`` are the cached pre-encoded append messages for the
        clip this PCM is a prefix of (chunk ``i`` at byte ``i * chunk_size``);
        the trailing partial chunk is encoded on the fly.
        """
        import websockets

        ttfc: Optional[float] = None
        audio_started_at: Optional[float] = None
        audio_end_at: Optional[float] = None
        time_to_first_visible_text: Optional[float] = None
        time_to_first_partial: Optional[float] = None
        time_to_final_transcript: Optional[float] = None
        partial_transcript: Optional[str] = None
        final_transcript = ""
        chunk_count = 0
        transcript_chunks: list[str] = []
        snapshots = TranscriptSnapshotRecorder()

        async with websockets.connect(
            self._ws_url,
            ping_interval=self._ws_ping_interval_s,
            ping_timeout=self._ws_ping_timeout_s,
            compression=self._ws_compression,
        ) as ws:
            await self._open_session(ws)

            async def _send() -> None:
                nonlocal audio_end_at, audio_started_at
                for byte_offset in range(0, len(pcm_bytes), self._ws_chunk_size):
                    if audio_started_at is None:
                        audio_started_at = time.monotonic()
                        snapshots.mark_audio_started(audio_started_at)
                    else:
                        await self._maybe_pace_until(
                            audio_started_at
                            + byte_offset / BYTES_PER_SAMPLE / self._sample_rate
                        )
                    chunk_end = byte_offset + self._ws_chunk_size
                    if wire_messages is not None and chunk_end <= len(pcm_bytes):
                        message = wire_messages[byte_offset // self._ws_chunk_size]
                    else:
                        message = self._encode_chunk(pcm_bytes[byte_offset:chunk_end])
                    await ws.send(message)
                if audio_started_at is not None:
                    await self._maybe_pace_until(
                        audio_started_at
                        + len(pcm_bytes) / BYTES_PER_SAMPLE / self._sample_rate
                    )
                await ws.send(self._eof())
                audio_end_at = time.monotonic()

            send_task = asyncio.ensure_future(_send())
            try:
                while True:
                    kind, text = self._parse_message(json.loads(await ws.recv()))
                    now = time.monotonic()
                    if kind == "delta":
                        # TTFC counts only deltas whose own payload carries
                        # transcript text after cleaning; empty progress /
                        # keepalive deltas (e.g. Vajra's priming delta) and
                        # pure control-token pads are skipped. See
                        # STTStreamResult for how this differs from
                        # time_to_first_visible_text below.
                        if ttfc is None and _clean_transcript(text):
                            assert (
                                audio_started_at is not None
                            ), "delta arrived before any audio was sent"
                            ttfc = (now - audio_started_at) * 1000
                        transcript_chunks.append(text)
                        chunk_count += 1
                        current_transcript = _clean_transcript(
                            "".join(transcript_chunks)
                        )
                        snapshots.add(now, current_transcript)
                        if time_to_first_visible_text is None and current_transcript:
                            assert (
                                audio_started_at is not None
                            ), "transcript arrived before any audio was sent"
                            time_to_first_visible_text = (now - audio_started_at) * 1000
                        if (
                            audio_end_at is not None
                            and time_to_first_partial is None
                            and current_transcript
                        ):
                            time_to_first_partial = (now - audio_end_at) * 1000
                            partial_transcript = current_transcript
                    elif kind == "done":
                        final_transcript = _clean_transcript(
                            text or "".join(transcript_chunks)
                        )
                        snapshots.add(now, final_transcript)
                        if final_transcript:
                            if ttfc is None:
                                assert (
                                    audio_started_at is not None
                                ), "completion arrived before any audio was sent"
                                ttfc = (now - audio_started_at) * 1000
                            if time_to_first_visible_text is None:
                                assert (
                                    audio_started_at is not None
                                ), "transcript arrived before any audio was sent"
                                time_to_first_visible_text = (
                                    now - audio_started_at
                                ) * 1000
                            if chunk_count == 0:
                                chunk_count = 1
                        if audio_end_at is not None:
                            time_to_final_transcript = (now - audio_end_at) * 1000
                        break
                    elif kind == "error":
                        raise RuntimeError(
                            f"{self._provider} streaming error: {text or 'unknown'}"
                        )
            finally:
                # Normal path: the sender is already done (servers only emit
                # "done" after EOF). Error path: cancel it. A CancelledError is
                # expected; any other error is a real send failure and must
                # propagate.
                if not send_task.done():
                    send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

        if not final_transcript:
            final_transcript = _clean_transcript("".join(transcript_chunks))

        return STTStreamResult(
            ttfc=ttfc,
            time_to_first_visible_text=time_to_first_visible_text,
            time_to_first_partial=time_to_first_partial,
            time_to_final_transcript=time_to_final_transcript,
            partial_transcript=partial_transcript,
            final_transcript=final_transcript,
            transcript_snapshots=snapshots.snapshots,
            chunk_count=chunk_count,
            pcm_byte_count=len(pcm_bytes),
        )

    # ------------------------------------------------------------------
    # Request lifecycle (shared)
    # ------------------------------------------------------------------

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
    ) -> RequestResult:
        """Stream an audio file to the STT API and collect transcription metrics."""

        audio_content = request.channels.get(ChannelModality.AUDIO)
        if not isinstance(audio_content, AudioChannelRequestContent):
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=400,
                error_msg="No AUDIO channel in request for STT",
                client_completed_at=time.monotonic(),
            )

        audio_path = audio_content.input_audio

        try:
            os.path.getsize(audio_path)
        except OSError as e:
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=404,
                error_msg=f"Failed to read audio file {audio_path}: {e}",
                client_completed_at=time.monotonic(),
            )

        logger.debug(
            "[STT %s] request_id=%d session_id=%d file=%s",
            self._provider,
            request.id,
            session_id,
            audio_path,
        )

        error_msg: Optional[str] = None
        error_code: Optional[int] = None
        stream_result: Optional[STTStreamResult] = None

        # Decode and slice before the latency clock so file I/O and DSP don't
        # inflate TTFC. Decode and append-message encoding are cached per
        # clip and run in an executor so they never stall the shared event
        # loop (a synchronous decode here freezes pacing for every session
        # on this worker's loop).
        try:
            loop = asyncio.get_running_loop()
            clip = await loop.run_in_executor(None, self._clip_assets, audio_path)
            start_ms = _metadata_ms(request.metadata, "input_audio_start_ms")
            pcm_bytes = _slice_pcm16_bytes(
                memoryview(clip.pcm),
                self._sample_rate,
                start_ms=start_ms,
                end_ms=_metadata_ms(request.metadata, "input_audio_end_ms"),
            )
            # Cached messages are aligned to the clip start; a non-zero slice
            # start shifts the chunk grid, so encode per chunk in that case.
            wire_messages = clip.wire_messages if not start_ms else None
        except Exception as e:
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=422,
                error_msg=f"Failed to decode audio file {audio_path}: {e}",
                client_completed_at=time.monotonic(),
            )

        t_start = time.monotonic()

        try:
            async with asyncio.timeout(self._request_timeout):
                stream_result = await self._stream(pcm_bytes, wire_messages)
        except TimeoutError:
            error_code = 408
            error_msg = f"STT request timed out after {self._request_timeout}s"
        except (OSError, ConnectionError) as e:
            # Includes ConnectionRefusedError when the server isn't up.
            error_code = 503
            error_msg = str(e)
        except Exception as e:
            error_code = 520
            error_msg = str(e)
            logger.error(
                "[STT %s] request_id=%d error: %s",
                self._provider,
                request.id,
                e,
                exc_info=True,
            )

        completed_at = time.monotonic()
        total_latency_ms = (completed_at - t_start) * 1000
        success = error_msg is None and error_code is None

        channels = {}
        if success and stream_result is not None:
            input_audio_duration_ms = _pcm_duration_ms(
                stream_result.pcm_byte_count, self._sample_rate
            )
            # Report the input clip's byte count; the evaluator derives
            # duration/RTF from pcm_byte_count + sample_rate.
            metrics_dict: dict = {
                "audio_task": AudioTask.STT,
                "ttfc": round(stream_result.ttfc or 0.0, 3),
                "end_to_end_latency": round(total_latency_ms, 3),
                "time_to_first_visible_text": (
                    round(stream_result.time_to_first_visible_text, 3)
                    if stream_result.time_to_first_visible_text is not None
                    else None
                ),
                "time_to_first_partial": (
                    round(stream_result.time_to_first_partial, 3)
                    if stream_result.time_to_first_partial is not None
                    else None
                ),
                "time_to_final_transcript": (
                    round(stream_result.time_to_final_transcript, 3)
                    if stream_result.time_to_final_transcript is not None
                    else None
                ),
                "chunk_count": stream_result.chunk_count,
                "pcm_byte_count": stream_result.pcm_byte_count,
                "raw_pcm": True,
                "input_tokens": len(stream_result.final_transcript.split()),
                "sample_rate": self._sample_rate,
                "input_audio_duration_ms": round(input_audio_duration_ms, 3),
                "partial_transcript": stream_result.partial_transcript,
                "final_transcript": stream_result.final_transcript,
                "transcript_snapshots": stream_result.transcript_snapshots,
            }

            # Ground truth and dataset metadata are guaranteed by the trace generator.
            for key, value in request.metadata.items():
                metrics_dict.setdefault(key, value)

            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=stream_result.final_transcript,
                metrics=metrics_dict,
            )

        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            channels=channels,
            success=success,
            error_code=error_code,
            error_msg=error_msg,
            client_completed_at=completed_at,
        )


class VllmRealtimeSTTClient(_STTClientBase):
    """vllm_realtime: WebSocket /v1/realtime — base64 PCM16 chunks, deltas.

    Server -> Client: ``{"type": "session.created"}`` then
    ``transcription.delta`` / ``transcription.done`` / ``error``.
    """

    ws_path = "/v1/realtime"

    async def _open_session(self, ws) -> None:
        msg = json.loads(await ws.recv())
        if msg.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got: {msg}")
        await ws.send(json.dumps({"type": "session.update", "model": self._model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    def _encode_chunk(self, chunk: bytes | memoryview) -> str:
        return json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            }
        )

    def _eof(self) -> str:
        return json.dumps({"type": "input_audio_buffer.commit", "final": True})

    def _parse_message(self, msg: dict) -> tuple[str, str]:
        msg_type = msg.get("type")
        if msg_type == "transcription.delta":
            return "delta", msg.get("delta", "")
        if msg_type == "transcription.done":
            return "done", msg.get("text", "")
        if msg_type == "error":
            return "error", str(msg.get("error", ""))
        return "", ""


class VajraOpenAIRealtimeSTTClient(_STTClientBase):
    """vajra_openai_realtime: WebSocket /openai/v1/realtime — OpenAI transcription.

    Drives Vajra's OpenAI-compatible realtime transcription endpoint. Server ->
    Client: ``transcription_session.created`` then
    ``conversation.item.input_audio_transcription.delta`` /
    ``conversation.item.input_audio_transcription.completed`` / ``error``. v1 is
    manual-commit, one transcript per connection (PCM16 mono 16 kHz).
    """

    ws_path = "/openai/v1/realtime?intent=transcription"

    async def _open_session(self, ws) -> None:
        msg = json.loads(await ws.recv())
        if msg.get("type") != "transcription_session.created":
            raise RuntimeError(f"Expected transcription_session.created, got: {msg}")
        # Configure the session; unlike vllm_realtime, do NOT commit here — a
        # commit before any audio would finalize an empty transcript.
        await ws.send(
            json.dumps(
                {
                    "type": "transcription_session.update",
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": self._model},
                }
            )
        )

    def _encode_chunk(self, chunk: bytes | memoryview) -> str:
        return json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            }
        )

    def _eof(self) -> str:
        return json.dumps({"type": "input_audio_buffer.commit"})

    def _parse_message(self, msg: dict) -> tuple[str, str]:
        msg_type = msg.get("type")
        if msg_type == "conversation.item.input_audio_transcription.delta":
            return "delta", msg.get("delta", "")
        if msg_type == "conversation.item.input_audio_transcription.completed":
            return "done", msg.get("transcript", "")
        # ".failed" is the terminal event for a committed item whose
        # transcription failed; treat it like a session-level "error" so the
        # request fails fast instead of waiting for the timeout.
        if msg_type in (
            "error",
            "conversation.item.input_audio_transcription.failed",
        ):
            error = msg.get("error")
            if isinstance(error, dict):
                return "error", str(error.get("message", ""))
            return "error", str(error or "")
        return "", ""


_PROVIDERS: dict[str, type[_STTClientBase]] = {
    "vllm_realtime": VllmRealtimeSTTClient,
    "vajra_openai_realtime": VajraOpenAIRealtimeSTTClient,
}


def STTClient(config: "STTClientConfig", **kwargs) -> _STTClientBase:
    """Factory: return the STT client for ``config.provider``."""
    try:
        cls = _PROVIDERS[config.provider]
    except KeyError:
        raise ValueError(f"Unsupported STT provider: {config.provider}")
    return cls(config, **kwargs)
