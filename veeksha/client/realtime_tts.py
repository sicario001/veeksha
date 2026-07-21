"""WebSocket client for text-to-speech over the OpenAI Realtime protocol.

It sends paced text conversation items at an emulated upstream-LLM decode rate,
receives audio chunks, and records the per-event millisecond timestamps consumed
by the audio interactivity evaluator.

Structurally mirrors the streaming HTTP :class:`~veeksha.client.tts.TTSClient`
(same metric contract), but the transport is a
WebSocket and the request is a paced sequence of ``conversation.item.create``
events followed by an audio-only ``response.create`` event.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote, urljoin

from websockets.asyncio.client import connect

from veeksha.client.base import BaseLLMClient
from veeksha.client.utils import (
    WS_TRANSPORT_ERROR_PRIORITY,
    TextDeltaPacer,
    flatten_ws_exception,
    map_ws_transport_error,
    segment_text,
)
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import RealtimeTTSClientConfig

logger = init_logger(__name__)


class RealtimeEventKind(StrEnum):
    SESSION_UPDATED = "session_updated"
    RESPONSE_CREATED = "response_created"
    AUDIO_DELTA = "audio_delta"
    AUDIO_DONE = "audio_done"
    RESPONSE_DONE = "response_done"
    ERROR = "error"
    OTHER = "other"


class RealtimeTTSProtocol:
    """OpenAI Realtime wire contract used by the benchmark client."""

    _EVENT_KINDS = {
        "session.updated": RealtimeEventKind.SESSION_UPDATED,
        "response.created": RealtimeEventKind.RESPONSE_CREATED,
        "response.output_audio.delta": RealtimeEventKind.AUDIO_DELTA,
        "response.output_audio.done": RealtimeEventKind.AUDIO_DONE,
        "response.done": RealtimeEventKind.RESPONSE_DONE,
        "error": RealtimeEventKind.ERROR,
    }

    def __init__(
        self, config: "RealtimeTTSClientConfig", api_key: Optional[str]
    ) -> None:
        self.config = config
        self._api_key = api_key

    @property
    def raw_pcm(self) -> bool:
        return True

    def build_ws_url(self, api_base: str) -> str:
        ws_base = api_base
        if ws_base.startswith("https://"):
            ws_base = "wss://" + ws_base[len("https://") :]
        elif ws_base.startswith("http://"):
            ws_base = "ws://" + ws_base[len("http://") :]
        normalized_base = ws_base.rstrip("/")
        path = "realtime" if normalized_base.endswith("/v1") else "v1/realtime"
        url = urljoin(f"{normalized_base}/", path)
        return f"{url}?model={quote(self.config.model, safe='')}"

    def headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    def session_update_json(self) -> str:
        output: dict = {
            "format": {"type": "audio/pcm", "rate": self.config.sample_rate}
        }
        if self.config.voice_id:
            output["voice"] = self.config.voice_id
        session = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {"output": output},
        }
        return json.dumps({"type": "session.update", "session": session})

    def conversation_item_create_json(self, text: str) -> str:
        return json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    def response_create_json(self) -> str:
        return json.dumps(
            {
                "type": "response.create",
                "response": {"output_modalities": ["audio"]},
            }
        )

    def classify(self, event_type: str) -> RealtimeEventKind:
        return self._EVENT_KINDS.get(event_type, RealtimeEventKind.OTHER)

    def extract_audio(self, event: dict) -> bytes:
        encoded = event.get("delta")
        return base64.b64decode(encoded) if encoded else b""


def _build_realtime_protocol(
    config: "RealtimeTTSClientConfig", api_key: Optional[str] = None
) -> RealtimeTTSProtocol:
    return RealtimeTTSProtocol(
        config, api_key=config.api_key if api_key is None else api_key
    )


# ---------------------------------------------------------------------------
# Server-side error signalling
# ---------------------------------------------------------------------------


class RealtimeServerError(Exception):
    """Raised when the server sends an ``error`` event mid-stream.

    Carries the decoded error event payload so the client can surface the
    server-provided message.
    """

    def __init__(self, event: dict) -> None:
        self.event = event
        super().__init__(str(event))


def _validate_completed_response(event: dict) -> None:
    """Raise when a terminal Realtime response did not complete successfully."""
    response = event.get("response")
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise RealtimeServerError(event)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _round_ms(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


def _extract_output_sample_rate(session: dict) -> Optional[int]:
    """Read ``session.audio.output.format.rate`` from a GA session object."""
    audio = session.get("audio")
    if isinstance(audio, dict):
        output = audio.get("output")
        if isinstance(output, dict):
            fmt = output.get("format")
            if isinstance(fmt, dict) and isinstance(fmt.get("rate"), int):
                return fmt["rate"]
    return None


class RealtimeTTSClient(BaseLLMClient):
    """Async WebSocket client for OpenAI Realtime text-to-speech."""

    def __init__(self, config: "RealtimeTTSClientConfig", **kwargs) -> None:
        # **kwargs swallows tokenizer_provider (matching TTSClient); realtime
        # TTS token counts come from whitespace segmentation instead.
        super().__init__(config)
        self._realtime_config = config
        self._protocol = _build_realtime_protocol(config, api_key=self.api_key)

    def _connect(self):
        """Return the websocket connect context manager.

        A seam so tests can override the transport. ``max_size=None`` lifts the
        inbound-frame cap (audio deltas can be large); ``compression=None``
        keeps binary PCM uncompressed. The asyncio transport sets ``TCP_NODELAY``
        by default, so no explicit socket option is required.
        """
        open_timeout = min(self._realtime_config.request_timeout, 30)
        return connect(
            self._protocol.build_ws_url(str(self.api_base)),
            max_size=None,
            compression=None,
            open_timeout=open_timeout,
            additional_headers=self._protocol.headers(),
        )

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent: Optional[Callable[[], None]] = None,
        on_request_dispatched: Optional[Callable[[], None]] = None,
    ) -> RequestResult:
        """Stream text deltas over a websocket and collect audio metrics."""
        text_content = request.channels.get(ChannelModality.TEXT)
        if not isinstance(text_content, TextChannelRequestContent):
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=400,
                error_msg="No TEXT channel in request for realtime TTS",
                client_completed_at=time.monotonic(),
            )

        input_text = text_content.input_text
        pacing = self._realtime_config.pacing
        segments = segment_text(input_text, pacing.tokens_per_delta)
        pacer = TextDeltaPacer(pacing, seed=pacing.seed + request.id)
        input_tokens = text_content.target_prompt_tokens or sum(
            seg.n_tokens for seg in segments
        )

        logger.debug(
            "[RealtimeTTS] request_id=%d session_id=%d chars=%d deltas=%d",
            request.id,
            session_id,
            len(input_text),
            len(segments),
        )

        # Collected state (shared with the nested send/recv loops via closure).
        audio_chunks: list[bytes] = []
        audio_chunk_ts: list[list[float]] = []  # [[offset_ms, n_bytes], ...]
        text_delta_ts: list[list[float]] = []  # [[offset_ms, n_chars], ...]
        ttfc: Optional[float] = None
        ws_connect_latency: Optional[float] = None
        session_ready_offset: Optional[float] = None
        response_created_offset: Optional[float] = None
        input_complete_offset: Optional[float] = None
        audio_done_offset: Optional[float] = None
        response_done_offset: Optional[float] = None
        sample_rate = self._realtime_config.sample_rate

        sent_fired = False

        def fire_sent_once() -> None:
            nonlocal sent_fired
            if not sent_fired and on_request_sent is not None:
                on_request_sent()
                sent_fired = True

        error_code: Optional[int] = None
        error_msg: Optional[str] = None
        # Absolute stamp taken immediately before the FIRST paced frame leaves;
        # recorded, not used, by the client (the preflight pairs it with the
        # server's receive stamp). None until the first send.
        request_sent_monotonic: Optional[float] = None

        t_start = time.monotonic()

        async def send_loop(ws) -> None:
            nonlocal input_complete_offset
            if pacer.initial_delay_s > 0:
                await asyncio.sleep(pacer.initial_delay_s)
            # Pace by absolute deadlines so ws.send backpressure never
            # accumulates drift into subsequent gaps.
            deadline = time.monotonic()
            for seg in segments:
                deadline += pacer.next_gap()
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                await ws.send(self._protocol.conversation_item_create_json(seg.text))
                text_delta_ts.append([(time.monotonic() - t_start) * 1000, seg.n_chars])
            input_complete_offset = (time.monotonic() - t_start) * 1000
            await ws.send(self._protocol.response_create_json())

        async def recv_loop(ws) -> None:
            nonlocal ttfc, session_ready_offset, response_created_offset
            nonlocal audio_done_offset, response_done_offset, sample_rate
            while True:
                raw = await ws.recv()
                # Stamp receipt BEFORE any json/base64 decode work.
                offset_ms = (time.monotonic() - t_start) * 1000
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                kind = self._protocol.classify(event.get("type", ""))

                if kind is RealtimeEventKind.AUDIO_DELTA:
                    chunk = self._protocol.extract_audio(event)
                    if chunk:
                        audio_chunks.append(chunk)
                        audio_chunk_ts.append([offset_ms, len(chunk)])
                        if ttfc is None:
                            ttfc = offset_ms
                        fire_sent_once()
                elif kind is RealtimeEventKind.SESSION_UPDATED:
                    if session_ready_offset is None:
                        session_ready_offset = offset_ms
                    session = event.get("session")
                    if isinstance(session, dict):
                        server_sr = _extract_output_sample_rate(session)
                        if (
                            isinstance(server_sr, int)
                            and server_sr > 0
                            and server_sr != sample_rate
                        ):
                            logger.warning(
                                "Realtime server echoed sample_rate=%d (configured "
                                "%d); using server value.",
                                server_sr,
                                self._realtime_config.sample_rate,
                            )
                            sample_rate = server_sr
                elif kind is RealtimeEventKind.RESPONSE_CREATED:
                    if response_created_offset is None:
                        response_created_offset = offset_ms
                elif kind is RealtimeEventKind.AUDIO_DONE:
                    if audio_done_offset is None:
                        audio_done_offset = offset_ms
                elif kind is RealtimeEventKind.RESPONSE_DONE:
                    response_done_offset = offset_ms
                    _validate_completed_response(event)
                    return  # Terminal: stop on response.done, not audio.done.
                elif kind is RealtimeEventKind.ERROR:
                    raise RealtimeServerError(event)
                # OTHER: keep receiving until response.done.

        try:
            async with asyncio.timeout(self._realtime_config.request_timeout):
                async with self._connect() as ws:
                    ws_connect_latency = (time.monotonic() - t_start) * 1000
                    # C2 client-side send stamp: the request starts going out at
                    # the FIRST application frame (session.update), which is what
                    # the mock stamps its received_at against. Stamping the first
                    # paced frame instead would land after the mock's receive and
                    # make the delivery lag negative.
                    request_sent_monotonic = time.monotonic()
                    await ws.send(self._protocol.session_update_json())
                    # Analog of the HTTP-200 ack: the scheduler's dispatch pacing
                    # advances on this callback.
                    if on_request_dispatched is not None:
                        on_request_dispatched()

                    async with asyncio.TaskGroup() as task_group:
                        task_group.create_task(send_loop(ws))
                        task_group.create_task(recv_loop(ws))
        except TimeoutError:
            error_code = 408
            error_msg = "Realtime TTS request timed out"
            logger.warning("Realtime TTS timeout: (%s) %s", error_code, error_msg)
        except Exception as exc:  # noqa: BLE001 - mapped to error codes below.
            error_code, error_msg = _map_error(
                flatten_ws_exception(exc, _ERROR_PRIORITY)
            )
            logger.warning("Realtime TTS error: (%s) %s", error_code, error_msg)

        completed_at = time.monotonic()
        total_latency_ms = (completed_at - t_start) * 1000
        success = error_code is None and error_msg is None

        # on_request_sent fallback: fire exactly once even on error paths.
        # With ordering="prefill" the client_runner's DispatchTracker only
        # advances via on_request_sent (or when send_request RAISES); a request
        # that fails *before* first audio would otherwise deadlock the tracker,
        # since we return an error result rather than raising. Firing here keeps
        # the sequential launch moving.
        fire_sent_once()

        audio_data = b"".join(audio_chunks) if audio_chunks else b""

        metrics = {
            "audio_task": AudioTask.TTS,
            AudioMetricKey.TTFC.value: round(ttfc or 0.0, 3),
            AudioMetricKey.END_TO_END_LATENCY.value: round(total_latency_ms, 3),
            AudioMetricKey.CHUNK_COUNT.value: len(audio_chunks),
            AudioMetricKey.RAW_PCM.value: self._protocol.raw_pcm,
            AudioMetricKey.SAMPLE_RATE.value: sample_rate,
            AudioMetricKey.INPUT_CHARS.value: len(input_text),
            AudioMetricKey.INPUT_TOKENS.value: input_tokens,
            AudioMetricKey.INPUT_TEXT.value: input_text,
            # Absolute anchors (recordings only). Chunk i arrived at
            # request_start_monotonic + audio_chunk_timestamps[i][0]/1000;
            # request_sent_monotonic is the C2 client-side send stamp.
            "request_start_monotonic": t_start,
            "request_sent_monotonic": request_sent_monotonic,
            AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value: text_delta_ts,
            AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: audio_chunk_ts,
            AudioMetricKey.WS_CONNECT_LATENCY_MS.value: _round_ms(ws_connect_latency),
            AudioMetricKey.SESSION_READY_OFFSET_MS.value: _round_ms(
                session_ready_offset
            ),
            AudioMetricKey.RESPONSE_CREATED_OFFSET_MS.value: _round_ms(
                response_created_offset
            ),
            AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: _round_ms(
                input_complete_offset
            ),
            AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: _round_ms(audio_done_offset),
            AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: _round_ms(
                response_done_offset
            ),
        }

        channels: dict = {}
        has_partial = bool(audio_chunks or text_delta_ts)
        if success or has_partial:
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=audio_data,
                metrics=metrics,
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


# ---------------------------------------------------------------------------
# Error flattening / mapping
# ---------------------------------------------------------------------------

_ERROR_PRIORITY: tuple[type[BaseException], ...] = (
    RealtimeServerError,
    *WS_TRANSPORT_ERROR_PRIORITY,
)


def _server_error_message(event: dict) -> str:
    """Extract the useful message from an error or failed response event."""
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message

    response = event.get("response")
    if isinstance(response, dict):
        status_details = response.get("status_details")
        if isinstance(status_details, dict):
            error = status_details.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            reason = status_details.get("reason")
            if isinstance(reason, str) and reason:
                return reason

        status = response.get("status")
        if isinstance(status, str) and status:
            return f"Realtime response ended with status {status}"

    return json.dumps(event)


def _map_error(exc: BaseException) -> tuple[int, str]:
    """Map a flattened exception to an (error_code, error_msg) pair."""
    if isinstance(exc, RealtimeServerError):
        return 500, _server_error_message(exc.event)
    return map_ws_transport_error(exc, "Realtime TTS request timed out")
