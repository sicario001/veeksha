"""WebSocket client for Vajra's native streaming-text TTS protocol.

Speaks the ``/v1/audio/speech/stream`` contract: the client sends one
``session.config`` JSON message, then paced ``input.text`` deltas (emulating
an upstream LLM's decode cadence), then a terminal ``input.done``. The server
replies with an ``audio.start`` JSON event, binary frames of raw int16 PCM,
and terminal ``audio.done`` / ``session.done`` JSON events.

Structurally mirrors :class:`~veeksha.client.realtime_tts.RealtimeTTSClient`
and records the same :class:`~veeksha.core.audio_contract.AudioMetricKey`
timeline data, so the audio interactivity evaluator works unchanged; only the
wire protocol differs (native Vajra events and binary PCM frames instead of
OpenAI Realtime events with base64 audio).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional
from urllib.parse import urljoin

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
    from veeksha.config.client import VajraTTSStreamClientConfig

logger = init_logger(__name__)


class VajraTTSStreamProtocol:
    """Vajra streaming-speech wire contract used by the benchmark client."""

    def __init__(
        self, config: "VajraTTSStreamClientConfig", api_key: Optional[str]
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
        path = (
            "audio/speech/stream"
            if normalized_base.endswith("/v1")
            else "v1/audio/speech/stream"
        )
        return urljoin(f"{normalized_base}/", path)

    def headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    def session_config_json(self) -> str:
        message: dict = {
            "type": "session.config",
            "response_format": "pcm",
            "stream_audio": True,
        }
        if self.config.voice_id:
            message["voice"] = self.config.voice_id
        if self.config.language is not None:
            message["language"] = self.config.language
        if self.config.instructions is not None:
            message["instructions"] = self.config.instructions
        if self.config.task_type is not None:
            message["task_type"] = self.config.task_type
        return json.dumps(message)

    def input_text_json(self, text: str) -> str:
        return json.dumps({"type": "input.text", "text": text})

    def input_done_json(self) -> str:
        return json.dumps({"type": "input.done"})


# ---------------------------------------------------------------------------
# Server-side error signalling
# ---------------------------------------------------------------------------


class VajraTTSServerError(Exception):
    """Raised when the server sends an ``error`` event or a failed ``audio.done``.

    Carries the decoded event payload so the client can surface the
    server-provided message.
    """

    def __init__(self, event: dict) -> None:
        self.event = event
        super().__init__(str(event))


_ERROR_PRIORITY: tuple[type[BaseException], ...] = (
    VajraTTSServerError,
    *WS_TRANSPORT_ERROR_PRIORITY,
)


def _server_error_message(event: dict) -> str:
    """Extract the useful message from an error or failed audio.done event."""
    message = event.get("message")
    if isinstance(message, str) and message:
        return message
    if event.get("type") == "audio.done":
        return "Vajra TTS stream reported audio.done with error=true"
    return json.dumps(event)


def _map_error(exc: BaseException) -> tuple[int, str]:
    """Map a flattened exception to an (error_code, error_msg) pair."""
    if isinstance(exc, VajraTTSServerError):
        return 500, _server_error_message(exc.event)
    return map_ws_transport_error(exc, "Vajra TTS stream request timed out")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _round_ms(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


class VajraTTSStreamClient(BaseLLMClient):
    """Async WebSocket client for Vajra native streaming-text text-to-speech."""

    def __init__(self, config: "VajraTTSStreamClientConfig", **kwargs) -> None:
        # **kwargs swallows tokenizer_provider (matching RealtimeTTSClient);
        # TTS token counts come from whitespace segmentation instead.
        super().__init__(config)
        self._stream_config = config
        self._protocol = VajraTTSStreamProtocol(config, api_key=self.api_key)

    def _connect(self):
        """Return the websocket connect context manager.

        A seam so tests can override the transport. ``max_size=None`` lifts the
        inbound-frame cap (PCM frames can be large); ``compression=None``
        keeps binary PCM uncompressed (permessage-deflate burns CPU on
        high-entropy audio). The asyncio transport sets ``TCP_NODELAY`` by
        default, so no explicit socket option is required.
        """
        open_timeout = min(self._stream_config.request_timeout, 30)
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
                error_msg="No TEXT channel in request for Vajra TTS stream",
                client_completed_at=time.monotonic(),
            )

        input_text = text_content.input_text
        pacing = self._stream_config.pacing
        segments = segment_text(input_text, pacing.tokens_per_delta)
        pacer = TextDeltaPacer(pacing, seed=pacing.seed + request.id)
        input_tokens = text_content.target_prompt_tokens or sum(
            seg.n_tokens for seg in segments
        )

        logger.debug(
            "[VajraTTSStream] request_id=%d session_id=%d chars=%d deltas=%d",
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
        input_complete_offset: Optional[float] = None
        audio_done_offset: Optional[float] = None
        session_done_offset: Optional[float] = None
        sample_rate = self._stream_config.sample_rate

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
                await ws.send(self._protocol.input_text_json(seg.text))
                text_delta_ts.append([(time.monotonic() - t_start) * 1000, seg.n_chars])
            input_complete_offset = (time.monotonic() - t_start) * 1000
            await ws.send(self._protocol.input_done_json())

        async def recv_loop(ws) -> None:
            nonlocal ttfc, session_ready_offset, audio_done_offset
            nonlocal session_done_offset, sample_rate
            while True:
                raw = await ws.recv()
                # Stamp receipt BEFORE any json decode work.
                offset_ms = (time.monotonic() - t_start) * 1000

                # Binary frames are raw int16 PCM audio.
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    chunk = bytes(raw)
                    if chunk:
                        audio_chunks.append(chunk)
                        audio_chunk_ts.append([offset_ms, len(chunk)])
                        if ttfc is None:
                            ttfc = offset_ms
                        fire_sent_once()
                    continue

                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                event_type = event.get("type", "")

                if event_type == "audio.start":
                    if session_ready_offset is None:
                        session_ready_offset = offset_ms
                    server_sr = event.get("sample_rate")
                    if (
                        isinstance(server_sr, int)
                        and server_sr > 0
                        and server_sr != sample_rate
                    ):
                        logger.warning(
                            "Vajra TTS server sent sample_rate=%d (configured "
                            "%d); using server value.",
                            server_sr,
                            self._stream_config.sample_rate,
                        )
                        sample_rate = server_sr
                elif event_type == "audio.done":
                    if audio_done_offset is None:
                        audio_done_offset = offset_ms
                    if event.get("error"):
                        raise VajraTTSServerError(event)
                elif event_type == "session.done":
                    session_done_offset = offset_ms
                    return  # Terminal: stop on session.done, not audio.done.
                elif event_type == "error":
                    raise VajraTTSServerError(event)
                # Anything else: keep receiving until session.done.

        try:
            async with asyncio.timeout(self._stream_config.request_timeout):
                async with self._connect() as ws:
                    ws_connect_latency = (time.monotonic() - t_start) * 1000
                    # C2 client-side send stamp: the request starts going out at
                    # the FIRST application frame (session.config), which is what
                    # the mock stamps its received_at against. Stamping the first
                    # paced frame instead would land after the mock's receive and
                    # make the delivery lag negative.
                    request_sent_monotonic = time.monotonic()
                    await ws.send(self._protocol.session_config_json())
                    # Analog of the HTTP-200 ack: the scheduler's dispatch pacing
                    # advances on this callback.
                    if on_request_dispatched is not None:
                        on_request_dispatched()

                    async with asyncio.TaskGroup() as task_group:
                        task_group.create_task(send_loop(ws))
                        task_group.create_task(recv_loop(ws))
        except TimeoutError:
            error_code = 408
            error_msg = "Vajra TTS stream request timed out"
            logger.warning("Vajra TTS stream timeout: (%s) %s", error_code, error_msg)
        except Exception as exc:  # noqa: BLE001 - mapped to error codes below.
            error_code, error_msg = _map_error(
                flatten_ws_exception(exc, _ERROR_PRIORITY)
            )
            logger.warning("Vajra TTS stream error: (%s) %s", error_code, error_msg)

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
            AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: _round_ms(
                input_complete_offset
            ),
            AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: _round_ms(audio_done_offset),
            AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: _round_ms(
                session_done_offset
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
