"""TTS client for streaming HTTP-based text-to-speech APIs."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import httpx

from veeksha.client.base import BaseLLMClient
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import TTSClientConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class OpenAISpeechRequest:
    """Canonical OpenAI Audio Speech HTTP request."""

    url: str
    headers: dict[str, str]
    payload: dict[str, str | bool]


class TTSProtocolError(Exception):
    """Raised when a server response violates the selected Speech stream format."""


def _build_audio_speech_url(api_base: str) -> str:
    """Build ``/v1/audio/speech`` from an API base with or without ``/v1``."""
    normalized_base = api_base.rstrip("/")
    if not normalized_base.endswith("/v1"):
        normalized_base = f"{normalized_base}/v1"
    return urljoin(f"{normalized_base}/", "audio/speech")


class TTSClient(BaseLLMClient):
    """Async client for streaming HTTP-based TTS APIs."""

    def __init__(self, config: TTSClientConfig, **kwargs) -> None:
        super().__init__(config)
        self._chunk_size = config.chunk_size
        self._speech_url = _build_audio_speech_url(str(self.api_base))
        self._client_storage = threading.local()

    def _build_request(self, text: str) -> OpenAISpeechRequest:
        """Build the one wire contract accepted by the HTTP TTS client."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return OpenAISpeechRequest(
            url=self._speech_url,
            headers=headers,
            payload={
                "model": self.config.model,
                "input": text,
                "voice": self.config.voice_id,
                "response_format": "pcm" if self.config.raw_pcm else "wav",
                "stream": True,
                "stream_format": "audio",
            },
        )

    def _validate_audio_response(self, response: httpx.Response) -> None:
        content_type = response.headers.get("Content-Type", "")
        media_type = content_type.partition(";")[0].strip().lower()
        allowed = {
            "application/octet-stream",
            "application/x-wav",
            "binary/octet-stream",
        }
        if (
            media_type
            and not media_type.startswith("audio/")
            and media_type not in allowed
        ):
            raise TTSProtocolError(
                "OpenAI Audio Speech stream_format=audio requires an audio byte "
                f"response, got Content-Type {content_type!r}"
            )

    def _get_client(self) -> httpx.AsyncClient:
        """Return a thread-local httpx client bound to the caller's event loop."""
        if not hasattr(self._client_storage, "client"):
            self._client_storage.client = httpx.AsyncClient(
                timeout=self.config.request_timeout
            )
        return self._client_storage.client

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent: Callable[[], None] | None = None,
        on_request_dispatched: Callable[[], None] | None = None,
    ) -> RequestResult:
        """Send a streaming TTS request and collect audio metrics."""
        text_content = request.channels.get(ChannelModality.TEXT)
        if not isinstance(text_content, TextChannelRequestContent):
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=400,
                error_msg="No TEXT channel in request for TTS",
                client_completed_at=time.monotonic(),
            )
        input_text = text_content.input_text
        speech_request = self._build_request(input_text)

        logger.debug(
            "[TTS] request_id=%d session_id=%d chars=%d text=%.80r",
            request.id,
            session_id,
            len(input_text),
            input_text,
        )

        error_msg: str | None = None
        error_code: int | None = None
        ttfc: float | None = None
        chunk_count = 0
        audio_chunks: list[bytes] = []

        t_start = time.monotonic()
        # If we have information about the scheduler_ready_at, dispatched_at, client_picked_up_at for this request,
        # we should make sure that t_cs is close to them
        # client request sent time t_cs
        # send t_cs along with the request
        # At the server, we should validate that server receive time i.e t_sr is close to t_cs

        try:
            async with self._get_client().stream(
                "POST",
                speech_request.url,
                headers=speech_request.headers,
                json=speech_request.payload,
                timeout=self.config.request_timeout,
            ) as response:
                response.raise_for_status()
                self._validate_audio_response(response)
                if on_request_dispatched is not None:
                    on_request_dispatched()

                sent_notified = False
                async for chunk in response.aiter_bytes(chunk_size=self._chunk_size):
                    if not chunk:
                        continue
                    receive_time = time.monotonic()
                    # client response receive time for ith chunk i.e t_cr_i 
                    # The mock server should send server chunk response sent time i.e t_ss_i alongside the audio chunk
                    # The client/preflight validation should validate t_cr_i and t_ss_i are close
                    # The server should also validate the time interval between t_ss_{i+1} - t_ss_{i} matches the
                    # precofigured time for sending a chunck of the audio
                    # The server should also validate the time interval betwen the first response t_ss_1 and the first segment
                    # receive time i.e t_sr_1 is close to the fixed precofigured value ttfc
                    # The above validation can be done on the client side as well i.e t_cr_1 - t_cs should be close to the configured ttfc
                    if ttfc is None:
                        ttfc = (receive_time - t_start) * 1000
                    if not sent_notified and on_request_sent is not None:
                        on_request_sent()
                        sent_notified = True

                    audio_chunks.append(chunk)
                    chunk_count += 1

                if not sent_notified and on_request_sent is not None:
                    on_request_sent()

        except TTSProtocolError as e:
            error_code = 502
            error_msg = str(e)
            logger.warning("TTS protocol error: (%s) %s", error_code, error_msg)
        except httpx.HTTPStatusError as e:
            error_code = e.response.status_code if e.response else 500
            error_msg = str(e)
            logger.warning("HTTP Error: status=%s msg=%s", error_code, error_msg)
        except httpx.ConnectError as e:
            error_code = 503
            error_msg = str(e)
            logger.warning("Connection Error: (%s) %s", error_code, error_msg)
        except httpx.TimeoutException:
            error_code = 408
            error_msg = "TTS request timed out"
            logger.warning("Timeout Error: (%s) %s", error_code, error_msg)
        except Exception as e:
            error_code = 520
            error_msg = str(e)
            logger.exception("Unexpected error: (%s) %s", error_code, error_msg)

        # request completed time. Validate that this is close to the t_ss_k and t_cr_k (i.e time for last chunk)
        completed_at = time.monotonic()
        total_latency_ms = (completed_at - t_start) * 1000
        success = error_msg is None and error_code is None
        audio_data = b"".join(audio_chunks) if audio_chunks else b""

        channels = {}
        if success:
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=audio_data,
                metrics={
                    "audio_task": AudioTask.TTS,
                    AudioMetricKey.TTFC.value: round(ttfc or 0.0, 3),
                    AudioMetricKey.END_TO_END_LATENCY.value: round(total_latency_ms, 3),
                    AudioMetricKey.CHUNK_COUNT.value: chunk_count,
                    AudioMetricKey.RAW_PCM.value: self.config.raw_pcm,
                    AudioMetricKey.SAMPLE_RATE.value: self.config.sample_rate,
                    AudioMetricKey.INPUT_CHARS.value: len(input_text),
                    AudioMetricKey.INPUT_TOKENS.value: text_content.target_prompt_tokens
                    or 0,
                    AudioMetricKey.INPUT_TEXT.value: input_text,
                },
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
