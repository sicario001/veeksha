"""Unified native transport — evaluator-ready RequestResults for every modality.

The native engines (SSE ``run_batch``, WS ``ws_stream``) own transport + timing;
this module is the seam that turns veeksha ``Request`` batches into the
``RequestResult`` shapes the *existing* per-modality evaluators consume, so a
benchmark can run text / realtime-TTS / STT entirely over the native path:

  * text  -> SSE receive; TEXT channel with inter_chunk_times + num_output_tokens
  * TTS   -> realtime WS (paced text in, audio deltas out); AUDIO channel with the
             per-chunk timeline the audio evaluator reads for TTFA / streaming-RTF
  * STT   -> realtime WS (paced audio in, transcript deltas out); AUDIO channel
             with final_transcript + snapshots + ground truth for WER

Python owns *what* each request is (payload, pacing, ground truth); native owns
concurrency + kernel-time receive/send timing. Native never touches per-event
Python code on the hot path.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from veeksha.core import audio_contract as ac
from veeksha.core.request import Request
from veeksha.core.request_content import (
    AudioChannelRequestContent,
    TextChannelRequestContent,
)
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.native.engine import (
    NativeChainTurnSpec,
    NativeReceiveEngine,
    NativeRequest,
    NativeWsEngine,
    native_available,
)
from veeksha.types import AudioTask, ChannelModality

__all__ = ["NativeTransport", "native_available"]


class NativeTransport:
    """Runs veeksha Request batches over the native engine, per modality."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sse = NativeReceiveEngine(host, port)
        self._ws = NativeWsEngine(host, port)

    # ------------------------------------------------------------------ text
    def run_text(
        self,
        requests: List[Request],
        concurrency: int,
        model: str = "mock",
        path: str = "/v1/chat/completions",
        default_max_tokens: int = 16,
        timeout_s: float = 120.0,
        dispatch_offsets_s: Optional[List[float]] = None,
        num_threads: int = 1,
        api_key: Optional[str] = None,
        additional_sampling_params: Optional[Dict[str, Any]] = None,
    ) -> List[RequestResult]:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        native_reqs = []
        for request in requests:
            content = request.channels.get(ChannelModality.TEXT)
            text = (
                content.input_text
                if isinstance(content, TextChannelRequestContent)
                else ""
            )
            payload: Dict[str, Any] = {
                "model": model,
                "stream": True,
                "max_completion_tokens": _target_tokens(request, default_max_tokens),
                "messages": [{"role": "user", "content": text}],
            }
            if additional_sampling_params:
                payload.update(additional_sampling_params)
            native_reqs.append(
                NativeRequest(path=path, body=json.dumps(payload), headers=headers)
            )

        anchor = time.monotonic()
        results = self._sse.run(
            native_reqs,
            concurrency=concurrency,
            sse=True,
            timeout_s=timeout_s,
            modality=ChannelModality.TEXT,
            dispatch_offsets_s=dispatch_offsets_s,
            num_threads=num_threads,
        )
        out = []
        for request, res in zip(requests, results):
            channels = {}
            success = res.success and len(res.stream) > 0
            if len(res.stream) > 0:
                tte = res.stream.time_to_first_event() or 0.0
                inter_chunk_times = [tte] + res.stream.inter_event_deltas()
                channels[ChannelModality.TEXT] = ChannelResponse(
                    modality=ChannelModality.TEXT,
                    content=res.content,
                    metrics={
                        "is_stream": True,
                        "inter_chunk_times": inter_chunk_times,
                        # SSE chunk count, not a tokenizer count: one delta per
                        # chunk is the streaming contract the mock + vLLM follow.
                        "num_output_tokens": len(res.stream),
                        "num_total_prompt_tokens": 0,
                        "num_delta_prompt_tokens": 0,
                    },
                )
            dispatched_at, completed_at = _anchored_times(
                anchor, res.dispatch_offset_s, res.stream
            )
            out.append(
                _request_result(
                    request,
                    channels,
                    success,
                    res.status,
                    res.error,
                    dispatched_at,
                    completed_at,
                )
            )
        return out

    # ------------------------------------------------------- text, multi-turn
    def run_text_chains(
        self,
        chains: List[List[Request]],
        chain_delays_s: List[List[float]],
        chain_history: List[List[bool]],
        concurrency: int,
        model: str = "mock",
        path: str = "/v1/chat/completions",
        default_max_tokens: int = 16,
        timeout_s: float = 120.0,
        start_offsets_s: Optional[List[float]] = None,
        num_threads: int = 1,
        api_key: Optional[str] = None,
        additional_sampling_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[RequestResult]]:
        """Multi-turn text conversations over the native chains engine.

        For each chain, turn K's request template carries the full
        conversation: the known user texts of every prior turn interleaved with
        HOLES for the assistant outputs, which NATIVE fills with turn results
        (JSON-escaped, Content-Length recomputed) the instant they exist — so
        history injection never touches Python between turns.
        ``chain_delays_s[c][k]`` is turn K's think time after turn K-1
        completes; ``chain_history[c][k]`` says whether turn K inherits history
        (mirrors the session graph's is_history_parent edges).

        Returns per-chain lists of per-turn RequestResults (evaluator-ready).
        """
        import uuid

        hole_token = f"@@VEEKSHA-HOLE-{uuid.uuid4().hex}"
        auth = f"Authorization: Bearer {api_key}\r\n" if api_key else ""
        header_prefix = (
            f"POST {path} HTTP/1.1\r\nHost: {self.host}\r\n"
            f"Content-Type: application/json\r\n{auth}Content-Length: "
        )
        header_suffix = "\r\nConnection: close\r\n\r\n"

        def turn_text(request: Request) -> str:
            content = request.channels.get(ChannelModality.TEXT)
            return (
                content.input_text
                if isinstance(content, TextChannelRequestContent)
                else ""
            )

        spec_chains: List[List[NativeChainTurnSpec]] = []
        for ci, chain in enumerate(chains):
            specs: List[NativeChainTurnSpec] = []
            texts = [turn_text(r) for r in chain]
            for k, request in enumerate(chain):
                messages: List[Dict[str, Any]] = []
                hole_refs: List[int] = []
                if chain_history[ci][k]:
                    for j in range(k):
                        messages.append({"role": "user", "content": texts[j]})
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"{hole_token}-{len(hole_refs)}@@",
                            }
                        )
                        hole_refs.append(j)
                messages.append({"role": "user", "content": texts[k]})
                payload: Dict[str, Any] = {
                    "model": model,
                    "stream": True,
                    "max_completion_tokens": _target_tokens(
                        request, default_max_tokens
                    ),
                    "messages": messages,
                }
                if additional_sampling_params:
                    payload.update(additional_sampling_params)
                serialized = json.dumps(payload)
                segments: List[str] = []
                rest = serialized
                for h in range(len(hole_refs)):
                    token = f"{hole_token}-{h}@@"
                    idx = rest.find(token)
                    segments.append(rest[:idx])
                    rest = rest[idx + len(token) :]
                segments.append(rest)
                specs.append(
                    NativeChainTurnSpec(
                        header_prefix=header_prefix,
                        header_suffix=header_suffix,
                        body_segments=segments,
                        hole_refs=hole_refs,
                        delay_s=chain_delays_s[ci][k],
                    )
                )
            spec_chains.append(specs)

        anchor = time.monotonic()
        results = self._sse.run_chains(
            spec_chains,
            concurrency=concurrency,
            timeout_s=timeout_s,
            modality=ChannelModality.TEXT,
            start_offsets_s=start_offsets_s,
            num_threads=num_threads,
        )
        out: List[List[RequestResult]] = []
        for chain, res in zip(chains, results):
            turn_results: List[RequestResult] = []
            for k, request in enumerate(chain):
                if k < len(res.turns):
                    turn = res.turns[k]
                    success = not res.error and len(turn.stream) > 0
                    channels = {}
                    if len(turn.stream) > 0:
                        tte = turn.stream.time_to_first_event() or 0.0
                        inter_chunk_times = [tte] + turn.stream.inter_event_deltas()
                        channels[ChannelModality.TEXT] = ChannelResponse(
                            modality=ChannelModality.TEXT,
                            content=turn.content,
                            metrics={
                                "is_stream": True,
                                "inter_chunk_times": inter_chunk_times,
                                "num_output_tokens": len(turn.stream),
                                "num_total_prompt_tokens": 0,
                                "num_delta_prompt_tokens": 0,
                            },
                        )
                    dispatched_at, completed_at = _anchored_times(
                        anchor, turn.dispatch_offset_s, turn.stream
                    )
                    turn_results.append(
                        _request_result(
                            request,
                            channels,
                            success,
                            turn.status,
                            res.error,
                            dispatched_at,
                            completed_at,
                        )
                    )
                else:
                    turn_results.append(
                        _request_result(
                            request, {}, False, 0, res.error or "turn not reached"
                        )
                    )
            out.append(turn_results)
        return out

    # ------------------------------------------------------------- realtime TTS
    def run_realtime_tts(
        self,
        requests: List[Request],
        concurrency: int,
        model: str = "tts-1",
        path: str = "/v1/realtime",
        sample_rate: int = 24000,
        tokens_per_delta: int = 3,
        timeout_s: float = 30.0,
        num_threads: int = 1,
    ) -> List[RequestResult]:
        """Pace each request's input text as realtime deltas and collect the
        audio-delta timeline for the audio evaluator (TTFA / RTF). All requests
        run concurrently (native owns the connection pool)."""
        req_messages: List[List[str]] = []
        req_offsets: List[List[float]] = []
        texts: List[str] = []
        for request in requests:
            content = request.channels.get(ChannelModality.TEXT)
            text = (
                content.input_text
                if isinstance(content, TextChannelRequestContent)
                else ""
            )
            texts.append(text)
            messages, offsets = _tts_messages(
                text, model, sample_rate, tokens_per_delta
            )
            req_messages.append(messages)
            req_offsets.append(offsets)

        anchor = time.monotonic()
        results = self._ws.stream_batch(
            path,
            req_messages,
            concurrency=concurrency,
            req_offsets_s=req_offsets,
            timeout_s=timeout_s,
            modality=ChannelModality.AUDIO,
            num_threads=num_threads,
            # a realtime server keeps the session open after finishing a
            # response, so the stream ends on the terminal event, exactly where
            # the Python realtime client stops.
            done_markers=['"response.done"'],
        )
        return [
            _tts_request_result(request, res, sample_rate, text, anchor)
            for request, res, text in zip(requests, results, texts)
        ]

    # --------------------------------------------------------------------- STT
    def run_stt(
        self,
        requests: List[Request],
        concurrency: int,
        path: str = "/v1/realtime",
        sample_rate: int = 16000,
        chunk_ms: float = 100.0,
        timeout_s: float = 30.0,
        num_threads: int = 1,
    ) -> List[RequestResult]:
        """Pace each request's input audio as append frames and collect the
        transcript deltas for the STT evaluator (WER + interactivity). All
        requests run concurrently (native owns the connection pool)."""
        req_messages: List[List[str]] = []
        req_offsets: List[List[float]] = []
        pcm_lens: List[int] = []
        for request in requests:
            content = request.channels.get(ChannelModality.AUDIO)
            pcm = _load_pcm(content, sample_rate)
            pcm_lens.append(len(pcm))
            messages, offsets = _stt_messages(pcm, sample_rate, chunk_ms)
            req_messages.append(messages)
            req_offsets.append(offsets)

        anchor = time.monotonic()
        results = self._ws.stream_batch(
            path,
            req_messages,
            concurrency=concurrency,
            req_offsets_s=req_offsets,
            timeout_s=timeout_s,
            modality=ChannelModality.AUDIO,
            num_threads=num_threads,
            done_markers=['"transcription.done"'],
        )
        return [
            _stt_request_result(request, res, sample_rate, pcm_len, anchor)
            for request, res, pcm_len in zip(requests, results, pcm_lens)
        ]


# --------------------------------------------------------------------- helpers
def _target_tokens(request: Request, default: int) -> int:
    spec = getattr(request, "requested_output", None)
    text_spec = getattr(spec, "text", None) if spec is not None else None
    target = getattr(text_spec, "target_tokens", None) if text_spec else None
    return int(target) if target else default


def _anchored_times(
    anchor: float, dispatch_offset_s: float, stream: Any
) -> Tuple[float, float]:
    """Monotonic dispatch/completion times for a native result.

    ``anchor`` is time.monotonic() taken just before the engine call (the engine
    starts its own run clock within call overhead of it). Completion adds the
    last event's stream offset; the connect phase between dispatch and send is
    not separately reported, so completion is a lower bound by that amount.
    """
    dispatched_at = anchor + dispatch_offset_s
    last_offset = stream.events[-1].offset_s if len(stream) > 0 else 0.0
    return dispatched_at, dispatched_at + last_offset


def _request_result(
    request: Request,
    channels: Dict[ChannelModality, ChannelResponse],
    success: bool,
    status: int,
    error: str,
    dispatched_at: Optional[float] = None,
    completed_at: Optional[float] = None,
) -> RequestResult:
    return RequestResult(
        request_id=request.id,
        session_id=getattr(request, "session_id", request.id),
        channels=channels,
        success=success,
        error_code=None if success else (status or 500),
        error_msg=error or None,
        scheduler_dispatched_at=dispatched_at,
        client_picked_up_at=dispatched_at,
        client_completed_at=completed_at,
    )


def _tts_messages(text, model, sample_rate, tokens_per_delta):
    """session.update, paced text deltas, response.create + their send offsets."""
    words = text.split()
    segments = [
        " ".join(words[i : i + tokens_per_delta])
        for i in range(0, max(1, len(words)), tokens_per_delta)
    ] or [text]
    session = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {"output": {"format": {"type": "audio/pcm", "rate": sample_rate}}},
        },
    }
    messages = [json.dumps(session)]
    offsets = [0.0]
    # pace deltas at ~50ms apart (a stand-in cadence; timing precision is native)
    t = 0.0
    for seg in segments:
        t += 0.05
        messages.append(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": seg}],
                    },
                }
            )
        )
        offsets.append(t)
    t += 0.05
    messages.append(
        json.dumps(
            {"type": "response.create", "response": {"output_modalities": ["audio"]}}
        )
    )
    offsets.append(t)
    return messages, offsets


def _tts_request_result(request, res, sample_rate, text, anchor) -> RequestResult:
    """Parse audio-delta frames into the audio evaluator's realtime dialect."""
    timeline: List[List[float]] = []
    audio = bytearray()
    for offset_s, payload in res.timed_frames():
        try:
            event = json.loads(payload)
        except json.JSONDecodeError, ValueError:
            continue
        if event.get("type") == "response.output_audio.delta":
            delta = event.get("delta")
            if delta:
                chunk = base64.b64decode(delta)
                audio.extend(chunk)
                timeline.append([offset_s * 1000.0, len(chunk)])
    success = res.success and bool(timeline)
    metrics = {
        ac.AUDIO_TASK: AudioTask.TTS,
        ac.AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: timeline,
        ac.SAMPLE_RATE: sample_rate,
        ac.AudioMetricKey.TTFC.value: timeline[0][0] if timeline else 0.0,
        ac.AudioMetricKey.CHUNK_COUNT.value: len(timeline),
        ac.AudioMetricKey.RAW_PCM.value: True,
    }
    channels = {}
    if timeline:
        channels[ChannelModality.AUDIO] = ChannelResponse(
            modality=ChannelModality.AUDIO, content=bytes(audio), metrics=metrics
        )
    dispatched_at, completed_at = _anchored_times(
        anchor, res.dispatch_offset_s, res.stream
    )
    return _request_result(
        request, channels, success, 0, res.error, dispatched_at, completed_at
    )


def _stt_messages(pcm: bytes, sample_rate: int, chunk_ms: float):
    """session.update, paced input_audio_buffer.append frames, commit + offsets."""
    bytes_per_chunk = max(2, int(sample_rate * 2 * chunk_ms / 1000.0))
    messages = [json.dumps({"type": "session.update", "model": "stt"})]
    offsets = [0.0]
    t = 0.0
    for i in range(0, len(pcm), bytes_per_chunk):
        chunk = pcm[i : i + bytes_per_chunk]
        messages.append(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )
        offsets.append(t)  # real-time pace: chunk i sent at i*chunk_ms
        t += chunk_ms / 1000.0
    messages.append(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
    offsets.append(t)
    return messages, offsets


def _audio_send_offsets_ms(sent_offsets_s: List[float]) -> List[float]:
    """Per-audio-frame send stamps, re-based on the first audio frame.

    Drops the leading session.update and the trailing commit that
    ``_stt_messages`` wraps the audio in, so index i is audio chunk i — the
    same contract the Python client's ``send_offsets_ms`` follows.
    """
    if len(sent_offsets_s) <= 2:
        return []
    audio = sent_offsets_s[1:-1]
    origin = audio[0]
    return [(s - origin) * 1000.0 for s in audio]


def _stt_request_result(request, res, sample_rate, pcm_bytes, anchor) -> RequestResult:
    """Parse transcript-delta frames into the STT evaluator's dialect."""
    final_transcript = ""
    snapshots: List[Dict[str, Any]] = []
    ttfc = None
    for offset_s, payload in res.timed_frames():
        try:
            event = json.loads(payload)
        except json.JSONDecodeError, ValueError:
            continue
        etype = event.get("type")
        if etype == "transcription.delta":
            delta = event.get("delta", "")
            final_transcript = (final_transcript + delta).strip()
            snapshots.append(
                {"elapsed_ms": offset_s * 1000.0, "transcript": final_transcript}
            )
            if ttfc is None:
                ttfc = offset_s * 1000.0
        elif etype == "transcription.done":
            text = event.get("text")
            if text:
                final_transcript = str(text).strip()
    success = res.success and bool(final_transcript)
    metrics: Dict[str, Any] = {
        ac.AUDIO_TASK: AudioTask.STT,
        ac.SAMPLE_RATE: sample_rate,
        ac.PCM_BYTE_COUNT: pcm_bytes,
        "final_transcript": final_transcript,
        "transcript_snapshots": snapshots,
        "time_to_first_visible_text": ttfc,
        "raw_pcm": True,
        # The engine's own per-chunk dispatch stamps, same field AND same
        # definition as the Python STT client: one stamp per AUDIO frame,
        # measured from the pacing origin. _stt_messages brackets the appends
        # with a session.update and a closing commit, which are not audio and
        # would shift every index by one if left in.
        "send_offsets_ms": _audio_send_offsets_ms(res.sent_offsets_s),
    }
    # ground truth + dataset metadata flow from the trace generator's request.
    for key, value in (request.metadata or {}).items():
        metrics.setdefault(key, value)
    channels = {}
    if final_transcript:
        channels[ChannelModality.AUDIO] = ChannelResponse(
            modality=ChannelModality.AUDIO, content=final_transcript, metrics=metrics
        )
    dispatched_at, completed_at = _anchored_times(
        anchor, res.dispatch_offset_s, res.stream
    )
    return _request_result(
        request, channels, success, 0, res.error, dispatched_at, completed_at
    )


def _load_pcm(content: Optional[Any], sample_rate: int) -> bytes:
    """Read 16-bit mono PCM from the request's audio (WAV path or raw bytes)."""
    if isinstance(content, AudioChannelRequestContent):
        audio = content.input_audio
    else:
        audio = content
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    if isinstance(audio, str):
        import wave

        with wave.open(audio, "rb") as w:
            return w.readframes(w.getnframes())
    return b""
