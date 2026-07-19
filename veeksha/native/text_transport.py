"""Native transport for text (LLM) requests — the P2 client-integration seam.

Bridges veeksha ``Request`` objects to the native receive engine and back to
``RequestResult``s the existing TextPerformanceEvaluator consumes unchanged. This
is the submit/drain model the docs call for: Python owns *what* the requests are
(payload), native owns *when* each chunk arrives (concurrency + kernel-time
receive timing). One native poll() loop replaces the per-stream asyncio coroutines
on the receive hot path.

The Python worker pipeline still drives open-loop per-request today; this seam is
how a native-backed text client submits a batch and drains finished timelines
without a coroutine per stream.
"""

from __future__ import annotations

import json
from typing import List, Optional

from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.native.engine import NativeReceiveEngine, NativeRequest
from veeksha.types import ChannelModality


def _chat_body(text: str, model: str, max_tokens: int) -> str:
    return json.dumps(
        {
            "model": model,
            "stream": True,
            "max_completion_tokens": max_tokens,
            "messages": [{"role": "user", "content": text}],
        }
    )


def _target_tokens(request: Request, default: int) -> int:
    spec = getattr(request, "requested_output", None)
    text_spec = getattr(spec, "text", None) if spec is not None else None
    target = getattr(text_spec, "target_tokens", None) if text_spec else None
    return int(target) if target else default


def run_text_requests(
    requests: List[Request],
    host: str,
    port: int,
    concurrency: int,
    model: str = "dummy",
    path: str = "/v1/chat/completions",
    default_max_tokens: int = 16,
    timeout_s: float = 120.0,
) -> List[RequestResult]:
    """Execute text requests through native transport; return RequestResults.

    Each result carries a TEXT ChannelResponse with the ``inter_chunk_times`` +
    ``num_output_tokens`` the text evaluator reads — derived from the native
    per-chunk timeline (offset[0] is TTFC; the rest are inter-chunk gaps).
    """
    native_requests: List[NativeRequest] = []
    for request in requests:
        text_content = request.channels.get(ChannelModality.TEXT)
        text = (
            text_content.input_text
            if isinstance(text_content, TextChannelRequestContent)
            else ""
        )
        body = _chat_body(text, model, _target_tokens(request, default_max_tokens))
        native_requests.append(NativeRequest(path=path, body=body))

    engine = NativeReceiveEngine(host, port)
    results = engine.run(
        native_requests,
        concurrency=concurrency,
        sse=True,
        timeout_s=timeout_s,
        modality=ChannelModality.TEXT,
    )

    request_results: List[RequestResult] = []
    for request, native_result in zip(requests, results):
        stream = native_result.stream
        success = native_result.success and len(stream) > 0
        channels = {}
        if len(stream) > 0:
            tte = stream.time_to_first_event() or 0.0
            inter_chunk_times = [tte] + stream.inter_event_deltas()
            channels[ChannelModality.TEXT] = ChannelResponse(
                modality=ChannelModality.TEXT,
                content=native_result.content,
                metrics={
                    "is_stream": True,
                    "inter_chunk_times": inter_chunk_times,
                    "num_output_tokens": len(stream),
                    "num_total_prompt_tokens": 0,
                    "num_delta_prompt_tokens": 0,
                },
            )
        request_results.append(
            RequestResult(
                request_id=request.id,
                session_id=getattr(request, "session_id", request.id),
                channels=channels,
                success=success,
                error_code=None if success else (native_result.status or 500),
                error_msg=native_result.error or None,
            )
        )
    return request_results
