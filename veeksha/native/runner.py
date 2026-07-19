"""Native benchmark execution path — run a request batch over the native
transport and score it with the normal evaluator.

This is the entry point a benchmark uses when ``client.use_native_transport`` is
set: it selects the per-modality native transport method, runs the requests
(native owns concurrency), feeds each result to the existing evaluator, and
finalizes — so the native path produces the SAME EvaluationResult a Python run
would, just with the Python per-event overhead removed from the hot path.

Selection is guarded: native is used only when the extension is built AND the
endpoint is plaintext (see ``should_use_native``); otherwise callers keep the
Python transport.
"""

from __future__ import annotations

from typing import Any, List, Optional
from urllib.parse import urlparse

from veeksha.core.request import Request
from veeksha.native.engine import native_available
from veeksha.native.transport import NativeTransport
from veeksha.types import ChannelModality, ClientType

# ClientTypes the native transport can serve, and how.
_TEXT_TYPES = {ClientType.OPENAI_CHAT_COMPLETIONS, ClientType.OPENAI_COMPLETIONS}


def should_use_native(client_config: Any) -> bool:
    """Whether a benchmark should route this client through the native transport."""
    if not getattr(client_config, "use_native_transport", False):
        return False
    if not native_available():
        return False
    api_base = getattr(client_config, "api_base", None)
    if not api_base:
        return True  # host/port supplied elsewhere; assume plaintext
    scheme = urlparse(api_base).scheme.lower()
    return scheme in ("http", "ws", "")  # plaintext only; TLS -> Python


def _host_port(api_base: str) -> tuple[str, int]:
    parsed = urlparse(api_base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme in ("https", "wss") else 80)
    return host, port


def _client_task(client_config: Any) -> str:
    ctype = client_config.get_type()
    if ctype in _TEXT_TYPES:
        return "text"
    if ctype == ClientType.REALTIME_TTS:
        return "tts"
    if ctype == ClientType.STT:
        return "stt"
    raise ValueError(
        f"native transport does not support client type {ctype!r} "
        "(supported: OpenAI chat/completions, realtime TTS, STT)"
    )


def compute_dispatch_offsets(
    sessions: List[Any], traffic_config: Any, seed_manager: Any
) -> Optional[List[float]]:
    """Per-request arrival offsets (seconds) for rate-based (open-loop) traffic.

    Reproduces the rate scheduler's schedule: each session starts at the running
    sum of interarrival intervals (from the same seeded interval generator), and
    each request inherits its node's wait_after_ready. Returns None for
    non-rate traffic (closed-loop), where native fills concurrency + refills.
    Aligned with the flattened request order used by ``execute_native``.
    """
    from veeksha.config.traffic import RateTrafficConfig

    if not isinstance(traffic_config, RateTrafficConfig):
        return None
    from veeksha.generator.interval.registry import IntervalGeneratorRegistry

    interval_gen = IntervalGeneratorRegistry.get(
        traffic_config.interval_generator.get_type(),
        traffic_config.interval_generator,
        rng=seed_manager.numpy_factory("interval")(),
    )
    offsets: List[float] = []
    start = 0.0
    for session in sessions:
        graph = getattr(session, "session_graph", None)
        for node_id in session.requests:
            wait = 0.0
            if graph is not None and hasattr(graph, "nodes"):
                node = graph.nodes.get(node_id)
                wait = getattr(node, "wait_after_ready", 0.0) if node else 0.0
            offsets.append(start + wait)
        start += interval_gen.get_next_interval()
    return offsets


def execute_native(
    requests: List[Request],
    client_config: Any,
    concurrency: int,
    timeout_s: float = 120.0,
    dispatch_offsets_s: Optional[List[float]] = None,
) -> List[Any]:
    """Run requests through the native transport for the client's modality."""
    host, port = _host_port(client_config.api_base or "http://127.0.0.1:80")
    transport = NativeTransport(host, port)
    task = _client_task(client_config)
    if task == "text":
        return transport.run_text(
            requests,
            concurrency=concurrency,
            model=getattr(client_config, "model", "dummy") or "dummy",
            timeout_s=timeout_s,
            dispatch_offsets_s=dispatch_offsets_s,
        )
    if task == "tts":
        return transport.run_realtime_tts(
            requests,
            concurrency=concurrency,
            model=getattr(client_config, "model", "tts-1") or "tts-1",
            sample_rate=getattr(client_config, "sample_rate", 24000),
            timeout_s=timeout_s,
        )
    return transport.run_stt(
        requests,
        concurrency=concurrency,
        sample_rate=getattr(client_config, "sample_rate", 16000),
        timeout_s=timeout_s,
    )


def feed_native_results(
    requests: List[Request],
    evaluator: Any,
    client_config: Any,
    concurrency: int,
    timeout_s: float = 120.0,
    dispatch_offsets_s: Optional[List[float]] = None,
) -> int:
    """Run a request batch natively and feed each result to ``evaluator`` (no
    finalize). Returns the number of successful requests."""
    results = execute_native(
        requests, client_config, concurrency, timeout_s, dispatch_offsets_s
    )
    n_ok = 0
    for result in results:
        evaluator.register_request(
            request_id=result.request_id,
            session_id=result.session_id,
            dispatched_at=0.0,
            channels=result.channels,
        )
        evaluator.record_request_completed(
            request_id=result.request_id,
            session_id=result.session_id,
            completed_at=0.0,
            response=result,
        )
        if getattr(result, "success", False):
            n_ok += 1
    return n_ok


def run_native_benchmark(
    requests: List[Request],
    evaluator: Any,
    client_config: Any,
    concurrency: int,
    timeout_s: float = 120.0,
) -> Any:
    """Execute a request batch natively and score it with ``evaluator``.

    Returns the evaluator's finalized EvaluationResult — identical shape to a
    Python-transport run.
    """
    feed_native_results(requests, evaluator, client_config, concurrency, timeout_s)
    return evaluator.finalize()
