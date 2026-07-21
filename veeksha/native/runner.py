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
    try:
        _client_task(client_config)
    except ValueError as exc:  # unsupported client type -> Python, not a crash
        import logging

        logging.getLogger(__name__).warning("Native transport skipped: %s", exc)
        return False
    except AttributeError:
        pass  # config exposes no get_type (e.g. test stubs); resolved at execute
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


# Concurrency at which a single poll loop services enough sockets per poll() that
# userspace read-batching drift becomes visible; above it a second shard roughly
# halves that drift (measured against a sub-ms native reference) while staying
# safe against a co-located server.
_AUTO_SHARD_CONCURRENCY = 200


def _resolve_native_threads(configured: int, concurrency: int) -> int:
    """Resolve native_threads, expanding 0 (=auto) by concurrency."""
    configured = int(configured)
    if configured > 0:
        return configured
    return 2 if concurrency >= _AUTO_SHARD_CONCURRENCY else 1


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


def session_chain(session: Any) -> Optional[List[int]]:
    """The session's node ids in chain order, or None if not a linear chain.

    Native multi-turn supports exactly the linear-conversation shape: one root
    with wait_after_ready 0, every node at most one child, every non-root
    exactly one parent. Anything else (DAGs, delayed roots) stays on Python.
    """
    from veeksha.core.session_graph import children, parents

    graph = getattr(session, "session_graph", None)
    if graph is None or not getattr(graph, "nodes", None):
        return None
    roots = [nid for nid in graph.nodes if not parents(graph, nid)]
    if len(roots) != 1:
        return None
    root = roots[0]
    if getattr(graph.nodes[root], "wait_after_ready", 0.0) != 0.0:
        return None
    order = [root]
    seen = {root}
    node = root
    while True:
        kids = children(graph, node)
        if not kids:
            break
        if len(kids) > 1:
            return None
        node = kids[0].dst
        if node in seen:
            return None  # cycle
        seen.add(node)
        order.append(node)
    if len(order) != len(graph.nodes) or len(order) != len(session.requests):
        return None
    return order


def sessions_chain_eligible(sessions: List[Any]) -> bool:
    """Whether every session is a linear TEXT chain the native engine can run."""
    for session in sessions:
        order = session_chain(session)
        if order is None:
            return False
        for nid in order:
            request = session.requests.get(nid)
            if request is None:
                return False
            channels = getattr(request, "channels", {})
            if list(channels.keys()) != [ChannelModality.TEXT]:
                return False
    return True


def build_chains(sessions: List[Any]):
    """Flatten linear sessions into (chains, delays, history_flags) for the
    native chains engine. Turn K's delay is its node's wait_after_ready (think
    time after turn K-1 completes); its history flag mirrors the incoming
    edge's is_history_parent."""
    from veeksha.core.session_graph import parents

    chains: List[List[Any]] = []
    delays: List[List[float]] = []
    history: List[List[bool]] = []
    for session in sessions:
        order = session_chain(session)
        assert order is not None, "caller must check sessions_chain_eligible"
        graph = session.session_graph
        chain, chain_delay, chain_hist = [], [], []
        for k, nid in enumerate(order):
            request = session.requests[nid]
            request.session_id = session.id
            chain.append(request)
            chain_delay.append(getattr(graph.nodes[nid], "wait_after_ready", 0.0))
            if k == 0:
                chain_hist.append(False)
            else:
                incoming = parents(graph, nid)
                chain_hist.append(bool(incoming and incoming[0].is_history_parent))
        chains.append(chain)
        delays.append(chain_delay)
        history.append(chain_hist)
    return chains, delays, history


def compute_chain_start_offsets(
    sessions: List[Any], traffic_config: Any, seed_manager: Any
) -> Optional[List[float]]:
    """Per-SESSION arrival offsets (seconds) for rate-based multi-turn traffic.

    Chain starts reproduce the rate scheduler's seeded interarrival schedule;
    intra-chain timing (think time) is owned by the engine's per-turn delays.
    Returns None for closed-loop traffic.
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
    for _session in sessions:
        offsets.append(start)
        start += interval_gen.get_next_interval()
    return offsets


def feed_native_chain_results(
    sessions: List[Any],
    evaluator: Any,
    client_config: Any,
    concurrency: int,
    timeout_s: float = 120.0,
    start_offsets_s: Optional[List[float]] = None,
) -> int:
    """Run linear multi-turn sessions natively (history flows inside the
    engine) and feed each turn's result to ``evaluator``. Returns the number of
    successful turns."""
    host, port = _host_port(client_config.api_base or "http://127.0.0.1:80")
    transport = NativeTransport(host, port)
    chains, delays, history = build_chains(sessions)
    num_threads = _resolve_native_threads(
        getattr(client_config, "native_threads", 0), concurrency
    )
    chain_results = transport.run_text_chains(
        chains,
        chain_delays_s=delays,
        chain_history=history,
        concurrency=concurrency,
        model=getattr(client_config, "model", "mock") or "mock",
        timeout_s=timeout_s,
        start_offsets_s=start_offsets_s,
        num_threads=num_threads,
        api_key=getattr(client_config, "api_key", None),
        additional_sampling_params=getattr(
            client_config, "additional_sampling_params_dict", None
        ),
    )
    n_ok = 0
    for turn_results in chain_results:
        for result in turn_results:
            dispatched_at = result.scheduler_dispatched_at or 0.0
            completed_at = result.client_completed_at or dispatched_at
            evaluator.register_request(
                request_id=result.request_id,
                session_id=result.session_id,
                dispatched_at=dispatched_at,
                channels=result.channels,
            )
            evaluator.record_request_completed(
                request_id=result.request_id,
                session_id=result.session_id,
                completed_at=completed_at,
                response=result,
            )
            if result.success:
                n_ok += 1
    return n_ok


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
    num_threads = _resolve_native_threads(
        getattr(client_config, "native_threads", 0), concurrency
    )
    if task == "text":
        return transport.run_text(
            requests,
            concurrency=concurrency,
            model=getattr(client_config, "model", "mock") or "mock",
            timeout_s=timeout_s,
            dispatch_offsets_s=dispatch_offsets_s,
            num_threads=num_threads,
            api_key=getattr(client_config, "api_key", None),
            additional_sampling_params=getattr(
                client_config, "additional_sampling_params_dict", None
            ),
        )
    if task == "tts":
        return transport.run_realtime_tts(
            requests,
            concurrency=concurrency,
            model=getattr(client_config, "model", "tts-1") or "tts-1",
            sample_rate=getattr(client_config, "sample_rate", 24000),
            timeout_s=timeout_s,
            num_threads=num_threads,
        )
    return transport.run_stt(
        requests,
        concurrency=concurrency,
        sample_rate=getattr(client_config, "sample_rate", 16000),
        timeout_s=timeout_s,
        num_threads=num_threads,
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
        # Real monotonic anchors (set by NativeTransport from the engine's
        # dispatch offsets) — duration/throughput metrics stay meaningful.
        dispatched_at = result.scheduler_dispatched_at or 0.0
        completed_at = result.client_completed_at or dispatched_at
        evaluator.register_request(
            request_id=result.request_id,
            session_id=result.session_id,
            dispatched_at=dispatched_at,
            channels=result.channels,
        )
        evaluator.record_request_completed(
            request_id=result.request_id,
            session_id=result.session_id,
            completed_at=completed_at,
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
