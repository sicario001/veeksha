"""Probes that measure Veeksha's timing preflight at a given concurrency.

`probe_receive_drift` drives the REAL client/dispatch/completion pipeline against
the dummy engine and reports how faithfully Veeksha recorded the (known) chunk
cadence. `probe_pacing` measures whether this system can sustain accurate
real-time send pacing (as used for streaming-audio / ASR benchmarks) at a given
concurrency — a pure asyncio scheduling probe.
"""

from __future__ import annotations

import asyncio
import threading
import time
from queue import Queue
from typing import Dict, List

from veeksha.config.client import OpenAIChatCompletionsClientConfig
from veeksha.config.traffic import ConcurrentTrafficConfig
from veeksha.core.context import WorkerContext
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import SessionGraph, SessionNode, add_node
from veeksha.core.tokenizer import TokenizerHandle, TokenizerProvider
from veeksha.traffic.concurrent import ConcurrentTrafficScheduler
from veeksha.types import ChannelModality
from veeksha.workers.client_runner import ClientRunnerManager
from veeksha.workers.completion import CompletionWorker
from veeksha.workers.dispatch import DispatchWorker


def _whitespace_tokenizer_provider() -> TokenizerProvider:
    """A model-free tokenizer so the preflight needs no download (token counts
    are irrelevant to timing)."""
    handle = TokenizerHandle(
        count_tokens=lambda t: len(str(t).split()),
        decode=lambda ids: " ".join("x" for _ in ids),
        encode=lambda t: list(range(len(str(t).split()))),
        get_vocab=lambda: [0],
    )
    return TokenizerProvider({ChannelModality.TEXT: handle}, model_name="preflight")


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def _single_request_session(sid: int, num_chunks: int) -> Session:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    req = Request(
        id=sid,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="hi", target_prompt_tokens=1
            )
        },
        requested_output=RequestedOutputSpec(
            text=TextOutputSpec(target_tokens=num_chunks)
        ),
    )
    return Session(id=sid, session_graph=graph, requests={0: req})


class _CollectEvaluator:
    """Minimal evaluator satisfying the worker interface; stores results."""

    def __init__(self):
        self.results = []
        self._lock = threading.Lock()

    def register_request(
        self, request_id, session_id, dispatched_at, channels, requested_output
    ):
        return

    def record_request_completed(
        self, request_id, session_id, completed_at, response, error=None
    ):
        with self._lock:
            self.results.append(response)


def probe_receive_drift(
    engine,
    concurrency: int,
    num_client_threads: int,
    num_dispatcher_threads: int,
    num_completion_threads: int,
    num_chunks: int,
    chunk_dt: float,
    prefill_s: float,
    n_requests: int,
    max_connections=None,
) -> Dict[str, float]:
    """Drive the real pipeline at `concurrency` and return drift metrics."""
    engine.reset_telemetry()

    client_cfg = OpenAIChatCompletionsClientConfig(
        api_base=f"http://127.0.0.1:{engine.port}/v1",
        api_key="none",
        model="dummy",
        request_timeout=120,
        max_connections=max_connections,
    )
    # Import lazily so this module imports even where the client's deps differ.
    from veeksha.client.openai_chat import OpenAIChatCompletionsClient

    client = OpenAIChatCompletionsClient(client_cfg, _whitespace_tokenizer_provider())

    sched = ConcurrentTrafficScheduler(
        ConcurrentTrafficConfig(
            target_concurrent_sessions=concurrency,
            rampup_seconds=0,
            cancel_session_on_failure=False,
        ),
        SeedManager(0),
    )
    sched.reset_reference_time()

    evaluator = _CollectEvaluator()
    client_queues = [Queue() for _ in range(num_client_threads)]
    output_queue: Queue = Queue()
    stop_event = threading.Event()

    client_runner = ClientRunnerManager(
        client=client,
        input_queues=client_queues,
        output_queue=output_queue,
        stop_event=stop_event,
        traffic_scheduler=sched,
    )
    dispatchers = [
        DispatchWorker(
            sched,
            client_queues,
            evaluator,
            WorkerContext(worker_id=i, stop_event=stop_event),
        )
        for i in range(num_dispatcher_threads)
    ]
    completers = [
        CompletionWorker(
            output_queue,
            sched,
            evaluator,
            WorkerContext(worker_id=i, stop_event=stop_event),
        )
        for i in range(num_completion_threads)
    ]
    threads = [
        threading.Thread(target=w.run, daemon=True) for w in (*dispatchers, *completers)
    ]

    client_runner.start()
    for t in threads:
        t.start()

    t0 = time.monotonic()
    for sid in range(n_requests):
        sched.schedule_session(_single_request_session(sid, num_chunks))

    deadline = (
        t0
        + 60.0
        + n_requests * (prefill_s + num_chunks * chunk_dt) / max(1, concurrency)
    )
    while len(evaluator.results) < n_requests and time.monotonic() < deadline:
        time.sleep(0.02)
    wall = time.monotonic() - t0
    completed = len(evaluator.results)

    stop_event.set()
    client_runner.stop()
    for _ in range(num_completion_threads):
        output_queue.put(None)
    for t in threads:
        t.join(timeout=2.0)
    client_runner.wait(timeout=3.0)

    results = list(evaluator.results)
    lo = int(0.10 * len(results))
    hi = int(0.95 * len(results)) or len(results)
    window = results[lo:hi]

    ivl_err: List[float] = []
    stretch: List[float] = []
    ttfc: List[float] = []
    # steady-state ideal: first-chunk -> last-chunk (excludes connect/prefill
    # noise, which is a separate concern from streaming-timing preflight).
    ideal_span = (num_chunks - 1) * chunk_dt
    for rr in window:
        if not rr.success:
            continue
        ch = rr.channels.get(ChannelModality.TEXT)
        if ch is None:
            continue
        ict = ch.metrics.get("inter_chunk_times") or []
        if len(ict) < 2:
            continue
        ttfc.append(ict[0] * 1000.0)
        for d in ict[1:]:
            ivl_err.append(abs(d - chunk_dt) * 1000.0)
        span = sum(ict[1:])  # first chunk -> last chunk
        stretch.append(span / ideal_span if ideal_span > 0 else float("nan"))

    return {
        "concurrency": concurrency,
        "achieved": engine.max_active_conns,
        "throughput": completed / wall if wall else 0.0,
        "ivl_err_p99_ms": _pct(ivl_err, 99),
        "stretch_p99": _pct(stretch, 99),
        "ttfc_p99_ms": _pct(ttfc, 99),
        "server_jitter_p99_ms": engine.server_jitter_p99_ms(),
    }


# --------------------------------------------------------------------- pacing
def _cpu(n: int) -> int:
    acc = 0
    for k in range(n):
        acc += k * k
    return acc


async def _paced_send(k_chunks: int, chunk_dt: float, cpu: int) -> float:
    """Absolute-deadline pacing (the veeksha stt.py design)."""
    start = time.monotonic()
    for i in range(k_chunks):
        _cpu(cpu)
        target = start + (i + 1) * chunk_dt
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
    return time.monotonic() - start


def probe_pacing(
    concurrency: int, clip_s: float, chunk_ms: float, cpu: int = 500
) -> float:
    """Return p99 of actual/ideal send time across `concurrency` paced senders.

    Models the ASR realtime-audio pacing loop: a clip of `clip_s` should take
    `clip_s` to send. Returns the p99 stretch ratio (1.0 = perfect).
    """
    chunk_dt = chunk_ms / 1000.0
    k = max(1, int(round(clip_s / chunk_dt)))

    async def _run():
        tasks = [
            asyncio.create_task(_paced_send(k, chunk_dt, cpu))
            for _ in range(concurrency)
        ]
        return await asyncio.gather(*tasks)

    totals = asyncio.run(_run())
    ratios = sorted(t / clip_s for t in totals)
    return ratios[min(len(ratios) - 1, int(0.99 * (len(ratios) - 1)))]
