"""Probes that measure Veeksha's timing preflight at a given concurrency.

`probe_receive_drift` drives the REAL client/dispatch/completion pipeline against
the mock engine and reports how faithfully Veeksha recorded the (known) chunk
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
        model="mock",
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


async def _paced_send(k_chunks: int, chunk_dt: float, cpu: int):
    """Absolute-deadline pacing (the veeksha stt.py design).

    Returns (total_send_time, per_chunk_send_drift_ms) where each drift is the
    signed lateness of that chunk's dispatch vs its schedule — the per-dispatch
    precision that ASR interactivity depends on.
    """
    start = time.monotonic()
    per_chunk_drift_ms = []
    for i in range(k_chunks):
        _cpu(cpu)
        target = start + (i + 1) * chunk_dt
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        # how late did this chunk actually go out vs its scheduled slot?
        per_chunk_drift_ms.append((time.monotonic() - target) * 1000.0)
    return time.monotonic() - start, per_chunk_drift_ms


def probe_pacing(
    concurrency: int, clip_s: float, chunk_ms: float, cpu: int = 500
) -> Dict[str, float]:
    """Measure real-time send-pacing precision across `concurrency` senders.

    Models the ASR realtime-audio pacing loop: a clip of `clip_s` sent in
    `chunk_ms` slots should take `clip_s`, with each chunk dispatched on time.
    Returns both the aggregate stretch (p99 of total/ideal) AND the per-chunk
    send-drift (p99/max of |actual - scheduled| per dispatch), the granularity
    that interactivity is sensitive to.
    """
    chunk_dt = chunk_ms / 1000.0
    k = max(1, int(round(clip_s / chunk_dt)))

    async def _run():
        tasks = [
            asyncio.create_task(_paced_send(k, chunk_dt, cpu))
            for _ in range(concurrency)
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run())
    ratios = [total / clip_s for total, _ in results]
    per_chunk_abs = [abs(d) for _, drifts in results for d in drifts]
    return {
        "stretch_p99": _pct(ratios, 99),
        "send_drift_p99_ms": _pct(per_chunk_abs, 99),
        "send_drift_max_ms": max(per_chunk_abs) if per_chunk_abs else float("nan"),
    }


def probe_audio_transport(
    concurrency: int,
    num_chunks: int = 20,
    chunk_ms: float = 20.0,
    request_timeout: float = 30.0,
) -> Dict[str, float]:
    """Per-chunk receive drift on the REAL realtime-audio WebSocket transport.

    The audio analogue of ``probe_receive_drift`` (which covers SSE): drive the
    actual ``RealtimeTTSClient`` against a fixed-cadence mock realtime server at
    ``concurrency`` and measure how faithfully the client's own per-chunk arrival
    timeline reproduces the known ``chunk_ms`` cadence. This exercises the real WS
    receive path (framing, base64 decode, timestamping) rather than a synthetic
    model, so it catches per-token/per-chunk receive drift the interactivity
    metric is sensitive to.

    Returns p99/max of ``|observed inter-chunk gap - chunk_ms|`` (ms) and the
    achieved request count.
    """
    # Imported lazily: the client package pulls heavier deps that the rest of the
    # preflight (pure SSE + pacing) does not need.
    from veeksha.client.realtime_tts import RealtimeTTSClient
    from veeksha.config.client import RealtimeTTSClientConfig
    from veeksha.core.audio_contract import AudioMetricKey
    from veeksha.preflight.audio_server import MockRealtimeAudioServer

    chunk_dt_ms = chunk_ms
    server = MockRealtimeAudioServer(
        num_chunks=num_chunks, chunk_dt=chunk_ms / 1000.0
    ).start()
    try:
        config = RealtimeTTSClientConfig(
            api_base=f"http://127.0.0.1:{server.port}",
            model="preflight-tts",
            request_timeout=request_timeout,
        )
        client = RealtimeTTSClient(config)

        async def _run():
            tasks = [
                asyncio.create_task(
                    client.send_request(_single_text_request(i), session_id=i)
                )
                for i in range(concurrency)
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(_run())
        # the server's own emit lateness — so callers can tell whether high drift
        # is the client (real) or the mock server saturating (server-limited).
        server_jitter = server.server_jitter_p99_ms()
    finally:
        server.stop()

    inter_chunk_drift_ms: List[float] = []
    achieved = 0
    for result in results:
        if isinstance(result, BaseException) or not getattr(result, "success", False):
            continue
        channel = result.channels.get(ChannelModality.AUDIO)
        if channel is None:
            continue
        timeline = channel.metrics.get(AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value, [])
        if len(timeline) < 2:
            continue
        achieved += 1
        offsets_ms = [row[0] for row in timeline]
        for a, b in zip(offsets_ms, offsets_ms[1:]):
            inter_chunk_drift_ms.append(abs((b - a) - chunk_dt_ms))

    return {
        "achieved": float(achieved),
        "recv_drift_p99_ms": _pct(inter_chunk_drift_ms, 99),
        "recv_drift_max_ms": (
            max(inter_chunk_drift_ms) if inter_chunk_drift_ms else float("nan")
        ),
        "server_jitter_p99_ms": server_jitter,
    }


def _single_text_request(rid: int) -> Request:
    return Request(
        id=rid,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="preflight realtime audio transport probe request"
            )
        },
    )


def probe_stt_transport(
    concurrency: int,
    clip_s: float = 2.0,
    sample_rate: int = 16000,
    request_timeout: float = 30.0,
) -> Dict[str, float]:
    """Per-audio-chunk SEND drift on the REAL STT client's realtime pacing.

    For ASR the interactivity-critical drift is on the SEND side: veeksha must
    stream the input audio at 1x real time (a 90s clip should take 90s to send).
    We measure this at the ground-truth point — a mock server timestamps each
    ``input_audio_buffer.append`` on arrival — while driving the actual STTClient
    at ``concurrency``. So this exercises the real WS send path (encode + paced
    ws.send), not a synthetic model.

    Returns p99/max of ``|arrival offset - i * chunk_period|`` (ms) and the
    aggregate stretch (send span / ideal span; 1.0 == perfect real-time pacing).
    """
    import os
    import tempfile
    import wave

    from veeksha.client.stt import STTClient
    from veeksha.config.client import STTClientConfig
    from veeksha.core.request_content import AudioChannelRequestContent
    from veeksha.preflight.audio_server import MockSTTPreflightServer

    server = MockSTTPreflightServer().start()
    tmpdir = tempfile.mkdtemp()
    wav_path = os.path.join(tmpdir, "clip.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * int(sample_rate * clip_s))

    try:
        config = STTClientConfig(
            api_base=f"http://127.0.0.1:{server.port}",
            model="preflight-stt",
            provider="vllm_realtime",
            sample_rate=sample_rate,
            ws_realtime_pacing=True,
            request_timeout=request_timeout,
        )
        client = STTClient(config)
        chunk_bytes = config.ws_chunk_size

        def _req(i: int) -> Request:
            return Request(
                id=i,
                channels={
                    ChannelModality.AUDIO: AudioChannelRequestContent(
                        input_audio=wav_path
                    )
                },
                metadata={"expected_transcript": "x", "dataset": "preflight"},
            )

        async def _run():
            tasks = [
                asyncio.create_task(client.send_request(_req(i), session_id=i))
                for i in range(concurrency)
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run(_run())
    finally:
        server.stop()
        try:
            os.remove(wav_path)
            os.rmdir(tmpdir)
        except OSError:
            pass

    # chunk i should arrive at server at ~ i * chunk_period (1x real-time pacing)
    chunk_period_ms = chunk_bytes / 2.0 / sample_rate * 1000.0
    send_drift_ms: List[float] = []
    stretches: List[float] = []
    for arrivals in server.append_arrivals:
        if len(arrivals) < 2:
            continue
        for i, arrival in enumerate(arrivals):
            send_drift_ms.append(abs(arrival - i * chunk_period_ms))
        span = arrivals[-1] - arrivals[0]
        ideal = (len(arrivals) - 1) * chunk_period_ms
        if ideal > 0:
            stretches.append(span / ideal)
    return {
        "achieved": float(len(server.append_arrivals)),
        "send_drift_p99_ms": _pct(send_drift_ms, 99),
        "send_drift_max_ms": max(send_drift_ms) if send_drift_ms else float("nan"),
        "stretch_p99": _pct(stretches, 99),
    }
