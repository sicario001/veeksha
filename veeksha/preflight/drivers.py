"""Per-modality preflight workloads that drive the REAL benchmark pipeline.

Requirement 1.1 of the design is that the preflight certifies the harness *as
it ships*: not a bespoke asyncio probe that happens to use the same client, but
``benchmark._run_main_loop`` itself. So every workload here builds a genuine
:class:`~veeksha.config.benchmark.BenchmarkConfig`, resolves the client through
``ClientRegistry`` and the scheduler through ``TrafficSchedulerRegistry``, and
hands them to ``_run_main_loop``. That means each measurement exercises, unchanged:

* ``PrefetchWorker`` (+ ``SharedSessionCounter``, generator lock),
* the real ``ConcurrentTrafficScheduler`` — its single ``Condition``, its
  ready-heap, its refill-on-completion and session-graph release logic,
* ``DispatchWorker`` x N — evaluator registration, power-of-two queue choice,
  and the ``scheduler_ready_at`` / ``scheduler_dispatched_at`` stamps,
* ``ClientWorker`` x N (one asyncio loop each) — the dispatch-ticket gate, the
  ``client_picked_up_at`` stamp, and the real ``send_request`` of the real
  client,
* ``CompletionWorker`` x N — ``result_processed_at``, ``notify_completion``
  before evaluator recording,
* ``ThreadPoolManager`` and the termination monitor with its timeout/grace
  drain.

Only two things are substituted, and both sit *outside* the timing window by
construction (design §2.2):

1. the **server** is a mock with a known emit schedule — that is the whole
   point of the exercise;
2. the **session source** yields fixed, pre-built sessions instead of running
   the synthetic generator, and the **evaluator** only retains results.
   Session content preparation happens before dispatch and metric scoring
   happens after completion; neither is what the preflight certifies, and
   keeping them trivial also keeps the HF ``tokenizers`` import (which
   re-enables the GIL on free-threaded builds) out of a measurement run.

Scoring lives in :mod:`veeksha.preflight.scorer`; nothing in this module does
drift arithmetic.
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Set

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import (
    BaseClientConfig,
    OpenAIChatCompletionsClientConfig,
    RealtimeTTSClientConfig,
    STTClientConfig,
    TextPacingConfig,
    VajraTTSStreamClientConfig,
)
from veeksha.config.generator.interval import PoissonIntervalGeneratorConfig
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.traffic import (
    BaseTrafficConfig,
    ConcurrentTrafficConfig,
    RateTrafficConfig,
)
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import (
    AudioChannelRequestContent,
    TextChannelRequestContent,
)
from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
from veeksha.core.response import RequestResult
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import (
    SessionEdge,
    SessionGraph,
    SessionNode,
    add_edge,
    add_node,
)
from veeksha.evaluator.base import BaseEvaluator, EvaluationResult
from veeksha.logger import init_logger
from veeksha.preflight.audio_server import (
    MockRealtimeAudioServer,
    MockSTTPreflightServer,
)
from veeksha.preflight.mock_engine import MockStreamingEngine
from veeksha.preflight.vajra_server import MockVajraTTSStreamServer
from veeksha.preflight.scorer import (
    absolute_times_from_offsets_ms,
    asr_send_schedule_ms,
    audio_arrival_times,
    delivery_lag_ms,
    delivery_lags_ms,
    dispatch_drift_ms,
    p50,
    p99,
    rate_schedule_offsets_ms,
    send_drift,
    stt_send_times,
    text_arrival_times,
    text_pacing_schedule_ms,
    think_time_drift_ms,
)
from veeksha.preflight.sharded_server import format_pfid
from veeksha.types import ChannelModality
from veeksha.workers.prefetch import PrefetchWorker

logger = init_logger(__name__)

__all__ = [
    "CheckMetrics",
    "DispatchWorkload",
    "MultiTurnTextWorkload",
    "PreflightWorkload",
    "Measurement",
    "SttWorkload",
    "TextWorkload",
    "TtsWorkload",
    "VajraTtsWorkload",
    "build_workloads",
]

NAN = float("nan")

#: Benchmark seed for every preflight measurement. The dispatch check REPRODUCES the
#: rate scheduler's seeded arrival schedule, so this constant is part of that
#: check's definition, not an arbitrary number.
PREFLIGHT_SEED = 42

#: ``PrefetchWorker`` generates sessions unthrottled for its first
#: ``_BURST_DURATION_S`` seconds and then falls back to one session per
#: ``_MAX_POLL_INTERVAL_S`` (~20/s). A dispatch measurement that outlives the burst
#: window at a higher arrival rate measures the prefetch throttle instead of
#: dispatch accuracy, so the workload sizes itself to finish inside a safety
#: fraction of it — and warns loudly if it cannot.
BURST_WINDOW_S = PrefetchWorker._BURST_DURATION_S
BURST_SAFETY = 0.8
PREFETCH_THROTTLED_RATE_PER_S = 1.0 / PrefetchWorker._MAX_POLL_INTERVAL_S

#: Text-delta pacing the realtime-TTS / Vajra checks drive the input at. Slow
#: enough that audio (which the mocks begin on the first text delta) overlaps
#: the still-sending input. Shared between the client config and the C1
#: text-pacing scorer so the intended schedule is exactly what the client paced
#: to — ``fixed`` cadence (default), so the schedule is deterministic.
_TTS_TEXT_PACING = TextPacingConfig(tokens_per_second=20.0)


# --------------------------------------------------------------------- results
@dataclass(frozen=True)
class CheckMetrics:
    """One reported check derived from one measurement of one workload.

    A single dispatch can answer more than one question — ASR measures the send
    and receive halves of word interactivity from the same traffic — so a
    workload may return several of these from one run. Two views over one run
    beat two runs, because the halves then describe the same traffic.
    """

    name: str
    #: Column disambiguation + include/exclude statement, printed by the report.
    notes: str
    metrics: Dict[str, float]
    #: Named honesty terms, evaluated by the validator against its thresholds.
    #: Returning named booleans (rather than one) is what lets the report say
    #: *which* property failed.
    gate: Callable[[Dict[str, float], Any], Dict[str, bool]]


@dataclass
class Measurement:
    """Raw output of one measurement: results, server telemetry, wall time."""

    results: List[RequestResult]
    achieved: int
    wall_s: float
    #: The monotonic instant ``reset_reference_time()`` made the scheduler's
    #: zero. Every scheduled arrival time is an offset from exactly this, so
    #: the dispatch check needs it and an approximation would not do.
    scheduler_epoch: float = NAN
    #: The mock's per-request stamps, keyed by PFID, pulled after the run. C2
    #: and C3 pair the client's stamps against these; a request missing here is
    #: a correlation failure, not zero data.
    server_records: Dict[int, Any] = dc_field(default_factory=dict)
    #: Connections the mock accepted but could not key to a unique PFID — a
    #: measurement error the scorer must see rather than silently lose.
    unidentified_conns: int = 0
    extras: Dict[str, Any] = dc_field(default_factory=dict)


def _stamp(value: Any) -> Optional[float]:
    """A usable absolute monotonic stamp, or None (missing/NaN/non-numeric)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


# ------------------------------------------------------------------- pipeline
class _CollectingEvaluator(BaseEvaluator):
    """Evaluator that only retains results.

    The preflight certifies *timing capture*, not metric derivation, so the
    evaluator is reduced to the minimum surface ``_run_main_loop`` and the
    termination monitor actually call. Keeping it trivial also keeps evaluator
    CPU off the completion workers, so the multi-turn think-time drift reflects
    the harness's own release latency rather than someone else's scoring cost.
    """

    def __init__(self) -> None:
        # Deliberately not calling super().__init__: there is no evaluator
        # config to honour, and channel filtering would drop results.
        self.config = None
        self.seed_manager = None
        self._target_channels = None
        self._lock = threading.Lock()
        self.results: List[RequestResult] = []
        self._registered: Set[int] = set()
        self._completed = 0
        self._errored = 0

    # ---- pipeline surface
    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        channels: Dict[ChannelModality, Any],
        requested_output: Any = None,
    ) -> None:
        with self._lock:
            self._registered.add(request_id)

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
        error: Optional[Exception] = None,
    ) -> None:
        with self._lock:
            if isinstance(response, RequestResult):
                self.results.append(response)
            if error is not None or not getattr(response, "success", False):
                self._errored += 1
            else:
                self._completed += 1

    def record_session_completed(
        self, session_id: int, completed_at: float, success: bool
    ) -> None:
        return

    # ---- monitor surface
    def get_session_counts(self) -> tuple:
        with self._lock:
            return (self._completed, self._errored, 0)

    def get_completed_request_count(self) -> int:
        with self._lock:
            return self._completed + self._errored

    def get_registered_request_ids(self) -> Set[int]:
        with self._lock:
            return set(self._registered)

    def set_included_requests(self, request_ids: Set[int]) -> None:
        return

    # ---- unused but abstract
    def finalize(self) -> EvaluationResult:
        return EvaluationResult(evaluator_type="preflight", channel=None, metrics={})

    def save(self, output_dir: str) -> None:
        return


class _FixedSessionSource:
    """Session source yielding one single-request session per call.

    Structurally identical to what ``SyntheticSessionGenerator`` produces (a
    one-node graph with ``wait_after_ready=0``), minus the content generation:
    the preflight's request content is fixed by design, because the mock
    servers answer on a schedule that does not depend on it. Implements only
    what ``PrefetchWorker`` uses.
    """

    def __init__(self, make_request: Callable[[int], Request]) -> None:
        self._make_request = make_request
        self._next_id = 0
        self.config = None
        self.seed_manager = None

    def generate_session(self) -> Session:
        request_id = self._next_id
        self._next_id += 1
        graph = SessionGraph()
        add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
        request = self._make_request(request_id)
        return Session(id=request_id, session_graph=graph, requests={0: request})

    def capacity(self) -> int:
        return -1


class _MultiTurnSessionSource:
    """Session source yielding 2-turn TEXT sessions for the think-time check.

    Each session is a 2-node graph where node 1 is a child of node 0 with
    ``wait_after_ready = think_time_s`` — so the real scheduler releases turn 2
    at ``turn1 completion + think_time`` (that IS the intended schedule the
    check scores). The edge is deliberately NOT a history parent: the scheduling
    dependency is all the check needs, and injecting turn 1's messages into turn
    2's body would put a SECOND ``PFID`` marker there and break the mock's
    first-match correlation. Each turn carries its own ``PFID`` so the mock
    records both; request ids are unique across turns (``2*session_id + turn``).
    """

    def __init__(
        self, make_request: Callable[[int], Request], think_time_s: float
    ) -> None:
        self._make_request = make_request
        self._think_time_s = think_time_s
        self._next_session = 0
        self.config = None
        self.seed_manager = None

    def generate_session(self) -> Session:
        session_id = self._next_session
        self._next_session += 1
        graph = SessionGraph()
        add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
        add_node(graph, SessionNode(id=1, wait_after_ready=self._think_time_s))
        add_edge(graph, SessionEdge(src=0, dst=1, is_history_parent=False))
        requests = {
            0: self._make_request(2 * session_id),
            1: self._make_request(2 * session_id + 1),
        }
        return Session(id=session_id, session_graph=graph, requests=requests)

    def capacity(self) -> int:
        return -1


def _measurement_timeout_s(lifetime_s: float, n_requests: int, concurrency: int) -> int:
    """Safety timeout for one measurement.

    Generous on purpose: it is a deadlock guard, not a measurement bound. A
    measurement normally ends when the prefetch worker is exhausted and the scheduler
    has no pending work.
    """
    waves = math.ceil(max(1, n_requests) / max(1, concurrency))
    return int(max(60.0, 3.0 * lifetime_s * (waves + 1) + 30.0))


def build_benchmark_config(
    client_config: BaseClientConfig,
    *,
    concurrency: int,
    n_requests: int,
    lifetime_s: float,
    config,
    traffic_config: Optional[BaseTrafficConfig] = None,
    num_client_threads: Optional[int] = None,
) -> BenchmarkConfig:
    """A real ``BenchmarkConfig`` describing this measurement.

    The workload is expressed exactly the way a user would express it: a
    concurrency-scheduled run of ``n_requests`` single-turn sessions against
    ``client_config``. Nothing about the pipeline is special-cased for the
    preflight; only the endpoint it points at is a mock.

    ``traffic_config`` overrides the closed-loop default with an explicit
    traffic strategy — the dispatch check needs genuinely open-loop
    (rate-based) arrivals, because a deterministic arrival schedule is the only
    thing a dispatch time can be scored against.

    ``num_client_threads`` overrides the configured value. It exists for the
    rate-scheduled check: ``_run_main_loop`` auto-sizes the client pool from
    the scheduler's ``target_concurrent_sessions``, which only a closed-loop
    scheduler has, so an open-loop measurement would silently fall back to three
    client workers no matter what concurrency it offers.
    """
    return BenchmarkConfig(
        output_dir=os.path.join(tempfile.gettempdir(), "veeksha_preflight"),
        seed=PREFLIGHT_SEED,
        traffic_scheduler=traffic_config
        or ConcurrentTrafficConfig(
            target_concurrent_sessions=concurrency,
            # No rampup: the preflight measures steady state at the requested
            # concurrency, and a ramp would spend the short measurement climbing.
            rampup_seconds=0,
        ),
        client=client_config,
        runtime=RuntimeConfig(
            max_sessions=n_requests,
            benchmark_timeout=_measurement_timeout_s(
                lifetime_s, n_requests, concurrency
            ),
            post_timeout_grace_seconds=10,
            num_dispatcher_threads=config.num_dispatcher_threads,
            num_completion_threads=config.num_completion_threads,
            num_client_threads=(
                num_client_threads
                if num_client_threads is not None
                else config.num_client_threads
            ),
        ),
    )


def run_real_pipeline(
    benchmark_config: BenchmarkConfig,
    session_source: _FixedSessionSource,
    tokenizer_provider,
) -> tuple[List[RequestResult], float, float]:
    """Run ``_run_main_loop`` for one measurement.

    Returns ``(results, wall_seconds, scheduler_epoch)``. The epoch is the
    monotonic instant ``reset_reference_time()`` set as the scheduler's zero —
    the origin every scheduled arrival time is measured from. It is read back
    from the scheduler rather than approximated with a stamp taken before
    client/scheduler construction: that gap is milliseconds wide and the
    dispatch check gates on milliseconds.

    Imported lazily so that importing the preflight package does not drag in
    the whole benchmark module graph.
    """
    from veeksha.benchmark import _run_main_loop
    from veeksha.client.registry import ClientRegistry
    from veeksha.traffic.registry import TrafficSchedulerRegistry

    seed_manager = SeedManager(benchmark_config.seed)
    traffic_scheduler = TrafficSchedulerRegistry.get(
        benchmark_config.traffic_scheduler.get_type(),
        config=benchmark_config.traffic_scheduler,
        seed_manager=seed_manager,
    )
    client = ClientRegistry.get(
        benchmark_config.client.get_type(),
        config=benchmark_config.client,
        tokenizer_provider=tokenizer_provider,
    )
    evaluator = _CollectingEvaluator()

    started_at = time.monotonic()
    traffic_scheduler.reset_reference_time()
    # Every scheduler keeps its zero in `_start_monotonic` (concurrent.py:33,
    # rate.py:35, sequential_launch.py:42) and `reset_reference_time` just
    # restamps it; reading it back is exact, where a stamp taken here would be
    # a few microseconds late and one taken before construction milliseconds
    # early.
    epoch = float(getattr(traffic_scheduler, "_start_monotonic", started_at))
    _run_main_loop(
        session_generator=session_source,
        traffic_scheduler=traffic_scheduler,
        evaluator=evaluator,
        client=client,
        runtime_config=benchmark_config.runtime,
        benchmark_start_time=started_at,
    )
    return evaluator.results, time.monotonic() - started_at, epoch


# --------------------------------------------------------------- paired scoring
#: The float metric keys every reported check fills. A check leaves the ones it
#: does not measure at NaN, so the report renders them as a dash rather than a
#: misleading zero, and the gates (``value < threshold``) fail safe on them.
def _empty_metrics(achieved: int) -> Dict[str, float]:
    return {
        "achieved": float(achieved),
        "completed": 0.0,
        "offered": 0.0,
        "served_fraction": NAN,
        "unpaired_fraction": NAN,
        "c2_lag_p99_ms": NAN,
        "c2_lag_p50_ms": NAN,
        "c3_lag_p99_ms": NAN,
        "c3_lag_p50_ms": NAN,
        # STT only: per-append delivery lag (client send -> server append),
        # gated separately from the request-level C2 in the c2 column.
        "append_lag_p99_ms": NAN,
        "dispatch_drift_p99_ms": NAN,
        "min_lag_ms": NAN,
        # Multi-turn only: late-only think-time drift (client-side, gated) and
        # its server-arrival variant (reported), both left NaN by every other
        # check so the report renders a dash.
        "think_time_drift_p99_ms": NAN,
        "think_time_drift_p50_ms": NAN,
        "think_time_arrival_p99_ms": NAN,
    }


def _finish_metrics(
    metrics: Dict[str, float],
    *,
    completed: int,
    offered: int,
    unpaired: int,
    n_records: int,
    c2_lags: List[float],
    c3_lags: List[float],
    invariant_lags: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Fold paired-lag samples and correlation counts into a metrics dict.

    ``served_fraction`` counts every connection the mock produced a record for
    — identified OR not, since an unidentified connection was still SERVED —
    against the requests offered: it measures whether the mock could sustain
    the load at all, and drives the engine-limited verdict. ``unpaired_fraction``
    counts requests whose id could NOT be found among the mock's records
    against the offered load: it measures correlation loss (a request served
    but not keyed lands here), a measurement error that fails the check rather
    than shrinking the sample silently. The two are distinct — a mock that
    served everything but keyed nothing has served_fraction 1.0 and
    unpaired_fraction 1.0, i.e. a plumbing failure, not a capacity one.

    ``c2_lags`` fill the c2 column and gate. ``invariant_lags`` is the set the
    positive-lag invariant guards (``min_lag_ms``); it defaults to the C3
    samples. The request-level C2 is not in it: it is two absolute stamps
    subtracted directly (client ``request_sent_monotonic`` at the first
    application frame -> mock ``received_at`` at that frame), with no offset
    reconstruction, so the reconstruction-bug guard the invariant exists for
    does not apply to it. Its p99 is gated in the c2 column all the same.
    """
    invariant = c3_lags if invariant_lags is None else invariant_lags
    metrics["completed"] = float(completed)
    metrics["offered"] = float(offered)
    metrics["served_fraction"] = (n_records / offered) if offered else NAN
    metrics["unpaired_fraction"] = (unpaired / offered) if offered else NAN
    metrics["c2_lag_p99_ms"] = p99(c2_lags)
    metrics["c2_lag_p50_ms"] = p50(c2_lags)
    metrics["c3_lag_p99_ms"] = p99(c3_lags)
    metrics["c3_lag_p50_ms"] = p50(c3_lags)
    metrics["min_lag_ms"] = min(invariant) if invariant else NAN
    return metrics


def _audio_arrivals(chan_metrics: Dict[str, Any]) -> List[float]:
    """Absolute arrival of each audio chunk from ``audio_chunk_timestamps``.

    The audio clients record ``(offset_ms, ...)`` rows; the first column is the
    chunk's arrival offset from ``request_start_monotonic``.
    """
    timeline = chan_metrics.get(AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value) or []
    offsets = [float(row[0]) for row in timeline]
    return audio_arrival_times(chan_metrics.get("request_start_monotonic"), offsets)


def _text_pacing_drift(results: List[RequestResult], pacing_config) -> List[float]:
    """Late-only text-delta pacing drift (C1) over every stream, in ms.

    The realtime-TTS / Vajra clients pace their input text deltas to emulate an
    upstream decode rate; each stream records its actual delta send offsets in
    ``text_delta_timestamps[i][0]`` (ms from ``t_start``). Re-anchor those on the
    first delta and score them against the schedule the pacer paced to
    (:func:`text_pacing_schedule_ms`, also first-delta-anchored), late-only via
    :func:`send_drift`. Needs >= 2 deltas to have a cadence to measure; a stream
    with fewer contributes nothing rather than a spurious zero.
    """
    drift: List[float] = []
    for result in results:
        if not result.success:
            continue
        channel = result.channels.get(ChannelModality.AUDIO)
        if channel is None:
            continue
        timeline = channel.metrics.get(AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value) or []
        if len(timeline) < 2:
            continue
        first = float(timeline[0][0])
        actual_rel = [float(row[0]) - first for row in timeline]
        schedule = text_pacing_schedule_ms(
            pacing_config, result.request_id, len(timeline)
        )
        drift.extend(send_drift(actual_rel, schedule))
    return drift


def _score_response_delivery(
    measurement: Measurement,
    *,
    channel_mod: ChannelModality,
    reconstruct_arrivals: Callable[[Dict[str, Any]], List[float]],
) -> Dict[str, float]:
    """C2 (request delivery, 1/request) + C3 (response delivery, per event).

    C2 is the client's ``request_sent_monotonic`` against the mock's
    ``received_at``, one lag per request, paired by PFID. C3 is one lag per
    streamed event — the mock's ``emitted_at[i]`` against the client's
    reconstructed absolute arrival of event *i*. Both are ``later - earlier``;
    C3 is always positive for a correctly paired event and feeds the positive-
    lag invariant, while the coarse request-level C2 does not (see
    :func:`_finish_metrics`).
    """
    records = measurement.server_records
    metrics = _empty_metrics(measurement.achieved)

    c2_lags: List[float] = []
    c3_lags: List[float] = []
    offered = unpaired = completed = 0
    for result in measurement.results:
        offered += 1
        record = records.get(result.request_id)
        if record is None:
            unpaired += 1
            continue
        if not result.success:
            continue
        channel = result.channels.get(channel_mod)
        if channel is None:
            continue
        chan_metrics = channel.metrics
        sent = _stamp(chan_metrics.get("request_sent_monotonic"))
        received = _stamp(getattr(record, "received_at", None))
        if sent is not None and received is not None:
            c2_lags.append(delivery_lag_ms(sent, received))
        arrivals = reconstruct_arrivals(chan_metrics)
        emitted = [float(x) for x in (getattr(record, "emitted_at", None) or [])]
        if arrivals and emitted:
            c3_lags.extend(delivery_lags_ms(emitted, arrivals))
            completed += 1

    return _finish_metrics(
        metrics,
        completed=completed,
        offered=offered,
        unpaired=unpaired,
        n_records=len(records),
        c2_lags=c2_lags,
        c3_lags=c3_lags,
    )


# ------------------------------------------------------------------- workloads
class PreflightWorkload(ABC):
    """One measurable transport: mock server + real client + scoring.

    Subclasses own their mock server's lifecycle, the requests they send, and
    how the recorded stamps become reported numbers. The pipeline they run on
    is the shared one above — there is no per-modality copy of the harness for
    the modalities to drift apart in.
    """

    name: str = ""

    def __init__(self, config) -> None:
        self.config = config
        self._server = None

    # -- lifecycle
    @abstractmethod
    def start(self) -> None: ...

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None

    def reset_telemetry(self) -> None:
        if self._server is not None:
            self._server.reset_telemetry()

    # -- sizing
    @abstractmethod
    def request_lifetime_s(self) -> float:
        """How long one request lives — the number that sets offered load.

        Requests/second = concurrency / lifetime, so a mock that answers far
        faster than the model you intend to benchmark reports a ceiling your
        benchmark will never hit.
        """

    def request_count(self, concurrency: int) -> int:
        """Requests for one measurement at ``concurrency``.

        At least 1.2x concurrency, so the measurement can actually *reach* the
        concurrency it claims to measure (achieved >= 0.95c is a gate term and
        is unreachable if the request count barely covers c). Above that floor
        the count follows the wall-time budget.
        """
        lifetime = self.request_lifetime_s()
        floor = max(concurrency, math.ceil(1.2 * concurrency))
        if lifetime <= 0:
            return floor
        budgeted = int(self.config.budget_s * concurrency / lifetime)
        return max(floor, min(4000, budgeted))

    def server_loops(self, concurrency: int) -> int:
        configured = int(getattr(self.config, "server_loops", 0) or 0)
        if configured > 0:
            return configured
        return min(24, max(2, math.ceil(concurrency / 150)))

    # -- measurement
    def _collect(
        self,
        results: List[RequestResult],
        wall: float,
        epoch: float,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Measurement:
        """Snapshot the server's post-run telemetry into a Measurement.

        ``records()`` and ``unidentified_connections()`` are read HERE, while
        the server is still up, so scoring is a pure function of the snapshot.
        """
        assert self._server is not None
        return Measurement(
            results=results,
            achieved=self._server.max_active_conns,
            wall_s=wall,
            scheduler_epoch=epoch,
            server_records=self._server.records(),
            unidentified_conns=self._server.unidentified_connections(),
            extras=extras or {},
        )

    @abstractmethod
    def dispatch(self, concurrency: int, n_requests: int) -> Measurement: ...

    @abstractmethod
    def score(self, measurement: Measurement) -> List[CheckMetrics]: ...

    def measure(self, concurrency: int) -> List[CheckMetrics]:
        self.reset_telemetry()
        measurement = self.dispatch(concurrency, self.request_count(concurrency))
        return self.score(measurement)


class TextWorkload(PreflightWorkload):
    """Text SSE at a known inter-token cadence, through the chat client."""

    name = "text response delivery C3 (SSE, OpenAIChatCompletionsClient)"
    NOTES = (
        "c3LagP99/c3LagP50 = p99/p50 RESPONSE-DELIVERY lag: the mock's send "
        "stamp for chunk i (emitted_at[i]) -> the client's reconstructed "
        "arrival of chunk i (request_start_monotonic + sum inter_chunk_times), "
        "one lag per chunk over every stream. c2LagP99/c2LagP50 = p99/p50 "
        "REQUEST-DELIVERY lag: request_sent_monotonic -> the mock's received_at, "
        "one per request. Both are later-minus-earlier on one host, so each "
        "contains the loopback path cost plus the receiving side's scheduling "
        "delay, and EXCLUDES parse cost on both sides (each stamp sits next to "
        "its syscall). The p50 is this machine's delivery floor. C3 pairs "
        "against the stamp the mock took when it ACTUALLY sent each chunk, so "
        "the mock's own emit schedule adherence never enters these numbers."
    )

    def start(self) -> None:
        self._server = MockStreamingEngine(
            chunk_ms=self.config.chunk_ms,
            prefill_ms=self.config.prefill_ms,
            default_chunks=self.config.num_chunks,
            num_loops=self.server_loops(self.config.target_concurrency),
        ).start()

    def request_lifetime_s(self) -> float:
        return (
            self.config.prefill_ms + self.config.num_chunks * self.config.chunk_ms
        ) / 1000.0

    def _client_config(self) -> OpenAIChatCompletionsClientConfig:
        assert self._server is not None
        return OpenAIChatCompletionsClientConfig(
            api_base=f"{self._server.http_base}/v1/",
            api_key="preflight",
            model="preflight-mock",
            request_timeout=self.config.request_timeout_s,
        )

    def _make_request(self, request_id: int) -> Request:
        return Request(
            id=request_id,
            channels={
                ChannelModality.TEXT: TextChannelRequestContent(
                    # The PFID marker rides in the prompt verbatim; the mock
                    # reads it out of the POST body to key its records() to this
                    # request. Without it C2/C3 cannot be paired.
                    input_text=f"{format_pfid(request_id)} preflight"
                )
            },
            # The mock reads max_completion_tokens off the body, so the output
            # spec is what makes the stream exactly num_chunks long.
            requested_output=RequestedOutputSpec(
                text=TextOutputSpec(target_tokens=self.config.num_chunks)
            ),
        )

    def dispatch(self, concurrency: int, n_requests: int) -> Measurement:
        from veeksha.core.tokenizer import build_word_split_tokenizer_provider

        assert self._server is not None
        benchmark_config = build_benchmark_config(
            self._client_config(),
            concurrency=concurrency,
            n_requests=n_requests,
            lifetime_s=self.request_lifetime_s(),
            config=self.config,
        )
        # Whitespace tokenizer rather than the config's HF default: tokenization
        # is prompt preparation (outside the timing window), and importing HF
        # `tokenizers` re-enables the GIL on free-threaded builds, which would
        # serialize the very worker threads under measurement.
        results, wall, epoch = run_real_pipeline(
            benchmark_config,
            _FixedSessionSource(self._make_request),
            build_word_split_tokenizer_provider("preflight-mock"),
        )
        return self._collect(results, wall, epoch)

    @staticmethod
    def _text_arrivals(chan_metrics: Dict[str, Any]) -> List[float]:
        return text_arrival_times(
            chan_metrics.get("request_start_monotonic"),
            [float(x) for x in (chan_metrics.get("inter_chunk_times") or [])],
        )

    def score(self, measurement: Measurement) -> List[CheckMetrics]:
        metrics = _score_response_delivery(
            measurement,
            channel_mod=ChannelModality.TEXT,
            reconstruct_arrivals=self._text_arrivals,
        )
        return [
            CheckMetrics(
                name=self.name,
                notes=self.NOTES,
                metrics=metrics,
                gate=_delivery_gate,
            )
        ]


class TtsWorkload(PreflightWorkload):
    """Realtime-TTS audio deltas at a known cadence, over a real WebSocket."""

    name = "audio response delivery C3 + text pacing C1 (realtime TTS WS, RealtimeTTSClient)"
    NOTES = (
        "c3LagP99/c3LagP50 = p99/p50 RESPONSE-DELIVERY lag: the mock's audio "
        "send stamp for delta i (emitted_at[i]) -> the client's reconstructed "
        "arrival (request_start_monotonic + audio_chunk_timestamps[i][0]/1000), "
        "one lag per delta. c2LagP99/c2LagP50 = REQUEST-DELIVERY lag (1/request): "
        "request_sent_monotonic -> the mock's received_at. Both later-minus-"
        "earlier on one host: loopback path + the receiving side's scheduling "
        "delay, no parse cost either side; p50 is the delivery floor. On this WS "
        "client request_sent is stamped at the first application frame (the "
        "session.update send), which is the frame the mock stamps received_at "
        "against, so c2 is a true positive request-delivery lag. dispP99 = C1 "
        "TEXT-DELTA PACING adherence (client-only): per delta, max(0, actual "
        "send offset - the pacer's intended offset), late-only against the "
        "TextDeltaPacer cadence the client emulates an upstream decode rate "
        "with, both anchored on the first delta. A saturated client streams the "
        "input late and the TTS server then sees it at the wrong rate. Gated on "
        "pacing_drift_threshold_ms (a scheduling quantity, like the STT audio "
        "pacing). C3 pairs against what the mock ACTUALLY sent, so the mock's "
        "own emit schedule adherence never enters these numbers."
    )

    def start(self) -> None:
        self._server = MockRealtimeAudioServer(
            num_chunks=self.config.audio_num_chunks,
            chunk_bytes=self.config.audio_chunk_bytes,
            first_delta_ms=self.config.audio_first_delta_ms,
            audio_chunk_ms=self.config.audio_chunk_ms,
            num_loops=self.server_loops(self.config.target_concurrency),
        ).start()

    def request_lifetime_s(self) -> float:
        return (
            self.config.audio_first_delta_ms
            + self.config.audio_num_chunks * self.config.audio_chunk_ms
        ) / 1000.0

    def _client_config(self) -> RealtimeTTSClientConfig:
        assert self._server is not None
        return RealtimeTTSClientConfig(
            api_base=self._server.http_base,
            api_key="preflight",
            model="preflight-realtime-tts",
            request_timeout=self.config.request_timeout_s,
            # Paced slowly enough that audio (which the mock starts on the
            # FIRST text delta, as a streaming TTS server does) overlaps the
            # input still going out — the interleaved send/receive path. Shared
            # with the C1 text-pacing scorer so the intended schedule matches.
            pacing=_TTS_TEXT_PACING,
        )

    def _make_request(self, request_id: int) -> Request:
        return Request(
            id=request_id,
            channels={
                ChannelModality.TEXT: TextChannelRequestContent(
                    input_text=(
                        f"{format_pfid(request_id)} preflight realtime audio "
                        "transport probe"
                    )
                )
            },
        )

    def dispatch(self, concurrency: int, n_requests: int) -> Measurement:
        from veeksha.core.tokenizer import build_word_split_tokenizer_provider

        assert self._server is not None
        benchmark_config = build_benchmark_config(
            self._client_config(),
            concurrency=concurrency,
            n_requests=n_requests,
            lifetime_s=self.request_lifetime_s(),
            config=self.config,
        )
        results, wall, epoch = run_real_pipeline(
            benchmark_config,
            _FixedSessionSource(self._make_request),
            build_word_split_tokenizer_provider("preflight-realtime-tts"),
        )
        return self._collect(results, wall, epoch)

    def score(self, measurement: Measurement) -> List[CheckMetrics]:
        metrics = _score_response_delivery(
            measurement,
            channel_mod=ChannelModality.AUDIO,
            reconstruct_arrivals=_audio_arrivals,
        )
        # C1 text-delta pacing adherence rides in the dispP99 column (a late-
        # only, client-side schedule-adherence number like dispatch/STT pacing).
        metrics["dispatch_drift_p99_ms"] = p99(
            _text_pacing_drift(measurement.results, _TTS_TEXT_PACING)
        )
        return [
            CheckMetrics(
                name=self.name,
                notes=self.NOTES,
                metrics=metrics,
                gate=_tts_gate,
            )
        ]


class VajraTtsWorkload(PreflightWorkload):
    """Vajra streaming-text TTS: binary PCM frames arriving WHILE text is sent.

    The sibling of :class:`TtsWorkload` for Vajra's native protocol, and
    deliberately the harder case. The OpenAI-realtime client sends every paced
    text delta and only then ``response.create``, so its audio can only arrive
    after the send loop is done; Vajra's server streams audio as text arrives,
    so this check exercises paced sends and timestamped receives interleaved on
    one event loop — the path where drift shows up first under concurrency.
    Audio also arrives as raw binary frames here, so the client's receive path
    skips base64 decode entirely.
    """

    name = (
        "audio response delivery C3 + text pacing C1 "
        "(Vajra streaming TTS WS, VajraTTSStreamClient)"
    )
    NOTES = (
        "c3LagP99/c3LagP50 = p99/p50 RESPONSE-DELIVERY lag: the mock's frame "
        "send stamp (emitted_at[i]) -> the client's reconstructed arrival "
        "(request_start_monotonic + audio_chunk_timestamps[i][0]/1000), one lag "
        "per frame, later-minus-earlier on one host (loopback + receiving-side "
        "scheduling, no parse cost either side). Audio overlaps the paced text "
        "input by design, so this measures response delivery WHILE the client "
        "is also sending — the path where lag grows first under concurrency. "
        "p50 is the delivery floor. c2LagP99/c2LagP50 = REQUEST-DELIVERY lag "
        "(1/request): request_sent_monotonic (the first application frame, the "
        "session.config send) -> the mock's received_at at that frame, a true "
        "positive request-delivery lag. dispP99 = C1 TEXT-DELTA PACING adherence "
        "(client-only): per delta, max(0, actual send offset - the pacer's "
        "intended offset), late-only against the TextDeltaPacer cadence, both "
        "anchored on the first delta — the interleaved-send path is exactly "
        "where the client falls behind its own input pacing first. Gated on "
        "pacing_drift_threshold_ms. C3 pairs against what the mock ACTUALLY "
        "sent, so the mock's own emit schedule adherence never enters these "
        "numbers."
    )

    def start(self) -> None:
        self._server = MockVajraTTSStreamServer(
            num_chunks=self.config.audio_num_chunks,
            chunk_bytes=self.config.audio_chunk_bytes,
            first_delta_ms=self.config.audio_first_delta_ms,
            audio_chunk_ms=self.config.audio_chunk_ms,
            num_loops=self.server_loops(self.config.target_concurrency),
        ).start()

    def request_lifetime_s(self) -> float:
        # Emission starts on the first text delta and overlaps the input, so
        # the audio timeline (not the text pacing) bounds the lifetime.
        return (
            self.config.audio_first_delta_ms
            + self.config.audio_num_chunks * self.config.audio_chunk_ms
        ) / 1000.0

    def _client_config(self) -> VajraTTSStreamClientConfig:
        assert self._server is not None
        return VajraTTSStreamClientConfig(
            api_base=self._server.http_base,
            api_key="preflight",
            model="preflight-vajra-tts",
            request_timeout=self.config.request_timeout_s,
            # Paced slowly enough that text is still going out while audio
            # comes back: overlap is the property this check exists to
            # exercise, so it must not finish before the first frame lands.
            # Shared with the C1 text-pacing scorer so the schedule matches.
            pacing=_TTS_TEXT_PACING,
        )

    def _make_request(self, request_id: int) -> Request:
        return Request(
            id=request_id,
            channels={
                ChannelModality.TEXT: TextChannelRequestContent(
                    input_text=(
                        f"{format_pfid(request_id)} preflight vajra streaming "
                        "audio transport probe with enough words to keep the "
                        "input paced"
                    )
                )
            },
        )

    def dispatch(self, concurrency: int, n_requests: int) -> Measurement:
        from veeksha.core.tokenizer import build_word_split_tokenizer_provider

        assert self._server is not None
        benchmark_config = build_benchmark_config(
            self._client_config(),
            concurrency=concurrency,
            n_requests=n_requests,
            lifetime_s=self.request_lifetime_s(),
            config=self.config,
        )
        results, wall, epoch = run_real_pipeline(
            benchmark_config,
            _FixedSessionSource(self._make_request),
            build_word_split_tokenizer_provider("preflight-vajra-tts"),
        )
        return self._collect(results, wall, epoch)

    def score(self, measurement: Measurement) -> List[CheckMetrics]:
        metrics = _score_response_delivery(
            measurement,
            channel_mod=ChannelModality.AUDIO,
            reconstruct_arrivals=_audio_arrivals,
        )
        # C1 text-delta pacing adherence rides in the dispP99 column.
        metrics["dispatch_drift_p99_ms"] = p99(
            _text_pacing_drift(measurement.results, _TTS_TEXT_PACING)
        )
        return [
            CheckMetrics(
                name=self.name,
                notes=self.NOTES,
                metrics=metrics,
                gate=_tts_gate,
            )
        ]


class SttWorkload(PreflightWorkload):
    """Realtime ASR: 1x-paced audio out, transcript deltas back.

    ASR word interactivity spans both directions, so this check pairs both. C2
    (request delivery) is per audio chunk: the client's paced send instant
    (audio_started_monotonic + send_offsets_ms[i]/1000) against the mock's
    per-append arrival stamp — the ASR pacing ground truth, now a delivery lag.
    C3 (response delivery) is per transcript delta: the mock's emit stamp
    against the client's reconstructed arrival. One dispatch answers both, so
    the two halves describe the same traffic rather than two runs that merely
    ran near each other.
    """

    name = "asr pacing C1 + delivery C2/C3 (realtime STT WS, STTClient)"
    NOTES = (
        "THREE numbers. dispP99 = C1 CHUNK-PACING ADHERENCE (client-only, the "
        "important one, and the only STT number that also runs in production): "
        "per chunk, max(0, send_offsets_ms[i] - i*chunk_bytes/(2*sample_rate)*"
        "1000), late-only against the intended 1x-realtime schedule the client "
        "paces on. It says whether veeksha kept up with generating realtime "
        "audio — a saturated client sends chunks late and silently benchmarks "
        "slower-than-1x audio. Gated on pacing_drift_threshold_ms (a scheduling "
        "quantity, distinct from the transport lags below). c2LagP99/c2LagP50 = "
        "C2 REQUEST-DELIVERY lag (1/request): request_sent_monotonic "
        "(=audio_started_monotonic) -> the mock's received_at. A SECOND C2, the "
        "per-append delivery lag (append_at[i] minus the client's absolute send "
        "audio_started_monotonic + send_offsets_ms[i]/1000), is also gated "
        "(term 'audio delivery lag') and feeds the positive-lag invariant — it "
        "is the ASR-critical delivery number, one lag per audio chunk. "
        "c3LagP99/c3LagP50 = C3 RESPONSE-DELIVERY lag per transcript delta: the "
        "mock's emit stamp -> the client's reconstructed arrival "
        "(audio_started_monotonic + transcript_delta_offsets_ms[i]/1000). All "
        "later-minus-earlier on one host: loopback + the receiving side's "
        "scheduling delay, no parse cost either side (the mock stamps the raw "
        "frame before base64/JSON). p50 is the delivery floor; the client is "
        "scored against the mock's actual receive/send stamps, so the mock's "
        "own schedule adherence never enters these numbers."
    )

    def __init__(self, config) -> None:
        super().__init__(config)
        self._tmpdir: Optional[str] = None
        self._wav: Optional[str] = None

    def start(self) -> None:
        self._server = MockSTTPreflightServer(
            transcript=self.config.stt_transcript,
            transcript_delta_ms=self.config.transcript_delta_ms,
            num_loops=self.server_loops(self.config.target_concurrency),
        ).start()
        # One shared silence clip for every request — the audio carries no id.
        # STT correlation rides a payload field, not the audio: _make_request
        # sets request.metadata["preflight_pfid"], the STT client echoes it into
        # its session.update as veeksha_request_id (omitted on real runs), and
        # the mock keys records() by it. (A file-baked PCM marker was tried and
        # rejected: the client decodes through librosa, whose float round-trip
        # corrupts the low bytes a marker would live in.) If an id fails to
        # round-trip, the request is unidentified and the check FAILS on
        # `request correlation` — surfaced, never silently dropped.
        self._tmpdir = tempfile.mkdtemp(prefix="veeksha-preflight-")
        self._wav = os.path.join(self._tmpdir, "clip.wav")
        n_frames = int(self.config.stt_sample_rate * self.config.pacing_clip_s)
        with wave.open(self._wav, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.config.stt_sample_rate)
            handle.writeframes(b"\x00\x00" * n_frames)

    def stop(self) -> None:
        super().stop()
        if self._tmpdir and os.path.isdir(self._tmpdir):
            import shutil

            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None
        self._wav = None

    def request_lifetime_s(self) -> float:
        # The clip is streamed at 1x real time, so its duration IS the request
        # lifetime (plus the short transcript tail).
        return self.config.pacing_clip_s + (
            len(self.config.stt_transcript.split())
            * self.config.transcript_delta_ms
            / 1000.0
        )

    def _client_config(self) -> STTClientConfig:
        assert self._server is not None
        return STTClientConfig(
            api_base=self._server.http_base,
            api_key="preflight",
            model="preflight-stt",
            provider="vllm_realtime",
            sample_rate=self.config.stt_sample_rate,
            ws_chunk_size=self.config.stt_ws_chunk_size,
            # The whole point of this check: exercise the client's absolute
            # deadline pacer. Without it there is no send schedule to score.
            ws_realtime_pacing=True,
            ws_ping_interval_s=None,
            request_timeout=self.config.request_timeout_s,
        )

    def _make_request(self, request_id: int) -> Request:
        assert self._wav is not None
        return Request(
            id=request_id,
            channels={
                ChannelModality.AUDIO: AudioChannelRequestContent(input_audio=self._wav)
            },
            # STT correlation rides in metadata, NOT in the PCM: the id would
            # not survive the client's librosa float round-trip. The STT client
            # reads this key and puts the id in its session.update message;
            # MockSTTPreflightServer reads it back and keys its ServerRecord by
            # it. Same integer the scorer pairs on (result.request_id).
            metadata={"dataset": "preflight", "preflight_pfid": request_id},
        )

    def dispatch(self, concurrency: int, n_requests: int) -> Measurement:
        from veeksha.core.tokenizer import build_word_split_tokenizer_provider

        assert self._server is not None
        benchmark_config = build_benchmark_config(
            self._client_config(),
            concurrency=concurrency,
            n_requests=n_requests,
            lifetime_s=self.request_lifetime_s(),
            config=self.config,
        )
        results, wall, epoch = run_real_pipeline(
            benchmark_config,
            _FixedSessionSource(self._make_request),
            build_word_split_tokenizer_provider("preflight-stt"),
        )
        return self._collect(results, wall, epoch)

    def score(self, measurement: Measurement) -> List[CheckMetrics]:
        chunk_bytes = self.config.stt_ws_chunk_size
        sample_rate = self.config.stt_sample_rate
        records = measurement.server_records
        metrics = _empty_metrics(measurement.achieved)

        pacing_drift: List[float] = []
        c2_lags: List[float] = []  # request-level C2 (1/request), the c2 column
        append_lags: List[float] = []  # per-append C2 (client send -> arrival)
        c3_lags: List[float] = []
        offered = unpaired = completed = 0
        for result in measurement.results:
            offered += 1
            record = records.get(result.request_id)
            if record is None:
                unpaired += 1
                continue
            if not result.success:
                continue
            channel = result.channels.get(ChannelModality.AUDIO)
            if channel is None:
                continue
            chan_metrics = channel.metrics
            anchor = _stamp(chan_metrics.get("audio_started_monotonic"))
            send_offsets = [
                float(x) for x in (chan_metrics.get("send_offsets_ms") or [])
            ]
            # C2 request-level (1/request): client send stamp -> mock receive.
            sent = _stamp(chan_metrics.get("request_sent_monotonic"))
            received = _stamp(getattr(record, "received_at", None))
            if sent is not None and received is not None:
                c2_lags.append(delivery_lag_ms(sent, received))
            # C1 chunk-pacing adherence vs the intended 1x schedule (client-
            # only, works in production). send_offsets_ms is already the actual
            # offset from the audio anchor, so it pairs against the schedule.
            if len(send_offsets) >= 2:
                schedule = asr_send_schedule_ms(
                    len(send_offsets), chunk_bytes, sample_rate
                )
                pacing_drift.extend(send_drift(send_offsets, schedule))
            # C2 per-append delivery: each paced send vs the mock's per-append
            # arrival — a true per-event pairing, so it feeds the invariant.
            send_times = stt_send_times(anchor, send_offsets)
            append_at = [float(x) for x in (getattr(record, "append_at", None) or [])]
            if send_times and append_at:
                append_lags.extend(delivery_lags_ms(send_times, append_at))
                completed += 1
            # C3: transcript deltas vs the mock's emit stamps.
            delta_offsets = [
                float(x)
                for x in (chan_metrics.get("transcript_delta_offsets_ms") or [])
            ]
            arrivals = absolute_times_from_offsets_ms(anchor, delta_offsets)
            emitted = [float(x) for x in (getattr(record, "emitted_at", None) or [])]
            if arrivals and emitted:
                c3_lags.extend(delivery_lags_ms(emitted, arrivals))

        metrics = _finish_metrics(
            metrics,
            completed=completed,
            offered=offered,
            unpaired=unpaired,
            n_records=len(records),
            c2_lags=c2_lags,
            c3_lags=c3_lags,
            # The per-event reconstructed pairings that must be positive:
            # transcript delivery and per-append audio delivery. The
            # request-level C2 is two absolute stamps subtracted directly (no
            # offset reconstruction), so the reconstruction-bug guard the
            # invariant exists for does not apply to it.
            invariant_lags=append_lags + c3_lags,
        )
        metrics["append_lag_p99_ms"] = p99(append_lags)
        # C1 chunk-pacing adherence rides in the dispP99 column: like request
        # dispatch it is a late-only, client-side C1 schedule-adherence number.
        metrics["dispatch_drift_p99_ms"] = p99(pacing_drift)
        return [
            CheckMetrics(
                name=self.name,
                notes=self.NOTES,
                metrics=metrics,
                gate=_stt_gate,
            )
        ]


class DispatchWorkload(PreflightWorkload):
    """Dispatch accuracy: did the harness START each request when it should?

    The other checks answer "when are we supposed to RECEIVE a packet" and
    "when are we supposed to SEND an audio packet". This one answers the third
    question — "when are we supposed to SEND A REQUEST" — and it is the only
    one whose error is attributable to the traffic scheduler rather than to a
    receive path.

    It needs genuinely open-loop traffic to be answerable at all: under
    closed-loop (``ConcurrentTrafficConfig``) scheduling a session's arrival
    time is defined by when an earlier one finished, so there is no
    independent "supposed to" to score against. ``RateTrafficConfig`` assigns
    every session a deterministic seeded offset before the run, which
    :func:`~veeksha.preflight.scorer.rate_schedule_offsets_ms` reproduces
    exactly.

    Definition, per session *i* (``i`` = the i-th session scheduled, which with
    the pipeline's single ``PrefetchWorker`` is session id order):

    * ``scheduled(i)`` — the seeded arrival offset ``RateTrafficScheduler``
      assigns, ms from the scheduler's epoch;
    * ``actual(i) = scheduler_dispatched_at - epoch``, ms, where the epoch is
      the instant ``reset_reference_time()`` set as the scheduler's zero;
    * ``drift(i) = max(0, actual - scheduled)`` — late-only, because the
      scheduler pops from a deadline heap and structurally cannot dispatch
      early.

    The server is the same text-SSE mock the text check uses, with a short
    response: the response content is irrelevant here — only offered load is.
    """

    name = "send schedule C1 (rate-scheduled arrivals, RateTrafficScheduler)"
    NOTES = (
        "dispP99 = p99 per-request DISPATCH lateness against the rate "
        "scheduler's seeded arrival schedule: max(0, (scheduler_dispatched_at "
        "- epoch) - scheduled(i)), where epoch is the instant "
        "reset_reference_time() zeroed the scheduler and scheduled(i) is the "
        "i-th session's arrival offset, reproduced from the same seed and "
        "interval generator. LATE-ONLY, because the ready-heap cannot pop "
        "before a deadline, so earliness does not exist to average against "
        "lateness. INCLUDES scheduler wake latency, ready-heap pop, contention "
        "on the single threading.Condition that guards every scheduler "
        "operation, and DispatchWorker thread scheduling. EXCLUDES everything "
        "client-side (connect, send, network, server): this is the one check "
        "whose error is attributable to the SCHEDULER rather than to a "
        "delivery path, and — being client-side only — the one that also works "
        "on a real production run. Failed requests are scored too: a request "
        "is dispatched before it can fail. The c2/c3 lag columns do not apply "
        "(this check pairs no server stamp)."
    )

    def start(self) -> None:
        self._server = MockStreamingEngine(
            chunk_ms=self.config.chunk_ms,
            prefill_ms=self.config.prefill_ms,
            default_chunks=self.config.dispatch_response_chunks,
            num_loops=self.server_loops(self.config.target_concurrency),
        ).start()

    def request_lifetime_s(self) -> float:
        # Deliberately SHORT (dispatch_response_chunks, not num_chunks): the
        # arrival rate that reaches the target concurrency is
        # concurrency / lifetime, and the whole measurement has to fit inside
        # PrefetchWorker's burst window. A 5 s lifetime at c=100 would need
        # 20 arrivals/s over a window that only leaves room for a handful of
        # requests; a 0.4 s lifetime needs 250/s and gets there in seconds.
        return (
            self.config.prefill_ms
            + self.config.dispatch_response_chunks * self.config.chunk_ms
        ) / 1000.0

    def arrival_rate(self, concurrency: int) -> float:
        """Sessions/second whose offered load equals ``concurrency``.

        Little's law, stated as sizing policy: with open-loop arrivals at rate
        ``r`` and request lifetime ``L``, the number of simultaneously live
        requests settles at ``r * L``. So ``r = concurrency / L`` is what makes
        this check certify the same concurrency the other checks do.
        """
        lifetime = self.request_lifetime_s()
        return concurrency / lifetime if lifetime > 0 else float(concurrency)

    def _plan(self, concurrency: int) -> tuple[int, float, str]:
        """``(n_requests, arrival_rate, warning)`` for one measurement.

        The measurement must end inside ``BURST_WINDOW_S``: after that PrefetchWorker
        throttles to ~20 sessions/s, which at any interesting rate would make
        the measurement a picture of the prefetch throttle rather than of
        dispatch accuracy. The request count is therefore derived from the
        window, not from ``budget_s`` alone — but the 1.2x concurrency floor
        every workload keeps (so a measurement can reach the concurrency it claims)
        wins over it, and when it does, the run cannot fit and we say so.
        """
        rate = self.arrival_rate(concurrency)
        lifetime = self.request_lifetime_s()
        window = min(self.config.budget_s, BURST_SAFETY * BURST_WINDOW_S)
        arrival_span_s = max(0.0, window - lifetime)
        floor = max(concurrency, math.ceil(1.2 * concurrency))
        n_requests = max(floor, min(4000, int(rate * arrival_span_s)))

        duration_s = (n_requests / rate if rate > 0 else 0.0) + lifetime
        warning = ""
        if duration_s > BURST_WINDOW_S:
            warning = (
                f"WARNING: this measurement is estimated to run {duration_s:.1f}s, "
                f"beyond PrefetchWorker's {BURST_WINDOW_S:.0f}s unthrottled "
                f"burst window, after which session generation throttles to "
                f"~{PREFETCH_THROTTLED_RATE_PER_S:.0f}/s. Arrivals scheduled "
                f"after that "
                f"point may be late because sessions were GENERATED late, not "
                f"because dispatch drifted: driftP99 is then an upper bound, "
                f"not a scheduler measurement. Shorten the measurement "
                f"(dispatch_response_chunks, budget_s) or lower "
                f"target_concurrency."
            )
        return n_requests, rate, warning

    def request_count(self, concurrency: int) -> int:
        return self._plan(concurrency)[0]

    def _client_config(self) -> OpenAIChatCompletionsClientConfig:
        assert self._server is not None
        return OpenAIChatCompletionsClientConfig(
            api_base=f"{self._server.http_base}/v1/",
            api_key="preflight",
            model="preflight-mock",
            request_timeout=self.config.request_timeout_s,
        )

    def _make_request(self, request_id: int) -> Request:
        return Request(
            id=request_id,
            channels={
                ChannelModality.TEXT: TextChannelRequestContent(
                    input_text="preflight", target_prompt_tokens=1
                )
            },
            requested_output=RequestedOutputSpec(
                text=TextOutputSpec(target_tokens=self.config.dispatch_response_chunks)
            ),
        )

    def dispatch(self, concurrency: int, n_requests: int) -> Measurement:
        from veeksha.core.tokenizer import build_word_split_tokenizer_provider

        assert self._server is not None
        _, rate, warning = self._plan(concurrency)
        if warning:
            logger.warning("preflight dispatch check: %s", warning)

        interval_generator = PoissonIntervalGeneratorConfig(arrival_rate=rate)
        traffic_config = RateTrafficConfig(interval_generator=interval_generator)
        benchmark_config = build_benchmark_config(
            self._client_config(),
            concurrency=concurrency,
            n_requests=n_requests,
            lifetime_s=self.request_lifetime_s(),
            config=self.config,
            traffic_config=traffic_config,
            # _run_main_loop auto-sizes the client pool from the scheduler's
            # target_concurrent_sessions, which RateTrafficScheduler does not
            # have — it would quietly run three client workers at any
            # concurrency. Apply the benchmark's own formula so this measurement
            # carries the same client-side shape as the other checks; dispatch
            # drift excludes the client either way, but a starved pool would
            # distort achieved concurrency and the reported pickup column.
            num_client_threads=(
                self.config.num_client_threads
                if self.config.num_client_threads is not None
                else max(3, math.ceil(concurrency / 8))
            ),
        )
        results, wall, epoch = run_real_pipeline(
            benchmark_config,
            _FixedSessionSource(self._make_request),
            build_word_split_tokenizer_provider("preflight-mock"),
        )
        return Measurement(
            results=results,
            achieved=self._server.max_active_conns,
            wall_s=wall,
            scheduler_epoch=epoch,
            extras={
                # Everything score() needs to REPRODUCE the schedule the run
                # was actually given — the schedule is not observable after the
                # fact, because the scheduler drops session state on completion.
                "interval_generator": interval_generator,
                "n_sessions": n_requests,
                "arrival_rate": rate,
                "burst_warning": warning,
            },
        )

    def score(self, measurement: Measurement) -> List[CheckMetrics]:
        interval_generator = measurement.extras.get("interval_generator")
        n_sessions = int(measurement.extras.get("n_sessions", 0) or 0)
        epoch = measurement.scheduler_epoch
        scheduled_ms = (
            rate_schedule_offsets_ms(PREFLIGHT_SEED, interval_generator, n_sessions)
            if interval_generator is not None
            else []
        )

        actual_ms: List[float] = []
        due_ms: List[float] = []
        for result in measurement.results:
            # NOT filtered on success: a request is dispatched before it can
            # fail, so its dispatch time is valid data either way. Filtering
            # would let a measurement that failed its late requests look punctual.
            index = result.session_id
            if not (0 <= index < len(scheduled_ms)):
                continue
            if result.scheduler_dispatched_at is None or epoch != epoch:
                continue
            actual_ms.append((result.scheduler_dispatched_at - epoch) * 1000.0)
            due_ms.append(scheduled_ms[index])

        drift = dispatch_drift_ms(actual_ms, due_ms)
        metrics = _empty_metrics(measurement.achieved)
        metrics["completed"] = float(len(drift))
        metrics["offered"] = float(len(measurement.results))
        # C1 is client-side only: it pairs no server stamp, so served_fraction
        # and the correlation counts stay NaN (engine-limited never applies).
        metrics["dispatch_drift_p99_ms"] = p99(drift)
        notes = self.NOTES
        warning = measurement.extras.get("burst_warning") or ""
        if warning:
            notes = f"{notes} {warning}"
        return [
            CheckMetrics(
                name=self.name,
                notes=notes,
                metrics=metrics,
                gate=_dispatch_gate,
            )
        ]


class MultiTurnTextWorkload(PreflightWorkload):
    """Multi-turn think time: did the harness RELEASE the next turn on time?

    The fold of the old completion-queue check, measured directly and end to
    end. Each session is two short TEXT turns (SSE, ``OpenAIChatCompletionsClient``)
    with turn 2 waiting ``think_time_s`` after turn 1 completes. The real
    scheduler releases turn 2 at ``client_completed_at[turn1] + think_time_s``,
    so scoring turn 2's actual send against that intended release certifies the
    harness's ability to release multi-turn follow-ups on schedule — a late
    release means the server sees the conversation's turns at the wrong cadence.

    It is the multi-turn analog of the dispatch C1 check, one level up: instead
    of "did request *i* start on its seeded arrival" it asks "did the NEXT turn
    start on its completion-relative schedule", and it structurally includes the
    completion-notify lag (the scheduler only learns turn 1 finished when the
    completion worker dequeues it) that the removed ``cqP99`` used as a proxy.

    Definition per session (its two turns ordered by ``scheduler_dispatched_at``
    — turn 2 is a child of turn 1 and cannot dispatch until turn 1 completes, so
    dispatch time strictly increases with turn order, which is robust regardless
    of how request ids are assigned):

    * ``intended        = client_completed_at[turn1] + think_time_s``
    * ``think_time_drift = max(0, (request_sent_monotonic[turn2] - intended))``
      — client-side, GATED (ttP99), floor REPORTED (ttP50);
    * ``server-arrival  = max(0, (received_at[turn2] - intended))`` — the mock's
      receive stamp for turn 2, which additionally carries the C2 delivery lag;
      REPORTED only (ttSrvP99), never gated.

    Responses are kept SHORT (``dispatch_response_chunks`` chunks/turn): the run
    measures turn CADENCE, not stream length.
    """

    name = "multi-turn think time C1 (SSE, OpenAIChatCompletionsClient)"
    NOTES = (
        "ttP99 = p99 late-only THINK-TIME DRIFT, GATED, client-side only (also "
        "holds in production): per 2-turn session, max(0, turn-2 "
        "request_sent_monotonic - (client_completed_at[turn1] + think_time_s)), "
        "i.e. how late the harness RELEASED the next turn against the schedule "
        "the scheduler itself committed to (child ready_at = parent completion + "
        "think). LATE-ONLY, because the scheduler cannot release turn 2 before "
        "that ready_at. INCLUDES the completion-notify lag (the scheduler learns "
        "turn 1 finished only when the completion worker dequeues it), the "
        "child-node release, and turn 2's client pickup/send; EXCLUDES the "
        "server (a client-side send stamp). ttP50 is the turnaround floor. "
        "ttSrvP99 = p99 of the SERVER-ARRIVAL variant (mock received_at[turn2] "
        "instead of the client send), which additionally carries the C2 "
        "delivery lag — the true end-to-end 'did the next turn ARRIVE at the "
        "server on time' — REPORTED, not gated. Gated on "
        "think_time_drift_threshold_ms (a scheduling quantity, its own knob). "
        "This is the multi-turn analog of the dispatch C1 check."
    )

    def start(self) -> None:
        self._server = MockStreamingEngine(
            chunk_ms=self.config.chunk_ms,
            prefill_ms=self.config.prefill_ms,
            # SHORT turns: the point is turn cadence, not stream length. Reuse
            # the dispatch check's short response length for the same reason.
            default_chunks=self.config.dispatch_response_chunks,
            num_loops=self.server_loops(self.config.target_concurrency),
        ).start()

    def request_lifetime_s(self) -> float:
        # A full 2-turn session's lifetime: two short turns plus the think time
        # between them. It sizes the offered load (sessions/s = concurrency /
        # lifetime) and the safety timeout, so it must reflect the whole session.
        turn_s = (
            self.config.prefill_ms
            + self.config.dispatch_response_chunks * self.config.chunk_ms
        ) / 1000.0
        return 2.0 * turn_s + self.config.think_time_s

    def _client_config(self) -> OpenAIChatCompletionsClientConfig:
        assert self._server is not None
        return OpenAIChatCompletionsClientConfig(
            api_base=f"{self._server.http_base}/v1/",
            api_key="preflight",
            model="preflight-mock",
            request_timeout=self.config.request_timeout_s,
        )

    def _make_request(self, request_id: int) -> Request:
        return Request(
            id=request_id,
            channels={
                ChannelModality.TEXT: TextChannelRequestContent(
                    # Each turn embeds its OWN PFID so the mock records both.
                    input_text=f"{format_pfid(request_id)} preflight turn"
                )
            },
            requested_output=RequestedOutputSpec(
                text=TextOutputSpec(target_tokens=self.config.dispatch_response_chunks)
            ),
        )

    def dispatch(self, concurrency: int, n_requests: int) -> Measurement:
        # ``n_requests`` here counts SESSIONS (each yields two turns).
        from veeksha.core.tokenizer import build_word_split_tokenizer_provider

        assert self._server is not None
        benchmark_config = build_benchmark_config(
            self._client_config(),
            concurrency=concurrency,
            n_requests=n_requests,
            lifetime_s=self.request_lifetime_s(),
            config=self.config,
        )
        results, wall, epoch = run_real_pipeline(
            benchmark_config,
            _MultiTurnSessionSource(self._make_request, self.config.think_time_s),
            build_word_split_tokenizer_provider("preflight-mock"),
        )
        return self._collect(results, wall, epoch)

    @staticmethod
    def _request_sent(result: RequestResult) -> Optional[float]:
        channel = result.channels.get(ChannelModality.TEXT)
        if channel is None:
            return None
        return _stamp(channel.metrics.get("request_sent_monotonic"))

    def score(self, measurement: Measurement) -> List[CheckMetrics]:
        records = measurement.server_records
        think = self.config.think_time_s
        metrics = _empty_metrics(measurement.achieved)

        # Correlation is per TURN (each turn opens a connection): a session whose
        # turns did not both key to a server record is a measurement error,
        # surfaced via the unpaired path rather than silently dropped.
        offered = len(measurement.results)
        unpaired = sum(1 for r in measurement.results if r.request_id not in records)

        by_session: Dict[int, List[RequestResult]] = {}
        for result in measurement.results:
            by_session.setdefault(result.session_id, []).append(result)

        client_drift: List[float] = []
        arrival_drift: List[float] = []
        for turns in by_session.values():
            if len(turns) != 2 or any(t.scheduler_dispatched_at is None for t in turns):
                continue  # not a complete, orderable 2-turn session
            turn1, turn2 = sorted(turns, key=lambda r: r.scheduler_dispatched_at)
            # Client-side (gated): turn-2 send vs turn-1 completion + think.
            cd = think_time_drift_ms(
                turn1.client_completed_at, self._request_sent(turn2), think
            )
            if not math.isnan(cd):
                client_drift.append(cd)
            # Server-arrival (reported): turn-2 mock receive vs the same intended.
            record2 = records.get(turn2.request_id)
            received2 = _stamp(getattr(record2, "received_at", None))
            ad = think_time_drift_ms(turn1.client_completed_at, received2, think)
            if not math.isnan(ad):
                arrival_drift.append(ad)

        metrics["completed"] = float(len(client_drift))
        metrics["offered"] = float(offered)
        metrics["served_fraction"] = (len(records) / offered) if offered else NAN
        metrics["unpaired_fraction"] = (unpaired / offered) if offered else NAN
        metrics["think_time_drift_p99_ms"] = p99(client_drift)
        metrics["think_time_drift_p50_ms"] = p50(client_drift)
        metrics["think_time_arrival_p99_ms"] = p99(arrival_drift)
        # The mock serves each turn as a separate connection over a session's
        # think-time duty cycle, so peak CONNECTIONS undershoots the SESSION
        # concurrency the harness actually holds; the achieved-concurrency gate
        # (peak connections) does not apply. Load application is verified by
        # served_fraction + correlation instead.
        metrics["gates_achieved_concurrency"] = 0.0
        return [
            CheckMetrics(
                name=self.name,
                notes=self.NOTES,
                metrics=metrics,
                gate=_multiturn_gate,
            )
        ]


# ----------------------------------------------------------------- gate terms
# Gate terms are named so the report can say WHICH property failed, and are
# written as `value < threshold` so a NaN (unmeasured) term is False — an
# unmeasured measurement can never read as honest.
def _delivery_gate(metrics: Dict[str, float], config) -> Dict[str, bool]:
    """Every paired check: request delivery (C2), response delivery (C3), the
    correlation fraction and the positive-lag invariant.

    ``request correlation`` fails when too many requests had no server record (a
    measurement error, not less data). ``positive-lag invariant`` fails only on
    an actual negative per-event lag — ``min_lag_ms < 0`` is False when
    NaN/empty, so an unmeasured check falls through to the lag-p99 terms
    instead, which fail safe on NaN.
    """
    return {
        "request delivery lag": (
            metrics["c2_lag_p99_ms"] < config.delivery_lag_threshold_ms
        ),
        "response delivery lag": (
            metrics["c3_lag_p99_ms"] < config.delivery_lag_threshold_ms
        ),
        "request correlation": (
            metrics["unpaired_fraction"] < config.max_unpaired_fraction
        ),
        "positive-lag invariant": not (metrics["min_lag_ms"] < 0.0),
    }


def _tts_gate(metrics: Dict[str, float], config) -> Dict[str, bool]:
    """Realtime-TTS / Vajra: the shared delivery gate plus the C1 text-delta
    pacing term.

    ``text pacing adherence`` is gated on ``pacing_drift_threshold_ms`` (a
    scheduling quantity — did the client keep the emulated decode cadence —
    distinct from the transport lags), and rides in the dispP99 column.
    """
    terms = _delivery_gate(metrics, config)
    terms["text pacing adherence"] = (
        metrics["dispatch_drift_p99_ms"] < config.pacing_drift_threshold_ms
    )
    return terms


def _stt_gate(metrics: Dict[str, float], config) -> Dict[str, bool]:
    """ASR: the shared delivery gate plus two ASR-specific terms.

    ``audio pacing adherence`` (C1, client-only) is gated on its OWN threshold —
    it is a scheduling quantity (did the client keep the 1x cadence), physically
    different from the transport lags, so it does not share their bar. ``audio
    delivery lag`` is the per-append C2 (client send -> server append arrival),
    the ASR-critical delivery number that has no dedicated column.
    """
    terms = _delivery_gate(metrics, config)
    terms["audio pacing adherence"] = (
        metrics["dispatch_drift_p99_ms"] < config.pacing_drift_threshold_ms
    )
    terms["audio delivery lag"] = (
        metrics["append_lag_p99_ms"] < config.delivery_lag_threshold_ms
    )
    return terms


def _dispatch_gate(metrics: Dict[str, float], config) -> Dict[str, bool]:
    # Only the dispatch term is gated here: C1 pairs no server stamp (no lag,
    # no correlation).
    return {
        "dispatch drift": (
            metrics["dispatch_drift_p99_ms"] < config.dispatch_drift_threshold_ms
        ),
    }


def _multiturn_gate(metrics: Dict[str, float], config) -> Dict[str, bool]:
    """Multi-turn think time (C1): the think-time drift term plus correlation.

    ``think-time drift`` is late-only and gated on ``think_time_drift_threshold_ms``
    (a scheduling quantity, its own knob, riding in the ttP99 column). ``request
    correlation`` guards the PFID plumbing — a session whose turns did not both
    key to a server record is a measurement error, surfaced here rather than
    silently dropped. The server-arrival variant (ttSrvP99) is reported, not
    gated.
    """
    return {
        "think-time drift": (
            metrics["think_time_drift_p99_ms"] < config.think_time_drift_threshold_ms
        ),
        "request correlation": (
            metrics["unpaired_fraction"] < config.max_unpaired_fraction
        ),
    }


def build_workloads(config) -> List[PreflightWorkload]:
    """Workloads enabled by ``config``, in report order."""
    workloads: List[PreflightWorkload] = []
    if config.check_text:
        workloads.append(TextWorkload(config))
    if config.check_tts:
        workloads.append(TtsWorkload(config))
    if config.check_vajra_tts:
        workloads.append(VajraTtsWorkload(config))
    if config.check_stt:
        workloads.append(SttWorkload(config))
    if config.check_dispatch:
        workloads.append(DispatchWorkload(config))
    if config.check_multiturn:
        workloads.append(MultiTurnTextWorkload(config))
    return workloads
