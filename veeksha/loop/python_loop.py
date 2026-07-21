"""Pure-Python MainLoop implementation wrapping today's worker machinery.

This is the same pipeline `_run_main_loop` used to run —
PrefetchWorker -> TrafficScheduler -> DispatchWorker pool ->
ClientRunnerManager -> CompletionWorker pool — with exactly two changes:

1. DispatchWorker emits ``LoopEvent.DISPATCHED`` into an internal event
   queue instead of calling ``evaluator.register_request`` / the trace
   recorder; CompletionWorker still calls ``scheduler.notify_completion``
   first (loop-internal, timing-critical), then emits
   ``LoopEvent.COMPLETED`` instead of calling
   ``evaluator.record_request_completed``.
2. The traffic scheduler and client are constructed inside ``start()`` from
   the plain configs in ``MainLoopConfig``.

Consumption is pull-based (``drain_events``), identical to the native
native loop's drain surface, so the scoring replay code is shared verbatim
(``veeksha.loop.drain.ResultDrain``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Set

from veeksha.core.response import RequestResult
from veeksha.core.thread_pool import ThreadPoolManager
from veeksha.loop.interface import (
    LoopCounters,
    LoopEvent,
    LoopEventKind,
    MainLoopConfig,
    SessionSource,
)
from veeksha.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _SessionState:
    """Loop-side per-session completion accounting."""

    total: int
    observed: int = 0
    errored: int = 0
    cancelled: int = 0


class LoopEventSink:
    """Loop-side event collector shared by dispatch/completion workers.

    Builds the ordered ``LoopEvent`` stream and keeps the exact loop-side
    bookkeeping the monitor needs: dispatched request ids, request counters,
    and per-session completion accounting (mirroring the evaluator's session
    semantics so ``LoopCounters`` matches ``evaluator.get_session_counts()``).

    The DISPATCHED-before-COMPLETED order guarantee holds naturally: a
    request's DISPATCHED event is enqueued by the dispatch worker before the
    request ever reaches a client, and its COMPLETED event is enqueued by a
    completion worker strictly after.
    """

    def __init__(self) -> None:
        self._events: Queue = Queue()
        self._lock = threading.Lock()
        self._dispatched_ids: Set[int] = set()
        self._requests_dispatched = 0
        self._requests_completed = 0
        self._sessions: Dict[int, _SessionState] = {}
        self._sessions_seen = 0
        self._sessions_completed = 0
        self._sessions_errored = 0

    # ---- worker-facing API -------------------------------------------------

    def record_dispatched(
        self,
        request: Any,
        session_id: int,
        session_size: int,
        scheduler_ready_at: float,
        dispatched_at: float,
    ) -> None:
        """Record a dispatch and emit the DISPATCHED event."""
        with self._lock:
            self._dispatched_ids.add(request.id)
            self._requests_dispatched += 1

        self._events.put(
            LoopEvent(
                kind=LoopEventKind.DISPATCHED,
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_size,
                ready_at=scheduler_ready_at,
                dispatched_at=dispatched_at,
                request=request,
            )
        )

    def record_completed(self, result: RequestResult) -> None:
        """Record a completion and emit the COMPLETED event.

        Session accounting mirrors the evaluator: a session is seen on its
        first observed completion, and finalized (completed or errored) when
        all its requests have been observed. Sessions cut short by
        cancellation stay unfinalized, exactly like the evaluator leaves them
        in-progress until finalize.
        """
        session_id = result.session_id
        cancelled = bool(getattr(result, "cancelled", False))
        # Match the evaluator's error derivation exactly (performance/base.py
        # builds `Exception(error_msg) if result.error_msg else None`): a
        # result with an error_code but empty error_msg counts as SUCCESS
        # there, so it must here too or the counters split diverges.
        errored = bool(result.error_msg)

        with self._lock:
            self._requests_completed += 1

            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState(total=result.session_total_requests)
                self._sessions[session_id] = state
                self._sessions_seen += 1
            else:
                state.total = max(state.total, result.session_total_requests)

            state.observed += 1
            if cancelled:
                state.cancelled += 1
            elif errored:
                state.errored += 1

            if state.observed >= state.total:
                del self._sessions[session_id]
                if state.errored or state.cancelled:
                    self._sessions_errored += 1
                else:
                    self._sessions_completed += 1

        self._events.put(
            LoopEvent(
                kind=LoopEventKind.COMPLETED,
                request_id=result.request_id,
                session_id=session_id,
                session_total_requests=result.session_total_requests,
                result=result,
            )
        )

    # ---- loop-facing API ---------------------------------------------------

    def drain_events(self, max_items: int, timeout_s: float) -> List[LoopEvent]:
        """Pop up to ``max_items`` events, blocking up to ``timeout_s``."""
        events: List[LoopEvent] = []
        try:
            events.append(self._events.get(timeout=timeout_s))
        except Empty:
            return events
        while len(events) < max_items:
            try:
                events.append(self._events.get_nowait())
            except Empty:
                break
        return events

    def dispatched_request_ids(self) -> Set[int]:
        with self._lock:
            return set(self._dispatched_ids)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "sessions_completed": self._sessions_completed,
                "sessions_errored": self._sessions_errored,
                "sessions_seen": self._sessions_seen,
                "requests_dispatched": self._requests_dispatched,
                "requests_completed": self._requests_completed,
            }


class PythonMainLoop:
    """MainLoop implementation running today's Python worker pipeline.

    Construction dependencies beyond ``MainLoopConfig`` are impl-specific:
    the seed manager (traffic schedulers draw seeded streams from it) and
    either a pre-built client or a tokenizer provider to build one from
    config. Passing the pre-built client keeps warm caches (tokenizers, STT
    clip frames) shared with the warmup phase, exactly as before the
    refactor.
    """

    _POOL_JOIN_TIMEOUT_S = 1.0

    def __init__(
        self,
        config: MainLoopConfig,
        *,
        seed_manager: Any,
        tokenizer_provider: Any = None,
        client: Any = None,
    ):
        """Initialize the loop.

        Args:
            config: Minimal plain-data loop configuration.
            seed_manager: SeedManager used to build the traffic scheduler.
            tokenizer_provider: Tokenizer provider used to build the client
                from ``config.client`` when no client instance is given.
            client: Optional pre-built client instance (skips construction
                from config; used by the benchmark to share the warmup
                client, and by tests to inject scripted clients).
        """
        self._config = config
        self._seed_manager = seed_manager
        self._tokenizer_provider = tokenizer_provider
        self._client = client

        self._sink = LoopEventSink()
        self._stop_event = threading.Event()
        self._wait_for_in_flight_on_join = True

        self._scheduler = None
        self._client_runner = None
        self._pool_manager: Optional[ThreadPoolManager] = None
        self._output_queue: Queue = Queue()
        self._started = False
        self._joined = False

    # ---- MainLoop protocol ---------------------------------------------

    def start(self, source: SessionSource) -> None:
        """Build scheduler/client/workers from configs and start them."""
        from veeksha.client.registry import ClientRegistry
        from veeksha.traffic.registry import TrafficSchedulerRegistry
        from veeksha.workers.client_runner import ClientRunnerManager
        from veeksha.workers.completion import CompletionWorker
        from veeksha.workers.dispatch import DispatchWorker
        from veeksha.workers.prefetch import PrefetchWorker

        if self._started:
            raise RuntimeError("PythonMainLoop.start() called twice")
        self._started = True

        runtime = self._config.runtime

        self._scheduler = TrafficSchedulerRegistry.get(
            self._config.traffic.get_type(),
            config=self._config.traffic,
            seed_manager=self._seed_manager,
        )
        # Anchor the arrival clock to the benchmark's reference time (taken
        # before evaluator/trace-recorder construction), matching the old
        # loop's ordering: build cost must not shift the arrival schedule.
        self._scheduler.reset_reference_time(self._config.monotonic_anchor)

        if self._client is None:
            self._client = ClientRegistry.get(
                self._config.client.get_type(),
                config=self._config.client,
                tokenizer_provider=self._tokenizer_provider,
            )

        num_client_threads = runtime.num_client_threads
        if num_client_threads is None:
            # Provision client workers for the offered load (the sweep planner
            # already does this; direct configs get the same protection): an
            # under-provisioned pool serializes per-session sends and shows up
            # as phantom server-side latency at high concurrency.
            target_sessions = getattr(
                self._scheduler, "target_concurrent_sessions", None
            ) or getattr(self._scheduler, "_target_concurrent", None)
            num_client_threads = (
                max(3, -(-int(target_sessions) // 8)) if target_sessions else 3
            )

        client_queues = [Queue() for _ in range(num_client_threads)]
        self._output_queue = Queue()

        self._client_runner = ClientRunnerManager(
            client=self._client,
            input_queues=client_queues,
            output_queue=self._output_queue,
            stop_event=self._stop_event,
            traffic_scheduler=self._scheduler,
        )

        self._pool_manager = ThreadPoolManager(stop_event=self._stop_event)

        self._pool_manager.create_pool(
            name="prefetch",
            worker_class=PrefetchWorker,
            worker_kwargs={
                "traffic_scheduler": self._scheduler,
                "session_source": source,
            },
            pool_size=1,
        )

        self._pool_manager.create_pool(
            name="dispatch",
            worker_class=DispatchWorker,
            worker_kwargs={
                "traffic_scheduler": self._scheduler,
                "client_queues": client_queues,
                "event_sink": self._sink,
            },
            pool_size=runtime.num_dispatcher_threads,
        )

        self._pool_manager.create_pool(
            name="completion",
            worker_class=CompletionWorker,
            worker_kwargs={
                "output_queue": self._output_queue,
                "traffic_scheduler": self._scheduler,
                "event_sink": self._sink,
            },
            pool_size=runtime.num_completion_threads,
        )

        self._client_runner.start()
        self._pool_manager.start_all()

        logger.info(
            f"Started {self._pool_manager.get_total_thread_count()} worker threads "
            f"and {self._client_runner.get_worker_count()} client workers"
        )

    def drain_events(
        self, max_items: int = 256, timeout_s: float = 0.1
    ) -> List[LoopEvent]:
        """Pop up to max_items ordered events (blocks up to timeout_s)."""
        return self._sink.drain_events(max_items=max_items, timeout_s=timeout_s)

    def counters(self) -> LoopCounters:
        """Loop-side progress snapshot for the monitor."""
        counts = self._sink.snapshot()
        # Sample prefetch liveness BEFORE pending-work: once prefetch is dead
        # no new session can appear, so a subsequent idle reading is
        # authoritative. The reverse order has a window where a final session
        # lands between the two samples and the monitor exits early (the old
        # loop sampled in this order too).
        intake_exhausted = False
        if self._pool_manager is not None:
            prefetch_threads = self._pool_manager.thread_pools.get("prefetch", [])
            intake_exhausted = all(not t.is_alive() for t in prefetch_threads)
        if self._scheduler is not None:
            in_flight = len(self._scheduler.get_in_flight_request_ids())
            idle = not self._scheduler.has_pending_work()
        else:
            in_flight = 0
            idle = not self._started
        return LoopCounters(
            sessions_completed=counts["sessions_completed"],
            sessions_errored=counts["sessions_errored"],
            sessions_seen=counts["sessions_seen"],
            requests_dispatched=counts["requests_dispatched"],
            requests_completed=counts["requests_completed"],
            in_flight=in_flight,
            intake_exhausted=intake_exhausted,
            idle=idle,
        )

    def in_flight_request_ids(self) -> Set[int]:
        """Request ids currently tracked by the scheduler (exact)."""
        if self._scheduler is None:
            return set()
        return self._scheduler.get_in_flight_request_ids()

    def dispatched_request_ids(self) -> Set[int]:
        """All request ids dispatched so far (loop-side, exact)."""
        return self._sink.dispatched_request_ids()

    def request_stop(self, grace_s: float) -> None:
        """Stop dispatching new work.

        ``grace_s < 0`` requests a graceful drain: ``join()`` waits for all
        in-flight client work (today's behavior when the monitor exited with
        nothing pending). ``grace_s >= 0`` means the monitor already spent
        the grace budget, so ``join()`` must not wait for still-in-flight
        client tasks (they are cancelled instead).
        """
        self._wait_for_in_flight_on_join = grace_s < 0
        self._stop_event.set()

    def join(self, timeout_s: float = _POOL_JOIN_TIMEOUT_S) -> bool:
        """Shut down workers with today's stop + drain + sentinel sequence."""
        if not self._started or self._joined:
            return True
        self._joined = True

        assert self._pool_manager is not None and self._client_runner is not None

        self._stop_event.set()
        self._pool_manager.join_pool("prefetch", timeout=timeout_s)
        self._pool_manager.join_pool("dispatch", timeout=timeout_s)

        logger.info("Stopping client runner...")
        self._client_runner.stop()
        if self._wait_for_in_flight_on_join:
            self._client_runner.wait()

        for _ in range(self._config.runtime.num_completion_threads):
            self._output_queue.put(None)
        self._pool_manager.join_pool("completion", timeout=timeout_s)

        threads = [t for pool in self._pool_manager.thread_pools.values() for t in pool]
        return all(not t.is_alive() for t in threads)
