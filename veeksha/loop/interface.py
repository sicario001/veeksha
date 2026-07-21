"""Minimal struct interface between the benchmark runner and the main loop.

The main loop (Python or native) is configured with plain, immutable configs
(``MainLoopConfig``), pulls sessions from a ``SessionSource``, and emits an
ordered ``LoopEvent`` stream. Everything that is scoring — evaluator,
trace recorder — lives *outside* the loop and consumes ``drain_events()``
(see ``veeksha.loop.drain.ResultDrain``).

Ordering guarantee: for any request, its ``DISPATCHED`` event is enqueued
before its ``COMPLETED`` event, and ``drain_events()`` preserves enqueue
order. This makes ``evaluator.register_request`` /
``record_request_completed`` replay valid.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from veeksha.config.client import BaseClientConfig
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.traffic import BaseTrafficConfig
from veeksha.core.request import Request
from veeksha.core.response import RequestResult
from veeksha.core.session import Session
from veeksha.types import ChannelModality


@dataclass(frozen=True)
class MainLoopConfig:
    """Plain-data configuration for a main loop implementation.

    The vidhi configs referenced here are frozen dataclasses that mirror 1:1
    onto the native structs (NativeRuntimeConfig / TrafficPlanConfig /
    EndpointConfig); the native adapter converts them once at start.

    Attributes:
        runtime: Runtime knobs (thread counts, timeouts, max_sessions).
        traffic: Traffic scheduling semantics (rate / concurrent / sequential).
            Each loop implementation builds its own scheduler from this config.
        client: Client/endpoint configuration. Each loop implementation builds its own
            transport from this config.
        monotonic_anchor: ``time.monotonic()`` reading taken at benchmark
            start (replaces the old ``benchmark_start_time`` argument).
    """

    runtime: RuntimeConfig
    traffic: BaseTrafficConfig
    client: BaseClientConfig
    monotonic_anchor: float


class LoopEventKind(enum.Enum):
    """Kind of a loop event."""

    DISPATCHED = "dispatched"
    COMPLETED = "completed"


@dataclass
class LoopEvent:
    """One entry of the ordered event stream emitted by a main loop.

    ``DISPATCHED`` events carry everything ``evaluator.register_request`` and
    ``trace_recorder.record_dispatch`` need; ``COMPLETED`` events carry the
    full ``RequestResult`` for ``evaluator.record_request_completed``.

    Attributes:
        kind: DISPATCHED or COMPLETED.
        request_id: Benchmark-scoped request id.
        session_id: Session the request belongs to.
        session_total_requests: Total number of requests in the session.
        ready_at: Monotonic timestamp when the scheduler marked the request
            ready (DISPATCHED only).
        dispatched_at: Monotonic timestamp of dispatch (DISPATCHED only).
        result: The completed request result (COMPLETED only).
        request: The original request object (DISPATCHED only). The replay
            derives everything the old direct call sites passed from it —
            ``request.channels`` for ``register_request``, and
            ``request.requested_output``; ``record_dispatch`` reads channels,
            history and session_context. The native main loop populates this
            from its compiler-side sidecar map.
    """

    kind: LoopEventKind
    request_id: int
    session_id: int
    session_total_requests: int
    ready_at: Optional[float] = None
    dispatched_at: Optional[float] = None
    result: Optional[RequestResult] = None
    request: Optional[Request] = None


@dataclass(frozen=True)
class LoopCounters:
    """Lock-free-ish snapshot of loop progress for the monitor.

    Session counts are loop-side and exact (the monitor no longer reads
    evaluator counts, which trail reality by the drain lag). Semantics match
    the evaluator's ``get_session_counts()``: a session is *seen* when its
    first request completes, *completed* when all its requests completed
    successfully, and *errored* when it finished with any errored or
    cancelled request.
    """

    sessions_completed: int = 0
    sessions_errored: int = 0
    sessions_seen: int = 0
    requests_dispatched: int = 0
    requests_completed: int = 0
    in_flight: int = 0
    intake_exhausted: bool = False
    idle: bool = False


@runtime_checkable
class SessionSource(Protocol):
    """Pull-based session intake.

    Replaces: session_generator + pregenerated_sessions + generator_lock +
    SharedSessionCounter. Implementations must be thread-safe and enforce
    ``max_sessions`` themselves.
    """

    def next_session(self) -> Optional[Session]:
        """Return the next session, or None when exhausted."""
        ...


@runtime_checkable
class MainLoop(Protocol):
    """One control surface for both loop implementations. Replaces ``_run_main_loop``."""

    def start(self, source: SessionSource) -> None:
        """Build internal machinery from configs and start dispatching."""
        ...

    def drain_events(
        self, max_items: int = 256, timeout_s: float = 0.1
    ) -> List[LoopEvent]:
        """Pop up to ``max_items`` ordered events; block up to ``timeout_s``
        for the first one. Empty list = nothing pending."""
        ...

    def counters(self) -> LoopCounters:
        """Snapshot of loop-side progress counters."""
        ...

    def in_flight_request_ids(self) -> Set[int]:
        """Request ids currently in flight (loop-side, exact)."""
        ...

    def dispatched_request_ids(self) -> Set[int]:
        """All request ids dispatched so far (timeout bookkeeping)."""
        ...

    def request_stop(self, grace_s: float) -> None:
        """Stop dispatching new work. ``grace_s < 0`` means wait for all
        in-flight requests during ``join``; ``grace_s >= 0`` means the grace
        budget was already spent by the monitor and ``join`` must not wait
        for still-in-flight client work."""
        ...

    def join(self, timeout_s: float) -> bool:
        """Shut down workers and wait for them. Returns True when all loop
        threads exited within the budget."""
        ...
