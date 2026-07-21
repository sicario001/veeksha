"""ResultDrain: replays loop events into the evaluator and trace recorder.

Shared by both loop implementations: the loop (Python or native) only emits an ordered
``LoopEvent`` stream; this drain consumes ``loop.drain_events()`` and calls
``evaluator.register_request`` / ``record_request_completed`` (and
``trace_recorder.record_dispatch``) off the loop's critical path.

Events are routed to scoring workers by ``hash(session_id)``: per-session
order (DISPATCHED before COMPLETED, turn order) is preserved because one
router thread pulls the globally ordered stream and each session always maps
to the same worker queue. Cross-session order is irrelevant to the
evaluator. Pool size defaults to ``runtime.num_completion_threads`` so ASR
scoring keeps the same parallelism it had on the completion workers.
"""

from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Any, List, Optional

from veeksha.loop.interface import LoopEvent, LoopEventKind, MainLoop
from veeksha.logger import init_logger

logger = init_logger(__name__)


class ResultDrain:
    """Thread pool consuming ``loop.drain_events()`` and scoring them."""

    _DRAIN_MAX_ITEMS = 256
    _DRAIN_TIMEOUT_S = 0.1

    def __init__(
        self,
        loop: MainLoop,
        evaluator: Any,
        trace_recorder: Any = None,
        num_threads: int = 8,
    ):
        """Initialize the drain.

        Args:
            loop: Main loop to drain events from.
            evaluator: Evaluator receiving the replayed events.
            trace_recorder: Optional trace recorder for DISPATCHED events.
            num_threads: Number of scoring worker threads (defaults should
                come from ``runtime.num_completion_threads``).
        """
        self._loop = loop
        self._evaluator = evaluator
        self._trace_recorder = trace_recorder
        self._num_threads = max(1, num_threads)

        self._queues: List[Queue] = [Queue() for _ in range(self._num_threads)]
        self._loop_finished = threading.Event()
        self._router_thread: Optional[threading.Thread] = None
        self._worker_threads: List[threading.Thread] = []
        self._started = False

    def start(self) -> None:
        """Start the router and scoring worker threads."""
        if self._started:
            raise RuntimeError("ResultDrain.start() called twice")
        self._started = True

        for i, queue in enumerate(self._queues):
            thread = threading.Thread(
                target=self._score_loop,
                args=(queue,),
                name=f"scoring-{i}",
                daemon=True,
            )
            self._worker_threads.append(thread)
            thread.start()

        self._router_thread = threading.Thread(
            target=self._route_loop, name="scoring-router", daemon=True
        )
        self._router_thread.start()

    def join(self, timeout_s: Optional[float] = None) -> None:
        """Flush and stop. Must be called after ``loop.join()`` returned.

        The router keeps draining until the loop's event queue is empty
        (guaranteed final because the loop's workers have exited), then each
        scoring worker drains its own queue to the sentinel — so every event
        emitted by the loop is replayed before this returns.
        """
        if not self._started:
            return
        self._loop_finished.set()
        if self._router_thread is not None:
            self._router_thread.join(timeout_s)
        for queue in self._queues:
            queue.put(None)
        for thread in self._worker_threads:
            thread.join(timeout_s)

    # ---- internals -------------------------------------------------------

    def _route_loop(self) -> None:
        """Pull ordered events from the loop and fan out by session."""
        while True:
            events = self._loop.drain_events(
                max_items=self._DRAIN_MAX_ITEMS, timeout_s=self._DRAIN_TIMEOUT_S
            )
            for event in events:
                queue = self._queues[hash(event.session_id) % self._num_threads]
                queue.put(event)
            if not events and self._loop_finished.is_set():
                # Loop has joined and its queue read empty — but a straggler
                # worker that survived a pool-join timeout could still emit.
                # Double-check after a beat; exit only if still dry.
                time.sleep(0.2)
                late = self._loop.drain_events(
                    max_items=self._DRAIN_MAX_ITEMS, timeout_s=0.05
                )
                if not late:
                    return
                for event in late:
                    queue = self._queues[hash(event.session_id) % self._num_threads]
                    queue.put(event)

    def _score_loop(self, queue: Queue) -> None:
        """Replay events for the sessions routed to this worker."""
        while True:
            event = queue.get()
            if event is None:
                return
            try:
                self._replay(event)
            except Exception:
                logger.exception(
                    "Scoring drain failed to replay event for request %s",
                    event.request_id,
                )

    def _replay(self, event: LoopEvent) -> None:
        """Replay one event into the evaluator (and trace recorder)."""
        if event.kind == LoopEventKind.DISPATCHED:
            self._evaluator.register_request(
                request_id=event.request_id,
                session_id=event.session_id,
                dispatched_at=event.dispatched_at,
                channels=event.request.channels if event.request else {},
                requested_output=(
                    event.request.requested_output if event.request else None
                ),
            )
            if self._trace_recorder is not None:
                self._trace_recorder.record_dispatch(
                    request=event.request,
                    session_id=event.session_id,
                    session_size=event.session_total_requests,
                    dispatched_at=event.dispatched_at,
                )
        else:
            result = event.result
            assert result is not None
            error = Exception(result.error_msg) if result.error_msg else None
            self._evaluator.record_request_completed(
                request_id=result.request_id,
                session_id=result.session_id,
                completed_at=result.client_completed_at,
                response=result,
                error=error,
            )
