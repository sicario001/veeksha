"""Dispatch worker for sending requests to client queues."""

import random
import time
from queue import Queue
from typing import TYPE_CHECKING, List

from veeksha.core.context import WorkerContext
from veeksha.logger import init_logger
from veeksha.traffic.base import BaseTrafficScheduler

if TYPE_CHECKING:
    from veeksha.loop.python_loop import LoopEventSink

logger = init_logger(__name__)


class DispatchWorker:
    """Worker that polls ready requests from scheduler and dispatches to client queues.

    This worker:
    1. Waits for ready requests from the traffic scheduler (wait_for_ready)
    2. Emits a DISPATCHED loop event (consumed outside the loop by the
       scoring drain, which replays it into the evaluator / trace recorder)
    3. Dispatches requests to client worker queues
    """

    _WAIT_TIMEOUT_S = 0.01

    def __init__(
        self,
        traffic_scheduler: BaseTrafficScheduler,
        client_queues: List[Queue],
        event_sink: "LoopEventSink",
        worker_context: WorkerContext,
    ):
        """Initialize the dispatch worker.

        Args:
            traffic_scheduler: Scheduler to poll for ready requests
            client_queues: Queues to dispatch requests to (one per client worker)
            event_sink: Loop-side sink receiving DISPATCHED events
            worker_context: Worker context with stop event
        """
        self.traffic_scheduler = traffic_scheduler
        self.client_queues = client_queues
        self.event_sink = event_sink
        self.worker_context = worker_context

    def _select_queue(self) -> Queue:
        """Select a client queue using power-of-two load balancing"""
        n = len(self.client_queues)
        if n == 1:
            return self.client_queues[0]

        idx1, idx2 = random.sample(range(n), 2)

        q1 = self.client_queues[idx1]
        q2 = self.client_queues[idx2]

        return q1 if q1.qsize() <= q2.qsize() else q2

    def _dispatch(self, request, session_id: int, session_size: int) -> None:
        """Stamp, record, and enqueue one ready request."""
        scheduler_ready_at = time.monotonic()
        dispatched_at = time.monotonic()

        self.event_sink.record_dispatched(
            request=request,
            session_id=session_id,
            session_size=session_size,
            scheduler_ready_at=scheduler_ready_at,
            dispatched_at=dispatched_at,
        )

        queue = self._select_queue()
        queue.put(
            (request, session_id, session_size, scheduler_ready_at, dispatched_at)
        )

    def run(self) -> None:
        """Main worker loop."""
        logger.debug("Dispatch worker %s starting", self.worker_context.worker_id)

        while not self.worker_context.stop_event.is_set():
            result = self.traffic_scheduler.wait_for_ready(timeout=self._WAIT_TIMEOUT_S)

            if result is None:
                continue

            request, session_id, session_size = result
            self._dispatch(request, session_id, session_size)

        # Drain remaining ready requests
        self._drain()

        logger.debug("Dispatch worker %s exiting", self.worker_context.worker_id)

    def _get_session_id(self, request) -> int:
        """Get session ID for a request."""
        return self.traffic_scheduler.get_session_id(request.id)

    def _get_session_size(self, request) -> int:
        """Get total number of requests in the session."""
        return self.traffic_scheduler.get_session_size(request.id)

    def _drain(self) -> None:
        """Drain any remaining ready requests."""
        logger.debug(
            "Dispatch worker %s: draining scheduler", self.worker_context.worker_id
        )
        while True:
            result = self.traffic_scheduler.pop_ready()
            if result is None:
                break

            request, session_id, session_size = result
            self._dispatch(request, session_id, session_size)
