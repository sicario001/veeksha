"""Completion worker for processing completed requests."""

import time
from queue import Empty, Queue
from typing import TYPE_CHECKING

from veeksha.core.context import WorkerContext
from veeksha.core.response import RequestResult
from veeksha.logger import init_logger
from veeksha.traffic.base import BaseTrafficScheduler

if TYPE_CHECKING:
    from veeksha.loop.python_loop import LoopEventSink

logger = init_logger(__name__)


QUEUE_GET_TIMEOUT_S = 0.1
DRAIN_MAX_EMPTY_POLLS = 5


class CompletionWorker:
    """Worker that processes completed requests from client output queue.

    This worker:
    1. Receives RequestResult from client output queue
    2. Notifies traffic scheduler of completion (loop-internal and
       timing-critical: refills concurrency / releases next turns)
    3. Emits a COMPLETED loop event (consumed outside the loop by the
       scoring drain, which replays it into the evaluator)
    """

    def __init__(
        self,
        output_queue: Queue,
        traffic_scheduler: BaseTrafficScheduler,
        event_sink: "LoopEventSink",
        worker_context: WorkerContext,
    ):
        """Initialize the completion worker.

        Args:
            output_queue: Queue receiving RequestResult from client workers
            traffic_scheduler: Scheduler to notify of completions
            event_sink: Loop-side sink receiving COMPLETED events
            worker_context: Worker context with stop event
        """
        self.output_queue = output_queue
        self.traffic_scheduler = traffic_scheduler
        self.event_sink = event_sink
        self.worker_context = worker_context

    def _process_result(self, result: RequestResult) -> None:
        """Process a single request result."""
        result.result_processed_at = time.monotonic()

        # Scheduler notification comes FIRST: it is on the refill /
        # next-turn-release path and must not wait on anything else.
        self.traffic_scheduler.notify_completion(
            request_id=result.request_id,
            completed_at_monotonic=result.client_completed_at,  # type: ignore
            success=result.success,
            channel_responses=result.channels if result.success else None,
        )

        self.event_sink.record_completed(result)

    def run(self) -> None:
        """Main worker loop."""
        logger.debug("Completion worker %s starting", self.worker_context.worker_id)

        while not self.worker_context.stop_event.is_set():
            try:
                item = self.output_queue.get(timeout=QUEUE_GET_TIMEOUT_S)
            except Empty:
                continue

            # sentinel
            if item is None:
                break

            self._process_result(item)

        self._drain()

        logger.debug("Completion worker %s exiting", self.worker_context.worker_id)

    def _drain(self) -> None:
        """Drain any remaining results from the queue."""
        logger.debug(
            "Completion worker %s: draining queue", self.worker_context.worker_id
        )
        empty_polls = 0
        while empty_polls < DRAIN_MAX_EMPTY_POLLS:
            try:
                item = self.output_queue.get_nowait()
                # sentinel
                if item is None:
                    break
                self._process_result(item)
                empty_polls = 0
            except Empty:
                empty_polls += 1
                time.sleep(0.01)
