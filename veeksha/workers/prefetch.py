"""Prefetch worker for session generation and scheduling."""

import time
from typing import TYPE_CHECKING

from veeksha.core.context import WorkerContext
from veeksha.logger import init_logger
from veeksha.traffic.base import BaseTrafficScheduler

if TYPE_CHECKING:
    from veeksha.loop.interface import SessionSource

logger = init_logger(__name__)


class PrefetchWorker:
    """Worker that pulls sessions from a source and schedules them.

    This worker pulls sessions from a ``SessionSource`` (which owns generator
    locking and max-sessions accounting) and feeds them to the traffic
    scheduler, which then manages the dispatch timing of individual requests.
    """

    # unthrottled for first _BURST_DURATION_S seconds, then throttles
    _BURST_DURATION_S = 5.0
    _MAX_POLL_INTERVAL_S = 0.05

    def __init__(
        self,
        traffic_scheduler: BaseTrafficScheduler,
        session_source: "SessionSource",
        worker_context: WorkerContext,
    ):
        """Initialize the prefetch worker.

        Args:
            traffic_scheduler: Scheduler to schedule sessions with
            session_source: Thread-safe source of sessions (max_sessions
                enforced by the source)
            worker_context: Worker context with stop event
        """
        self.traffic_scheduler = traffic_scheduler
        self.session_source = session_source
        self.worker_context = worker_context

    def _get_poll_interval(self) -> float:
        """Calculate poll interval based on runtime duration.

        Unthrottled for the first _BURST_DURATION_S seconds, then throttles
        to _MAX_POLL_INTERVAL_S.

        Returns:
            Poll interval in seconds.
        """
        if time.monotonic() - self._start_time < self._BURST_DURATION_S:
            return 0.0
        return self._MAX_POLL_INTERVAL_S

    def run(self) -> None:
        """Main worker loop."""
        logger.debug("Prefetch worker %s starting", self.worker_context.worker_id)

        self._start_time = time.monotonic()
        scheduled = 0

        while not self.worker_context.stop_event.is_set():
            session = self.session_source.next_session()
            if session is None:
                logger.info(
                    "Prefetch worker %s: no more sessions to generate",
                    self.worker_context.worker_id,
                )
                break

            # Schedule the session with traffic scheduler
            self.traffic_scheduler.schedule_session(session)
            scheduled += 1

            if scheduled % 100 == 0:
                logger.debug("Prefetch progress: %d sessions scheduled", scheduled)

            # Throttle (burst at start, then steady-state)
            time.sleep(self._get_poll_interval())

        logger.debug("Prefetch worker %s exiting", self.worker_context.worker_id)
