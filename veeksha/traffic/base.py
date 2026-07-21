from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Mapping, Optional, Set, Tuple

from veeksha.config.traffic import BaseTrafficConfig
from veeksha.core.request import Request
from veeksha.core.response import ChannelResponse
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.types import ChannelModality

if TYPE_CHECKING:
    from veeksha.traffic.dispatch_tracker import DispatchTracker


class BaseTrafficScheduler:
    def __init__(self, config: BaseTrafficConfig, seed_manager: SeedManager):
        self.config = config
        self.seed_manager = seed_manager

    @abstractmethod
    def schedule_session(self, session: Session) -> None:
        """Schedule a session for dispatch."""
        raise NotImplementedError

    @abstractmethod
    def pop_ready(self) -> Optional[Tuple[Request, int, int]]:
        """Pop a ready request from the scheduler.

        Returns:
            Tuple of (request, session_id, session_size) if a request is ready,
            None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def wait_for_ready(
        self, timeout: float = 0.001
    ) -> Optional[Tuple[Request, int, int]]:
        """Wait for a ready request with timeout.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Tuple of (request, session_id, session_size) if ready, None if timeout.
        """
        raise NotImplementedError

    @abstractmethod
    def notify_completion(
        self,
        request_id: int,
        completed_at_monotonic: float,
        success: bool,
        channel_responses: Optional[Mapping[ChannelModality, ChannelResponse]] = None,
    ) -> None:
        """Notify the scheduler that a request has completed."""
        raise NotImplementedError

    @abstractmethod
    def get_session_id(self, request_id: int) -> int:
        """Get the session ID for a given request ID.

        Returns -1 if the request is not found.
        """
        raise NotImplementedError

    @abstractmethod
    def get_session_size(self, request_id: int) -> int:
        """Get the total number of requests in the session for a given request ID.

        Returns 1 if the request is not found.
        """
        raise NotImplementedError

    @abstractmethod
    def has_pending_work(self) -> bool:
        """Check if there are pending sessions or in-flight requests."""
        raise NotImplementedError

    @abstractmethod
    def get_in_flight_request_ids(self) -> Set[int]:
        """Return the set of request IDs currently in-flight."""
        raise NotImplementedError

    @property
    def dispatch_tracker(self) -> Optional[DispatchTracker]:
        """Optional dispatch tracker for ticket-based ordering."""
        return None

    def notify_request_sent(self, request_id: int) -> None:
        """Called when the server acknowledges a request (HTTP 200 received).

        The default implementation is a no-op.  Subclasses (e.g.
        :class:`SequentialLaunchTrafficScheduler`) may use this to gate
        activation of pending sessions.
        """
        return

    def reset_reference_time(self, anchor: Optional[float] = None) -> None:
        """Optional hook invoked before the benchmark starts dispatching.

        ``anchor`` is the benchmark's ``time.monotonic()`` reference; ``None``
        means "now". Schedulers that pace against a virtual clock re-align to
        it so arrival schedules are anchored at benchmark start, not at
        scheduler construction.
        """
        return
