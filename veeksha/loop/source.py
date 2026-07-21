"""SessionSource implementations.

These absorb the old PrefetchWorker intake plumbing: the generator lock, the
``SharedSessionCounter`` max-sessions accounting, and the pregenerated-list
indexing all live here, behind one thread-safe ``next_session()`` call.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from veeksha.core.session import Session
from veeksha.generator.session.base import BaseSessionGenerator
from veeksha.logger import init_logger

logger = init_logger(__name__)


class GeneratorSessionSource:
    """Pulls sessions from a session generator, enforcing ``max_sessions``.

    Thread-safe: generation happens under an internal lock (session
    generators are not required to be thread-safe).
    """

    def __init__(self, session_generator: BaseSessionGenerator, max_sessions: int = -1):
        """Initialize the source.

        Args:
            session_generator: Generator to pull sessions from.
            max_sessions: Maximum number of sessions to produce. -1 for
                unlimited (until the generator raises StopIteration).
        """
        self._generator = session_generator
        self._max_sessions = max_sessions
        self._lock = threading.Lock()
        self._count = 0
        self._exhausted = False

    def next_session(self) -> Optional[Session]:
        """Return the next session, or None when exhausted."""
        with self._lock:
            if self._exhausted:
                return None
            if self._max_sessions >= 0 and self._count >= self._max_sessions:
                self._exhausted = True
                return None
            try:
                session = self._generator.generate_session()
            except StopIteration:
                logger.debug("Session source: generator exhausted at %d", self._count)
                self._exhausted = True
                return None
            self._count += 1
            return session

    @property
    def count(self) -> int:
        """Number of sessions produced so far."""
        return self._count


class PregeneratedSessionSource:
    """Serves sessions from a pre-generated list.

    Thread-safe. ``max_sessions`` caps consumption when the list is longer
    (the benchmark pregenerates at most ``max_sessions`` entries already, so
    the cap is a safety net for direct users).
    """

    def __init__(self, sessions: List[Session], max_sessions: int = -1):
        """Initialize the source.

        Args:
            sessions: Pre-generated sessions, served in order.
            max_sessions: Maximum number of sessions to serve. -1 serves the
                whole list.
        """
        self._sessions = sessions
        self._max_sessions = max_sessions
        self._lock = threading.Lock()
        self._index = 0

    def next_session(self) -> Optional[Session]:
        """Return the next session, or None when exhausted."""
        with self._lock:
            if self._index >= len(self._sessions):
                return None
            if self._max_sessions >= 0 and self._index >= self._max_sessions:
                return None
            session = self._sessions[self._index]
            self._index += 1
            return session

    @property
    def count(self) -> int:
        """Number of sessions served so far."""
        return self._index
