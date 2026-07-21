"""Unit tests for SessionSource implementations."""

import threading

import pytest

from tests.unit.loop.helpers import make_linear_session
from veeksha.loop.source import GeneratorSessionSource, PregeneratedSessionSource


class FakeGenerator:
    """Session generator producing up to ``limit`` sessions."""

    def __init__(self, limit: int = -1):
        self.limit = limit
        self.generated = 0

    def generate_session(self):
        if self.limit >= 0 and self.generated >= self.limit:
            raise StopIteration
        self.generated += 1
        return make_linear_session(self.generated, 1)


@pytest.mark.unit
def test_generator_source_enforces_max_sessions() -> None:
    source = GeneratorSessionSource(FakeGenerator(), max_sessions=3)

    sessions = [source.next_session() for _ in range(5)]

    assert [s is not None for s in sessions] == [True, True, True, False, False]
    assert source.count == 3


@pytest.mark.unit
def test_generator_source_unlimited_until_generator_exhausts() -> None:
    source = GeneratorSessionSource(FakeGenerator(limit=4), max_sessions=-1)

    produced = 0
    while source.next_session() is not None:
        produced += 1

    assert produced == 4
    # Once exhausted, stays exhausted.
    assert source.next_session() is None


@pytest.mark.unit
def test_generator_source_thread_safe_max_sessions() -> None:
    max_sessions = 50
    source = GeneratorSessionSource(FakeGenerator(), max_sessions=max_sessions)
    results = []
    lock = threading.Lock()

    def pull():
        while True:
            session = source.next_session()
            if session is None:
                return
            with lock:
                results.append(session.id)

    threads = [threading.Thread(target=pull) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == max_sessions
    assert len(set(results)) == max_sessions  # no duplicates


@pytest.mark.unit
def test_pregenerated_source_serves_in_order() -> None:
    sessions = [make_linear_session(i, 1) for i in range(1, 4)]
    source = PregeneratedSessionSource(sessions)

    served = [source.next_session() for _ in range(4)]

    assert [s.id for s in served[:3]] == [1, 2, 3]
    assert served[3] is None
    assert source.count == 3


@pytest.mark.unit
def test_pregenerated_source_respects_max_sessions() -> None:
    sessions = [make_linear_session(i, 1) for i in range(1, 6)]
    source = PregeneratedSessionSource(sessions, max_sessions=2)

    assert source.next_session() is not None
    assert source.next_session() is not None
    assert source.next_session() is None
