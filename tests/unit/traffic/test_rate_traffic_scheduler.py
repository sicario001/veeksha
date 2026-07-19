"""Unit tests for RateTrafficScheduler."""

import time
from typing import Any, Dict, Optional

import pytest

from veeksha.config.generator.interval import FixedIntervalGeneratorConfig
from veeksha.config.traffic import RateTrafficConfig
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import (
    SessionEdge,
    SessionGraph,
    SessionNode,
    add_edge,
    add_node,
)
from veeksha.traffic.rate import RateTrafficScheduler
from veeksha.types import ChannelModality


def wait_until(predicate, timeout_s=0.5, interval_s=0.005):
    """Wait until predicate returns True or timeout."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def pop_ready_with_timeout(scheduler, timeout_s=0.2) -> Optional[Request]:
    """Helper to pop a ready request with a timeout."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        result = scheduler.pop_ready()
        if result is not None:
            return result[0]
        time.sleep(0.005)
    return None


def make_request(request_id: int, model: str = "mock") -> Request:
    """Create a simple test request."""
    return Request(
        id=request_id,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text=f"test_{request_id}"
            )
        },
    )


def make_linear_session(
    session_id: int, num_requests: int, wait_times: Optional[list] = None
) -> Session:
    """Create a session with a linear chain of requests: 0 -> 1 -> 2 -> ..."""
    graph = SessionGraph()
    requests: Dict[int, Request] = {}

    if wait_times is None:
        wait_times = [0.0] * num_requests

    for i in range(num_requests):
        add_node(graph, SessionNode(id=i, wait_after_ready=wait_times[i]))
        requests[i] = make_request(request_id=session_id * 100 + i)

    for i in range(num_requests - 1):
        add_edge(graph, SessionEdge(src=i, dst=i + 1))

    return Session(id=session_id, session_graph=graph, requests=requests)


def make_scheduler(interval: float = 0.1) -> RateTrafficScheduler:
    """Create a scheduler with fixed interval generator."""
    config = RateTrafficConfig(
        interval_generator=FixedIntervalGeneratorConfig(interval=interval)
    )
    seed_manager = SeedManager(seed=42)
    return RateTrafficScheduler(config, seed_manager)


@pytest.mark.unit
def test_single_request_session_dispatches() -> None:
    """Single-request session dispatches at the generated start time."""
    scheduler = make_scheduler(interval=0.05)
    session = make_linear_session(session_id=1, num_requests=1)

    scheduler.schedule_session(session)

    # Should not be ready immediately (first session starts at t=0, but need tiny delay for processing)
    # With interval=0.05, first session starts at t=0
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.2)


@pytest.mark.unit
def test_linear_chain_respects_dependencies() -> None:
    """Linear chain waits for each request to complete before releasing next."""
    scheduler = make_scheduler(interval=0.01)
    session = make_linear_session(
        session_id=2, num_requests=3, wait_times=[0.0, 0.02, 0.02]
    )

    scheduler.schedule_session(session)

    # First request ready
    req1 = None
    assert wait_until(
        lambda: (req1 := scheduler.pop_ready()) is not None, timeout_s=0.2
    )

    # Second not ready yet (first not completed)
    assert scheduler.pop_ready() is None

    # Complete first request
    scheduler.notify_completion(
        request_id=200, completed_at_monotonic=time.monotonic(), success=True
    )

    # Second becomes ready after its wait_after_ready
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.2)


@pytest.mark.unit
def test_cancellation_drops_pending_requests() -> None:
    """Failure with cancel_on_failure drops remaining session requests."""
    scheduler = make_scheduler(interval=0.01)
    session = make_linear_session(session_id=3, num_requests=3)

    scheduler.schedule_session(session)

    # First request ready
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.2)

    # Fail the first request
    scheduler.notify_completion(
        request_id=300, completed_at_monotonic=time.monotonic(), success=False
    )

    # No more requests should become ready
    time.sleep(0.05)
    assert scheduler.pop_ready() is None


@pytest.mark.unit
def test_multiple_sessions_interleave() -> None:
    """Multiple sessions interleave without interference."""
    scheduler = make_scheduler(interval=0.01)
    session1 = make_linear_session(session_id=10, num_requests=2)
    session2 = make_linear_session(session_id=20, num_requests=2)

    scheduler.schedule_session(session1)
    scheduler.schedule_session(session2)

    # Both first requests should become ready
    ready_count = 0
    for _ in range(10):
        if wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.1):
            ready_count += 1
        if ready_count >= 2:
            break

    assert ready_count >= 2, "Expected both root requests to become ready"


@pytest.mark.unit
def test_fan_out_session() -> None:
    """Session with fan-out: one root spawns multiple parallel children."""
    scheduler = make_scheduler(interval=0.01)

    # Create graph: 0 -> 1, 0 -> 2 (fan-out)
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    add_node(graph, SessionNode(id=1, wait_after_ready=0.0))
    add_node(graph, SessionNode(id=2, wait_after_ready=0.0))
    add_edge(graph, SessionEdge(src=0, dst=1))
    add_edge(graph, SessionEdge(src=0, dst=2))

    requests = {
        0: make_request(400),
        1: make_request(401),
        2: make_request(402),
    }
    session = Session(id=40, session_graph=graph, requests=requests)

    scheduler.schedule_session(session)

    # Root (0) should be ready
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.2)

    # Children not ready yet
    assert scheduler.pop_ready() is None

    # Complete root
    scheduler.notify_completion(
        request_id=400, completed_at_monotonic=time.monotonic(), success=True
    )

    # Both children should become ready
    child1 = None
    child2 = None
    assert wait_until(
        lambda: (child1 := scheduler.pop_ready()) is not None, timeout_s=0.2
    )
    assert wait_until(
        lambda: (child2 := scheduler.pop_ready()) is not None, timeout_s=0.2
    )


@pytest.mark.unit
def test_fan_in_session() -> None:
    """Session with fan-in: multiple parents must complete before child is ready."""
    scheduler = make_scheduler(interval=0.01)

    # Create graph: 0 -> 2, 1 -> 2 (fan-in to node 2)
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    add_node(graph, SessionNode(id=1, wait_after_ready=0.0))
    add_node(graph, SessionNode(id=2, wait_after_ready=0.0))
    add_edge(graph, SessionEdge(src=0, dst=2, is_history_parent=False))
    add_edge(graph, SessionEdge(src=1, dst=2, is_history_parent=True))

    requests = {
        0: make_request(500),
        1: make_request(501),
        2: make_request(502),
    }
    session = Session(id=50, session_graph=graph, requests=requests)

    scheduler.schedule_session(session)

    # Both roots should be ready (0 and 1 have no parents)
    root1 = None
    root2 = None
    assert wait_until(
        lambda: (root1 := scheduler.pop_ready()) is not None, timeout_s=0.2
    )
    assert wait_until(
        lambda: (root2 := scheduler.pop_ready()) is not None, timeout_s=0.2
    )

    # Child (2) not ready yet
    assert scheduler.pop_ready() is None

    # Complete only one parent
    scheduler.notify_completion(
        request_id=500, completed_at_monotonic=time.monotonic(), success=True
    )

    # Child still not ready (other parent not done)
    time.sleep(0.02)
    assert scheduler.pop_ready() is None

    # Complete second parent
    scheduler.notify_completion(
        request_id=501, completed_at_monotonic=time.monotonic(), success=True
    )

    # Now child should be ready
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.2)


@pytest.mark.unit
def test_session_garbage_collected_when_complete() -> None:
    """Session state is cleaned up after all requests complete."""
    scheduler = make_scheduler(interval=0.01)
    session = make_linear_session(session_id=60, num_requests=1)

    scheduler.schedule_session(session)
    assert len(scheduler._sessions) == 1

    # Pop and complete the request
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.2)
    scheduler.notify_completion(
        request_id=6000, completed_at_monotonic=time.monotonic(), success=True
    )

    # Session should be garbage collected
    assert len(scheduler._sessions) == 0


@pytest.mark.unit
def test_history_inheritance() -> None:
    """Requests inherit history from their history parent."""
    scheduler = make_scheduler(interval=0.01)

    # Linear: 0 -> 1 -> 2
    session = make_linear_session(session_id=1, num_requests=3)
    scheduler.schedule_session(session)

    # 0 ready
    req0 = pop_ready_with_timeout(scheduler)
    assert req0 is not None
    assert req0.history == []

    # Complete 0 with history
    response_0 = ChannelResponse(modality=ChannelModality.TEXT, content="response_0")
    scheduler.notify_completion(
        request_id=req0.id,
        completed_at_monotonic=time.monotonic(),
        success=True,
        channel_responses={ChannelModality.TEXT: response_0},
    )

    # 1 ready
    req1 = pop_ready_with_timeout(scheduler)
    assert req1 is not None
    assert len(req1.history) == 2
    assert req1.history[0] == {"role": "user", "content": "test_100"}
    assert req1.history[1] == {"role": "assistant", "content": "response_0"}

    # Complete 1 with history
    response_1 = ChannelResponse(modality=ChannelModality.TEXT, content="response_1")
    scheduler.notify_completion(
        request_id=req1.id,
        completed_at_monotonic=time.monotonic(),
        success=True,
        channel_responses={ChannelModality.TEXT: response_1},
    )

    # 2 ready
    req2 = pop_ready_with_timeout(scheduler)
    assert req2 is not None
    assert len(req2.history) == 4


@pytest.mark.unit
def test_ambiguous_history_inheritance() -> None:
    """Raises ValueError if multiple history parents exist."""
    scheduler = make_scheduler(interval=0.01)

    # Graph: 0 -> 2, 1 -> 2
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    add_node(graph, SessionNode(id=1, wait_after_ready=0.0))
    add_node(graph, SessionNode(id=2, wait_after_ready=0.0))

    # Both are history parents
    add_edge(graph, SessionEdge(src=0, dst=2, is_history_parent=True))
    add_edge(graph, SessionEdge(src=1, dst=2, is_history_parent=True))

    requests = {
        0: make_request(200),
        1: make_request(201),
        2: make_request(202),
    }
    session = Session(id=2, session_graph=graph, requests=requests)
    scheduler.schedule_session(session)

    # Complete parents (0 and 1)
    # We expect exactly 2 requests to be processed successfully
    for _ in range(2):
        req = pop_ready_with_timeout(scheduler)
        assert req is not None
        scheduler.notify_completion(req.id, time.monotonic(), success=True)

    # 2 should be ready now, but pop_ready checks history ambiguity
    # Wait for it to be in ready queue (internal) but popping fails
    # Since pop_ready raises, we can just check:
    def check_raises():
        try:
            req = scheduler.pop_ready()
            return False  # Should have raised or returned None if not ready yet
        except ValueError:
            return True

    assert wait_until(check_raises, timeout_s=1.0)


@pytest.mark.unit
def test_no_history_parent() -> None:
    """Requests with no history parent start with empty history."""
    scheduler = make_scheduler(interval=0.01)

    # Graph: 0 -> 1 (but is_history_parent=False)
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    add_node(graph, SessionNode(id=1, wait_after_ready=0.0))
    add_edge(graph, SessionEdge(src=0, dst=1, is_history_parent=False))

    requests = {
        0: make_request(300),
        1: make_request(301),
    }
    session = Session(id=3, session_graph=graph, requests=requests)
    scheduler.schedule_session(session)

    # Complete 0
    req0 = pop_ready_with_timeout(scheduler)
    assert req0 is not None

    response_0 = ChannelResponse(modality=ChannelModality.TEXT, content="response_0")
    scheduler.notify_completion(
        request_id=req0.id,
        completed_at_monotonic=time.monotonic(),
        success=True,
        channel_responses={ChannelModality.TEXT: response_0},
    )

    # 1 ready, but no history inherited (empty list)
    req1 = pop_ready_with_timeout(scheduler)
    assert req1 is not None
    assert req1.history == []
