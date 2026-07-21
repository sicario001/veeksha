"""Shared helpers for engine unit tests."""

import asyncio
import time
from typing import Dict, Iterable, List, Optional, Set

from veeksha.config.client import OpenAIChatCompletionsClientConfig
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.traffic import ConcurrentTrafficConfig
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import RequestResult
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import (
    SessionEdge,
    SessionGraph,
    SessionNode,
    add_edge,
    add_node,
)
from veeksha.loop.interface import MainLoopConfig
from veeksha.loop.python_loop import PythonMainLoop
from veeksha.loop.source import PregeneratedSessionSource
from veeksha.types import ChannelModality


def make_request(request_id: int) -> Request:
    return Request(
        id=request_id,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text=f"test_{request_id}"
            )
        },
    )


def make_linear_session(session_id: int, num_requests: int) -> Session:
    """Linear multi-turn session; request ids are session_id * 100 + turn."""
    graph = SessionGraph()
    requests: Dict[int, Request] = {}
    for i in range(num_requests):
        add_node(graph, SessionNode(id=i, wait_after_ready=0.0))
        requests[i] = make_request(request_id=session_id * 100 + i)
    for i in range(num_requests - 1):
        add_edge(graph, SessionEdge(src=i, dst=i + 1))
    return Session(id=session_id, session_graph=graph, requests=requests)


class ScriptedClient:
    """Fake client with scripted per-request outcomes.

    Requests in ``fail_ids`` return an errored RequestResult; requests in
    ``hang_ids`` block until cancelled (for timeout-path tests); everything
    else succeeds after ``delay_s``.
    """

    def __init__(
        self,
        fail_ids: Optional[Iterable[int]] = None,
        hang_ids: Optional[Iterable[int]] = None,
        delay_s: float = 0.0,
    ):
        self.fail_ids: Set[int] = set(fail_ids or ())
        self.hang_ids: Set[int] = set(hang_ids or ())
        self.delay_s = delay_s
        self.sent_request_ids: List[int] = []

    async def send_request(
        self,
        request,
        session_id,
        session_total_requests,
        on_request_sent=None,
        on_request_dispatched=None,
    ) -> RequestResult:
        self.sent_request_ids.append(request.id)
        if on_request_dispatched is not None:
            on_request_dispatched()
        if on_request_sent is not None:
            on_request_sent()

        if request.id in self.hang_ids:
            await asyncio.sleep(3600.0)

        if self.delay_s:
            await asyncio.sleep(self.delay_s)

        if request.id in self.fail_ids:
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=500,
                error_msg="scripted failure",
                client_completed_at=time.monotonic(),
            )

        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            success=True,
            client_completed_at=time.monotonic(),
        )


def make_loop(
    client,
    *,
    target_concurrent: int = 4,
    num_dispatcher_threads: int = 2,
    num_completion_threads: int = 2,
    num_client_threads: int = 2,
    max_sessions: int = -1,
) -> PythonMainLoop:
    runtime = RuntimeConfig(
        max_sessions=max_sessions,
        benchmark_timeout=60,
        num_dispatcher_threads=num_dispatcher_threads,
        num_completion_threads=num_completion_threads,
        num_client_threads=num_client_threads,
    )
    traffic = ConcurrentTrafficConfig(
        target_concurrent_sessions=target_concurrent, rampup_seconds=0
    )
    config = MainLoopConfig(
        runtime=runtime,
        traffic=traffic,
        client=OpenAIChatCompletionsClientConfig(),
        monotonic_anchor=time.monotonic(),
    )
    return PythonMainLoop(config, seed_manager=SeedManager(seed=42), client=client)


def wait_until(predicate, timeout_s: float = 10.0, interval_s: float = 0.01) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def run_loop_to_completion(loop: PythonMainLoop, sessions: List[Session]):
    """Start the loop over the sessions, wait for settle, shut down cleanly.

    Returns the drained events (in emission order).
    """
    source = PregeneratedSessionSource(sessions)
    loop.start(source)
    settled = wait_until(
        lambda: (lambda c: c.intake_exhausted and c.idle)(loop.counters())
    )
    loop.request_stop(grace_s=-1.0)
    loop.join(timeout_s=2.0)
    events = []
    while True:
        batch = loop.drain_events(max_items=1024, timeout_s=0.05)
        if not batch:
            break
        events.extend(batch)
    assert settled, "loop did not settle before timeout"
    return events
