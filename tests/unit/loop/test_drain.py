"""Unit tests for ResultDrain: routing, ordering, and full flush."""

import threading
import time
from queue import Empty, Queue

import pytest

from tests.unit.loop.helpers import (
    ScriptedClient,
    make_linear_session,
    make_loop,
    make_request,
    wait_until,
)
from veeksha.core.response import RequestResult
from veeksha.loop.interface import LoopEvent, LoopEventKind
from veeksha.loop.drain import ResultDrain
from veeksha.loop.source import PregeneratedSessionSource


class FakeLoop:
    """Feeds a pre-scripted ordered event stream through drain_events()."""

    def __init__(self, events):
        self._queue = Queue()
        for event in events:
            self._queue.put(event)

    def drain_events(self, max_items: int = 256, timeout_s: float = 0.1):
        events = []
        try:
            events.append(self._queue.get(timeout=timeout_s))
        except Empty:
            return events
        while len(events) < max_items:
            try:
                events.append(self._queue.get_nowait())
            except Empty:
                break
        return events


class RecordingEvaluator:
    """Thread-safe evaluator stub recording replayed calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = []  # (kind, request_id, session_id, thread_name)

    def register_request(
        self, request_id, session_id, dispatched_at, channels, requested_output=None
    ):
        with self._lock:
            self.calls.append(
                ("register", request_id, session_id, threading.current_thread().name)
            )

    def record_request_completed(
        self, request_id, session_id, completed_at, response, error=None
    ):
        with self._lock:
            self.calls.append(
                ("completed", request_id, session_id, threading.current_thread().name)
            )


class RecordingTraceRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self.dispatches = []

    def record_dispatch(self, request, session_id, session_size, dispatched_at):
        with self._lock:
            self.dispatches.append((request.id, session_id, session_size))


def _scripted_events(num_sessions: int, requests_per_session: int):
    """Interleaved DISPATCHED/COMPLETED stream, per-request order valid."""
    events = []
    now = time.monotonic()
    for turn in range(requests_per_session):
        for session_id in range(1, num_sessions + 1):
            request_id = session_id * 100 + turn
            request = make_request(request_id)
            events.append(
                LoopEvent(
                    kind=LoopEventKind.DISPATCHED,
                    request_id=request_id,
                    session_id=session_id,
                    session_total_requests=requests_per_session,
                    ready_at=now,
                    dispatched_at=now,
                    request=request,
                )
            )
        for session_id in range(1, num_sessions + 1):
            request_id = session_id * 100 + turn
            events.append(
                LoopEvent(
                    kind=LoopEventKind.COMPLETED,
                    request_id=request_id,
                    session_id=session_id,
                    session_total_requests=requests_per_session,
                    result=RequestResult(
                        request_id=request_id,
                        session_id=session_id,
                        session_total_requests=requests_per_session,
                        success=True,
                        client_completed_at=time.monotonic(),
                    ),
                )
            )
    return events


@pytest.mark.unit
def test_full_flush_before_join_returns() -> None:
    """join() only returns after every emitted event has been replayed."""
    events = _scripted_events(num_sessions=10, requests_per_session=4)
    evaluator = RecordingEvaluator()
    drain = ResultDrain(FakeLoop(events), evaluator, num_threads=4)

    drain.start()
    # The "loop" is already finished (all events scripted); join must flush.
    drain.join()

    assert len(evaluator.calls) == len(events)


@pytest.mark.unit
def test_per_session_order_preserved_across_workers() -> None:
    """Per session: register before completed for each request, turn order
    kept, and all of one session's events handled by a single thread."""
    num_sessions, turns = 8, 3
    events = _scripted_events(num_sessions, turns)
    evaluator = RecordingEvaluator()
    drain = ResultDrain(FakeLoop(events), evaluator, num_threads=3)

    drain.start()
    drain.join()

    per_session = {}
    for kind, request_id, session_id, thread_name in evaluator.calls:
        per_session.setdefault(session_id, []).append((kind, request_id, thread_name))

    assert len(per_session) == num_sessions
    for session_id, calls in per_session.items():
        threads = {thread_name for _, _, thread_name in calls}
        assert len(threads) == 1  # hash routing: one worker per session

        expected = []
        for turn in range(turns):
            request_id = session_id * 100 + turn
            expected.append(("register", request_id))
            expected.append(("completed", request_id))
        assert [(kind, request_id) for kind, request_id, _ in calls] == expected


@pytest.mark.unit
def test_trace_recorder_receives_dispatch_replay() -> None:
    events = _scripted_events(num_sessions=2, requests_per_session=2)
    evaluator = RecordingEvaluator()
    trace_recorder = RecordingTraceRecorder()
    drain = ResultDrain(
        FakeLoop(events), evaluator, trace_recorder=trace_recorder, num_threads=2
    )

    drain.start()
    drain.join()

    assert sorted(trace_recorder.dispatches) == [
        (100, 1, 2),
        (101, 1, 2),
        (200, 2, 2),
        (201, 2, 2),
    ]


@pytest.mark.unit
def test_result_drain_over_real_loop() -> None:
    """End-to-end: PythonMainLoop events replay into the evaluator via the
    drain, fully flushed by the time drain.join() returns."""
    sessions = [make_linear_session(i, 2) for i in range(1, 5)]
    loop = make_loop(ScriptedClient(delay_s=0.002), target_concurrent=4)
    evaluator = RecordingEvaluator()
    drain = ResultDrain(loop, evaluator, num_threads=2)

    source = PregeneratedSessionSource(sessions)
    loop.start(source)
    drain.start()
    assert wait_until(
        lambda: (lambda c: c.intake_exhausted and c.idle)(loop.counters())
    )
    loop.request_stop(grace_s=-1.0)
    loop.join(timeout_s=2.0)
    drain.join()

    registers = [c for c in evaluator.calls if c[0] == "register"]
    completions = [c for c in evaluator.calls if c[0] == "completed"]
    assert len(registers) == 8
    assert len(completions) == 8

    # Per-session replay order: register precedes completed per request.
    per_session_positions = {}
    for idx, (kind, request_id, session_id, _) in enumerate(evaluator.calls):
        per_session_positions.setdefault(session_id, {}).setdefault(request_id, {})[
            kind
        ] = idx
    for session_id, requests in per_session_positions.items():
        for request_id, positions in requests.items():
            assert positions["register"] < positions["completed"]
