"""Unit tests for PythonMainLoop: event ordering, counters, timeout path."""

import time

import pytest

from tests.unit.loop.helpers import (
    ScriptedClient,
    make_linear_session,
    make_loop,
    run_loop_to_completion,
    wait_until,
)
from veeksha.benchmark_utils import _monitor_for_completion
from veeksha.loop.interface import LoopEventKind
from veeksha.loop.source import PregeneratedSessionSource


@pytest.mark.unit
def test_dispatched_precedes_completed_per_request() -> None:
    """Under concurrency, every request's DISPATCHED event is drained before
    its COMPLETED event."""
    sessions = [make_linear_session(i, 2) for i in range(1, 9)]
    client = ScriptedClient(delay_s=0.005)
    loop = make_loop(client, target_concurrent=8)

    events = run_loop_to_completion(loop, sessions)

    dispatched_index = {}
    completed_index = {}
    for idx, event in enumerate(events):
        if event.kind == LoopEventKind.DISPATCHED:
            dispatched_index[event.request_id] = idx
        else:
            completed_index[event.request_id] = idx

    expected_requests = 8 * 2
    assert len(dispatched_index) == expected_requests
    assert len(completed_index) == expected_requests
    for request_id, completed_at in completed_index.items():
        assert request_id in dispatched_index
        assert dispatched_index[request_id] < completed_at


@pytest.mark.unit
def test_dispatched_events_carry_register_and_trace_payload() -> None:
    sessions = [make_linear_session(1, 1)]
    loop = make_loop(ScriptedClient(), target_concurrent=1)

    events = run_loop_to_completion(loop, sessions)

    dispatched = [e for e in events if e.kind == LoopEventKind.DISPATCHED]
    assert len(dispatched) == 1
    event = dispatched[0]
    assert event.request_id == 100
    assert event.session_id == 1
    assert event.session_total_requests == 1
    assert event.ready_at is not None and event.dispatched_at is not None
    assert event.request.channels  # register_request payload derives from it
    assert event.request is not None and event.request.id == 100  # trace payload

    completed = [e for e in events if e.kind == LoopEventKind.COMPLETED]
    assert len(completed) == 1
    assert completed[0].result is not None
    assert completed[0].result.request_id == 100
    assert completed[0].result.result_processed_at is not None


@pytest.mark.unit
def test_counters_match_scripted_outcomes() -> None:
    """Loop-side counters match a scripted client: successes and failures."""
    # 6 single-request sessions; sessions 2 and 5 fail (request ids 200, 500).
    sessions = [make_linear_session(i, 1) for i in range(1, 7)]
    client = ScriptedClient(fail_ids={200, 500})
    loop = make_loop(client, target_concurrent=6)

    events = run_loop_to_completion(loop, sessions)
    counters = loop.counters()

    assert counters.requests_dispatched == 6
    assert counters.requests_completed == 6
    assert counters.sessions_seen == 6
    assert counters.sessions_completed == 4
    assert counters.sessions_errored == 2
    assert counters.in_flight == 0
    assert counters.intake_exhausted
    assert counters.idle
    assert loop.dispatched_request_ids() == {100, 200, 300, 400, 500, 600}
    assert loop.in_flight_request_ids() == set()

    completed_events = [e for e in events if e.kind == LoopEventKind.COMPLETED]
    errored = {e.result.request_id for e in completed_events if e.result.error_msg}
    assert errored == {200, 500}


@pytest.mark.unit
def test_multi_turn_sessions_counted_once_on_final_completion() -> None:
    sessions = [make_linear_session(i, 3) for i in range(1, 4)]
    loop = make_loop(ScriptedClient(), target_concurrent=3)

    run_loop_to_completion(loop, sessions)
    counters = loop.counters()

    assert counters.requests_completed == 9
    assert counters.sessions_seen == 3
    assert counters.sessions_completed == 3
    assert counters.sessions_errored == 0


class _RecordingEvaluator:
    """Evaluator stub capturing set_included_requests for the timeout path."""

    def __init__(self):
        self.included_requests = None

    def set_included_requests(self, request_ids) -> None:
        self.included_requests = set(request_ids)


@pytest.mark.unit
def test_timeout_path_reports_in_flight_and_sets_included_requests() -> None:
    """Timeout with grace 0: in-flight ids are reported from the loop and the
    evaluator's included-request filter is dispatched-minus-remaining."""
    # 2 hanging sessions + 1 fast session.
    sessions = [make_linear_session(i, 1) for i in range(1, 4)]
    client = ScriptedClient(hang_ids={100, 200})
    loop = make_loop(client, target_concurrent=3)
    evaluator = _RecordingEvaluator()

    source = PregeneratedSessionSource(sessions)
    loop.start(source)
    try:
        # Wait until everything is dispatched and the fast request finished.
        assert wait_until(lambda: loop.counters().requests_completed >= 1)
        assert wait_until(lambda: loop.counters().requests_dispatched == 3)

        remaining = _monitor_for_completion(
            loop,
            evaluator,
            benchmark_start=time.monotonic(),
            benchmark_timeout=0.3,
            max_sessions=3,
            post_timeout_grace_seconds=0,
        )

        assert remaining == {100, 200}
        assert remaining == loop.in_flight_request_ids()
        assert evaluator.included_requests == {300}
        assert loop.dispatched_request_ids() == {100, 200, 300}
    finally:
        # Grace expired with requests still in flight: join must not wait
        # for the hanging client tasks.
        loop.request_stop(grace_s=0.0)
        loop.join(timeout_s=2.0)

    counters = loop.counters()
    assert counters.requests_completed == 1
    assert counters.sessions_completed == 1
