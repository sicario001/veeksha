"""End-to-end native loop tests over the TEXT_SSE transport: closed loop,
multi-turn history splicing, RATE open loop, CONCURRENT rampup, SEQUENTIAL
ticket orderings, advance-on-error, request_stop grace."""

from __future__ import annotations

import json
import time

import pytest

vn = pytest.importorskip(
    "veeksha.native.veeksha_native",
    reason="veeksha_native extension not built (run veeksha/native/build.sh)",
)

from tests.unit.native.mock_servers import (  # noqa: E402
    MalformedChunkSseServer,
    SseChatServer,
    drain_until_done,
    make_loop,
    request_plan,
    session_plan,
    split_events,
    text_transport,
)


def _single_turn_session(port, session_id, request_id, body=None):
    body = body or json.dumps(
        {"messages": [{"role": "user", "content": f"prompt-{session_id}"}]}
    )
    t = text_transport(vn, port, [body])
    return session_plan(vn, session_id, [request_plan(vn, request_id, 0, t)])


def test_closed_loop_text_end_to_end():
    srv = SseChatServer(num_chunks=6, chunk_gap_s=0.03, prefill_s=0.03).start()
    n = 6
    try:
        loop = make_loop(vn, vn.TrafficKind.CONCURRENT, srv.port, target=3)
        plans = [_single_turn_session(srv.port, i, 100 + i) for i in range(n)]
        assert loop.feed_sessions(plans) == n
        loop.close_intake()
        events = drain_until_done(loop, expect_completed=n)
    finally:
        srv.stop()

    dispatched, completed = split_events(vn, events)
    assert len(dispatched) == n
    assert len(completed) == n

    # DISPATCHED strictly precedes COMPLETED for every request in drain order
    disp_pos = {
        e.request_id: i
        for i, e in enumerate(events)
        if e.kind == vn.EventKind.DISPATCHED
    }
    comp_pos = {
        e.request_id: i
        for i, e in enumerate(events)
        if e.kind == vn.EventKind.COMPLETED
    }
    assert set(disp_pos) == set(comp_pos) == {100 + i for i in range(n)}
    for rid in disp_pos:
        assert disp_pos[rid] < comp_pos[rid]

    # counters exact
    c = loop.counters()
    assert c.sessions_seen == n
    assert c.sessions_completed == n
    assert c.sessions_errored == 0
    assert c.requests_dispatched == n
    assert c.requests_completed == n
    assert c.in_flight == 0
    assert c.intake_exhausted is True
    assert c.idle is True

    expected_content = srv.full_reply()
    for e in completed:
        r = e.result
        assert r.error == "", r.error
        assert r.status == 200
        assert r.content == expected_content
        assert r.session_total_requests == 1
        # five lifecycle stamps, ordered like the Python loop
        assert 0 <= r.scheduler_ready_ms <= r.scheduler_dispatched_ms
        assert r.scheduler_dispatched_ms <= r.client_picked_up_ms
        assert r.client_picked_up_ms < r.client_completed_ms
        assert r.client_completed_ms <= r.result_processed_ms
        # recv stamps: one per SSE event, monotonic, ~cadence
        offs = [s.offset_ms for s in r.recv_stamps]
        assert len(offs) == 6
        assert offs == sorted(offs)
        gaps = [b - a for a, b in zip(offs, offs[1:])]
        assert all(0 <= g < 150 for g in gaps), gaps
        avg = sum(gaps) / len(gaps)
        assert 10 < avg < 80, gaps  # server cadence is 30 ms
        assert r.recv_bytes == sum(s.size for s in r.recv_stamps)


def test_multi_turn_history_splice_utf8_round_trip():
    # tokens exercise escaping: quotes, backslash, newline, non-BMP emoji
    srv = SseChatServer(
        num_chunks=3,
        chunk_gap_s=0.01,
        prefill_s=0.01,
        token_text='hé"l\\lo\n💡{i} ',
    ).start()
    try:
        loop = make_loop(vn, vn.TrafficKind.CONCURRENT, srv.port, target=2)
        plans = []
        for s in range(2):
            t0 = text_transport(
                vn,
                srv.port,
                [json.dumps({"messages": [{"role": "user", "content": f"turn1-{s}"}]})],
            )
            # turn 2: hole filled with turn 1's extracted assistant content
            t1 = text_transport(
                vn,
                srv.port,
                [
                    '{"messages":[{"role":"assistant","content":"',
                    '"},{"role":"user","content":"turn2-marker"}]}',
                ],
                history_refs=[0],
            )
            reqs = [
                request_plan(vn, 10 * s, 0, t0),
                request_plan(vn, 10 * s + 1, 1, t1, parents=[(0, True)]),
            ]
            plans.append(session_plan(vn, s, reqs))
        loop.feed_sessions(plans)
        loop.close_intake()
        events = drain_until_done(loop, expect_completed=4)
    finally:
        srv.stop()

    _, completed = split_events(vn, events)
    assert len(completed) == 4
    for e in completed:
        assert e.result.error == "", e.result.error

    expected_parent = srv.full_reply()
    turn2_bodies = [b for b in srv.bodies if "turn2-marker" in b]
    assert len(turn2_bodies) == 2
    for body in turn2_bodies:
        msgs = json.loads(body)["messages"]
        # the spliced content round-trips exactly (unescape -> escape)
        assert msgs[0] == {"role": "assistant", "content": expected_parent}

    # child dispatched only after parent completed
    by_rid = {e.request_id: e.result for e in completed}
    for s in range(2):
        parent = by_rid[10 * s]
        child = by_rid[10 * s + 1]
        assert child.scheduler_dispatched_ms >= parent.client_completed_ms - 1.0


def test_rate_open_loop_dispatch_matches_fed_intervals():
    srv = SseChatServer(num_chunks=2, chunk_gap_s=0.005, prefill_s=0.0).start()
    n = 5
    gap_s = 0.08
    try:
        loop = make_loop(vn, vn.TrafficKind.RATE, srv.port)
        loop.feed_intervals([gap_s] * n)
        plans = [_single_turn_session(srv.port, i, i) for i in range(n)]
        loop.feed_sessions(plans)
        loop.close_intake()
        events = drain_until_done(loop, expect_completed=n)
    finally:
        srv.stop()

    dispatched, completed = split_events(vn, events)
    assert len(completed) == n
    assert all(e.result.error == "" for e in completed)
    at = {e.request_id: e.dispatched_ms for e in dispatched}
    # session i starts at i*gap; relative spacing must match the schedule
    base = at[0]
    for i in range(n):
        expected = i * gap_s * 1000.0
        actual = at[i] - base
        assert abs(actual - expected) < 10.0, (i, actual, expected)


def test_concurrent_rampup_staggers_first_sessions():
    # cap(t) = int(4 * t / 0.8): one activation every 200 ms. Streams must
    # outlive the rampup window (~1s vs 0.8s) or completions free capacity
    # and activations decouple from the ramp (same as the Python scheduler).
    srv = SseChatServer(num_chunks=20, chunk_gap_s=0.05, prefill_s=0.02).start()
    n = 4
    rampup = 0.8
    try:
        loop = make_loop(
            vn, vn.TrafficKind.CONCURRENT, srv.port, target=n, rampup=rampup
        )
        plans = [_single_turn_session(srv.port, i, i) for i in range(n)]
        loop.feed_sessions(plans)
        loop.close_intake()
        events = drain_until_done(loop, expect_completed=n)
    finally:
        srv.stop()

    dispatched, completed = split_events(vn, events)
    assert all(e.result.error == "" for e in completed)
    times = sorted(e.dispatched_ms for e in dispatched)
    step_ms = rampup / n * 1000.0  # 200 ms
    # no session dispatches before the cap admits one
    assert times[0] > step_ms * 0.5
    # the first-C stagger is observable: consecutive activations ~step apart
    gaps = [b - a for a, b in zip(times, times[1:])]
    for g in gaps:
        assert step_ms * 0.4 < g < step_ms * 2.0, gaps


@pytest.mark.parametrize("ordering", ["request", "prefill"])
def test_sequential_orderings_serialize(ordering):
    prefill_s = 0.12
    srv = SseChatServer(num_chunks=2, chunk_gap_s=0.01, prefill_s=prefill_s).start()
    n = 3
    try:
        loop = make_loop(
            vn,
            vn.TrafficKind.SEQUENTIAL_LAUNCH,
            srv.port,
            ordering=(
                vn.TicketOrdering.REQUEST
                if ordering == "request"
                else vn.TicketOrdering.PREFILL
            ),
        )
        plans = [_single_turn_session(srv.port, i, i) for i in range(n)]
        loop.feed_sessions(plans)
        loop.close_intake()
        events = drain_until_done(loop, expect_completed=n)
    finally:
        srv.stop()

    _, completed = split_events(vn, events)
    assert all(e.result.error == "" for e in completed)
    picked = {e.request_id: e.result.client_picked_up_ms for e in completed}
    # tickets were assigned in feed order -> pickups serialize in that order,
    # spaced by at least the advance point (prefill for PREFILL, full request
    # for REQUEST)
    order = sorted(picked, key=lambda rid: picked[rid])
    assert order == list(range(n))
    gaps = [picked[i + 1] - picked[i] for i in range(n - 1)]
    for g in gaps:
        assert g >= prefill_s * 1000.0 * 0.7, gaps


def test_sequential_dispatch_ordering_completes_without_serializing_on_prefill():
    prefill_s = 0.15
    srv = SseChatServer(num_chunks=2, chunk_gap_s=0.01, prefill_s=prefill_s).start()
    n = 3
    try:
        loop = make_loop(
            vn,
            vn.TrafficKind.SEQUENTIAL_LAUNCH,
            srv.port,
            ordering=vn.TicketOrdering.DISPATCH,
        )
        plans = [_single_turn_session(srv.port, i, i) for i in range(n)]
        loop.feed_sessions(plans)
        loop.close_intake()
        events = drain_until_done(loop, expect_completed=n)
    finally:
        srv.stop()

    _, completed = split_events(vn, events)
    assert all(e.result.error == "" for e in completed)
    picked = sorted(e.result.client_picked_up_ms for e in completed)
    # DISPATCH advances on HTTP 200 (before prefill): total pickup span must
    # be well under the serialized-prefill span (2 * 150 ms)
    assert picked[-1] - picked[0] < 2 * prefill_s * 1000.0 * 0.7


def test_ticket_gate_advances_on_error():
    srv = SseChatServer(
        num_chunks=2, chunk_gap_s=0.01, prefill_s=0.02, fail_marker="FAIL-ME"
    ).start()
    try:
        loop = make_loop(
            vn,
            vn.TrafficKind.SEQUENTIAL_LAUNCH,
            srv.port,
            ordering=vn.TicketOrdering.REQUEST,
        )
        bad = _single_turn_session(
            srv.port,
            0,
            0,
            body=json.dumps({"messages": [{"role": "user", "content": "FAIL-ME"}]}),
        )
        good = [_single_turn_session(srv.port, i, i) for i in (1, 2)]
        loop.feed_sessions([bad] + good)
        loop.close_intake()
        events = drain_until_done(loop, expect_completed=3)
    finally:
        srv.stop()

    _, completed = split_events(vn, events)
    by_rid = {e.request_id: e.result for e in completed}
    assert by_rid[0].error == "http status 500"
    assert by_rid[0].status == 500
    # the failed ticket advanced the gate: later tickets still ran
    assert by_rid[1].error == ""
    assert by_rid[2].error == ""
    c = loop.counters()
    assert c.sessions_errored == 1
    assert c.sessions_completed == 2


def test_malformed_chunked_response_fails_request_engine_keeps_running():
    srv = MalformedChunkSseServer(bad_marker="BAD-ME").start()
    try:
        loop = make_loop(vn, vn.TrafficKind.CONCURRENT, srv.port, target=2)
        bad = _single_turn_session(
            srv.port,
            0,
            0,
            body=json.dumps({"messages": [{"role": "user", "content": "BAD-ME"}]}),
        )
        good = _single_turn_session(srv.port, 1, 1)
        loop.feed_sessions([bad, good])
        loop.close_intake()
        events = drain_until_done(loop, expect_completed=2)
    finally:
        srv.stop()

    _, completed = split_events(vn, events)
    assert len(completed) == 2
    by_rid = {e.request_id: e.result for e in completed}
    # malformed framing fails the request with an llhttp reason — never a
    # silent completion
    assert by_rid[0].error.startswith("malformed http response ("), by_rid[0].error
    assert "HPE_" in by_rid[0].error, by_rid[0].error
    assert by_rid[0].status == 200  # headers parsed before the framing broke
    # the native loop kept running: the well-formed request completed normally
    assert by_rid[1].error == "", by_rid[1].error
    c = loop.counters()
    assert c.sessions_errored == 1
    assert c.sessions_completed == 1
    assert c.in_flight == 0
    assert c.idle is True


def test_request_stop_grace_fails_stragglers_keeping_partials():
    # long stream: 60 chunks x 50 ms = 3 s; stopped long before the end
    srv = SseChatServer(num_chunks=60, chunk_gap_s=0.05, prefill_s=0.02).start()
    n = 2
    try:
        loop = make_loop(
            vn,
            vn.TrafficKind.CONCURRENT,
            srv.port,
            target=n,
            request_timeout_s=60.0,
        )
        plans = [_single_turn_session(srv.port, i, i) for i in range(n)]
        loop.feed_sessions(plans)
        loop.close_intake()
        time.sleep(0.6)  # let a few chunks arrive
        loop.request_stop(0.2)
        assert loop.join(10.0)
        events = loop.drain_events(max_items=8192, timeout_s=0.0)
    finally:
        srv.stop()

    dispatched, completed = split_events(vn, events)
    assert len(dispatched) == n
    assert len(completed) == n
    for e in completed:
        r = e.result
        assert r.error == "timeout"
        # partial stream preserved
        assert len(r.recv_stamps) >= 3
        assert len(r.recv_stamps) < 60
        assert r.content  # partial extracted text kept
    c = loop.counters()
    assert c.in_flight == 0
