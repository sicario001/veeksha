"""Struct/enum round-trips across the pybind boundary (prototype doc §1-3).

Everything crossing the boundary is copyable plain data: def_readwrite fields
must accept Python-side assignment and read back identically, including
nested vectors/pairs (note: list-valued fields have COPY semantics — tests
and callers assign whole lists, never mutate in place).
"""

from __future__ import annotations

import pytest

vn = pytest.importorskip(
    "veeksha.native.veeksha_native",
    reason="veeksha_native extension not built (run veeksha/native/build.sh)",
)


def test_enums_exist_and_are_distinct():
    assert vn.TrafficKind.RATE != vn.TrafficKind.CONCURRENT
    assert {
        int(k)
        for k in (
            vn.TrafficKind.RATE,
            vn.TrafficKind.CONCURRENT,
            vn.TrafficKind.SEQUENTIAL_LAUNCH,
        )
    } == {0, 1, 2}
    assert {
        int(k)
        for k in (
            vn.TicketOrdering.DISPATCH,
            vn.TicketOrdering.PREFILL,
            vn.TicketOrdering.REQUEST,
        )
    } == {0, 1, 2}
    assert {
        int(k)
        for k in (
            vn.TransportKind.TEXT_SSE,
            vn.TransportKind.TTS_HTTP,
            vn.TransportKind.TTS_REALTIME_WS,
            vn.TransportKind.STT_WS,
        )
    } == {0, 1, 2, 3}
    assert {int(k) for k in (vn.EventKind.DISPATCHED, vn.EventKind.COMPLETED)} == {
        0,
        1,
    }


def test_runtime_config_round_trip():
    c = vn.NativeRuntimeConfig()
    assert c.num_dispatcher_threads == 2
    assert c.num_completion_threads == 8
    c.max_sessions = 77
    c.benchmark_timeout_s = 12.5
    c.post_timeout_grace_s = -1.0
    c.num_dispatcher_threads = 3
    c.num_completion_threads = 4
    c.num_client_threads = 5
    assert c.max_sessions == 77
    assert c.benchmark_timeout_s == 12.5
    assert c.post_timeout_grace_s == -1.0
    assert (
        c.num_dispatcher_threads,
        c.num_completion_threads,
        c.num_client_threads,
    ) == (3, 4, 5)


def test_traffic_plan_config_round_trip():
    t = vn.TrafficPlanConfig()
    assert t.cancel_session_on_failure is True
    t.kind = vn.TrafficKind.SEQUENTIAL_LAUNCH
    t.target_concurrent_sessions = 9
    t.rampup_seconds = 3.5
    t.ordering = vn.TicketOrdering.PREFILL
    t.cancel_session_on_failure = False
    assert t.kind == vn.TrafficKind.SEQUENTIAL_LAUNCH
    assert t.target_concurrent_sessions == 9
    assert t.rampup_seconds == 3.5
    assert t.ordering == vn.TicketOrdering.PREFILL
    assert t.cancel_session_on_failure is False


def test_endpoint_config_round_trip():
    e = vn.EndpointConfig()
    e.host = "10.0.0.5"
    e.port = 8123
    e.base_path = "/v1"
    e.headers = [("Authorization", "Bearer x"), ("X-A", "b")]
    e.request_timeout_s = 30.0
    assert e.host == "10.0.0.5"
    assert e.port == 8123
    assert e.base_path == "/v1"
    assert e.headers == [("Authorization", "Bearer x"), ("X-A", "b")]
    assert e.request_timeout_s == 30.0


def test_transport_plan_round_trip_http():
    t = vn.TransportPlan()
    t.kind = vn.TransportKind.TEXT_SSE
    t.header_prefix = "POST / HTTP/1.1\r\nContent-Length: "
    t.header_suffix = "\r\n\r\n"
    t.body_segments = ['{"a":"', '","b":"', '"}']
    t.history_refs = [0, 1]
    assert t.body_segments == ['{"a":"', '","b":"', '"}']
    assert t.history_refs == [0, 1]


def test_transport_plan_round_trip_ws_with_blob_frames():
    b = vn.BlobRef()
    b.blob_id = 3
    b.offset = 128
    b.len = 64
    f = vn.WsFrame()
    f.payload = '{"type":"x"}'
    f.blob = b
    f.send_offset_ms = 240.0
    t = vn.TransportPlan()
    t.kind = vn.TransportKind.STT_WS
    t.ws_path = "/v1/realtime"
    t.setup_frames = [f]
    t.paced_frames = [f, f]
    t.finish_frames = [f]
    t.done_markers = ["transcription.done", "closed"]
    assert t.ws_path == "/v1/realtime"
    assert len(t.paced_frames) == 2
    got = t.setup_frames[0]
    assert got.payload == '{"type":"x"}'
    assert (got.blob.blob_id, got.blob.offset, got.blob.len) == (3, 128, 64)
    assert got.send_offset_ms == 240.0
    assert t.done_markers == ["transcription.done", "closed"]
    # default: asap sentinel
    assert vn.WsFrame().send_offset_ms < 0
    assert vn.BlobRef().blob_id == -1


def test_session_plan_round_trip():
    t = vn.TransportPlan()
    t.kind = vn.TransportKind.TEXT_SSE
    r = vn.RequestPlan()
    r.request_id = 42
    r.node_id = 7
    r.wait_after_ready_s = 1.25
    r.parents = [(3, True), (5, False)]
    r.transport = t
    s = vn.SessionPlan()
    s.session_id = 99
    s.requests = [r]
    s.dispatch_ticket_base = 11
    assert s.session_id == 99
    assert s.dispatch_ticket_base == 11
    got = s.requests[0]
    assert got.request_id == 42
    assert got.node_id == 7
    assert got.wait_after_ready_s == 1.25
    assert got.parents == [(3, True), (5, False)]
    assert got.transport.kind == vn.TransportKind.TEXT_SSE
    # default ticket base = -1 (native assigns)
    assert vn.SessionPlan().dispatch_ticket_base == -1


def test_result_and_event_round_trip():
    st = vn.ChunkStamp()
    st.offset_ms = 12.5
    st.size = 400
    r = vn.RawRequestResult()
    r.request_id = 1
    r.session_id = 2
    r.session_total_requests = 3
    r.status = 200
    r.error = "boom"
    r.scheduler_ready_ms = 1.0
    r.scheduler_dispatched_ms = 2.0
    r.client_picked_up_ms = 3.0
    r.client_completed_ms = 4.0
    r.result_processed_ms = 5.0
    r.recv_stamps = [st]
    r.send_offsets_ms = [0.0, 40.0]
    r.content = "héllo"
    r.recv_bytes = 1234
    r.event_offsets_ms = [("session.updated", 5.5)]
    assert r.recv_stamps[0].offset_ms == 12.5
    assert r.recv_stamps[0].size == 400
    assert r.send_offsets_ms == [0.0, 40.0]
    assert r.content == "héllo"
    assert r.recv_bytes == 1234
    assert r.event_offsets_ms == [("session.updated", 5.5)]

    ev = vn.NativeLoopEvent()
    ev.kind = vn.EventKind.COMPLETED
    ev.request_id = 1
    ev.session_id = 2
    ev.session_total_requests = 3
    ev.ready_ms = 0.5
    ev.dispatched_ms = 0.75
    ev.result = r
    assert ev.kind == vn.EventKind.COMPLETED
    assert ev.result.error == "boom"
    assert ev.result.recv_stamps[0].size == 400


def test_counters_defaults():
    c = vn.NativeLoopCounters()
    assert c.sessions_completed == 0
    assert c.in_flight == 0
    assert c.intake_exhausted is False
    assert c.idle is False


def test_register_blob_returns_sequential_ids():
    import time as _time

    rt = vn.NativeRuntimeConfig()
    rt.num_client_threads = 1
    rt.num_dispatcher_threads = 1
    rt.num_completion_threads = 1
    tp = vn.TrafficPlanConfig()
    ep = vn.EndpointConfig()
    ep.host = "127.0.0.1"
    ep.port = 9
    loop = vn.NativeBenchmarkLoop(rt, tp, ep, _time.monotonic())
    try:
        assert loop.register_blob(b"abc") == 0
        assert loop.register_blob(b"\x00\x01\x02" * 100) == 1
        loop.close_intake()
    finally:
        loop.request_stop(0.0)
        assert loop.join(5.0)
