"""Native WS transports: TTS_REALTIME_WS paced sends + event offsets, and
STT_WS blob-sliced paced appends + transcript accumulation."""

from __future__ import annotations

import json

import pytest

vn = pytest.importorskip(
    "veeksha.native.veeksha_native",
    reason="veeksha_native extension not built (run veeksha/native/build.sh)",
)

from tests.unit.native.mock_servers import (  # noqa: E402
    RealtimeTtsServer,
    SttServer,
    drain_until_done,
    make_loop,
    request_plan,
    session_plan,
    split_events,
    ws_frame,
)


def _realtime_transport(offsets_ms):
    t = vn.TransportPlan()
    t.kind = vn.TransportKind.TTS_REALTIME_WS
    t.ws_path = "/v1/realtime"
    t.setup_frames = [ws_frame(vn, json.dumps({"type": "session.update"}))]
    t.paced_frames = [
        ws_frame(
            vn,
            json.dumps({"type": "conversation.item.create", "i": i}),
            send_offset_ms=off,
        )
        for i, off in enumerate(offsets_ms)
    ]
    t.finish_frames = [ws_frame(vn, json.dumps({"type": "response.create"}))]
    t.done_markers = ["response.done"]
    return t


def test_tts_realtime_paced_sends_and_event_offsets():
    srv = RealtimeTtsServer(num_chunks=5, chunk_gap_s=0.03).start()
    offsets = [0.0, 40.0, 80.0, 120.0]
    n = 2
    try:
        loop = make_loop(vn, vn.TrafficKind.CONCURRENT, srv.port, target=n)
        plans = [
            session_plan(vn, s, [request_plan(vn, s, 0, _realtime_transport(offsets))])
            for s in range(n)
        ]
        loop.feed_sessions(plans)
        loop.close_intake()
        events = drain_until_done(loop, expect_completed=n)
    finally:
        srv.stop()

    _, completed = split_events(vn, events)
    assert len(completed) == n
    for e in completed:
        r = e.result
        assert r.error == "", r.error
        assert r.status == 101  # ws handshake ok
        # paced sends: one recorded offset per paced frame, stamped BEFORE
        # send, anchored post-handshake; drift < 20 ms per dispatch
        assert len(r.send_offsets_ms) == len(offsets)
        drift = [abs(a - p) for a, p in zip(r.send_offsets_ms, offsets)]
        assert max(drift) < 20.0, r.send_offsets_ms
        # named protocol events observed with offsets
        types = [t for t, _ in r.event_offsets_ms]
        assert "session.updated" in types
        assert "response.done" in types
        assert types.count("response.output_audio.delta") == 5
        offs = dict()
        for t, off in r.event_offsets_ms:
            offs.setdefault(t, off)
        assert offs["session.updated"] < offs["response.done"]
        # recv stamps: only the audio deltas (content), monotonic
        assert len(r.recv_stamps) == 5
        arr = [s.offset_ms for s in r.recv_stamps]
        assert arr == sorted(arr)
        assert r.recv_bytes == sum(s.size for s in r.recv_stamps)
        # lifecycle ordering holds on the WS path too
        assert r.scheduler_dispatched_ms <= r.client_picked_up_ms
        assert r.client_picked_up_ms < r.client_completed_ms


def test_stt_blob_sliced_appends_pacing_and_transcript():
    srv = SttServer(num_deltas=3, delta_gap_s=0.03).start()
    # pre-encoded append frames share ONE registered blob; frames are slices
    append_payloads = [
        json.dumps({"type": "input_audio_buffer.append", "audio": ("A%d" % i) * 40})
        for i in range(4)
    ]
    blob_bytes = "".join(append_payloads).encode("utf-8")
    schedule = [0.0, 60.0, 120.0, 180.0]  # 1x-realtime style absolute offsets
    try:
        loop = make_loop(vn, vn.TrafficKind.CONCURRENT, srv.port, target=1)
        blob_id = loop.register_blob(blob_bytes)

        t = vn.TransportPlan()
        t.kind = vn.TransportKind.STT_WS
        t.ws_path = "/v1/realtime?intent=transcription"
        t.setup_frames = [
            ws_frame(vn, json.dumps({"type": "transcription_session.update"}))
        ]
        paced = []
        off = 0
        for payload, when in zip(append_payloads, schedule):
            b = vn.BlobRef()
            b.blob_id = blob_id
            b.offset = off
            b.len = len(payload.encode("utf-8"))
            off += b.len
            paced.append(ws_frame(vn, None, blob=b, send_offset_ms=when))
        t.paced_frames = paced
        t.finish_frames = [
            ws_frame(vn, json.dumps({"type": "input_audio_buffer.commit"}))
        ]
        t.done_markers = ["transcription.done"]

        plan = session_plan(vn, 0, [request_plan(vn, 0, 0, t)])
        loop.feed_sessions([plan])
        loop.close_intake()
        events = drain_until_done(loop, expect_completed=1)
    finally:
        srv.stop()

    _, completed = split_events(vn, events)
    r = completed[0].result
    assert r.error == "", r.error

    # blob slicing correct: server received the exact append payloads
    got_appends = [p for _, p in srv.appends]
    assert got_appends == append_payloads

    # send offsets: stamped BEFORE each send, anchored at the FIRST paced
    # send (audio_started_at semantics); drift vs the 1x schedule stays small
    assert len(r.send_offsets_ms) == len(schedule)
    assert r.send_offsets_ms[0] == 0.0  # anchor defines t=0
    drift = [abs(a - p) for a, p in zip(r.send_offsets_ms, schedule)]
    assert max(drift) < 20.0, r.send_offsets_ms

    # transcript deltas: stamped + extracted delta text accumulated in order
    assert r.content == srv.transcript()
    assert len(r.recv_stamps) == 3
    arr = [s.offset_ms for s in r.recv_stamps]
    assert arr == sorted(arr)

    c = loop.counters()
    assert c.sessions_completed == 1
    assert c.requests_completed == 1
