"""Native multi-turn chains: receive->dispatch coupling + inter-turn content flow.

Each chain is a sequence of dependent turns built from templates; native fires
turn N+1 at (turn N complete + think time), splicing prior turns' outputs into
the next request body (JSON-escaped, Content-Length recomputed) — a faithful
conversation with no Python between receive and next dispatch.
"""

from __future__ import annotations

import json

import pytest

from veeksha.native.engine import (
    NativeChainTurnSpec,
    NativeReceiveEngine,
    native_available,
)

pytestmark = pytest.mark.skipif(
    not native_available(), reason="veeksha_native extension not built"
)


def _mock_engine(num_chunks=5, **kwargs):
    from veeksha.preflight.mock_engine import MockStreamingEngine

    return MockStreamingEngine(
        chunk_dt=0.02,
        prefill_s=0.02,
        default_chunks=num_chunks,
        num_loops=4,
        **kwargs,
    ).start()


def _spec(num_chunks: int, hole_refs=None, delay_s: float = 0.0):
    """A turn template; with hole_refs, prior turn outputs land in messages."""
    hole_refs = hole_refs or []
    sentinel = "\x00HOLE\x00"
    messages = []
    for _ in hole_refs:
        messages.append({"role": "assistant", "content": sentinel})
    messages.append({"role": "user", "content": "hi"})
    body = json.dumps(
        {
            "model": "d",
            "stream": True,
            "max_completion_tokens": num_chunks,
            "messages": messages,
        }
    )
    # json.dumps renders the sentinel as \u0000-escaped text inside the string;
    # split the serialized body on that escaped form to get the segments.
    escaped_sentinel = json.dumps(sentinel)[1:-1]
    segments = body.split(escaped_sentinel)
    assert len(segments) == len(hole_refs) + 1
    return NativeChainTurnSpec(
        header_prefix=(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            "Content-Type: application/json\r\nContent-Length: "
        ),
        header_suffix="\r\nConnection: close\r\n\r\n",
        body_segments=segments,
        hole_refs=list(hole_refs),
        delay_s=delay_s,
    )


def test_native_chains_run_all_dependent_turns():
    engine_srv = _mock_engine(num_chunks=5)
    try:
        engine = NativeReceiveEngine("127.0.0.1", engine_srv.port)
        chains = [[_spec(5), _spec(5), _spec(5)] for _ in range(6)]
        results = engine.run_chains(chains, concurrency=4, timeout_s=30.0)
    finally:
        engine_srv.stop()

    assert len(results) == 6
    ok = [r for r in results if r.success]
    assert len(ok) == 6
    for r in ok:
        assert len(r.turns) == 3
        assert all(len(t.stream) == 5 for t in r.turns)
        assert len(r.handoff_s) == 2  # two dependent-turn launches


def test_native_coupling_handoff_is_sub_millisecond():
    """Zero-think dependent dispatch stays native (no Python queue jitter)."""
    engine_srv = _mock_engine(num_chunks=4)
    try:
        engine = NativeReceiveEngine("127.0.0.1", engine_srv.port)
        chains = [[_spec(4)] * 4 for _ in range(8)]
        results = engine.run_chains(chains, concurrency=4, timeout_s=30.0)
    finally:
        engine_srv.stop()

    handoffs = [h for r in results if r.success for h in r.handoff_s]
    assert handoffs
    handoffs.sort()
    median = handoffs[len(handoffs) // 2]
    assert median < 0.005  # < 5ms (typically ~0.05ms); no Python in the loop


def test_native_chains_inject_prior_turn_content():
    """Turn K's request body carries turn K-1's output — filled by NATIVE."""
    engine_srv = _mock_engine(num_chunks=3, store_bodies=True)
    try:
        engine = NativeReceiveEngine("127.0.0.1", engine_srv.port)
        chains = [[_spec(3), _spec(3, hole_refs=[0])]]
        results = engine.run_chains(chains, concurrency=1, timeout_s=30.0)
    finally:
        engine_srv.stop()

    assert results[0].success and len(results[0].turns) == 2
    turn0_output = results[0].turns[0].content
    assert turn0_output  # the mock streamed real content
    bodies = [b.decode() for b in engine_srv.request_bodies]
    assert len(bodies) == 2
    # the second request's messages contain the first turn's assistant output
    second = json.loads(bodies[1])
    assistant = [m for m in second["messages"] if m["role"] == "assistant"]
    assert assistant and assistant[0]["content"] == turn0_output


def test_native_chains_json_escape_special_content():
    """Outputs containing quotes/backslashes splice into valid JSON."""
    engine_srv = _mock_engine(
        num_chunks=2, store_bodies=True, chunk_content='he said \\"hi\\"\nok'
    )
    try:
        engine = NativeReceiveEngine("127.0.0.1", engine_srv.port)
        chains = [[_spec(2), _spec(2, hole_refs=[0])]]
        results = engine.run_chains(chains, concurrency=1, timeout_s=30.0)
    finally:
        engine_srv.stop()

    assert results[0].success and len(results[0].turns) == 2
    bodies = [b.decode() for b in engine_srv.request_bodies]
    second = json.loads(bodies[1])  # must parse: escaping was correct
    assistant = [m for m in second["messages"] if m["role"] == "assistant"]
    assert assistant[0]["content"] == results[0].turns[0].content


def test_native_chains_honor_think_time():
    """Turn N+1 fires at (turn N complete + delay), on time (not early/late)."""
    delay_s = 0.15
    engine_srv = _mock_engine(num_chunks=2)
    try:
        engine = NativeReceiveEngine("127.0.0.1", engine_srv.port)
        chains = [[_spec(2), _spec(2, delay_s=delay_s)] for _ in range(4)]
        results = engine.run_chains(chains, concurrency=4, timeout_s=30.0)
    finally:
        engine_srv.stop()

    for r in results:
        assert r.success and len(r.turns) == 2
        # dispatch lateness vs (complete + delay) stays small...
        assert all(abs(h) < 0.02 for h in r.handoff_s)
        # ...and the actual inter-turn gap includes the think time
        gap = r.turns[1].dispatch_offset_s - r.turns[0].dispatch_offset_s
        assert gap >= delay_s * 0.9


def test_native_chains_open_loop_starts():
    """start_offsets_s makes chain STARTS fire on their arrival schedule."""
    engine_srv = _mock_engine(num_chunks=2)
    try:
        engine = NativeReceiveEngine("127.0.0.1", engine_srv.port)
        offsets = [0.0, 0.1, 0.2, 0.3]
        chains = [[_spec(2), _spec(2)] for _ in offsets]
        results = engine.run_chains(
            chains, concurrency=16, timeout_s=30.0, start_offsets_s=offsets
        )
    finally:
        engine_srv.stop()

    for r, off in zip(results, offsets):
        assert r.success
        assert abs(r.turns[0].dispatch_offset_s - off) < 0.02


def test_native_chains_shard_across_threads():
    """Sharded chains produce the same completions as a single loop."""
    engine_srv = _mock_engine(num_chunks=3)
    try:
        engine = NativeReceiveEngine("127.0.0.1", engine_srv.port)
        chains = [[_spec(3), _spec(3, hole_refs=[0])] for _ in range(12)]
        results = engine.run_chains(
            chains, concurrency=8, timeout_s=30.0, num_threads=4
        )
    finally:
        engine_srv.stop()

    assert len(results) == 12
    assert all(r.success and len(r.turns) == 2 for r in results)
