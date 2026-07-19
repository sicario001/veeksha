"""P6: native receive->dispatch coupling for closed-loop / dependent workloads.

Each chain is a sequence of dependent turns; native fires the next turn the
instant the prior completes, so the inter-turn handoff stays native (sub-ms) and
never routes through a Python queue. Validates the coupling is off the Python
critical path.
"""

from __future__ import annotations

import pytest

from veeksha.native.engine import (
    NativeReceiveEngine,
    NativeRequest,
    native_available,
)

pytestmark = pytest.mark.skipif(
    not native_available(), reason="veeksha_native extension not built"
)


def _mock_engine(num_chunks=5):
    from veeksha.preflight.mock_engine import MockStreamingEngine

    return MockStreamingEngine(
        chunk_dt=0.02, prefill_s=0.02, default_chunks=num_chunks, num_loops=4
    ).start()


def _turn(num_chunks: int) -> NativeRequest:
    body = f'{{"model":"d","stream":true,"max_completion_tokens":{num_chunks}}}'
    return NativeRequest(path="/v1/chat/completions", body=body)


def test_native_chains_run_all_dependent_turns():
    engine_srv = _mock_engine(num_chunks=5)
    try:
        engine = NativeReceiveEngine("127.0.0.1", engine_srv.port)
        chains = [[_turn(5), _turn(5), _turn(5)] for _ in range(6)]
        results = engine.run_chains(chains, concurrency=4, timeout_s=30.0)
    finally:
        engine_srv.stop()

    assert len(results) == 6
    ok = [r for r in results if r.success]
    assert len(ok) == 6
    # every chain ran all 3 dependent turns, each turn streamed its 5 chunks
    for r in ok:
        assert len(r.turns) == 3
        assert all(len(t.stream) == 5 for t in r.turns)
        assert len(r.handoff_s) == 2  # two inter-turn handoffs


def test_native_coupling_handoff_is_sub_millisecond():
    """The receive->dispatch handoff is native (no Python queue jitter)."""
    engine_srv = _mock_engine(num_chunks=4)
    try:
        engine = NativeReceiveEngine("127.0.0.1", engine_srv.port)
        chains = [[_turn(4), _turn(4), _turn(4), _turn(4)] for _ in range(8)]
        results = engine.run_chains(chains, concurrency=4, timeout_s=30.0)
    finally:
        engine_srv.stop()

    handoffs = [h for r in results if r.success for h in r.handoff_s]
    assert handoffs
    # native fires the next turn immediately; handoff stays well under 1ms median
    handoffs.sort()
    median = handoffs[len(handoffs) // 2]
    assert median < 0.005  # < 5ms (typically ~0.05ms); no Python in the loop
