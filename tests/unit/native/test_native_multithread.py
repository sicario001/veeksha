"""Multi-threaded native engine: connections sharded across N poll-loop threads.

Validates correctness — the same results, index-aligned and complete, regardless
of thread count. (Throughput scaling only shows on a dedicated server host; a
co-located mock server saturates first, so we assert correctness, not speedup.)
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


def _dummy_engine(num_chunks=8):
    from veeksha.preflight.dummy_engine import DummyStreamingEngine

    return DummyStreamingEngine(
        chunk_dt=0.02, prefill_s=0.02, default_chunks=num_chunks, num_loops=16
    ).start()


def _chat(num_chunks):
    body = f'{{"model":"d","stream":true,"max_completion_tokens":{num_chunks}}}'
    return NativeRequest(path="/v1/chat/completions", body=body)


@pytest.mark.parametrize("num_threads", [1, 2, 4, 8])
def test_sharded_threads_preserve_correctness(num_threads):
    engine_srv = _dummy_engine(num_chunks=8)
    try:
        native = NativeReceiveEngine("127.0.0.1", engine_srv.port)
        n = 120
        results = native.run(
            [_chat(8) for _ in range(n)],
            concurrency=48,
            timeout_s=30.0,
            num_threads=num_threads,
        )
    finally:
        engine_srv.stop()

    assert len(results) == n
    # every request index appears exactly once (sharding merged correctly)
    assert sorted(r.index for r in results) == list(range(n))
    ok = [r for r in results if r.success]
    assert len(ok) == n
    # each streamed its full 8-chunk timeline in ascending order
    for r in ok:
        assert len(r.stream) == 8
        offs = [e.offset_s for e in r.stream.events]
        assert offs == sorted(offs)


def test_config_native_threads_threads_through(monkeypatch):
    """client.native_threads reaches the engine via the runner."""
    from veeksha.config.client import OpenAIChatCompletionsClientConfig
    from veeksha.native import runner

    captured = {}

    def fake_run_text(self, requests, concurrency, **kw):
        captured["num_threads"] = kw.get("num_threads")
        return []

    monkeypatch.setattr(
        "veeksha.native.transport.NativeTransport.run_text", fake_run_text
    )
    cfg = OpenAIChatCompletionsClientConfig(
        api_base="http://127.0.0.1:9/v1", model="d", native_threads=6
    )
    runner.execute_native([], cfg, concurrency=10)
    assert captured["num_threads"] == 6
