"""Tests for the optional native (C++) receive path.

Skipped unless the veeksha_native extension has been built
(``veeksha/native/build.sh <python>``).
"""

from __future__ import annotations

import pytest

from veeksha import native

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="veeksha_native extension not built"
)


def test_native_sse_reference_server_is_punctual_and_faithful():
    """The native reference server emits with sub-ms jitter, and the native
    client measures its known cadence with low drift — the validation harness
    that isolates client fidelity from the Python reference's punctuality floor.
    """
    chunk_ms, num_chunks = 20.0, 12
    server = native._ext.NativeSseServer(num_chunks, chunk_ms, 20.0, 1)
    port = server.start()
    try:
        m = native.batch_receive_drift(
            "127.0.0.1",
            port,
            concurrency=32,
            num_chunks=num_chunks,
            chunk_ms=chunk_ms,
            total_requests=64,
            timeout_s=30.0,
        )
    finally:
        server.stop()

    assert m["completed"] == 64  # every request finishes
    # A native emitter with no GIL/coroutine overhead stays punctual even
    # co-located with the client; well under the Python reference's ~2-4 ms.
    assert server.server_jitter_p99_ms() < 2.0
    # And the native client tracks that cadence closely at modest concurrency.
    assert m["ivl_err_p99_ms"] < 6.0


def test_native_receive_drift_against_mock_engine():
    from veeksha.preflight.mock_engine import MockStreamingEngine

    chunk_ms, prefill_ms, num_chunks = 20.0, 30.0, 12
    engine = MockStreamingEngine(
        chunk_dt=chunk_ms / 1000,
        prefill_s=prefill_ms / 1000,
        default_chunks=num_chunks,
        num_loops=2,
    ).start()
    try:
        m = native.receive_drift(
            "127.0.0.1",
            engine.port,
            concurrency=20,
            num_chunks=num_chunks,
            chunk_ms=chunk_ms,
            prefill_ms=prefill_ms,
            total_requests=60,
        )
        assert m["completed"] >= 1
        # low-concurrency native drift must be small and stream ~real-time
        assert m["ivl_err_p99_ms"] < 25.0
        assert m["stretch_p99"] < 1.2
    finally:
        engine.stop()
