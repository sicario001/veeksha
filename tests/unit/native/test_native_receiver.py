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
