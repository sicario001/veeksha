"""Tests for the optional native (C++) receive path.

Skipped unless the veeksha_native extension has been built
(``veeksha/native/build.sh <python>``).
"""

from __future__ import annotations

import pytest

from veeksha import native
from veeksha.preflight.drivers import NATIVE, TextWorkload, measure_rung

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="veeksha_native extension not built"
)


class _EngineFacade:
    """Adapts the native reference SSE server to the mock engine's interface so
    the SAME preflight workload can be pointed at either one — both report peak
    connections and emit jitter, so the honesty gate applies unchanged."""

    def __init__(self, port, server):
        self.port = port
        self._server = server

    @property
    def max_active_conns(self):
        return self._server.max_active_conns()

    def reset_telemetry(self):
        self._server.reset_telemetry()

    def server_jitter_p99_ms(self):
        return self._server.server_jitter_p99_ms()


def _preflight_config(**kw):
    from veeksha.config.preflight import PreflightCheckConfig

    base = dict(
        target_concurrency=32,
        num_client_threads=2,
        num_dispatcher_threads=1,
        num_completion_threads=1,
        num_chunks=12,
        budget_s=2.0,
        check_pacing=False,
    )
    base.update(kw)
    return PreflightCheckConfig(**base)


def test_native_sse_reference_server_is_punctual_and_faithful():
    """The native reference server emits with sub-ms jitter, and the native
    client measures its known cadence with low drift — the validation harness
    that isolates client fidelity from the Python reference's punctuality floor.
    """
    chunk_ms, num_chunks = 20.0, 12
    server = native._ext.NativeSseServer(num_chunks, chunk_ms, 20.0, 1)
    port = server.start()
    try:
        workload = TextWorkload(
            _EngineFacade(port, server),
            _preflight_config(num_chunks=num_chunks),
            chunk_dt=chunk_ms / 1000.0,
            prefill_s=0.02,
        )
        point = measure_rung(workload, NATIVE, 32, 10.0, 1.05, n_requests=64)[
            workload.name
        ]
    finally:
        server.stop()

    # native fills the concurrency window; a connection or two may close
    # before the last opens, so peak overlap need not hit the cap exactly.
    assert point.achieved >= 30
    # A native emitter with no GIL/coroutine overhead stays punctual even
    # co-located with the client. Typically sub-ms; the bound leaves headroom
    # for idle-box timer slop (macOS parks fresh threads on efficiency cores),
    # while still far under the Python reference's saturation regime.
    assert server.server_jitter_p99_ms() < 5.0
    # And the native client tracks that cadence closely at modest concurrency.
    # (10 ms leaves headroom for idle-box timer slop — macOS parks light loads
    # on efficiency cores — while staying far below the Python path's 20-30 ms.)
    assert point.ivl_err_p99_ms < 10.0


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
        workload = TextWorkload(
            engine,
            _preflight_config(num_chunks=num_chunks),
            chunk_dt=chunk_ms / 1000.0,
            prefill_s=prefill_ms / 1000.0,
        )
        point = measure_rung(workload, NATIVE, 20, 25.0, 1.05, n_requests=60)[
            workload.name
        ]
        assert point.achieved >= 19  # engine may drop a connection under load
        # low-concurrency native drift must be small
        assert point.ivl_err_p99_ms < 25.0
    finally:
        engine.stop()
