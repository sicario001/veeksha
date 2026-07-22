"""Measurement-fidelity preflight: mock servers with deterministic schedules.

The preflight answers one question — *can I trust the numbers this harness is
about to produce at this concurrency, on this machine?* — by running the real
benchmark pipeline against servers whose emit schedule is known exactly. Any
deviation between what a mock was *supposed* to send and what the client
*recorded* is harness drift, not model behaviour.

The package is layered so that measurement and scoring never mix:

* :mod:`veeksha.preflight.sharded_server` — shared accept-sharding, phase
  spreading and lock-free self-lateness telemetry.
* :mod:`veeksha.preflight.mock_engine` — text SSE mock
  (``POST /v1/chat/completions``).
* :mod:`veeksha.preflight.audio_server` — realtime-TTS and STT WebSocket mocks.
* :mod:`veeksha.preflight.drivers` — per-modality workloads that run the REAL
  ``_run_main_loop`` (real scheduler, workers and clients) against those mocks,
  plus the dispatch-accuracy workload, which scores the rate scheduler's own
  arrival times rather than a transport.
* :mod:`veeksha.preflight.scorer` — the drift math: pure functions over the
  recorded stamps, no I/O, no servers.
* :mod:`veeksha.preflight.validator` — the gates and the tri-state verdict
  (honest / dishonest / engine-limited), applied to one measurement per check
  at the target concurrency.
* :mod:`veeksha.preflight.report` — the rendered table and its column legend.
* :mod:`veeksha.preflight.runner` — the ``veeksha preflight`` CLI entry point.

Scoring is imported eagerly; the drivers/validator/runner are not, so importing
this package never drags in the benchmark or client module graphs.
"""

from veeksha.preflight.audio_server import (
    MockRealtimeAudioServer,
    MockSTTPreflightServer,
)
from veeksha.preflight.mock_engine import MockStreamingEngine
from veeksha.preflight.scorer import (
    absolute_times_from_offsets_ms,
    asr_send_schedule_ms,
    audio_arrival_times,
    delivery_lag_ms,
    delivery_lags_ms,
    dispatch_drift_ms,
    p50,
    p99,
    rate_schedule_offsets_ms,
    send_drift,
    stt_send_times,
    text_arrival_times,
    text_pacing_schedule_ms,
    think_time_drift_ms,
)
from veeksha.preflight.sharded_server import (
    ACCEPT_BACKLOG,
    PhaseSpreader,
    ShardedLoopServer,
    ShardedTelemetry,
    percentile,
)

__all__ = [
    "ACCEPT_BACKLOG",
    "MockRealtimeAudioServer",
    "MockSTTPreflightServer",
    "MockStreamingEngine",
    "PhaseSpreader",
    "ShardedLoopServer",
    "ShardedTelemetry",
    "absolute_times_from_offsets_ms",
    "asr_send_schedule_ms",
    "audio_arrival_times",
    "delivery_lag_ms",
    "delivery_lags_ms",
    "dispatch_drift_ms",
    "p50",
    "p99",
    "percentile",
    "rate_schedule_offsets_ms",
    "send_drift",
    "stt_send_times",
    "text_arrival_times",
    "text_pacing_schedule_ms",
    "think_time_drift_ms",
]
