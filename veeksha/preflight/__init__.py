"""Preflight validation for Veeksha.

Before running a real benchmark on a system, ``veeksha preflight`` measures how
far *this machine* can be trusted: the maximum concurrency at which Veeksha's
own timing instrumentation stays accurate. It drives the real client/dispatch/
completion pipeline against built-in mock servers with known emit schedules, so
any drift between a server's schedule and Veeksha's recorded timings is
attributable to the harness — not a real model.

Checks:
  * receive-drift accuracy — max concurrency where recorded inter-chunk timings
    (which feed TTFT/TPOT) track the engine's true cadence; also reports the
    completion-queue drift (client-done -> processed).
  * pacing accuracy — max concurrency where real-time send pacing (as used for
    streaming-audio/ASR benchmarks) stays on schedule.
  * audio-transport checks — the same questions over the REAL WebSocket
    transports (TTS receive drift; STT/ASR send drift measured at the server).
  * native comparison — the native (C++) engine on the same ladder.

Every rung's verdict is tri-state: honest, dishonest, or engine-limited (the
mock server saturated — client fidelity unmeasurable there, never misreported
as a client failure).

No GPU, no model download, no network required.

Imports are lazy (PEP 562) so lightweight pieces (e.g. the sharded mock-server
infrastructure used by test helpers) don't pull the full client stack.
"""

__all__ = ["PreflightReport", "run_preflight_check"]


def __getattr__(name):
    if name == "PreflightReport":
        from veeksha.preflight.report import PreflightReport

        return PreflightReport
    if name == "run_preflight_check":
        from veeksha.preflight.validator import run_preflight_check

        return run_preflight_check
    raise AttributeError(name)
