"""preflight validation for Veeksha.

Before running a real benchmark on a system, `veeksha validate-preflight` measures
how far *this machine* can be trusted: the maximum concurrency at which Veeksha's
own timing instrumentation stays accurate. It drives the real client/dispatch/
completion pipeline against a built-in streaming dummy engine with a known emit
schedule, so any drift between the engine's schedule and Veeksha's recorded
timings is attributable to the harness — not a real model.

Two checks:
  * receive-drift accuracy — max concurrency where recorded inter-chunk timings
    (which feed TTFT/TPOT) track the engine's true cadence.
  * pacing accuracy — max concurrency where real-time send pacing (as used for
    streaming-audio/ASR benchmarks) stays on schedule.

No GPU, no model download, no network required.
"""

from veeksha.preflight.report import PreflightReport
from veeksha.preflight.validator import run_preflight_check

__all__ = ["PreflightReport", "run_preflight_check"]
