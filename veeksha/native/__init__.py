"""Optional native (C++) receive path + drift metrics over its output.

The `veeksha_native` extension (build with ``veeksha/native/build.sh``) runs a
single-thread poll() receive loop that timestamps SSE chunks at true socket-read
time. This module wraps it and computes the SAME drift metrics the Python
pipeline reports, so the two are directly comparable. If the extension is not
built, ``is_available()`` returns False and callers fall back to Python.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:  # the compiled extension sits next to this file after build.sh
    from veeksha.native import veeksha_native as _ext  # type: ignore
except Exception:  # pragma: no cover - extension optional / not built
    _ext = None


def is_available() -> bool:
    return _ext is not None


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def receive_drift(
    host: str,
    port: int,
    concurrency: int,
    num_chunks: int,
    chunk_ms: float,
    prefill_ms: float,
    total_requests: int,
    timeout_s: float = 120.0,
) -> Dict[str, float]:
    """Run the native receiver and return drift metrics matching the Python probe.

    Metrics: achieved concurrency is not observable client-side here (the engine
    reports it), so we return the completed count, the p99 inter-chunk error vs
    the known cadence, and the p99 steady-state stretch.
    """
    if _ext is None:
        raise RuntimeError(
            "veeksha_native not built; run veeksha/native/build.sh <python>"
        )
    timelines: List[List[float]] = _ext.receive(
        host, port, concurrency, num_chunks, total_requests, timeout_s
    )
    # drop first 10% warmup, mirror the Python probe's windowing
    lo = int(0.10 * len(timelines))
    window = timelines[lo:]
    ivl_err: List[float] = []
    stretch: List[float] = []
    ideal_span = (num_chunks - 1) * chunk_ms
    for offs in window:
        if len(offs) < 2:
            continue
        for a, b in zip(offs, offs[1:]):
            ivl_err.append(abs((b - a) - chunk_ms))
        span = offs[-1] - offs[0]  # first chunk -> last chunk (ms)
        stretch.append(span / ideal_span if ideal_span > 0 else float("nan"))
    return {
        "concurrency": float(concurrency),
        "completed": float(len(timelines)),
        "measured": float(len(stretch)),
        "ivl_err_p99_ms": _pct(ivl_err, 99),
        "stretch_p99": _pct(stretch, 99),
    }
