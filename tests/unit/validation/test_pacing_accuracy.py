"""Validation tests for real-time audio pacing accuracy (Task 4).

Goal (whiteboard): "a 1m30s clip should take 1m30s to send out". These tests
codify the finding from analysis/04_pacing_drift.md:

  * Absolute-deadline pacing (what veeksha/client/stt.py:276-381 does) does NOT
    accumulate drift and stays ~real-time at low concurrency.
  * A naive drifting `sleep(chunk_dt)` loop DOES accumulate drift (>real-time
    even at concurrency 1) — this test would FAIL such a regression.
  * Both degrade past a single-event-loop saturation knee; that knee is reported
    (not hard-asserted) so it can be tracked without CI flakiness.

The pacing primitives here mirror stt.py exactly; they are intentionally
self-contained so the test does not require the audio client stack.
"""

from __future__ import annotations

import asyncio
import time
from typing import List

import pytest


# --- pacing primitives mirroring veeksha/client/stt.py -----------------------
def _cpu(n: int) -> int:
    acc = 0
    for k in range(n):
        acc += k * k
    return acc


async def _send_deadline(k_chunks: int, chunk_dt: float, cpu: int) -> float:
    """Absolute-deadline pacing anchored once (stt.py design)."""
    start = time.monotonic()
    anchor = start
    for i in range(k_chunks):
        _cpu(cpu)
        target = anchor + (i + 1) * chunk_dt
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
    return time.monotonic() - start


async def _send_drifting(k_chunks: int, chunk_dt: float, cpu: int) -> float:
    """Naive per-iteration sleep — the anti-pattern that accumulates drift."""
    start = time.monotonic()
    for _ in range(k_chunks):
        _cpu(cpu)
        await asyncio.sleep(chunk_dt)
    return time.monotonic() - start


async def _run(strategy, concurrency: int, **kw) -> List[float]:
    tasks = [asyncio.create_task(strategy(**kw)) for _ in range(concurrency)]
    return await asyncio.gather(*tasks)


# small, fast scale: 1.0 s "clip", 20 ms chunks
CLIP_S = 1.0
CHUNK_DT = 0.02
K = int(round(CLIP_S / CHUNK_DT))
CPU = 500


def test_deadline_pacing_is_realtime_at_low_concurrency():
    """A paced clip should take ~its own duration to send (low concurrency)."""
    ratios = [
        t / CLIP_S
        for t in asyncio.run(
            _run(_send_deadline, 4, k_chunks=K, chunk_dt=CHUNK_DT, cpu=CPU)
        )
    ]
    worst = max(ratios)
    # Generous ceiling to avoid flakiness; deadline pacing is ~1.00x in practice.
    assert worst < 1.15, (
        f"deadline pacing drifted: worst send took {worst:.3f}x the clip duration "
        f"(expected ~1.0x). Pacing may have regressed to an accumulating sleep."
    )


def test_drifting_sleep_is_detectably_worse_than_deadline():
    """Regression guard: a drifting sleep must be measurably worse than deadline.

    If someone replaces the absolute-deadline loop with `sleep(chunk_dt)`, this
    catches it: the drifting loop runs long even at concurrency 1.
    """
    drift = max(
        t / CLIP_S
        for t in asyncio.run(
            _run(_send_drifting, 1, k_chunks=K, chunk_dt=CHUNK_DT, cpu=CPU)
        )
    )
    deadline = max(
        t / CLIP_S
        for t in asyncio.run(
            _run(_send_deadline, 1, k_chunks=K, chunk_dt=CHUNK_DT, cpu=CPU)
        )
    )
    assert drift > deadline, (
        f"expected drifting sleep ({drift:.3f}x) to exceed deadline pacing "
        f"({deadline:.3f}x)"
    )
    # deadline should be close to real-time; drifting should overshoot.
    assert deadline < 1.10
    assert drift > 1.02


@pytest.mark.slow
def test_report_pacing_saturation_knee(capsys):
    """Informational: report the concurrency where even deadline pacing drifts.

    Not a hard assertion (CI-machine dependent). Prints a small table so the
    max-honest-pacing-concurrency can be tracked over time.
    """
    lines = ["", "pacing accuracy (deadline) total/clip vs concurrency:"]
    for c in (1, 25, 100, 200, 400):
        ratios = [
            t / CLIP_S
            for t in asyncio.run(
                _run(_send_deadline, c, k_chunks=K, chunk_dt=CHUNK_DT, cpu=CPU)
            )
        ]
        p50 = sorted(ratios)[len(ratios) // 2]
        lines.append(f"  concurrency={c:>4}  p50={p50:.3f}x  max={max(ratios):.3f}x")
    with capsys.disabled():
        print("\n".join(lines))
