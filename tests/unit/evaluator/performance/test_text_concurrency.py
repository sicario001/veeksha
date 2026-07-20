"""Concurrency test for the lock-free (sharded) TextPerformanceEvaluator path.

Completing many requests from many threads must not lose updates, misalign the
parallel request-row store, or crash — proving the sharded-sketch +
narrowed-lock design is thread-safe.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from veeksha.config.evaluator import (
    PerformanceEvaluatorConfig,
    TextChannelPerformanceConfig,
)
from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
from veeksha.evaluator.performance.text import TextPerformanceEvaluator
from veeksha.evaluator.sharded_sketch import ShardedCDFSketch
from veeksha.types import ChannelModality


@dataclass
class _Chan:
    metrics: Dict[str, Any]


@dataclass
class _Resp:
    channels: Dict[ChannelModality, _Chan]
    session_total_requests: int = 1
    scheduler_ready_at: Optional[float] = None
    scheduler_dispatched_at: Optional[float] = None
    client_picked_up_at: Optional[float] = None
    client_completed_at: Optional[float] = None
    result_processed_at: Optional[float] = None


@dataclass
class _Content:
    target_prompt_tokens: int = 5


def _make_response() -> _Resp:
    inter_chunk_times = [0.5] + [0.1] * 9  # ttfc=0.5, 10 chunks
    return _Resp(
        channels={
            ChannelModality.TEXT: _Chan(
                metrics={
                    "is_stream": True,
                    "inter_chunk_times": inter_chunk_times,
                    "num_delta_prompt_tokens": 5,
                    "num_total_prompt_tokens": 20,
                    "num_output_tokens": 10,
                }
            )
        }
    )


def test_concurrent_completions_are_thread_safe():
    evaluator = TextPerformanceEvaluator(
        PerformanceEvaluatorConfig(), TextChannelPerformanceConfig()
    )
    n = 400
    # request-level summaries are the sharded ones
    assert isinstance(evaluator.summaries["ttfc"], ShardedCDFSketch)

    for rid in range(n):
        evaluator.register_request(
            rid,
            rid,
            float(rid),
            _Content(),
            RequestedOutputSpec(text=TextOutputSpec(target_tokens=10)),
        )

    def complete(rid: int):
        evaluator.record_request_completed(rid, rid, float(rid) + 2.0, _make_response())

    threads = [threading.Thread(target=complete, args=(rid,)) for rid in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # no lost updates: every request landed in the sharded sketches
    assert len(evaluator.summaries["ttfc"]) == n
    assert len(evaluator.summaries["num_output_tokens"]) == n
    # parallel row store stayed aligned (same length across columns)
    assert len(evaluator.request_ids) == n
    assert len(evaluator.ttfc) == n
    assert len(evaluator.num_output_tokens) == n
    assert set(evaluator.request_ids) == set(range(n))
    # ttfc is exactly 0.5 for every request -> merged mean ~= 0.5
    summary = evaluator.summaries["ttfc"].get_summary()
    mean = summary["Time to First Chunk (Mean)"]
    assert abs(mean - 0.5) < 1e-2
