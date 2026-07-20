import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from veeksha.config.evaluator import (
    DecodeWindowConfig,
    PerformanceEvaluatorConfig,
    TextChannelPerformanceConfig,
)
from veeksha.core.timed_event_stream import TimedEventStream
from veeksha.evaluator.base import EvaluationResult
from veeksha.evaluator.cdf_sketch import CDFSketch
from veeksha.evaluator.performance.channel_base import BaseChannelPerformanceEvaluator
from veeksha.evaluator.sharded_sketch import ShardedCDFSketch
from veeksha.evaluator.chrome_trace import generate_chrome_trace
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

logger = init_logger(__name__)


@dataclass
class TextRequestMetrics:
    """Metrics for a single text request.

    Headline timings (TTFC / TPOT / TBC / E2E) derive from the shared
    ``TimedEventStream`` primitive — the same derivations the audio evaluator
    uses — with ``inter_chunk_times`` kept as the stored wire format (the
    stream is its cumulative sum).
    """

    request_id: int
    session_id: int
    request_dispatched_at: float
    client_completed_at: float
    num_prompt_tokens: int
    num_output_tokens: int
    inter_chunk_times: List[float]
    is_stream: bool = False
    num_requested_output_tokens: Optional[int] = None
    session_total_requests: Optional[int] = None
    num_delta_prompt_tokens: Optional[int] = None
    num_total_prompt_tokens: Optional[int] = None
    target_num_delta_prompt_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        self._stream = TimedEventStream.from_inter_chunk_times(
            self.inter_chunk_times, modality=ChannelModality.TEXT
        )

    @property
    def stream(self) -> TimedEventStream:
        """The request's receive timeline as the shared timing primitive."""
        return self._stream

    @property
    def num_total_tokens(self) -> int:
        if self.num_total_prompt_tokens is not None:
            return self.num_total_prompt_tokens + self.num_output_tokens
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def end_to_end_latency(self) -> float:
        return self._stream.end_to_end() or 0.0

    @property
    def normalized_end_to_end_latency(self) -> float:
        if self.num_output_tokens == 0:
            return 0.0
        return self.end_to_end_latency / self.num_output_tokens

    @property
    def ttfc(self) -> float:
        return self._stream.time_to_first_event() or 0.0

    @property
    def tpot(self) -> float:
        # (E2E - TTFC) / (OutputTokens - 1)
        return self._stream.time_per_unit(self.num_output_tokens) or 0.0

    @property
    def tbc(self) -> float:
        return self._stream.mean_inter_event() or 0.0

    @property
    def output_throughput(self) -> float:
        if self.end_to_end_latency == 0:
            return 0.0
        return self.num_output_tokens / self.end_to_end_latency


class TextPerformanceEvaluator(BaseChannelPerformanceEvaluator):
    """Performance evaluator for text generation (implements legacy MetricStore)

    - CDFSketch-based metric aggregation
    - Request-level metrics tracking
    - Session metrics (size, duration, dispatch gap, think time)
    - Throughput metrics
    - Output storage (CSV, JSON, plots)
    - WandB integration
    - Streaming metrics support
    """

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: Optional[TextChannelPerformanceConfig] = None,
        benchmark_start_time: float = 0.0,
    ):
        self.config = config
        self.channel_config = channel_config or TextChannelPerformanceConfig()
        self.benchmark_start_time = benchmark_start_time

        self.lock = threading.Lock()

        # request tracking
        self._pending_requests: Dict[int, Dict[str, Any]] = (
            {}
        )  # request_id -> dispatch info

        # aggregate metrics
        self.summaries: Dict[str, Any] = {
            "num_prompt_tokens": CDFSketch(
                metric_name="Number of Prompt Tokens",
            ),
            "num_output_tokens": CDFSketch(
                metric_name="Number of Output Tokens",
            ),
            "num_total_tokens": CDFSketch(
                metric_name="Number of Total Tokens",
            ),
            "tpot": CDFSketch(
                metric_name="Time per Output Token",
                unit="s",
            ),
            "ttfc": CDFSketch(
                metric_name="Time to First Chunk",
                unit="s",
            ),
            "tbc": CDFSketch(
                metric_name="Time Between Chunks",
                unit="s",
            ),
            "end_to_end_latency": CDFSketch(
                metric_name="End to End Latency",
                unit="s",
            ),
            "normalized_end_to_end_latency": CDFSketch(
                metric_name="Normalized End to End Latency",
                unit="s/token",
            ),
            "output_throughput": CDFSketch(
                metric_name="Output Throughput",
            ),
            "session_size": CDFSketch(
                metric_name="Requests per Session",
            ),
            "session_duration": CDFSketch(
                metric_name="Session Duration",
                unit="s",
            ),
            "session_think_time": CDFSketch(
                metric_name="Intra-session Think Time",
                unit="s",
            ),
        }

        # request-level metrics
        self._request_level_summary_keys = {
            "num_prompt_tokens",
            "num_output_tokens",
            "num_total_tokens",
            "tpot",
            "ttfc",
            "tbc",
            "end_to_end_latency",
            "normalized_end_to_end_latency",
            "output_throughput",
        }

        # Request-level sketches are written per-completion across worker threads;
        # make them lock-free (sharded) so completion processing scales on
        # free-threaded Python. Session sketches keep the plain CDFSketch (written
        # under the light lock). See veeksha/evaluator/sharded_sketch.py.
        for _key in self._request_level_summary_keys:
            _c = self.summaries[_key]
            self.summaries[_key] = ShardedCDFSketch(
                _c.metric_name,
                should_write_to_wandb=_c.should_write_to_wandb,
                unit=_c.unit,
            )

        self.request_dispatched_at: List[float] = []
        self.completed_at: List[float] = []
        self.num_prompt_tokens: List[int] = []
        self.num_output_tokens: List[int] = []
        self.num_requested_output_tokens: List[Optional[int]] = []
        self.num_delta_prompt_tokens: List[Optional[int]] = []
        self.num_total_prompt_tokens: List[Optional[int]] = []
        self.target_num_delta_prompt_tokens: List[Optional[int]] = []
        self.num_total_tokens: List[int] = []
        self.tpot: List[float] = []
        self.ttfc: List[float] = []
        self.tbc: List[List[float]] = []
        self.end_to_end_latency: List[float] = []
        self.normalized_end_to_end_latency: List[float] = []
        self.output_throughput: List[float] = []
        self.session_ids: List[Optional[int]] = []
        self.session_total_requests: List[Optional[int]] = []
        self.request_ids: List[int] = []

        # Lifecycle timestamps
        self.scheduler_ready_at: List[Optional[float]] = []
        self.scheduler_dispatched_at: List[Optional[float]] = []
        self.client_picked_up_at: List[Optional[float]] = []
        self.client_completed_at: List[Optional[float]] = []
        self.result_processed_at: List[Optional[float]] = []

        # streaming
        self.is_stream: List[bool] = []
        self._request_rows_streamed: int = 0
        self._request_time_reference: float = self.benchmark_start_time

        # session tracking
        self._session_last_completion: Dict[int, float] = {}

    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        content: Any,
        requested_output: Any = None,
    ) -> None:
        """Register a text request that was dispatched."""
        with self.lock:
            if self._request_time_reference == 0.0:
                self._request_time_reference = dispatched_at

            target_output_tokens = None
            if requested_output is not None and hasattr(requested_output, "text"):
                text_spec = requested_output.text
                target_output_tokens = text_spec.target_tokens

            target_prompt_tokens = getattr(content, "target_prompt_tokens", None)

            self._pending_requests[request_id] = {
                "session_id": session_id,
                "dispatched_at": dispatched_at,
                "target_output_tokens": target_output_tokens,
                "target_prompt_tokens": target_prompt_tokens,
            }

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
    ) -> None:
        """Record that a text request completed.

        The expensive per-chunk work (sketch inserts in ``_update_summaries``)
        runs lock-free via sharded sketches so completion-worker threads scale on
        free-threaded Python. Only the cheap shared state — the pending-request
        map, per-session think-time, and the parallel request-row store (whose
        appends must stay index-aligned) — is taken under ``self.lock``.
        """
        with self.lock:
            dispatch_info = self._pending_requests.pop(request_id, None)
        if dispatch_info is None:
            logger.warning(f"Request {request_id} completed but was not registered")
            return

        dispatched_at = dispatch_info["dispatched_at"]
        target_output_tokens = dispatch_info.get("target_output_tokens")
        target_prompt_tokens = dispatch_info.get("target_prompt_tokens")

        # Extract metrics from the text channel response (pure, no shared state)
        channel_response = response.channels.get(ChannelModality.TEXT)

        if channel_response is not None:
            channel_metrics = channel_response.metrics or {}
            num_total_prompt_tokens = channel_metrics.get("num_total_prompt_tokens")
            num_delta_prompt_tokens = channel_metrics.get("num_delta_prompt_tokens")
            num_prompt_tokens = num_delta_prompt_tokens or 0
            num_output_tokens = channel_metrics.get("num_output_tokens", 0)
            inter_chunk_times = channel_metrics.get("inter_chunk_times", [])
            is_stream_val = channel_metrics.get("is_stream")
            request_is_stream = bool(is_stream_val)
        else:
            num_prompt_tokens = 0
            num_output_tokens = 0
            inter_chunk_times = []
            num_delta_prompt_tokens = None
            num_total_prompt_tokens = None
            request_is_stream = False

        session_total_requests = getattr(response, "session_total_requests", None)

        # Create metrics object (pure)
        metrics = TextRequestMetrics(
            request_id=request_id,
            session_id=session_id,
            request_dispatched_at=dispatched_at,
            client_completed_at=completed_at,
            num_prompt_tokens=num_prompt_tokens,
            num_output_tokens=num_output_tokens,
            inter_chunk_times=inter_chunk_times,
            is_stream=bool(request_is_stream),
            num_requested_output_tokens=target_output_tokens,
            session_total_requests=session_total_requests,
            num_delta_prompt_tokens=num_delta_prompt_tokens,
            num_total_prompt_tokens=num_total_prompt_tokens,
            target_num_delta_prompt_tokens=target_prompt_tokens,
        )

        # Update request-level CDF sketches — lock-free (sharded).
        self._update_summaries(metrics)

        # Shared state: session think-time + the parallel row store, under lock.
        with self.lock:
            prev_completion = self._session_last_completion.get(session_id)
            if prev_completion is not None:
                think_time = dispatched_at - prev_completion
                if think_time >= 0:
                    self.summaries["session_think_time"].put(think_time)
            self._session_last_completion[session_id] = completed_at

            # Store request-level metrics (incl. lifecycle timestamps from response)
            self._store_request_metrics(metrics, dispatched_at, response)

    def _update_summaries(self, metrics: TextRequestMetrics) -> None:
        """Update CDF sketches with request metrics."""
        is_streaming = metrics.is_stream
        for metric_name, cdf_sketch in self.summaries.items():
            if metric_name not in self._request_level_summary_keys:
                continue

            if not is_streaming and metric_name in {"tpot", "tbc"}:
                continue

            if metric_name == "tbc":
                # TBC is the inter-chunk times excluding TTFC
                cdf_sketch.extend(metrics.inter_chunk_times[1:])
            else:
                cdf_sketch.put(getattr(metrics, metric_name))

    def _store_request_metrics(
        self, metrics: TextRequestMetrics, dispatched_at: float, response: Any
    ) -> None:
        """Store request-level metrics for detailed output."""
        normalized_dispatched_at = max(
            0.0, dispatched_at - self._request_time_reference
        )

        self.request_dispatched_at.append(normalized_dispatched_at)
        self.completed_at.append(
            max(0.0, metrics.client_completed_at - self._request_time_reference)
        )
        self.num_prompt_tokens.append(metrics.num_prompt_tokens)
        self.num_output_tokens.append(metrics.num_output_tokens)
        self.num_requested_output_tokens.append(metrics.num_requested_output_tokens)
        self.num_delta_prompt_tokens.append(metrics.num_delta_prompt_tokens)
        self.num_total_prompt_tokens.append(metrics.num_total_prompt_tokens)
        self.target_num_delta_prompt_tokens.append(
            metrics.target_num_delta_prompt_tokens
        )
        self.num_total_tokens.append(metrics.num_total_tokens)
        self.is_stream.append(metrics.is_stream)
        self.tpot.append(metrics.tpot)
        self.ttfc.append(metrics.ttfc)
        self.tbc.append(metrics.inter_chunk_times[1:])
        self.end_to_end_latency.append(metrics.end_to_end_latency)
        self.normalized_end_to_end_latency.append(metrics.normalized_end_to_end_latency)
        self.output_throughput.append(metrics.output_throughput)
        self.session_ids.append(metrics.session_id)
        self.session_total_requests.append(metrics.session_total_requests)
        self.request_ids.append(metrics.request_id)

        # Extract and store lifecycle timestamps from response
        def normalize_ts(ts: Optional[float]) -> Optional[float]:
            if ts is None:
                return None
            return round(max(0.0, ts - self._request_time_reference), 5)

        self.scheduler_ready_at.append(
            normalize_ts(getattr(response, "scheduler_ready_at", None))
        )
        self.scheduler_dispatched_at.append(
            normalize_ts(getattr(response, "scheduler_dispatched_at", None))
        )
        self.client_picked_up_at.append(
            normalize_ts(getattr(response, "client_picked_up_at", None))
        )
        self.client_completed_at.append(
            normalize_ts(getattr(response, "client_completed_at", None))
        )
        self.result_processed_at.append(
            normalize_ts(getattr(response, "result_processed_at", None))
        )

    def record_session_completed(
        self,
        session_id: int,
        session_size: int,
        first_dispatch_at: Optional[float],
        last_completion_at: Optional[float],
    ) -> None:
        """Record session-level metrics."""
        with self.lock:
            # Session size
            self.summaries["session_size"].put(session_size)

            # Session duration
            if first_dispatch_at is not None and last_completion_at is not None:
                duration = max(0.0, last_completion_at - first_dispatch_at)
                self.summaries["session_duration"].put(duration)

            # Clean up session state
            self._session_last_completion.pop(session_id, None)

    def get_summary(self) -> Dict[str, float]:
        """Get summary metrics from all CDF sketches."""
        perf_summary = {}
        for cdf_sketch in self.summaries.values():
            perf_summary.update(cdf_sketch.get_summary())

        return perf_summary

    def finalize(self) -> EvaluationResult:
        """Finalize evaluation and return results."""
        with self.lock:
            return EvaluationResult(
                evaluator_type="text_performance",
                channel=ChannelModality.TEXT,
                metrics=self.get_summary(),
            )

    def get_streaming_metrics(self) -> Optional[Dict[str, Any]]:
        """Return current metrics for streaming."""
        with self.lock:
            return self.get_summary()

    def save(self, output_dir: str) -> None:
        """Save all evaluation artifacts."""
        with self.lock:
            self._save_prefill_stats(output_dir)
            self._save_request_level_metrics(output_dir)
            self._save_cdf_csvs(output_dir)
            self._save_throughput_metrics(output_dir)
            self._maybe_save_decode_window_metrics(output_dir)
            self._plot_cdfs(output_dir)
            self._store_ttfc_violin_plots(output_dir)

            self._save_chrome_trace(output_dir)
            self._log_wandb_metrics(output_dir)

    def flush_streaming_outputs(self, output_dir: str) -> None:
        """Flush current metrics for streaming."""
        with self.lock:
            # Export new request-level rows
            rows = self._export_request_rows(self._request_rows_streamed)
            if rows:
                self._append_request_level_rows(output_dir, rows)
                self._request_rows_streamed = len(self.ttfc)

            # Save current CDF summaries
            self._save_cdf_csvs(output_dir)

    # -------------------------------------------------------------------------
    # Output methods
    # -------------------------------------------------------------------------

    def _save_chrome_trace(self, output_dir: str) -> None:
        """Save Chrome Trace Event Format JSON for timeline visualization."""
        rows = self._export_request_rows(0)
        path = os.path.join(output_dir, "chrome_trace.json")
        generate_chrome_trace(rows, path)

    def _save_request_level_metrics(self, output_dir: str) -> None:
        """Save request-level metrics as JSONL."""
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        rows = self._export_request_rows(0)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def _export_request_rows(self, start_index: int = 0) -> List[Dict[str, Any]]:
        """Export request-level metrics as list of dicts."""
        rows: List[Dict[str, Any]] = []
        for idx in range(start_index, len(self.ttfc)):
            rows.append(
                {
                    "request_id": self.request_ids[idx],
                    "session_id": self.session_ids[idx],
                    "session_total_requests": self.session_total_requests[idx],
                    # Lifecycle timestamps
                    "scheduler_ready_at": self.scheduler_ready_at[idx],
                    "scheduler_dispatched_at": round(
                        self.request_dispatched_at[idx], 5
                    ),
                    "client_picked_up_at": self.client_picked_up_at[idx],
                    "client_completed_at": round(self.completed_at[idx], 5),
                    "result_processed_at": self.result_processed_at[idx],
                    # Token metrics
                    "num_delta_prompt_tokens": self.num_delta_prompt_tokens[idx],
                    "num_total_prompt_tokens": self.num_total_prompt_tokens[idx],
                    "target_num_delta_prompt_tokens": self.target_num_delta_prompt_tokens[
                        idx
                    ],
                    "num_output_tokens": self.num_output_tokens[idx],
                    "num_requested_output_tokens": self.num_requested_output_tokens[
                        idx
                    ],
                    "num_total_tokens": self.num_total_tokens[idx],
                    "is_stream": self.is_stream[idx],
                    # Latency metrics
                    "tpot": round(self.tpot[idx], 5),
                    "ttfc": round(self.ttfc[idx], 5),
                    "end_to_end_latency": round(self.end_to_end_latency[idx], 5),
                    "normalized_end_to_end_latency": round(
                        self.normalized_end_to_end_latency[idx], 5
                    ),
                    "output_throughput": round(self.output_throughput[idx], 5),
                    "tbc": [round(t, 5) for t in self.tbc[idx]],
                }
            )
        return rows

    def _append_request_level_rows(
        self, output_dir: str, rows: List[Dict[str, Any]]
    ) -> None:
        """Append request-level rows to JSONL file."""
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        with open(path, "a") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def _save_cdf_csvs(self, output_dir: str) -> None:
        """Save CDF data as CSV files."""
        for metric_name, cdf_sketch in self.summaries.items():
            df = cdf_sketch._to_df()
            df.to_csv(os.path.join(output_dir, f"{metric_name}.csv"), index=False)

    def _save_throughput_metrics(self, output_dir: str) -> None:
        """Save throughput metrics."""
        if not self.tpot:
            tpot_based = 0.0
        else:
            mean_tpot = float(np.mean(self.tpot))
            tpot_based = float("inf") if mean_tpot == 0 else float(1 / mean_tpot)

        tbc_flat: List[float] = []
        for per_request in self.tbc:
            tbc_flat.extend(per_request)
        tbc_based = float(1 / float(np.quantile(tbc_flat, 0.99))) if tbc_flat else 0.0
        metrics = {
            "tpot_based_throughput": tpot_based,
            "tbc_based_throughput": tbc_based,
        }
        path = os.path.join(output_dir, "throughput_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def _save_prefill_stats(self, output_dir: str) -> None:
        """Save TTFC statistics grouped by prompt length.

        Output:
            `prefill_stats.json` in the metrics directory.
        """
        if not self.ttfc:
            return

        # Prefer the intended prompt length from the generator when available
        group_key_name = "target_num_delta_prompt_tokens"
        groups: dict[int, list[float]] = defaultdict(list)

        for i in range(len(self.ttfc)):
            key = self.target_num_delta_prompt_tokens[i]
            if key is None:
                # Fallbacks, in order:
                # - observed delta prompt tokens (if populated)
                # - evaluator's num_prompt_tokens (delta, default 0)
                if self.num_delta_prompt_tokens[i] is not None:
                    key = int(self.num_delta_prompt_tokens[i])  # type: ignore[arg-type]
                    group_key_name = "num_delta_prompt_tokens"
                else:
                    key = int(self.num_prompt_tokens[i])
                    group_key_name = "num_prompt_tokens"

            if key is None:
                continue
            groups[int(key)].append(float(self.ttfc[i]))

        prefill_stats: Dict[str, Any] = {
            "metric": "ttfc",
            "group_by": group_key_name,
            "groups": {},
        }

        for prompt_len in sorted(groups.keys()):
            times = groups[prompt_len]
            if not times:
                prefill_stats["groups"][str(prompt_len)] = {"count": 0}
                continue
            arr = np.asarray(times, dtype=float)
            prefill_stats["groups"][str(prompt_len)] = {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p90": float(np.quantile(arr, 0.9)),
                "p99": float(np.quantile(arr, 0.99)),
            }

        path = os.path.join(output_dir, "prefill_stats.json")
        with open(path, "w") as f:
            json.dump(prefill_stats, f, indent=2)

    def _maybe_save_decode_window_metrics(self, output_dir: str) -> None:
        """Optionally write decode-window analysis artifacts.

        Written as a separate JSON artifact.
        """
        if not getattr(self.channel_config, "decode_window_enabled", False):
            return
        decode_cfg = getattr(self.channel_config, "decode_window_config", None)
        if decode_cfg is None:
            logger.warning(
                "decode_window_enabled=True but decode_window_config is missing; skipping."
            )
            return
        try:
            self._save_decode_window_metrics(output_dir, decode_cfg=decode_cfg)
        except Exception as exc:
            logger.warning("Decode window analysis failed: %s", exc)

    def _save_decode_window_metrics(
        self, output_dir: str, *, decode_cfg: DecodeWindowConfig
    ) -> None:
        """Compute and persist decode-window stats for text streaming requests."""
        # Per-request decoding interval is approximated as:
        #   [anchor + TTFC, anchor + sum(inter_chunk_times)]
        # where `anchor` is either client_picked_up_at or scheduler_dispatched_at.
        #
        # We then compute time intervals where at least `min_active_requests`
        # decoding intervals overlap, select a window, and filter TBC samples to
        # chunk-arrival times within that window.
        eligible = []
        skipped = {
            "non_stream": 0,
            "missing_anchor": 0,
            "no_decode_chunks": 0,
            "empty": 0,
        }

        num_rows = len(self.ttfc)
        if num_rows == 0:
            skipped["empty"] += 1
        for i in range(num_rows):
            if decode_cfg.require_streaming and not bool(self.is_stream[i]):
                skipped["non_stream"] += 1
                continue

            inter_chunk_times = [float(self.ttfc[i])] + [float(x) for x in self.tbc[i]]
            if len(inter_chunk_times) < 2:
                skipped["no_decode_chunks"] += 1
                continue

            dispatched_at = float(self.request_dispatched_at[i])
            anchor = dispatched_at
            if (
                decode_cfg.anchor_to_client_pickup
                and self.client_picked_up_at[i] is not None
            ):
                anchor = float(self.client_picked_up_at[i])  # type: ignore[arg-type]

            if anchor is None:
                skipped["missing_anchor"] += 1
                continue

            start = anchor + inter_chunk_times[0]
            end = anchor + float(sum(inter_chunk_times))
            if end <= start:
                skipped["no_decode_chunks"] += 1
                continue

            eligible.append(
                {
                    "index": i,
                    "anchor": anchor,
                    "inter_chunk_times": inter_chunk_times,
                    "decode_start": start,
                    "decode_end": end,
                }
            )

        events: list[tuple[float, int]] = []
        for r in eligible:
            events.append((float(r["decode_start"]), 1))
            events.append((float(r["decode_end"]), -1))

        # Sort by time, and process starts before ends at the same time.
        events.sort(key=lambda x: (x[0], -x[1]))

        # Resolve min_active_requests threshold
        min_active_threshold: int
        if decode_cfg.min_active_requests == "max_observed":
            # First pass: find peak concurrent decoding
            peak_active = 0
            current_active = 0
            for _, delta in events:
                current_active += delta
                peak_active = max(peak_active, current_active)
            min_active_threshold = max(1, peak_active)  # At least 1
        else:
            min_active_threshold = int(decode_cfg.min_active_requests)

        segments: list[dict[str, float]] = []
        active = 0
        last_t: Optional[float] = None
        for t, delta in events:
            if last_t is not None and t > last_t and active >= min_active_threshold:
                segments.append(
                    {
                        "start": float(last_t),
                        "end": float(t),
                        "duration_s": float(t - last_t),
                    }
                )
            active += delta
            last_t = t

        # Select window(s) based on strategy
        selected_segments: list[dict[str, float]] = []
        if segments:
            if decode_cfg.selection_strategy == "all":
                selected_segments = segments
            elif decode_cfg.selection_strategy == "first":
                selected_segments = [
                    min(segments, key=lambda s: (s["start"], -s["duration_s"]))
                ]
            else:
                # "longest" (default): prefer longest, then earliest.
                selected_segments = [
                    max(segments, key=lambda s: (s["duration_s"], -s["start"]))
                ]

        # Filter TBC samples from selected window(s)
        filtered_tbc: list[float] = []
        per_window_tbc: list[list[float]] = []

        for seg in selected_segments:
            window_start = seg["start"]
            window_end = seg["end"]
            window_tbc: list[float] = []

            for r in eligible:
                anchor = float(r["anchor"])
                inter_chunk_times = r["inter_chunk_times"]
                cumulative = 0.0
                for j, dt in enumerate(inter_chunk_times):
                    cumulative += float(dt)
                    arrival = anchor + cumulative
                    if j == 0:
                        continue  # TTFC
                    if window_start <= arrival <= window_end:
                        window_tbc.append(float(dt))

            per_window_tbc.append(window_tbc)
            filtered_tbc.extend(window_tbc)

        # Compute aggregate stats
        stats: Dict[str, Any] = {"count": int(len(filtered_tbc))}
        if filtered_tbc:
            arr = np.asarray(filtered_tbc, dtype=float)
            stats.update(
                {
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "p90": float(np.quantile(arr, 0.9)),
                    "p99": float(np.quantile(arr, 0.99)),
                }
            )

        # Build per-window stats for JSON
        per_window_stats: list[Dict[str, Any]] = []
        for i, (seg, w_tbc) in enumerate(zip(selected_segments, per_window_tbc)):
            w_stats: Dict[str, Any] = {
                "window_index": i,
                "start": seg["start"],
                "end": seg["end"],
                "duration_s": seg["duration_s"],
                "tbc_count": len(w_tbc),
            }
            if w_tbc:
                w_arr = np.asarray(w_tbc, dtype=float)
                w_stats.update(
                    {
                        "tbc_mean": float(np.mean(w_arr)),
                        "tbc_median": float(np.median(w_arr)),
                        "tbc_p99": float(np.quantile(w_arr, 0.99)),
                    }
                )
            per_window_stats.append(w_stats)

        # Compute total window duration
        total_window_duration = sum(seg["duration_s"] for seg in selected_segments)

        artifact: Dict[str, Any] = {
            "config": {
                "min_active_requests": decode_cfg.min_active_requests,
                "resolved_min_active_requests": min_active_threshold,
                "selection_strategy": decode_cfg.selection_strategy,
                "anchor_to_client_pickup": decode_cfg.anchor_to_client_pickup,
                "require_streaming": decode_cfg.require_streaming,
            },
            "eligible_requests": {
                "total_request_rows": num_rows,
                "eligible": len(eligible),
                "skipped": skipped,
            },
            "windows": {
                "num_candidate_segments": len(segments),
                "num_selected_segments": len(selected_segments),
                "total_duration_s": total_window_duration,
                "per_window": per_window_stats,
            },
            "tbc_in_window_stats": stats,
            "notes": [
                "Times are relative to the benchmark's request-level time reference.",
                "Decode interval is approximated using TTFC and total stream duration.",
                "TBC samples are included based on chunk-arrival time within the window(s).",
            ],
        }

        # Backward-compatible single-window view for legacy consumers/tests.
        if len(selected_segments) == 1:
            seg0 = selected_segments[0]
            artifact["window"] = {
                "start": seg0["start"],
                "end": seg0["end"],
                "duration_s": seg0["duration_s"],
            }

        out_path = os.path.join(output_dir, "decode_window_metrics.json")
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2)

        self._plot_decode_window(
            output_dir=output_dir,
            eligible=eligible,
            selected_segments=selected_segments,
            filtered_tbc=filtered_tbc,
            min_active_requests=min_active_threshold,
        )

    def _plot_decode_window(
        self,
        output_dir: str,
        eligible: list,
        selected_segments: list[dict[str, float]],
        filtered_tbc: list[float],
        min_active_requests: int,
    ) -> None:
        """Generate decode window visualization plot.

        Creates a two-panel figure:
        - Top: Timeline of per-request decode intervals with highlighted window(s)
        - Bottom: Histogram of TBC samples within the window(s)
        """
        if not eligible:
            logger.debug("No eligible requests for decode window plot")
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            num_windows = len(selected_segments)
            title_suffix = f"({num_windows} window{'s' if num_windows != 1 else ''})"

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
            fig.suptitle(
                f"Decode window analysis (min_active_requests={min_active_requests}) {title_suffix}",
                fontsize=12,
            )

            # --- Top panel: Timeline of decode intervals ---
            ax_timeline = axes[0]

            # Sort eligible by decode_start for cleaner visualization
            sorted_eligible = sorted(eligible, key=lambda r: float(r["decode_start"]))

            y_positions = list(range(len(sorted_eligible)))
            for i, r in enumerate(sorted_eligible):
                start = float(r["decode_start"])
                end = float(r["decode_end"])
                duration = end - start

                # Draw request decode interval as horizontal bar
                bar_color = "steelblue"
                ax_timeline.barh(
                    i,
                    duration,
                    left=start,
                    height=0.6,
                    color=bar_color,
                    alpha=0.7,
                    edgecolor="darkblue",
                    linewidth=0.5,
                )

            # Highlight all selected windows with color cycling
            window_colors = ["#2ecc71", "#27ae60", "#1abc9c", "#16a085"]  # Greens
            for idx, seg in enumerate(selected_segments):
                window_start = seg["start"]
                window_end = seg["end"]
                color = window_colors[idx % len(window_colors)]

                window_rect = Rectangle(
                    (window_start, -0.5),
                    window_end - window_start,
                    len(sorted_eligible),
                    alpha=0.15,
                    color=color,
                    zorder=0,
                )
                ax_timeline.add_patch(window_rect)
                ax_timeline.axvline(
                    window_start, color=color, linestyle="--", linewidth=1.5, alpha=0.8
                )
                ax_timeline.axvline(
                    window_end, color=color, linestyle="--", linewidth=1.5, alpha=0.8
                )
                # Add window label
                window_duration = window_end - window_start
                label = (
                    f"W{idx + 1}: {window_duration:.2f}s"
                    if num_windows > 1
                    else f"Window: {window_duration:.3f}s"
                )
                ax_timeline.annotate(
                    label,
                    xy=((window_start + window_end) / 2, len(sorted_eligible) - 0.5),
                    ha="center",
                    va="bottom",
                    fontsize=8 if num_windows > 2 else 9,
                    color="darkgreen",
                    fontweight="bold",
                )

            ax_timeline.set_xlabel("Time (s, relative to first request)")
            ax_timeline.set_ylabel("Request index")
            ax_timeline.set_title("Per-request decode intervals")
            # Add extra padding at top for window labels
            top_padding = 2 if num_windows > 1 else 1
            ax_timeline.set_ylim(-0.5, len(sorted_eligible) - 0.5 + top_padding)
            ax_timeline.grid(axis="x", alpha=0.3)

            # --- Bottom panel: TBC histogram ---
            ax_hist = axes[1]

            if filtered_tbc:
                arr = np.asarray(filtered_tbc, dtype=float)
                bins = min(50, max(10, len(arr) // 5))
                ax_hist.hist(
                    arr * 1000,  # Convert to ms for readability
                    bins=bins,
                    color="steelblue",
                    edgecolor="white",
                    alpha=0.8,
                )
                ax_hist.axvline(
                    float(np.mean(arr)) * 1000,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Mean: {float(np.mean(arr)) * 1000:.2f} ms",
                )
                ax_hist.axvline(
                    float(np.median(arr)) * 1000,
                    color="orange",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Median: {float(np.median(arr)) * 1000:.2f} ms",
                )
                ax_hist.legend(loc="upper right")
                ax_hist.set_title(f"TBC distribution in windows (n={len(arr)})")
            else:
                ax_hist.text(
                    0.5,
                    0.5,
                    "No TBC samples in window",
                    ha="center",
                    va="center",
                    transform=ax_hist.transAxes,
                    fontsize=12,
                    color="gray",
                )
                ax_hist.set_title("TBC distribution in windows")

            ax_hist.set_xlabel("Time Between Chunks (ms)")
            ax_hist.set_ylabel("Frequency")
            ax_hist.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            out_path = os.path.join(output_dir, "decode_window_plot.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except ImportError as e:
            logger.warning("matplotlib not available for decode window plot: %s", e)
        except Exception as e:
            logger.warning("Failed to generate decode window plot: %s", e)

    def _plot_cdfs(self, output_dir: str) -> None:
        """Generate CDF plots for all metrics."""
        for metric_name, cdf_sketch in self.summaries.items():
            cdf_sketch.plot_cdf(output_dir, metric_name)

    def _store_ttfc_violin_plots(self, output_dir: str) -> None:
        """Save TTFC distribution vs prompt length plots."""
        if not self.ttfc:
            return

        try:
            import rekha as rk  # type: ignore[import-not-found]

            from veeksha.evaluator.plot_utils import (
                apply_axis_scale,
                format_axis_label,
                recommend_axis_scale,
            )

            prompt_lengths = [int(n) for n in self.num_prompt_tokens]
            ttfcs = list(self.ttfc)
            if len(prompt_lengths) != len(ttfcs):
                return

            base_df = pd.DataFrame({"prompt_length": prompt_lengths, "ttfc": ttfcs})

            min_len = cast(int, base_df["prompt_length"].min())
            max_len = cast(int, base_df["prompt_length"].max())

            if max_len <= min_len:
                base_df["prompt_length_bin"] = pd.Series(
                    [f"{min_len}-{max_len}"] * len(base_df)
                ).astype("category")
            else:
                target_bins = 12
                bins = max(5, min(20, target_bins))
                raw_edges = np.linspace(min_len, max_len, bins + 1)
                int_edges = np.unique(np.round(raw_edges).astype(int))

                if int_edges.size < 2:
                    base_df["prompt_length_bin"] = pd.Series(
                        [f"{min_len}-{max_len}"] * len(base_df)
                    ).astype("category")
                else:
                    edges = int_edges
                    bins = edges.size - 1
                    labels = [f"{edges[i]}-{edges[i + 1]}" for i in range(bins)]
                    base_df["prompt_length_bin"] = pd.cut(
                        base_df["prompt_length"],
                        bins=edges,
                        include_lowest=True,
                        labels=labels,
                        right=True,
                    )

            df = base_df[["prompt_length_bin", "ttfc"]].copy()
            df["prompt_length_bin"] = df["prompt_length_bin"].astype("category")

            ttfc_scale = recommend_axis_scale(df["ttfc"], kind="numeric")
            y_label = "TTFC (s)"

            fig = rk.box(
                df,
                x="prompt_length_bin",
                y="ttfc",
                labels={
                    "prompt_length_bin": "Number of Prompt Tokens",
                    "ttfc": y_label,
                },
            )
            fig.save(os.path.join(output_dir, "ttfc_violin_plot.png"))

            if ttfc_scale != "linear":
                fig_scaled = rk.box(
                    df,
                    x="prompt_length_bin",
                    y="ttfc",
                    labels={
                        "prompt_length_bin": "Number of Prompt Tokens",
                        "ttfc": format_axis_label("TTFC", "s", ttfc_scale),
                    },
                )
                apply_axis_scale(fig_scaled, axis="y", scale=ttfc_scale)
                suffix = "log" if ttfc_scale == "log" else "symlog"
                fig_scaled.save(
                    os.path.join(output_dir, f"ttfc_violin_plot_{suffix}_y.png")
                )

        except Exception as e:
            logger.warning(f"Failed to generate TTFC violin plots: {e}")

    def _log_wandb_metrics(self, output_dir: str) -> None:
        """Log metrics to Weights & Biases."""
        try:
            from typing import Any, cast

            import wandb  # type: ignore[import-not-found]

            wandb = cast(Any, wandb)

            if not getattr(wandb, "run", None):
                return

            # Log summary table
            summary_path = os.path.join(output_dir, "summary_stats.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                numeric_rows = [
                    {"Metric": k, "Value": float(v)}
                    for k, v in summary.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                if numeric_rows:
                    df = pd.DataFrame.from_records(numeric_rows)
                    wandb.log({"summary_stats_table": wandb.Table(dataframe=df)})

            # Log throughput metrics
            throughput_path = os.path.join(output_dir, "throughput_metrics.json")
            if os.path.exists(throughput_path):
                with open(throughput_path, "r") as f:
                    throughput = json.load(f)
                data = {
                    "Metric Type": ["TPOT Based", "TBC Based"],
                    "Throughput (tok/s)": [
                        throughput.get("tpot_based_throughput", 0),
                        throughput.get("tbc_based_throughput", 0),
                    ],
                }
                df = pd.DataFrame(data)
                wandb.log(
                    {
                        "throughput_metrics": wandb.plot.bar(
                            table=wandb.Table(dataframe=df),
                            label="Metric Type",
                            value="Throughput (tok/s)",
                            title="Token Throughput",
                        )
                    }
                )

            # Log TTFC/TBC scalar charts
            self._log_ttfc_tbc_scalar_charts()

            # Log images
            for plot_name in ["ttfc_violin_plot.png"]:
                plot_path = os.path.join(output_dir, plot_name)
                if os.path.exists(plot_path):
                    wandb.log({plot_name.replace(".png", ""): wandb.Image(plot_path)})

        except Exception as e:
            logger.warning(f"Failed to log WandB metrics: {e}")

    def _log_ttfc_tbc_scalar_charts(self) -> None:
        """Log TTFC and TBC scalar charts to WandB."""
        try:
            from typing import Any, cast

            import wandb  # type: ignore[import-not-found]

            wandb = cast(Any, wandb)

            if not getattr(wandb, "run", None):
                return

            def log_for_sketch(sketch_key: str, short_name: str) -> None:
                if sketch_key not in self.summaries:
                    return
                sketch = self.summaries[sketch_key].sketch
                if sketch.count == 0:
                    return
                try:
                    stats = {
                        "Min": sketch._min,
                        "Mean": sketch.avg,
                        "Median": sketch.get_quantile_value(0.5),
                        "P90": sketch.get_quantile_value(0.9),
                        "P99": sketch.get_quantile_value(0.99),
                        "Max": sketch._max,
                    }
                    for stat_name, stat_value in stats.items():
                        df = pd.DataFrame(
                            {"Label": [short_name], "Value": [float(stat_value)]}
                        )
                        wandb.log(
                            {
                                f"{short_name} {stat_name}": wandb.plot.bar(
                                    table=wandb.Table(dataframe=df),
                                    label="Label",
                                    value="Value",
                                    title=f"{short_name} {stat_name}",
                                )
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to compute stats for {short_name}: {e}")

            log_for_sketch("ttfc", "TTFC (s)")
            log_for_sketch("tbc", "TBC (s)")

        except Exception as e:
            logger.warning(f"Failed to log scalar charts: {e}")
