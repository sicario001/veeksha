"""Base performance evaluator that delegates to channel-specific evaluators."""

import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Optional

from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.core.seeding import SeedManager
from veeksha.evaluator.base import BaseEvaluator, EvaluationResult
from veeksha.logger import init_logger
from veeksha.slo.runner import evaluate_and_save_slos
from veeksha.types import ChannelModality

logger = init_logger(__name__)


@dataclass
class SessionAggregate:
    """Aggregate metrics for a single session."""

    session_id: int
    session_total_requests: int
    requests_observed: int = 0
    completed_requests: int = 0
    errored_requests: int = 0
    cancelled_requests: int = 0
    first_dispatch_at: Optional[float] = None
    last_completion_at: Optional[float] = None
    last_completion_time: Optional[float] = None


class PerformanceEvaluator(BaseEvaluator):
    """Performance evaluator that measures latency, throughput, and deadlines.

    This is a composite evaluator that delegates to channel-specific evaluators
    for modality-specific metrics (e.g., TTFC/TBC for text, audio latency, etc.).
    It also tracks aggregate session-level metrics.
    """

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        seed_manager: Optional[SeedManager] = None,
        output_dir: Optional[str] = None,
        benchmark_start_time: float = 0.0,
    ):
        super().__init__(config, seed_manager)
        self.config: PerformanceEvaluatorConfig = config
        self.output_dir = output_dir
        self.benchmark_start_time = benchmark_start_time

        self._channel_evaluators: Dict[ChannelModality, Any] = {}

        self.lock = threading.Lock()

        # All evaluators share a bunch of common metrics

        # aggregate request tracking
        self.num_requests: int = 0
        self.num_errored_requests: int = 0
        self.num_completed_requests: int = 0
        self.num_cancelled_requests: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.error_code_freq: DefaultDict[int, int] = defaultdict(int)
        self._registered_request_ids: set = set()
        self._included_request_ids: Optional[set] = None  # None means include all

        # session tracking
        self.session_stats: Dict[int, SessionAggregate] = {}
        self.num_sessions_seen: int = 0
        self.num_sessions_successful: int = 0
        self.num_sessions_cancelled: int = 0
        self.num_sessions_errored: int = 0
        self.num_sessions_incomplete: int = 0
        self._first_session_start_time: Optional[float] = None
        self._last_session_start_time: Optional[float] = None

        # streaming support
        self._stream_trigger = threading.Event()
        self._stream_stop_event = threading.Event()
        self._stream_has_updates = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._request_time_reference: float = time.monotonic()

        if config.stream_metrics and self.output_dir:
            self._start_metric_streamer()

        from veeksha.evaluator.registry import ChannelPerformanceEvaluatorRegistry

        for channel in self.config.target_channels:
            channel_config = self.config.get_channel_config(channel)
            if channel_config:
                self._channel_evaluators[channel] = (
                    ChannelPerformanceEvaluatorRegistry.get(
                        channel,
                        config=self.config,
                        channel_config=channel_config,
                        benchmark_start_time=self.benchmark_start_time,
                    )
                )

    def _get_channel_evaluator(self, channel: ChannelModality) -> Any:
        """Get a channel-specific evaluator."""
        return self._channel_evaluators[channel]

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return (
            self.num_errored_requests / self.num_requests
            if self.num_requests > 0
            else 0.0
        )

    def _session_dispatch_rate(self) -> float:
        """Calculate observed session dispatch rate."""
        if (
            self.num_sessions_seen <= 1
            or self._first_session_start_time is None
            or self._last_session_start_time is None
        ):
            return 0.0
        span = max(0.0, self._last_session_start_time - self._first_session_start_time)
        if span == 0:
            return 0.0
        return (self.num_sessions_seen - 1) / span

    def _normalize_request_time(self, dispatched_at: float) -> float:
        """Normalize request time relative to start."""
        if dispatched_at >= self._request_time_reference:
            return max(0.0, dispatched_at - self._request_time_reference)
        return dispatched_at

    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        channels: Dict[ChannelModality, Any],
        requested_output: Any = None,
    ) -> None:
        """Register a request that was dispatched."""
        with self.lock:
            self.num_requests += 1
            self._registered_request_ids.add(request_id)
            if self.start_time is None:
                self.start_time = dispatched_at

            # register with channel evaluators
            for channel, content in channels.items():
                if self.should_evaluate_channel(channel):
                    evaluator = self._get_channel_evaluator(channel)
                    evaluator.register_request(
                        request_id=request_id,
                        session_id=session_id,
                        dispatched_at=dispatched_at,
                        content=content,
                        requested_output=requested_output,
                    )

    def get_registered_request_ids(self) -> set:
        """Return set of all request IDs that have been registered."""
        with self.lock:
            return self._registered_request_ids.copy()

    def get_completed_request_count(self) -> int:
        """Return count of completed requests for progress tracking."""
        with self.lock:
            return self.num_completed_requests

    def get_session_counts(self) -> tuple[int, int, int]:
        """Return session counts for progress tracking.

        Returns:
            Tuple of (completed_sessions, errored_sessions, in_progress_sessions)
        """
        with self.lock:
            completed = self.num_sessions_successful
            errored = self.num_sessions_errored + self.num_sessions_cancelled
            in_progress = len(self.session_stats)
            return completed, errored, in_progress

    def set_included_requests(self, request_ids: set) -> None:
        """Set which requests to include in final metrics.

        Args:
            request_ids: Set of request IDs to include. Only these will be
                        counted in the final summary metrics.
        """
        with self.lock:
            self._included_request_ids = request_ids.copy()
            logger.info(f"Set included requests filter: {len(request_ids)} requests")

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
        error: Optional[Exception] = None,
    ) -> None:
        """Record that a request completed.

        Only the cheap aggregate bookkeeping (session tracking, counters) is taken
        under ``self.lock``. The expensive channel-evaluator delegation runs
        *outside* the lock — each channel evaluator is internally thread-safe (its
        hot path is lock-free via sharded sketches) — so completion-worker threads
        process results in parallel instead of serializing on one lock.
        """
        with self.lock:
            self.end_time = completed_at

            # Update session tracking
            self._update_session_metrics_for_request(
                session_id=session_id,
                request_id=request_id,
                dispatched_at=getattr(response, "request_dispatched_at", completed_at),
                completed_at=completed_at,
                error=error,
                cancelled=getattr(response, "cancelled", False),
                session_total_requests=getattr(response, "session_total_requests", 1),
            )

            if getattr(response, "cancelled", False):
                self.num_cancelled_requests += 1
                return

            if error is not None or getattr(response, "error_code", None) is not None:
                error_code = getattr(response, "error_code", None)
                if error_code is not None:
                    self.error_code_freq[error_code] += 1
                self.num_errored_requests += 1
                return

            self.num_completed_requests += 1

        # Delegate to channel evaluators outside the lock (they self-synchronize).
        for channel, evaluator in self._channel_evaluators.items():
            evaluator.record_request_completed(
                request_id=request_id,
                session_id=session_id,
                completed_at=completed_at,
                response=response,
            )

        if self.config.stream_metrics and self._stream_thread:
            self._stream_has_updates.set()

    def _update_session_metrics_for_request(
        self,
        session_id: int,
        request_id: int,
        dispatched_at: float,
        completed_at: float,
        error: Optional[Exception],
        cancelled: bool,
        session_total_requests: int,
    ) -> None:
        """Update session-level metrics for a request.

        The evaluator observes request outcomes but does not decide when to cancel
        sessions - that is the traffic scheduler's responsibility. Sessions are
        finalized when all expected requests have been observed.
        """
        state = self.session_stats.get(session_id)
        if state is None:
            state = SessionAggregate(
                session_id=session_id,
                session_total_requests=session_total_requests,
            )
            self.session_stats[session_id] = state
            self._record_session_start(state, dispatched_at)
        else:
            state.session_total_requests = max(
                session_total_requests, state.session_total_requests
            )

        state.requests_observed += 1
        state.last_completion_time = completed_at
        state.last_completion_at = completed_at

        if cancelled:
            state.cancelled_requests += 1
        elif error is not None:
            state.errored_requests += 1
        else:
            state.completed_requests += 1

        if state.requests_observed >= state.session_total_requests:
            if state.cancelled_requests:
                termination = "cancelled"
            elif state.errored_requests:
                termination = "errored"
            else:
                termination = "success"
            self._finalize_session(session_id, termination)

    def _record_session_start(
        self, state: SessionAggregate, dispatch_time: float
    ) -> None:
        """Record session start metrics."""
        state.first_dispatch_at = dispatch_time
        if self._first_session_start_time is None:
            self._first_session_start_time = dispatch_time
        self._last_session_start_time = dispatch_time
        self.num_sessions_seen += 1

    def _finalize_session(self, session_id: int, termination: str) -> None:
        """Finalize session and update aggregate counters."""
        state = self.session_stats.pop(session_id, None)
        if state is None:
            return

        # Notify channel evaluators of session completion
        for evaluator in self._channel_evaluators.values():
            evaluator.record_session_completed(
                session_id=session_id,
                session_size=state.requests_observed,
                first_dispatch_at=state.first_dispatch_at,
                last_completion_at=state.last_completion_at,
            )

        if termination == "success":
            self.num_sessions_successful += 1
        elif termination == "errored":
            self.num_sessions_errored += 1
        elif termination == "cancelled":
            self.num_sessions_cancelled += 1
        elif termination == "incomplete":
            self.num_sessions_incomplete += 1

    def _finalize_remaining_sessions(self) -> None:
        """Finalize any sessions still in progress at end of benchmark."""
        for session_id in list(self.session_stats.keys()):
            state = self.session_stats.get(session_id)
            if state is None:
                continue
            if state.cancelled_requests:
                termination = "cancelled"
            elif state.errored_requests:
                termination = "errored"
            elif state.requests_observed >= state.session_total_requests:
                termination = "success"
            else:
                termination = "incomplete"
            self._finalize_session(session_id, termination)

    def record_session_completed(
        self,
        session_id: int,
        completed_at: float,
        success: bool,
    ) -> None:
        """Record that a session completed (explicit call)."""
        with self.lock:
            termination = "success" if success else "errored"
            if session_id in self.session_stats:
                self._finalize_session(session_id, termination)

    def get_aggregated_summary(self) -> Dict[str, float]:
        """Get aggregate summary metrics."""
        return {
            "Number of Requests": self.num_requests,
            "Number of Errored Requests": self.num_errored_requests,
            "Number of Completed Requests": self.num_completed_requests,
            "Number of Cancelled Requests": self.num_cancelled_requests,
            "Error Rate": self.error_rate,
            "Cancellation Rate": (
                self.num_cancelled_requests / self.num_requests
                if self.num_requests > 0
                else 0.0
            ),
            "Number of Sessions Seen": float(self.num_sessions_seen),
            "Successful Sessions": float(self.num_sessions_successful),
            "Errored Sessions": float(self.num_sessions_errored),
            "Cancelled Sessions": float(self.num_sessions_cancelled),
            "Incomplete Sessions": float(self.num_sessions_incomplete),
            "Observed Session Dispatch Rate": self._session_dispatch_rate(),
        }

    def _build_summary_stats(self) -> Dict[str, Any]:
        """Combine aggregate stats with error code frequencies."""
        summary_stats: Dict[str, Any] = {
            **self.get_aggregated_summary(),
            "error_code_freq": dict(self.error_code_freq),
        }
        return summary_stats

    def finalize(self) -> EvaluationResult:
        """Finalize evaluation and return results."""
        if self.config.stream_metrics:
            self._shutdown_metric_streamer()
        self._finalize_remaining_sessions()

        # Collect metrics from all channel evaluators
        combined_metrics = self.get_aggregated_summary()
        channel_metrics = {}

        for channel, evaluator in self._channel_evaluators.items():
            channel_result = evaluator.finalize()
            channel_metrics[str(channel)] = channel_result.metrics
            combined_metrics.update(channel_result.metrics)

        return EvaluationResult(
            evaluator_type="performance",
            channel=None,
            metrics=combined_metrics,
            raw_data={"channel_metrics": channel_metrics},
        )

    def save(self, output_dir: str) -> None:
        """Save evaluation artifacts to the output directory."""
        os.makedirs(output_dir, exist_ok=True)

        # Save aggregate summary
        summary = self._build_summary_stats()
        summary_path = os.path.join(output_dir, "summary_stats.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Delegate to channel evaluators
        for channel, evaluator in self._channel_evaluators.items():
            evaluator.save(output_dir)

        # request-level metrics are persisted now
        evaluate_and_save_slos(slo_configs=self.config.slos, metrics_dir=output_dir)

    def get_streaming_metrics(self) -> Optional[Dict[str, Any]]:
        """Return current metrics for streaming updates."""
        with self.lock:
            metrics = self.get_aggregated_summary()
            for channel, evaluator in self._channel_evaluators.items():
                channel_metrics = evaluator.get_streaming_metrics()
                if channel_metrics:
                    metrics.update(channel_metrics)
            return metrics

    # -------------------------------------------------------------------------
    # Streaming support
    # -------------------------------------------------------------------------

    def _start_metric_streamer(self) -> None:
        """Start background thread for metric streaming."""
        if not self.output_dir:
            logger.warning(
                "stream_metrics enabled but output_dir not set; disabling streaming."
            )
            return

        os.makedirs(self.output_dir, exist_ok=True)

        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="performance-evaluator-streamer",
            daemon=True,
        )
        self._stream_thread.start()
        logger.info(
            "Metric streaming enabled; flushing every %.1fs",
            self.config.stream_metrics_interval,
        )

    def _stream_loop(self) -> None:
        """Background loop for streaming metrics."""
        while True:
            triggered = self._stream_trigger.wait(
                timeout=self.config.stream_metrics_interval
            )
            if triggered:
                self._stream_trigger.clear()
            if triggered or self._stream_has_updates.is_set():
                try:
                    self._flush_streaming_outputs()
                    self._stream_has_updates.clear()
                except Exception as exc:
                    logger.warning(f"Streaming metrics flush failed: {exc}")
            if self._stream_stop_event.is_set():
                try:
                    self._flush_streaming_outputs()
                except Exception as exc:
                    logger.warning(f"Final streaming flush failed: {exc}")
                break

    def _flush_streaming_outputs(self) -> None:
        """Flush current metrics to output files."""
        if not self.output_dir:
            return

        with self.lock:
            # Write current summary
            summary = self._build_summary_stats()
            summary_path = os.path.join(self.output_dir, "summary_stats.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            # Delegate to channel evaluators
            for evaluator in self._channel_evaluators.values():
                if hasattr(evaluator, "flush_streaming_outputs"):
                    evaluator.flush_streaming_outputs(self.output_dir)

    def _shutdown_metric_streamer(self) -> None:
        """Shutdown the metric streaming thread."""
        if not self._stream_thread:
            return
        self._stream_trigger.set()
        self._stream_stop_event.set()
        self._stream_thread.join()
        self._stream_thread = None
