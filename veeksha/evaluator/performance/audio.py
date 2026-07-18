"""Audio-channel performance evaluator.

Computes the audio timing metrics (TTFA, RTF, generated-audio-duration, E2E)
from the SAME primitives as text: a ``TimedEventStream`` over the delivered audio
chunks, aggregated with lock-free ``ShardedCDFSketch``, behind the shared channel
ABC. This is the modality-unification payoff of doc analysis/05 — audio and text
share one timing primitive and one accumulation path.

It reads the audio chunk timeline from ``ChannelResponse.metrics`` using the
shared ``core.audio_contract`` keys. WER/interactivity scoring from the voice
branches is out of scope here (needs extra deps); this covers the timing family.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.core import audio_contract as ac
from veeksha.core.timed_event_stream import StreamEvent, TimedEventStream
from veeksha.evaluator.base import EvaluationResult
from veeksha.evaluator.performance.channel_base import BaseChannelPerformanceEvaluator
from veeksha.evaluator.sharded_sketch import ShardedCDFSketch
from veeksha.types import ChannelModality


class AudioPerformanceEvaluator(BaseChannelPerformanceEvaluator):
    """Timing-metric evaluator for the audio output channel (e.g. TTS)."""

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: Optional[AudioChannelPerformanceConfig] = None,
    ):
        self.config = config
        self.channel_config = channel_config or AudioChannelPerformanceConfig()
        self._lock = threading.Lock()
        self._pending: Dict[int, float] = {}
        self._num_completed = 0
        self.summaries: Dict[str, ShardedCDFSketch] = {
            "ttfa": ShardedCDFSketch("Time to First Audio", unit="s"),
            "rtf": ShardedCDFSketch("Real Time Factor"),
            "generated_audio_duration": ShardedCDFSketch(
                "Generated Audio Duration", unit="s"
            ),
            "end_to_end_latency": ShardedCDFSketch("End to End Latency", unit="s"),
            "session_size": ShardedCDFSketch("Requests per Session"),
        }

    # ---- lifecycle ----------------------------------------------------------
    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        content: Any,
        requested_output: Any = None,
    ) -> None:
        with self._lock:
            self._pending[request_id] = dispatched_at

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
    ) -> None:
        with self._lock:
            self._pending.pop(request_id, None)

        channel = response.channels.get(ChannelModality.AUDIO)
        if channel is None:
            return
        metrics = channel.metrics or {}
        stream = self._build_stream(metrics)
        if stream is None or len(stream) == 0:
            return

        # lock-free sketch puts (sharded)
        ttfa = stream.time_to_first_event()
        e2e = stream.end_to_end()
        duration = stream.produced_content_duration_s()
        rtf = stream.real_time_factor()
        if ttfa is not None:
            self.summaries["ttfa"].put(ttfa)
        if e2e is not None:
            self.summaries["end_to_end_latency"].put(e2e)
        if duration is not None:
            self.summaries["generated_audio_duration"].put(duration)
        if rtf is not None:
            self.summaries["rtf"].put(rtf)
        with self._lock:
            self._num_completed += 1

    def record_session_completed(
        self,
        session_id: int,
        session_size: int,
        first_dispatch_at: Optional[float],
        last_completion_at: Optional[float],
    ) -> None:
        self.summaries["session_size"].put(session_size)

    def finalize(self) -> EvaluationResult:
        summary: Dict[str, Any] = {"num_completed_requests": self._num_completed}
        for sketch in self.summaries.values():
            if len(sketch) > 0:
                summary.update(sketch.get_summary())
        return EvaluationResult(
            evaluator_type="audio_performance",
            channel=ChannelModality.AUDIO,
            metrics=summary,
        )

    # ---- helpers ------------------------------------------------------------
    def _build_stream(self, metrics: Dict[str, Any]) -> Optional[TimedEventStream]:
        sample_rate = int(metrics.get(ac.SAMPLE_RATE, ac.DEFAULT_AUDIO_SAMPLE_RATE))

        def bytes_to_seconds(n_bytes: int) -> float:
            return ac.pcm_bytes_to_duration_s(n_bytes, sample_rate)

        timeline = metrics.get(ac.AUDIO_CHUNK_TIMESTAMPS)
        if timeline:
            events: List[StreamEvent] = [
                StreamEvent(offset_s=float(offset_ms) / 1000.0, size=int(n_bytes))
                for offset_ms, n_bytes in timeline
            ]
            return TimedEventStream(
                ChannelModality.AUDIO, events, unit_duration_fn=bytes_to_seconds
            )

        # Aggregate fallback (HTTP-streaming dialect: total bytes + e2e, no timeline)
        pcm_bytes = metrics.get(ac.PCM_BYTE_COUNT)
        e2e = metrics.get(ac.END_TO_END_LATENCY)
        if pcm_bytes and e2e is not None:
            return TimedEventStream(
                ChannelModality.AUDIO,
                [StreamEvent(offset_s=float(e2e), size=int(pcm_bytes))],
                unit_duration_fn=bytes_to_seconds,
            )
        return None
