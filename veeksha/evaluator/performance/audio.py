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
from veeksha.types import AudioTask, ChannelModality


class AudioPerformanceEvaluator(BaseChannelPerformanceEvaluator):
    """Timing-metric evaluator for the audio output channel (e.g. TTS)."""

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: Optional[AudioChannelPerformanceConfig] = None,
        benchmark_start_time: float = 0.0,
    ):
        self.config = config
        self.channel_config = channel_config or AudioChannelPerformanceConfig()
        self.benchmark_start_time = benchmark_start_time
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
            "streaming_rtf": ShardedCDFSketch("Streaming Real Time Factor"),
            "session_size": ShardedCDFSketch("Requests per Session"),
        }
        # STT (ASR) distributions: per-request WER + interactivity + first-text
        # latency, alongside the corpus-level WER aggregates in _asr_accumulator.
        self.asr_summaries: Dict[str, ShardedCDFSketch] = {
            "final_wer": ShardedCDFSketch("Final WER", unit="%"),
            "partial_wer": ShardedCDFSketch("Partial WER", unit="%"),
            "interactivity": ShardedCDFSketch("Word Interactivity Latency", unit="ms"),
            "time_to_first_visible_text": ShardedCDFSketch(
                "Time to First Visible Text", unit="ms"
            ),
            "time_to_final_transcript": ShardedCDFSketch(
                "Time to Final Transcript", unit="ms"
            ),
        }
        # Lazily created on the first STT request so the TTS-only path never
        # imports jiwer / the normalizer (heavier deps).
        self._asr_accumulator: Optional[Any] = None
        self._asr_rows: List[Dict[str, Any]] = []
        self._num_asr_scored = 0

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

        # STT (speech-to-text) responses carry transcripts + ground truth; score
        # WER + interactivity. Everything else is a timing-only (TTS) response.
        if metrics.get(ac.AUDIO_TASK) == AudioTask.STT and (
            metrics.get("final_transcript") is not None
            and metrics.get("expected_transcript") is not None
        ):
            self._score_stt_request(request_id, metrics)
            with self._lock:
                self._num_completed += 1
            return

        derived = self._compute_metrics(channel, metrics)  # all in seconds
        if derived is None:
            return

        # lock-free sketch puts (sharded)
        for key in (
            "ttfa",
            "end_to_end_latency",
            "generated_audio_duration",
            "rtf",
            "streaming_rtf",
        ):
            value = derived.get(key)
            if value is not None:
                self.summaries[key].put(value)
        with self._lock:
            self._num_completed += 1

    # ---- STT scoring --------------------------------------------------------
    def _score_stt_request(self, request_id: int, metrics: Dict[str, Any]) -> None:
        """WER + interactivity for one realtime STT response (lock-free sketches).

        jiwer / the normalizer are imported here so the TTS-only path never pays
        for them. Corpus-level WER accumulates in the thread-safe accumulator;
        per-request WER/interactivity/latency feed the sharded distributions.
        """
        from veeksha.evaluator.performance.asr import (
            ASRMetricAccumulator,
            score_asr_request,
        )

        if self._asr_accumulator is None:
            with self._lock:
                if self._asr_accumulator is None:
                    self._asr_accumulator = ASRMetricAccumulator()

        # STT duration = the streamed input clip (bytes / sample_rate), falling
        # back to the client-reported input_audio_duration_ms.
        sample_rate = int(metrics.get(ac.SAMPLE_RATE, ac.DEFAULT_AUDIO_SAMPLE_RATE))
        byte_count = metrics.get(ac.PCM_BYTE_COUNT)
        if byte_count:
            duration_s = ac.pcm_bytes_to_duration_s(int(byte_count), sample_rate)
        else:
            duration_s = float(metrics.get("input_audio_duration_ms", 0.0)) / 1000.0

        scored = score_asr_request(
            request_id=request_id,
            channel_metrics=metrics,
            duration_s=duration_s,
            accumulator=self._asr_accumulator,
        )

        # per-request distributions (lock-free sharded puts)
        if scored.final_wer is not None:
            self.asr_summaries["final_wer"].put(scored.final_wer)
        if scored.partial_wer is not None:
            self.asr_summaries["partial_wer"].put(scored.partial_wer)
        if scored.interactivity is not None:
            self.asr_summaries["interactivity"].put(scored.interactivity)
        if scored.time_to_first_visible_text is not None:
            self.asr_summaries["time_to_first_visible_text"].put(
                scored.time_to_first_visible_text
            )
        if scored.time_to_final_transcript is not None:
            self.asr_summaries["time_to_final_transcript"].put(
                scored.time_to_final_transcript
            )

        with self._lock:
            self._asr_rows.append(scored.to_request_row())
            self._num_asr_scored += 1

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
        # STT: per-request WER/interactivity distributions + corpus WER aggregates.
        for sketch in self.asr_summaries.values():
            if len(sketch) > 0:
                summary.update(sketch.get_summary())
        if self._asr_accumulator is not None:
            summary["num_asr_scored_requests"] = self._num_asr_scored
            summary.update(self._asr_accumulator.get_summary())
        return EvaluationResult(
            evaluator_type="audio_performance",
            channel=ChannelModality.AUDIO,
            metrics=summary,
        )

    # ---- helpers ------------------------------------------------------------
    def _compute_metrics(
        self, channel: Any, metrics: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """Return {ttfa, end_to_end_latency, generated_audio_duration, rtf} in seconds.

        Two dialects: a realtime per-chunk timeline (offsets in ms), or the HTTP
        aggregate (TTFC + end_to_end_latency in ms + total audio bytes).
        """
        sample_rate = int(metrics.get(ac.SAMPLE_RATE, ac.DEFAULT_AUDIO_SAMPLE_RATE))

        def bytes_to_seconds(n_bytes: int) -> float:
            return ac.pcm_bytes_to_duration_s(n_bytes, sample_rate)

        # Realtime dialect: a per-chunk timeline (key present, even if empty).
        if ac.AUDIO_CHUNK_TIMESTAMPS in metrics:
            timeline = metrics.get(ac.AUDIO_CHUNK_TIMESTAMPS) or []
            events: List[StreamEvent] = [
                StreamEvent(offset_s=float(offset_ms) / 1000.0, size=int(n_bytes))
                for offset_ms, n_bytes in timeline
            ]
            stream = TimedEventStream(
                ChannelModality.AUDIO, events, unit_duration_fn=bytes_to_seconds
            )
            if len(stream) == 0:
                return None
            return {
                "ttfa": stream.time_to_first_event(),
                "end_to_end_latency": stream.end_to_end(),
                "generated_audio_duration": stream.produced_content_duration_s(),
                "rtf": stream.real_time_factor(),
                "streaming_rtf": stream.streaming_real_time_factor(),
            }

        # Aggregate (HTTP dialect): TTFC + end_to_end_latency in ms + audio bytes.
        e2e_ms = metrics.get(ac.END_TO_END_LATENCY)
        byte_count = metrics.get(ac.PCM_BYTE_COUNT)
        content = getattr(channel, "content", None)
        if byte_count is None and isinstance(content, (bytes, bytearray)):
            byte_count = len(content)
            if not metrics.get(ac.AudioMetricKey.RAW_PCM, True):  # WAV -> drop header
                byte_count = max(0, byte_count - ac.WAV_HEADER_BYTES)
        if e2e_ms is None or not byte_count:
            return None
        duration = bytes_to_seconds(int(byte_count))
        e2e_s = float(e2e_ms) / 1000.0
        ttfc_ms = metrics.get(ac.AudioMetricKey.TTFC)
        return {
            "ttfa": (float(ttfc_ms) / 1000.0) if ttfc_ms is not None else None,
            "end_to_end_latency": e2e_s,
            "generated_audio_duration": duration,
            "rtf": (e2e_s / duration) if duration else None,
        }
