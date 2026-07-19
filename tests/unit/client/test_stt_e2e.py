"""End-to-end: the real ASR/STT pipeline (WS client + workers + audio evaluator).

Proves an ASR benchmark runs from this branch: STTClient streams a WAV clip to
the mock realtime STT server, receives the transcript, and the audio metrics
(RTF from processing-latency / input-audio-duration) land in the
AudioPerformanceEvaluator. Needs `transformers` importable (client package);
runs in CI, and locally with PYTHONPATH=analysis/bench/_shim.
"""

from __future__ import annotations

import math
import tempfile
import threading
import time
import wave
from pathlib import Path
from queue import Queue

from tests.helpers.mock_stt_server import MockSTTServer
from veeksha.config.client import STTClientConfig
from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.config.traffic import ConcurrentTrafficConfig
from veeksha.core.context import WorkerContext
from veeksha.core.request import Request
from veeksha.core.request_content import AudioChannelRequestContent
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import SessionGraph, SessionNode, add_node
from veeksha.evaluator.performance.audio import AudioPerformanceEvaluator
from veeksha.traffic.concurrent import ConcurrentTrafficScheduler
from veeksha.types import ChannelModality
from veeksha.workers.client_runner import ClientRunnerManager
from veeksha.workers.completion import CompletionWorker
from veeksha.workers.dispatch import DispatchWorker


class _AudioAdapter:
    def __init__(self, audio_eval: AudioPerformanceEvaluator):
        self.audio = audio_eval
        self.transcripts = []

    def register_request(
        self, request_id, session_id, dispatched_at, channels, requested_output
    ):
        self.audio.register_request(
            request_id, session_id, dispatched_at, channels, requested_output
        )

    def record_request_completed(
        self, request_id, session_id, completed_at, response, error=None
    ):
        ch = response.channels.get(ChannelModality.AUDIO)
        if ch is not None:
            self.transcripts.append(ch.content)
        self.audio.record_request_completed(
            request_id, session_id, completed_at, response
        )


def _make_wav(path: str, seconds: float = 1.0, sr: int = 16000) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * seconds))


def _stt_session(sid: int, wav_path: str, expected_transcript: str = "") -> Session:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    metadata = {}
    if expected_transcript:
        metadata = {"expected_transcript": expected_transcript, "dataset": "e2e"}
    req = Request(
        id=sid,
        channels={
            ChannelModality.AUDIO: AudioChannelRequestContent(input_audio=wav_path)
        },
        metadata=metadata,
    )
    return Session(id=sid, session_graph=graph, requests={0: req})


def test_stt_end_to_end_produces_transcript_and_metrics(tmp_path: Path):
    wav = str(tmp_path / "clip.wav")
    _make_wav(wav, seconds=1.0, sr=16000)  # 1s -> 32000 PCM bytes

    srv = MockSTTServer(
        transcript="the quick brown fox", first_delta_delay=0.04, delta_dt=0.02
    ).start()
    try:
        from veeksha.client.stt import STTClient

        cfg = STTClientConfig(
            api_base=f"http://127.0.0.1:{srv.port}",
            model="m",
            provider="vllm_realtime",
            sample_rate=16000,
            ws_realtime_pacing=False,
        )
        client = STTClient(cfg)
        sched = ConcurrentTrafficScheduler(
            ConcurrentTrafficConfig(
                target_concurrent_sessions=3,
                rampup_seconds=0,
                cancel_session_on_failure=False,
            ),
            SeedManager(0),
        )
        sched.reset_reference_time()

        audio_eval = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
        evaluator = _AudioAdapter(audio_eval)
        n = 8
        client_queues = [Queue() for _ in range(2)]
        output_queue: Queue = Queue()
        stop_event = threading.Event()

        runner = ClientRunnerManager(
            client=client,
            input_queues=client_queues,
            output_queue=output_queue,
            stop_event=stop_event,
            traffic_scheduler=sched,
        )
        workers = [
            DispatchWorker(
                sched, client_queues, evaluator, WorkerContext(0, stop_event)
            ),
            CompletionWorker(
                output_queue, sched, evaluator, WorkerContext(0, stop_event)
            ),
        ]
        threads = [threading.Thread(target=w.run, daemon=True) for w in workers]
        runner.start()
        for t in threads:
            t.start()

        for sid in range(n):
            sched.schedule_session(_stt_session(sid, wav))

        deadline = time.monotonic() + 30.0
        while audio_eval._num_completed < n and time.monotonic() < deadline:
            time.sleep(0.02)

        stop_event.set()
        runner.stop()
        output_queue.put(None)
        for t in threads:
            t.join(timeout=2.0)
        runner.wait(timeout=3.0)

        m = audio_eval.finalize().metrics
        assert m["num_completed_requests"] == n
        # input clip is 1.0s -> generated_audio_duration (input duration for STT) ~= 1.0s
        assert math.isclose(m["Generated Audio Duration (Mean)"], 1.0, rel_tol=1e-2)
        assert "Real Time Factor (Mean)" in m
        assert m["Time to First Audio (Mean)"] > 0
        # transcript came back correctly
        assert all(t == "the quick brown fox" for t in evaluator.transcripts)
    finally:
        srv.stop()


def test_stt_end_to_end_scores_wer_when_ground_truth_present(tmp_path: Path):
    """With expected_transcript on the request, the evaluator scores WER + interactivity.

    This is the I10 acceptance path: ground-truth-carrying STT requests flow
    through the real pipeline and produce corpus/sample WER aggregates.
    """
    wav = str(tmp_path / "clip.wav")
    _make_wav(wav, seconds=1.0, sr=16000)

    srv = MockSTTServer(
        transcript="the quick brown fox", first_delta_delay=0.04, delta_dt=0.02
    ).start()
    try:
        from veeksha.client.stt import STTClient

        cfg = STTClientConfig(
            api_base=f"http://127.0.0.1:{srv.port}",
            model="m",
            provider="vllm_realtime",
            sample_rate=16000,
            ws_realtime_pacing=False,
        )
        client = STTClient(cfg)
        sched = ConcurrentTrafficScheduler(
            ConcurrentTrafficConfig(
                target_concurrent_sessions=3,
                rampup_seconds=0,
                cancel_session_on_failure=False,
            ),
            SeedManager(0),
        )
        sched.reset_reference_time()

        audio_eval = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
        evaluator = _AudioAdapter(audio_eval)
        n = 6
        client_queues = [Queue() for _ in range(2)]
        output_queue: Queue = Queue()
        stop_event = threading.Event()

        runner = ClientRunnerManager(
            client=client,
            input_queues=client_queues,
            output_queue=output_queue,
            stop_event=stop_event,
            traffic_scheduler=sched,
        )
        workers = [
            DispatchWorker(
                sched, client_queues, evaluator, WorkerContext(0, stop_event)
            ),
            CompletionWorker(
                output_queue, sched, evaluator, WorkerContext(0, stop_event)
            ),
        ]
        threads = [threading.Thread(target=w.run, daemon=True) for w in workers]
        runner.start()
        for t in threads:
            t.start()

        # half exact (0% WER), half one-word-off (25% WER) => corpus WER 12.5%
        for sid in range(n):
            expected = "the quick brown fox" if sid % 2 == 0 else "the quick brown cat"
            sched.schedule_session(_stt_session(sid, wav, expected_transcript=expected))

        deadline = time.monotonic() + 30.0
        while audio_eval._num_completed < n and time.monotonic() < deadline:
            time.sleep(0.02)

        stop_event.set()
        runner.stop()
        output_queue.put(None)
        for t in threads:
            t.join(timeout=2.0)
        runner.wait(timeout=3.0)

        m = audio_eval.finalize().metrics
        assert m["num_asr_scored_requests"] == n
        assert m["asr_final_sample_count"] == float(n)
        # 3 perfect (0%) + 3 at 25% => sample-mean WER = 12.5%
        assert math.isclose(m["asr_final_sample_mean_wer"], 12.5, rel_tol=1e-6)
    finally:
        srv.stop()
