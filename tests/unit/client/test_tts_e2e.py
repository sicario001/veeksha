"""End-to-end: the real TTS pipeline (client + workers + audio evaluator).

Proves a TTS benchmark runs from this branch: TTSClient streams audio from the
mock TTS server through the real dispatch/completion workers into the
AudioPerformanceEvaluator, which produces TTFA/RTF/duration. Needs `transformers`
importable (the veeksha.client package pulls it in); runs in CI, and locally with
PYTHONPATH=analysis/bench/_shim.
"""

from __future__ import annotations

import math
import threading
import time
from queue import Queue

from tests.helpers.mock_tts_server import MockTTSServer
from veeksha.config.client import TTSClientConfig
from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.config.traffic import ConcurrentTrafficConfig
from veeksha.core.context import WorkerContext
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
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
    """Adapt the top-level evaluator interface to the channel evaluator."""

    def __init__(self, audio_eval: AudioPerformanceEvaluator):
        self.audio = audio_eval

    def register_request(
        self, request_id, session_id, dispatched_at, channels, requested_output
    ):
        self.audio.register_request(
            request_id, session_id, dispatched_at, channels, requested_output
        )

    def record_request_completed(
        self, request_id, session_id, completed_at, response, error=None
    ):
        self.audio.record_request_completed(
            request_id, session_id, completed_at, response
        )


def _tts_session(sid: int) -> Session:
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    req = Request(
        id=sid,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="the quick brown fox", target_prompt_tokens=4
            )
        },
    )
    return Session(id=sid, session_graph=graph, requests={0: req})


def test_tts_end_to_end_produces_audio_metrics():
    srv = MockTTSServer(
        chunk_bytes=4800, num_chunks=10, chunk_dt=0.02, prefill_s=0.02
    ).start()
    try:
        cfg = TTSClientConfig(
            api_base=f"http://127.0.0.1:{srv.port}/v1",
            model="m",
            voice_id="v",
            raw_pcm=True,
            sample_rate=24000,
        )
        from veeksha.client.tts import TTSClient

        client = TTSClient(cfg)
        sched = ConcurrentTrafficScheduler(
            ConcurrentTrafficConfig(
                target_concurrent_sessions=4,
                rampup_seconds=0,
                cancel_session_on_failure=False,
            ),
            SeedManager(0),
        )
        sched.reset_reference_time()

        audio_eval = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
        evaluator = _AudioAdapter(audio_eval)
        n = 12
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
            sched.schedule_session(_tts_session(sid))

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
        # 48000 bytes @ 24kHz mono 16-bit = 1.0s of audio
        assert math.isclose(m["Generated Audio Duration (Mean)"], 1.0, rel_tol=1e-3)
        assert m["Time to First Audio (Mean)"] > 0
        assert "Real Time Factor (Mean)" in m
    finally:
        srv.stop()
