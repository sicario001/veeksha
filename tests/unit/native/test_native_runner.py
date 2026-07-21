"""Native benchmark execution path + config selection.

run_native_benchmark drives a request batch through the native transport into the
top-level PerformanceEvaluator (the same evaluator a Python run uses) and returns
the finalized EvaluationResult — for each modality.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from veeksha.native.engine import native_available
from veeksha.native.runner import should_use_native

pytestmark = pytest.mark.skipif(
    not native_available(), reason="veeksha_native extension not built"
)


class _Cfg:
    def __init__(self, use_native, api_base):
        self.use_native_transport = use_native
        self.api_base = api_base


def test_should_use_native_respects_flag_and_scheme():
    assert should_use_native(_Cfg(True, "http://127.0.0.1:8000"))
    assert should_use_native(_Cfg(True, "ws://127.0.0.1:8000"))
    # flag off -> Python
    assert not should_use_native(_Cfg(False, "http://127.0.0.1:8000"))
    # TLS -> Python fallback
    assert not should_use_native(_Cfg(True, "https://api.example.com"))
    assert not should_use_native(_Cfg(True, "wss://api.example.com"))


def test_resolve_native_threads_auto_and_override():
    from veeksha.native.runner import (
        _AUTO_SHARD_CONCURRENCY,
        _resolve_native_threads,
    )

    # 0 = auto: single loop below the batching threshold, 2 shards at/above it.
    assert _resolve_native_threads(0, 10) == 1
    assert _resolve_native_threads(0, _AUTO_SHARD_CONCURRENCY - 1) == 1
    assert _resolve_native_threads(0, _AUTO_SHARD_CONCURRENCY) == 2
    assert _resolve_native_threads(0, 1600) == 2
    # explicit value always wins over auto.
    assert _resolve_native_threads(1, 1600) == 1
    assert _resolve_native_threads(4, 10) == 4


def _text_request(rid, tokens=10):
    from veeksha.core.request import Request
    from veeksha.core.request_content import TextChannelRequestContent
    from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
    from veeksha.types import ChannelModality

    return Request(
        id=rid,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(input_text="hi there")
        },
        requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=tokens)),
    )


def test_run_native_benchmark_text_produces_evaluation_result():
    from veeksha.config.client import OpenAIChatCompletionsClientConfig
    from veeksha.config.evaluator import PerformanceEvaluatorConfig
    from veeksha.evaluator.performance.base import PerformanceEvaluator
    from veeksha.native.runner import run_native_benchmark
    from veeksha.preflight.mock_engine import MockStreamingEngine

    engine = MockStreamingEngine(
        chunk_dt=0.02, prefill_s=0.03, default_chunks=10, num_loops=4
    ).start()
    try:
        client_config = OpenAIChatCompletionsClientConfig(
            api_base=f"http://127.0.0.1:{engine.port}/v1",
            model="d",
            use_native_transport=True,
        )
        assert should_use_native(client_config)
        evaluator = PerformanceEvaluator(
            PerformanceEvaluatorConfig(target_channels=["text"], stream_metrics=False)
        )
        result = run_native_benchmark(
            [_text_request(i) for i in range(12)],
            evaluator,
            client_config,
            concurrency=6,
            timeout_s=30.0,
        )
    finally:
        engine.stop()

    # a real EvaluationResult with TTFT surfaced from the native path
    assert any("Time to First" in k for k in result.metrics)


def test_run_native_benchmark_stt_scores_wer(tmp_path: Path):
    from tests.helpers.mock_stt_server import MockSTTServer
    from veeksha.config.client import STTClientConfig
    from veeksha.config.evaluator import PerformanceEvaluatorConfig
    from veeksha.core.request import Request
    from veeksha.core.request_content import AudioChannelRequestContent
    from veeksha.evaluator.performance.base import PerformanceEvaluator
    from veeksha.native.runner import run_native_benchmark
    from veeksha.types import ChannelModality

    wav = str(tmp_path / "c.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)

    srv = MockSTTServer(transcript="the quick brown fox").start()
    try:
        client_config = STTClientConfig(
            api_base=f"http://127.0.0.1:{srv.port}",
            model="m",
            provider="vllm_realtime",
            sample_rate=16000,
            use_native_transport=True,
        )
        reqs = [
            Request(
                id=i,
                channels={
                    ChannelModality.AUDIO: AudioChannelRequestContent(input_audio=wav)
                },
                metadata={
                    "expected_transcript": "the quick brown fox",
                    "dataset": "e2e",
                },
            )
            for i in range(4)
        ]
        evaluator = PerformanceEvaluator(
            PerformanceEvaluatorConfig(target_channels=["audio"], stream_metrics=False)
        )
        result = run_native_benchmark(
            reqs, evaluator, client_config, concurrency=4, timeout_s=10.0
        )
    finally:
        srv.stop()

    assert result.metrics["num_asr_scored_requests"] == 4
    assert result.metrics["asr_final_corpus_wer"] == 0.0
