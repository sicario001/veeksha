"""N4: end-to-end native transport for ALL modalities into the real evaluators.

Each modality's requests run over the native engine (against the dummy servers)
and are scored by the same evaluator a Python-transport run would use:
  text -> TextPerformanceEvaluator (TTFT/TPOT)
  TTS  -> AudioPerformanceEvaluator (TTFA/RTF from the audio-delta timeline)
  STT  -> AudioPerformanceEvaluator STT branch (WER from transcript + ground truth)
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from veeksha.native.engine import native_available

pytestmark = pytest.mark.skipif(
    not native_available(), reason="veeksha_native extension not built"
)


def _text_request(rid, tokens=10):
    from veeksha.core.request import Request
    from veeksha.core.request_content import TextChannelRequestContent
    from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
    from veeksha.types import ChannelModality

    return Request(
        id=rid,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(input_text="hello world")
        },
        requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=tokens)),
    )


# --------------------------------------------------------------------- text
def test_native_transport_text_e2e():
    from veeksha.config.evaluator import PerformanceEvaluatorConfig
    from veeksha.evaluator.performance.text import TextPerformanceEvaluator
    from veeksha.native.transport import NativeTransport
    from veeksha.preflight.dummy_engine import DummyStreamingEngine
    from veeksha.types import ChannelModality

    engine = DummyStreamingEngine(
        chunk_dt=0.02, prefill_s=0.03, default_chunks=10, num_loops=4
    ).start()
    try:
        transport = NativeTransport("127.0.0.1", engine.port)
        results = transport.run_text(
            [_text_request(i, 10) for i in range(10)], concurrency=5, model="d"
        )
    finally:
        engine.stop()

    ok = [r for r in results if r.success]
    assert len(ok) == 10
    ev = TextPerformanceEvaluator(PerformanceEvaluatorConfig())
    for r in ok:
        ev.register_request(
            r.request_id, r.session_id, 0.0, r.channels[ChannelModality.TEXT]
        )
        ev.record_request_completed(r.request_id, r.session_id, 1.0, r)
    m = ev.finalize().metrics
    assert any("Time to First" in k for k in m)


# --------------------------------------------------------------------- TTS
def test_native_transport_realtime_tts_e2e():
    from tests.helpers.dummy_realtime_tts_server import DummyRealtimeTTSServer
    from veeksha.config.evaluator import PerformanceEvaluatorConfig
    from veeksha.evaluator.performance.audio import AudioPerformanceEvaluator
    from veeksha.native.transport import NativeTransport

    srv = DummyRealtimeTTSServer(
        num_chunks=6, chunk_bytes=4800, sample_rate=24000
    ).start()
    try:
        transport = NativeTransport("127.0.0.1", srv.port)
        results = transport.run_realtime_tts(
            [_text_request(i) for i in range(4)],
            concurrency=4,
            sample_rate=24000,
            timeout_s=5.0,
        )
    finally:
        srv.stop()

    ok = [r for r in results if r.success]
    assert len(ok) == 4
    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    for r in ok:
        ev.register_request(r.request_id, r.session_id, 0.0, None)
        ev.record_request_completed(r.request_id, r.session_id, 1.0, r)
    m = ev.finalize().metrics
    assert m["num_completed_requests"] == 4
    assert "Time to First Audio (Mean)" in m
    assert "Streaming Real Time Factor (Mean)" in m


# --------------------------------------------------------------------- STT
def _stt_request(rid, wav_path, expected):
    from veeksha.core.request import Request
    from veeksha.core.request_content import AudioChannelRequestContent
    from veeksha.types import ChannelModality

    return Request(
        id=rid,
        channels={
            ChannelModality.AUDIO: AudioChannelRequestContent(input_audio=wav_path)
        },
        metadata={"expected_transcript": expected, "dataset": "e2e"},
    )


def test_native_transport_stt_e2e(tmp_path: Path):
    from tests.helpers.dummy_stt_server import DummySTTServer
    from veeksha.config.evaluator import PerformanceEvaluatorConfig
    from veeksha.evaluator.performance.audio import AudioPerformanceEvaluator
    from veeksha.native.transport import NativeTransport

    wav = str(tmp_path / "clip.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)  # 1s of silence

    srv = DummySTTServer(
        transcript="the quick brown fox", first_delta_delay=0.03, delta_dt=0.02
    ).start()
    try:
        transport = NativeTransport("127.0.0.1", srv.port)
        results = transport.run_stt(
            [_stt_request(i, wav, "the quick brown fox") for i in range(4)],
            concurrency=4,
            sample_rate=16000,
            timeout_s=5.0,
        )
    finally:
        srv.stop()

    ok = [r for r in results if r.success]
    assert len(ok) == 4
    ev = AudioPerformanceEvaluator(PerformanceEvaluatorConfig())
    for r in ok:
        ev.register_request(r.request_id, r.session_id, 0.0, None)
        ev.record_request_completed(r.request_id, r.session_id, 1.0, r)
    m = ev.finalize().metrics
    assert m["num_asr_scored_requests"] == 4
    assert m["asr_final_corpus_wer"] == 0.0  # transcript matches ground truth
