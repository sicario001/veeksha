"""Wire-fidelity tests for the native plan compiler.

For every transport, the bytes/frames the native main loop puts on the wire
from a compiled plan must equal what the real Python clients send for the
same request — captured over real sockets by the servers in
``native_test_utils`` and compared byte-for-byte (HTTP: full head + body;
WS: every frame payload in order).
"""

from __future__ import annotations

import asyncio

import pytest

vn = pytest.importorskip(
    "veeksha.native.veeksha_native",
    reason="veeksha_native extension not built (run veeksha/native/build.sh)",
)

from tests.unit.loop.native_test_utils import (  # noqa: E402
    CaptureHttpServer,
    RealtimeTtsCaptureServer,
    SttCaptureServer,
    make_image_history_session,
    make_linear_text_session,
    make_stt_session,
    make_text_request,
    make_text_session,
    word_split_provider,
)
from tests.unit.native.mock_servers import (  # noqa: E402
    SseChatServer,
    drain_until_done,
    make_loop,
)
from veeksha.config.client import (  # noqa: E402
    OpenAIChatCompletionsClientConfig,
    RealtimeTTSClientConfig,
    STTClientConfig,
    TTSClientConfig,
)
from veeksha.core.seeding import SeedManager  # noqa: E402
from veeksha.loop.plan_compiler import (  # noqa: E402
    NativeIneligible,
    PlanCompiler,
)


def _run_native_session(port, plan, expect_completed=1, loop=None):
    """Run one compiled session plan through the native main loop."""
    if loop is None:
        loop = make_loop(vn, vn.TrafficKind.CONCURRENT, port, target=4)
    loop.feed_sessions([plan])
    loop.close_intake()
    events = drain_until_done(loop, expect_completed=expect_completed)
    completed = [e for e in events if e.kind == vn.EventKind.COMPLETED]
    for e in completed:
        assert e.result.error == "", e.result.error
    return completed


@pytest.mark.unit
def test_single_turn_text_bytes_match_python_client() -> None:
    """The compiled TEXT_SSE request is byte-identical (head + body) to the
    httpx request the Python chat client sends: same headers in the same
    order, same JSON encoding, sampling params merge, and max/min-token
    params."""
    srv = CaptureHttpServer(mode="sse", num_chunks=2).start()
    try:
        config = OpenAIChatCompletionsClientConfig(
            api_base=f"http://127.0.0.1:{srv.port}/v1",
            api_key="test-key",
            model="test-model",
            min_tokens_param="min_tokens",
            additional_sampling_params='{"temperature": 0.25}',
        )
        request = make_text_request(
            11,
            'hello "world" backslash \\ newline \n emoji 💡',
            target_tokens=7,
            metadata={"sampling_params": {"top_p": 0.9}},
        )
        session = make_text_session(3, request)

        from veeksha.client.openai_chat import OpenAIChatCompletionsClient

        client = OpenAIChatCompletionsClient(
            config, tokenizer_provider=word_split_provider()
        )
        result = asyncio.run(client.send_request(request, session_id=3))
        assert result.success, result.error_msg

        compiler = PlanCompiler(config, SeedManager(1), vn=vn)
        plan, sidecars = compiler.compile(session)
        _run_native_session(srv.port, plan)

        assert len(srv.requests) == 2
        python_bytes, native_bytes = srv.requests
        assert native_bytes == python_bytes
        assert b'"top_p": 0.9' in python_bytes or b'"top_p":0.9' in python_bytes
        assert 11 in sidecars
    finally:
        srv.stop()


@pytest.mark.unit
def test_multi_turn_dynamic_history_bodies_match_python_engine() -> None:
    """A 3-turn dynamic-history session produces the same request bodies on
    both loop implementations: user turns baked, assistant turns spliced natively from
    the parent turns' extracted content (tricky characters round-trip)."""
    import time

    from veeksha.config.runtime import RuntimeConfig
    from veeksha.config.traffic import ConcurrentTrafficConfig
    from veeksha.loop import MainLoopConfig, create_main_loop
    from veeksha.loop.source import PregeneratedSessionSource
    from tests.unit.loop.native_test_utils import drain_loop_until_idle

    token_text = 'hé"l\\lo\n💡{i} '

    def run(kind):
        srv = SseChatServer(
            num_chunks=3, chunk_gap_s=0.01, prefill_s=0.01, token_text=token_text
        ).start()
        try:
            config = MainLoopConfig(
                runtime=RuntimeConfig(
                    num_dispatcher_threads=1,
                    num_completion_threads=1,
                    num_client_threads=1,
                ),
                traffic=ConcurrentTrafficConfig(
                    target_concurrent_sessions=1, rampup_seconds=0
                ),
                client=OpenAIChatCompletionsClientConfig(
                    api_base=f"http://127.0.0.1:{srv.port}/v1",
                    api_key="k",
                    model="m",
                ),
                monotonic_anchor=time.monotonic(),
            )
            loop = create_main_loop(
                kind,
                config,
                seed_manager=SeedManager(7),
                tokenizer_provider=word_split_provider(),
            )
            sessions = [
                make_linear_text_session(1, ["turn one", "turn two", "turn three"])
            ]
            loop.start(PregeneratedSessionSource(sessions))
            drain_loop_until_idle(loop, expect_completed=3)
            return list(srv.bodies)
        finally:
            srv.stop()

    python_bodies = run("python")
    native_bodies = run("native")
    assert len(python_bodies) == len(native_bodies) == 3
    assert native_bodies == python_bodies


@pytest.mark.unit
def test_tts_http_bytes_match_python_client() -> None:
    """The compiled TTS_HTTP request (OpenAISpeechRequest payload + speech
    URL + headers) is byte-identical to the Python TTS client's request."""
    srv = CaptureHttpServer(mode="audio", num_chunks=2).start()
    try:
        config = TTSClientConfig(
            api_base=f"http://127.0.0.1:{srv.port}",
            api_key="tts-key",
            model="tts-model",
            voice_id="voice-a",
            raw_pcm=True,
        )
        request = make_text_request(21, "speak this now please")
        session = make_text_session(5, request)

        from veeksha.client.tts import TTSClient

        client = TTSClient(config)
        result = asyncio.run(client.send_request(request, session_id=5))
        assert result.success, result.error_msg

        compiler = PlanCompiler(config, SeedManager(1), vn=vn)
        plan, _ = compiler.compile(session)
        _run_native_session(srv.port, plan)

        assert len(srv.requests) == 2
        python_bytes, native_bytes = srv.requests
        assert native_bytes == python_bytes
        assert b"/v1/audio/speech" in python_bytes
    finally:
        srv.stop()


@pytest.mark.unit
def test_tts_realtime_frames_match_python_client() -> None:
    """The compiled TTS_REALTIME_WS frame sequence — session.update, one
    conversation.item.create per segment_text(...) segment, response.create
    — matches the Python realtime client's frames payload-for-payload."""
    srv = RealtimeTtsCaptureServer(num_chunks=3).start()
    try:
        config = RealtimeTTSClientConfig(
            api_base=f"http://127.0.0.1:{srv.port}",
            api_key="rt-key",
            model="rt-model",
            voice_id="rt-voice",
        )
        request = make_text_request(31, "hello world from realtime tts")
        session = make_text_session(6, request)

        from veeksha.client.realtime_tts import RealtimeTTSClient

        client = RealtimeTTSClient(config)
        result = asyncio.run(client.send_request(request, session_id=6))
        assert result.success, result.error_msg

        compiler = PlanCompiler(config, SeedManager(1), vn=vn)
        plan, _ = compiler.compile(session)
        completed = _run_native_session(srv.port, plan)

        assert len(srv.frames) == 2
        python_frames, native_frames = srv.frames
        assert native_frames == python_frames
        # 5 words -> 5 paced deltas at tokens_per_delta=1
        assert len(completed[0].result.send_offsets_ms) == 5
    finally:
        srv.stop()


@pytest.mark.unit
def test_stt_frames_match_python_client(tmp_path) -> None:
    """The compiled STT_WS flow — session.update + initial commit, blob-
    sliced input_audio_buffer.append frames from the clip's PCM, final
    commit — matches the Python vllm_realtime client's frames exactly."""
    from tests.unit.loop.native_test_utils import write_test_wav

    wav = write_test_wav(str(tmp_path / "clip.wav"))
    srv = SttCaptureServer(num_deltas=3).start()
    try:
        config = STTClientConfig(
            provider="vllm_realtime",
            api_base=f"http://127.0.0.1:{srv.port}",
            api_key="stt-key",
            model="stt-model",
            ws_chunk_size=1024,
            ws_realtime_pacing=True,
        )
        session = make_stt_session(7, wav)
        request = session.requests[0]

        from veeksha.client.stt import STTClient

        client = STTClient(config)
        result = asyncio.run(client.send_request(request, session_id=7))
        assert result.success, result.error_msg

        compiler = PlanCompiler(config, SeedManager(1), vn=vn)
        loop = make_loop(vn, vn.TrafficKind.CONCURRENT, srv.port, target=1)
        plan, _ = compiler.compile(session, loop.register_blob)
        completed = _run_native_session(srv.port, plan, loop=loop)

        assert len(srv.frames) == 2
        python_frames, native_frames = srv.frames
        assert native_frames == python_frames
        # frames: session.update, initial commit, N appends, final commit
        assert native_frames[0] == '{"type": "session.update", "model": "stt-model"}'
        assert native_frames[1] == '{"type": "input_audio_buffer.commit"}'
        assert (
            native_frames[-1] == '{"type": "input_audio_buffer.commit", "final": true}'
        )
        assert all('"input_audio_buffer.append"' in f for f in native_frames[2:-1])
        # send offsets follow the 1x schedule from the first paced send
        offsets = completed[0].result.send_offsets_ms
        assert offsets[0] == 0.0
        assert offsets == sorted(offsets)
        assert completed[0].result.content == srv.transcript()
    finally:
        srv.stop()


@pytest.mark.unit
def test_stt_shares_one_blob_per_clip(tmp_path) -> None:
    """Two sessions replaying the same clip register exactly ONE blob."""
    from tests.unit.loop.native_test_utils import write_test_wav

    wav = write_test_wav(str(tmp_path / "clip.wav"))
    config = STTClientConfig(
        provider="vllm_realtime",
        api_base="http://127.0.0.1:9",
        model="stt-model",
        ws_chunk_size=1024,
    )
    compiler = PlanCompiler(config, SeedManager(1), vn=vn)

    registered = []

    def registrar(data: bytes) -> int:
        registered.append(data)
        return len(registered) - 1

    plan_a, _ = compiler.compile(make_stt_session(1, wav), registrar)
    plan_b, _ = compiler.compile(make_stt_session(2, wav), registrar)
    assert len(registered) == 1
    for plan in (plan_a, plan_b):
        frames = plan.requests[0].transport.paced_frames
        assert all(f.blob.blob_id == 0 for f in frames)


@pytest.mark.unit
def test_rate_interval_chain_matches_scheduler_exactly() -> None:
    """The compiler-side RATE interval draws are bit-identical to the
    ``RateTrafficScheduler``'s seeded chain."""
    from veeksha.config.generator.interval import PoissonIntervalGeneratorConfig
    from veeksha.config.traffic import RateTrafficConfig
    from veeksha.loop.plan_compiler import build_interval_generator
    from veeksha.traffic.rate import RateTrafficScheduler

    traffic = RateTrafficConfig(
        interval_generator=PoissonIntervalGeneratorConfig(arrival_rate=3.0)
    )
    scheduler = RateTrafficScheduler(traffic, SeedManager(1234))
    compiler_gen = build_interval_generator(traffic, SeedManager(1234))

    scheduler_draws = [scheduler._interval_gen.get_next_interval() for _ in range(32)]
    compiler_draws = [compiler_gen.get_next_interval() for _ in range(32)]
    assert compiler_draws == scheduler_draws


@pytest.mark.unit
def test_compile_rejects_non_text_dynamic_history() -> None:
    """A dynamic-history parent carrying an IMAGE channel is ineligible."""
    config = OpenAIChatCompletionsClientConfig(api_base="http://127.0.0.1:9")
    compiler = PlanCompiler(config, SeedManager(1), vn=vn)
    with pytest.raises(NativeIneligible, match="non-text channels"):
        compiler.compile(make_image_history_session(1))
