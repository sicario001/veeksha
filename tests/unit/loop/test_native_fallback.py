"""Fallback behavior for main_loop=native: ineligible configs and the unbuilt
extension fall back to PythonMainLoop with a warning; the compiler raises
NativeIneligible for sessions the native path cannot serve faithfully."""

from __future__ import annotations

import logging
import time

import pytest

from veeksha.config.client import (
    OpenAIChatCompletionsClientConfig,
    OpenAICompletionsClientConfig,
    STTClientConfig,
)
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.traffic import ConcurrentTrafficConfig
from veeksha.core.seeding import SeedManager
from veeksha.loop import MainLoopConfig, create_main_loop
from veeksha.loop.python_loop import PythonMainLoop


def _config(client_config) -> MainLoopConfig:
    return MainLoopConfig(
        runtime=RuntimeConfig(),
        traffic=ConcurrentTrafficConfig(target_concurrent_sessions=1, rampup_seconds=0),
        client=client_config,
        monotonic_anchor=time.monotonic(),
    )


def _create_native(config, caplog):
    with caplog.at_level(logging.WARNING, logger="veeksha.loop"):
        return create_main_loop("native", config, seed_manager=SeedManager(1))


@pytest.mark.unit
def test_native_unbuilt_falls_back_with_warning(monkeypatch, caplog) -> None:
    import veeksha.native as native_pkg

    monkeypatch.setattr(native_pkg, "is_available", lambda: False)
    loop = _create_native(
        _config(OpenAIChatCompletionsClientConfig(api_base="http://127.0.0.1:9")),
        caplog,
    )
    assert isinstance(loop, PythonMainLoop)
    assert "not built" in caplog.text


@pytest.mark.unit
def test_vajra_stt_provider_falls_back_with_reason(caplog) -> None:
    pytest.importorskip("veeksha.native.veeksha_native")
    loop = _create_native(
        _config(
            STTClientConfig(
                provider="vajra_openai_realtime",
                api_base="http://127.0.0.1:9",
                model="m",
            )
        ),
        caplog,
    )
    assert isinstance(loop, PythonMainLoop)
    assert "vajra_openai_realtime" in caplog.text
    assert "falling back to the Python main loop" in caplog.text


@pytest.mark.unit
def test_tls_endpoint_falls_back_with_reason(caplog) -> None:
    pytest.importorskip("veeksha.native.veeksha_native")
    loop = _create_native(
        _config(OpenAIChatCompletionsClientConfig(api_base="https://example.com/v1")),
        caplog,
    )
    assert isinstance(loop, PythonMainLoop)
    assert "TLS" in caplog.text


@pytest.mark.unit
def test_unsupported_client_type_falls_back_with_reason(caplog) -> None:
    pytest.importorskip("veeksha.native.veeksha_native")
    loop = _create_native(
        _config(OpenAICompletionsClientConfig(api_base="http://127.0.0.1:9")),
        caplog,
    )
    assert isinstance(loop, PythonMainLoop)
    assert "no native transport" in caplog.text


@pytest.mark.unit
def test_eligible_config_returns_native_loop() -> None:
    pytest.importorskip("veeksha.native.veeksha_native")
    from veeksha.loop.native_loop import NativeMainLoop

    loop = create_main_loop(
        "native",
        _config(OpenAIChatCompletionsClientConfig(api_base="http://127.0.0.1:9")),
        seed_manager=SeedManager(1),
    )
    assert isinstance(loop, NativeMainLoop)


@pytest.mark.unit
def test_native_feeder_skips_ineligible_session_and_run_survives(caplog) -> None:
    """Session-level ineligibility (discovered mid-run, after config-level
    checks passed): the feeder skips the session with a warning and the run
    completes with the remaining sessions."""
    pytest.importorskip("veeksha.native.veeksha_native")
    from tests.unit.loop.native_test_utils import (
        drain_loop_until_idle,
        make_image_history_session,
        make_text_request,
        make_text_session,
        word_split_provider,
    )
    from tests.unit.native.mock_servers import SseChatServer
    from veeksha.loop.interface import LoopEventKind
    from veeksha.loop.source import PregeneratedSessionSource

    srv = SseChatServer(num_chunks=2, chunk_gap_s=0.005, prefill_s=0.005).start()
    try:
        config = _config(
            OpenAIChatCompletionsClientConfig(
                api_base=f"http://127.0.0.1:{srv.port}/v1", api_key="k", model="m"
            )
        )
        loop = create_main_loop(
            "native",
            config,
            seed_manager=SeedManager(1),
            tokenizer_provider=word_split_provider(),
        )
        sessions = [
            make_text_session(1, make_text_request(100, "good session")),
            make_image_history_session(2),  # natively ineligible
            make_text_session(3, make_text_request(300, "another good one")),
        ]
        with caplog.at_level(logging.WARNING, logger="veeksha.loop"):
            loop.start(PregeneratedSessionSource(sessions))
            events = drain_loop_until_idle(loop, expect_completed=2)
    finally:
        srv.stop()

    completed = [e for e in events if e.kind == LoopEventKind.COMPLETED]
    assert sorted(e.request_id for e in completed) == [100, 300]
    assert all(e.result.success for e in completed)
    assert "skipped session 2" in caplog.text
