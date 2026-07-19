"""E2E Test for Capacity Search Benchmark."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch, MagicMock

import pytest
import yaml

from veeksha.capacity_search import run_capacity_search
from veeksha.config.capacity_search import CapacitySearchConfig
from vidhi import create_class_from_dict
from veeksha.core.request_content import TextChannelRequestContent

SAMPLE_CONFIG_URL = "veeksha/sample_configs/capacity_search_rate.yml"


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def _write_json(self, status_code: int, payload: dict) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        if self.path.endswith("/health"):
            return self._write_json(200, {"status": "ok"})
        return self._write_json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            return self._write_json(404, {"error": "not found"})

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        data = json.loads(body or "{}")

        max_tokens = data.get("max_completion_tokens", 10)
        response = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "mock " * max_tokens},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": max_tokens,
                "total_tokens": 10 + max_tokens,
            },
        }

        self._write_json(200, response)

    def log_message(self, format, *args):
        pass  # Silence logs


class LocalHTTPServerManager:
    """Lightweight test server manager wiring through managed_server."""

    def __init__(self, config, handler_cls, output_dir: str):
        self.config = config
        self.handler_cls = handler_cls
        self.output_dir = output_dir
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def launch(self):
        self._server = HTTPServer(
            (self.config.host, self.config.port), self.handler_cls
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return True, None

    def wait_for_ready(self, timeout=None):
        return True

    def shutdown(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=1)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


# @pytest.mark.e2e
def test_capacity_search_rate_benchmark(tmp_path) -> None:
    # 1. Load config
    with open(SAMPLE_CONFIG_URL, "r") as f:
        config_dict = yaml.safe_load(f)

    # 2. Modify config for test
    config_dict["output_dir"] = str(tmp_path)

    # Speed up the test
    config_dict["max_iterations"] = 3
    config_dict["start_value"] = 5  # Start higher to try to find something
    config_dict["expansion_factor"] = 2.0
    config_dict["benchmark_config"]["runtime"]["max_sessions"] = 10
    config_dict["benchmark_config"]["runtime"]["benchmark_timeout"] = 10
    # Exit promptly after timeout if any requests hang
    config_dict["benchmark_config"]["runtime"]["post_timeout_grace_seconds"] = 0

    # Inject server config to trigger managed_server logic
    free_port = _find_free_port()
    config_dict["benchmark_config"]["server"] = {
        "type": "vllm",
        "model": "mock-model",
        "host": "localhost",
        "port": free_port,
        "api_key": "mock",
    }

    # Force fixed body length to avoid ambiguity/mocks
    config_dict["benchmark_config"]["session_generator"]["channels"][0][
        "body_length_generator"
    ] = {"type": "fixed", "value": 10}

    # 3. Create CapacitySearchConfig
    capacity_config = create_class_from_dict(CapacitySearchConfig, config_dict)

    # 4. Spin up a real HTTP server via the managed_server stack by patching the registry
    def _make_manager(_key, config, output_dir):
        return LocalHTTPServerManager(
            config, handler_cls=MockOpenAIHandler, output_dir=output_dir
        )

    with (
        patch(
            "veeksha.orchestration.benchmark_orchestrator.ServerManagerRegistry.get",
            side_effect=_make_manager,
        ),
        patch(
            "veeksha.benchmark.build_hf_tokenizer_handle_from_model"
        ) as mock_build_tok_bench,
        patch(
            "veeksha.core.tokenizer.build_hf_tokenizer_handle_from_model"
        ) as mock_build_tok_core,
    ):

        mock_handle = MagicMock()
        mock_handle.encode.return_value = [1] * 10
        mock_handle.decode.return_value = "mock_text"
        mock_handle.count_tokens.return_value = 0
        mock_build_tok_bench.return_value = mock_handle
        mock_build_tok_core.return_value = mock_handle

        # Mock the channel generator to return a realistic TextChannelRequestContent
        mock_channel_gen = MagicMock()
        mock_channel_gen.generate_content.return_value = TextChannelRequestContent(
            input_text="mock prompt",
            target_prompt_tokens=5,
        )

        with patch(
            "veeksha.generator.channel.registry.ChannelGeneratorRegistry.get",
            return_value=mock_channel_gen,
        ):
            # 5. Run
            result = run_capacity_search(capacity_config)

            # 6. Verify
            # Search should run iterations.
            assert len(result["history"]) > 0  # At least one attempt should run

        # Check result structure
        assert "best_value" in result
        assert "history" in result
        assert len(result["history"]) >= 1

        # We expect it to find *some* value since our mock server is very fast
        # (It returns instantly, so it should handle high rates)
        # However, due to startup overhead etc, it might fail early.
        # But at least one iteration should run.
