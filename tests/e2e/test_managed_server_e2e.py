"""E2E Test for Managed Server Benchmark."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest
import yaml

from veeksha.benchmark import manage_benchmark_run
from veeksha.config.benchmark import BenchmarkConfig
from vidhi import create_class_from_dict
from veeksha.core.request_content import TextChannelRequestContent

SAMPLE_CONFIG_PATH = "veeksha/sample_configs/managed_server.yml"


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        data = json.loads(body)

        # Determine number of tokens requested
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

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Silence logs


@pytest.fixture
def mock_openai_server():
    server = HTTPServer(("localhost", 0), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield server
    server.shutdown()


# @pytest.mark.e2e
def test_managed_server_benchmark(mock_openai_server, tmp_path) -> None:
    # 1. Load config
    with open(SAMPLE_CONFIG_PATH, "r") as f:
        config_dict = yaml.safe_load(f)

    # 2. Modify config for test
    config_dict["runtime"]["max_sessions"] = 5
    config_dict["runtime"]["benchmark_timeout"] = 10
    config_dict["output_dir"] = str(tmp_path)

    # 3. Create BenchmarkConfig
    benchmark_config = create_class_from_dict(BenchmarkConfig, config_dict)

    # 4. Mock managed_server context manager
    port = mock_openai_server.server_port
    server_info = {
        "api_base": f"http://localhost:{port}/v1",
        "api_key": "mock",
        "model": config_dict["server"]["model"],
    }

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = server_info
    mock_ctx.__exit__.return_value = None

    # We patch the call to managed_server in veeksha.benchmark
    with (
        patch("veeksha.benchmark.managed_server", return_value=mock_ctx) as mocked_ms,
        patch(
            "veeksha.benchmark.build_hf_tokenizer_handle_from_model"
        ) as mock_build_tok,
        patch(
            "veeksha.core.tokenizer.build_hf_tokenizer_handle_from_model"
        ) as mock_build_tok_core,
    ):

        # Mock tokenizer handle
        mock_handle = MagicMock()
        # Mock encode to return a list of token IDs
        mock_handle.encode.return_value = [1] * 10
        # Mock decode
        mock_handle.decode.return_value = "mock_text"
        mock_build_tok.return_value = mock_handle
        mock_build_tok_core.return_value = mock_handle

        # Stub the channel generator to avoid heavy prompt generation
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
            result = manage_benchmark_run(benchmark_config)

            # 6. Verify
            mocked_ms.assert_called_once()
            assert result.metrics.get("Successful Sessions", 0) == 5
            assert result.metrics.get("Number of Completed Requests", 0) > 0
            assert result.metrics.get("Error Rate", 0) == 0
