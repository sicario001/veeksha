"""E2E Test for External (Mock) Server Benchmark."""

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

SAMPLE_CONFIG_URL = "veeksha/sample_configs/managed_server.yml"


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
def test_external_server_benchmark(mock_openai_server, tmp_path) -> None:
    # 1. Load config
    with open(SAMPLE_CONFIG_URL, "r") as f:
        config_dict = yaml.safe_load(f)

    # 2. Modify config for test
    # Remove managed server config to force external server mode
    if "server" in config_dict:
        del config_dict["server"]

    config_dict["runtime"]["max_sessions"] = 5
    config_dict["runtime"]["benchmark_timeout"] = 10
    config_dict["output_dir"] = str(tmp_path)

    # Point client to our mock server
    port = mock_openai_server.server_port
    if "client" not in config_dict:
        config_dict["client"] = {}
    config_dict["client"]["api_base"] = f"http://localhost:{port}/v1"
    config_dict["client"]["api_key"] = "mock"
    config_dict["client"]["model"] = "mock-model"

    # 3. Create BenchmarkConfig
    benchmark_config = create_class_from_dict(BenchmarkConfig, config_dict)

    # 4. Mock tokenizer to avoid HF download
    with (
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
            result = manage_benchmark_run(benchmark_config)

            # 6. Verify
            # Check result metrics: all sessions succeeded
            assert result.metrics["Successful Sessions"] == 5
            assert result.metrics["Number of Completed Requests"] > 0
            assert result.metrics["Error Rate"] == 0
