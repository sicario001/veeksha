"""E2E Test for Trace-based Benchmark (SharedPrefix Flavor)."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest
import yaml

from veeksha.benchmark import manage_benchmark_run
from veeksha.config.benchmark import BenchmarkConfig
from vidhi import create_class_from_dict
from veeksha.generator.session.trace.prompt_builder import TracePromptBuilder
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
def test_trace_benchmark_shared_prefix(mock_openai_server, tmp_path) -> None:
    # 1. Create mock corpus
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        "Hello world this is a test corpus line one.\nLine two for corpus content generation.\n"
    )

    # 2. Create mock trace
    trace_file = tmp_path / "trace.jsonl"
    trace_data = [
        {
            "session_id": 1,
            "input_length": 10,
            "output_length": 5,
            "new_input_length": 10,
            "hash_ids": [101, 102],
            "timestamp": 1000,
        },
        {
            "session_id": 1,
            "input_length": 15,
            "output_length": 5,
            "new_input_length": 5,
            "hash_ids": [101, 102, 103],
            "timestamp": 1005,
        },
        {
            "session_id": 2,
            "input_length": 10,
            "output_length": 5,
            "new_input_length": 10,
            "hash_ids": [101, 104],
            "timestamp": 1010,
        },
    ]
    with open(trace_file, "w") as f:
        for item in trace_data:
            f.write(json.dumps(item) + "\n")

    # 3. Load sample config pattern
    with open(SAMPLE_CONFIG_URL, "r") as f:
        config_dict = yaml.safe_load(f)

    # 4. Modify config for trace test
    if "server" in config_dict:
        del config_dict["server"]

    config_dict["runtime"][
        "max_sessions"
    ] = 2  # Matches number of unique sessions in trace
    config_dict["runtime"]["benchmark_timeout"] = 10
    config_dict["output_dir"] = str(tmp_path / "output")

    # Configure Session Generator as TRACE
    config_dict["session_generator"] = {
        "type": "trace",
        "trace_file": str(trace_file),
        "wrap_mode": False,
        "flavor": {
            "type": "shared_prefix",
            "block_size": 4,  # Small block size
            "corpus_file": str(corpus_file),
        },
    }

    # Point client to our mock server
    port = mock_openai_server.server_port
    if "client" not in config_dict:
        config_dict["client"] = {}
    config_dict["client"]["api_base"] = f"http://localhost:{port}/v1"
    config_dict["client"]["api_key"] = "mock"
    config_dict["client"]["model"] = "mock-model"

    # 5. Create BenchmarkConfig
    benchmark_config = create_class_from_dict(BenchmarkConfig, config_dict)

    # 6. Mock dependencies
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
        mock_handle.get_vocab.return_value = ["a", "b"]
        mock_build_tok_bench.return_value = mock_handle
        mock_build_tok_core.return_value = mock_handle

        with patch.object(
            TracePromptBuilder, "build_from_hash_ids", return_value="mock prompt"
        ):
            result = manage_benchmark_run(benchmark_config)

            # Verify basic metrics: sessions succeeded, no errors
            assert result.metrics.get("Successful Sessions", 0) == 2
            assert result.metrics.get("Number of Completed Requests", 0) >= 3
            assert result.metrics.get("Error Rate", 0) == 0
