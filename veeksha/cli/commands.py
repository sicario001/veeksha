"""Veeksha top-level CLI command group.

veeksha benchmark [options]
veeksha capacity-search [options]
veeksha prefill [options]
veeksha decode [options]
veeksha stress [options]
veeksha score-tts-longform [options]
veeksha preflight [options]
"""

from __future__ import annotations

import sys
import warnings

# Suppress GIL re-enable warnings from C extensions (e.g. tokenizers)
# that haven't declared free-threaded support yet.
warnings.filterwarnings(
    "ignore", message=".*global interpreter lock.*", category=RuntimeWarning
)

from vidhi import parse_cli_sweep

from veeksha.capacity_search import run_capacity_search
from veeksha.cli.base import VeekshaCommand
from veeksha.cli.benchmarks import run_cli as run_benchmark
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.config.preflight import PreflightCheckConfig
from veeksha.config.score_tts_longform import ScoreTtsLongformConfig
from veeksha.microbench.config import (
    DecodeMicrobenchmarkConfig,
    PrefillMicrobenchmarkConfig,
    StressMicrobenchmarkConfig,
)
from veeksha.microbench.decode import run_decode
from veeksha.microbench.diff import DiffConfig, run_diff
from veeksha.microbench.prefill import run_prefill
from veeksha.microbench.stress import run_stress
from veeksha.preflight.runner import run_preflight_cli
from veeksha.verification.longform import run_score_tts_longform_cli
from veeksha.version import __version__

_RUNNERS = {
    BenchmarkConfig: run_benchmark,
    CapacitySearchConfig: lambda configs: [run_capacity_search(c) for c in configs],
    PrefillMicrobenchmarkConfig: lambda configs: [run_prefill(c) for c in configs],
    DecodeMicrobenchmarkConfig: lambda configs: [run_decode(c) for c in configs],
    StressMicrobenchmarkConfig: lambda configs: [run_stress(c) for c in configs],
    DiffConfig: lambda configs: [run_diff(c) for c in configs],
    ScoreTtsLongformConfig: run_score_tts_longform_cli,
    PreflightCheckConfig: run_preflight_cli,
}

_VERSION_FLAGS = {"--version", "-V"}


def main() -> None:
    """Entry point for the veeksha CLI."""
    if len(sys.argv) == 2 and sys.argv[1] in _VERSION_FLAGS:
        print(f"veeksha {__version__}")
        return

    configs = parse_cli_sweep(VeekshaCommand)
    if not configs:
        sys.exit("No configuration resolved from CLI arguments.")
    runner = _RUNNERS.get(type(configs[0]))
    if runner is None:
        sys.exit(f"Unknown command config type: {type(configs[0]).__name__}")
    runner(configs)
