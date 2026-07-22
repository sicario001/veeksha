"""CLI entry point for ``veeksha preflight``."""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List

from veeksha.config.preflight import PreflightCheckConfig
from veeksha.logger import init_logger
from veeksha.preflight.validator import run_preflight_check

logger = init_logger(__name__)

__all__ = ["run_preflight_cli"]


def run_preflight_cli(configs: List[PreflightCheckConfig]) -> None:
    """Run each config, print its report, exit non-zero if any check failed.

    Exit status is the point of the command: a script can gate a benchmark on
    ``veeksha preflight``. Non-zero means a gated check was measured DISHONEST
    at the target concurrency — an inconclusive (mock-saturated) check does not
    fail the run, because nothing was proven either way.
    """
    # The preflight spins worker pools up and down once per check; quiet the
    # per-pool chatter so the report is the output.
    for name in (
        "veeksha.workers.client_runner",
        "veeksha.workers.prefetch",
        "veeksha.core.thread_pool",
        "veeksha.benchmark",
        "veeksha.benchmark_utils",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    any_failed = False
    reports_per_dir: Dict[str, int] = {}
    for config in configs:
        report = run_preflight_check(config)
        text = report.format_text()
        print(text)
        if config.output_dir:
            os.makedirs(config.output_dir, exist_ok=True)
            out_dir = os.path.abspath(config.output_dir)
            index = reports_per_dir.get(out_dir, 0)
            reports_per_dir[out_dir] = index + 1
            filename = (
                "preflight_report.txt"
                if index == 0
                else f"preflight_report_{index}.txt"
            )
            path = os.path.join(config.output_dir, filename)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            logger.info("Preflight report written to %s", path)
        any_failed = any_failed or not report.passed

    if any_failed:
        sys.exit(1)
