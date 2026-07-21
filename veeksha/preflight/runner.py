"""CLI runner for `veeksha validate-preflight`."""

from __future__ import annotations

import os
import sys
from typing import List

from veeksha.config.preflight import PreflightCheckConfig
from veeksha.preflight.validator import run_preflight_check
from veeksha.logger import init_logger

logger = init_logger(__name__)


def run_preflight_cli(configs: List[PreflightCheckConfig]) -> None:
    # The preflight spins worker pools up/down many times; quiet the per-pool
    # INFO chatter so the report is readable.
    import logging

    logging.getLogger("veeksha.workers.client_runner").setLevel(logging.WARNING)

    any_failed = False
    # A config sweep may reuse one output_dir; suffix repeat reports with an
    # index so earlier ones are not overwritten.
    reports_per_dir: dict = {}
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
            with open(path, "w") as f:
                f.write(text)
            logger.info("Preflight report written to %s", path)
        any_failed = any_failed or not report.passed
    # Non-zero exit so this can gate a benchmark in a script.
    if any_failed:
        sys.exit(1)
