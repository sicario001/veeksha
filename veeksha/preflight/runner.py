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
    for config in configs:
        report = run_preflight_check(config)
        text = report.format_text()
        print(text)
        if config.output_dir:
            os.makedirs(config.output_dir, exist_ok=True)
            path = os.path.join(config.output_dir, "preflight_report.txt")
            with open(path, "w") as f:
                f.write(text)
            logger.info("Preflight report written to %s", path)
        any_failed = any_failed or not report.passed
    # Non-zero exit so this can gate a benchmark in a script.
    if any_failed:
        sys.exit(1)
