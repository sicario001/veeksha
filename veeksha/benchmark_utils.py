"""Utilities used by the benchmark runner."""

import hashlib
import math
import os
import shutil
import time
from datetime import datetime
from typing import Any, Dict, Set, Tuple

import yaml
from tqdm import tqdm
from vidhi import dataclass_to_dict

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.core.seeding import SeedManager
from veeksha.evaluator.base import BaseEvaluator
from veeksha.evaluator.composite import CompositeEvaluator
from veeksha.evaluator.registry import EvaluatorRegistry
from veeksha.logger import init_logger
from veeksha.types import EvaluationType

logger = init_logger(__name__)

__all__ = [
    "_persist_config_yaml",
    "_init_output_dir",
    "build_evaluator",
    "maybe_run_warmup",
    "_monitor_for_completion",
    "recommended_client_threads",
    "maybe_warn_client_thread_sizing",
]

# One Python asyncio event loop stays timing-accurate up to roughly this many
# concurrent streams before drift climbs (measured, analysis/bench/RESULTS.md).
STREAMS_PER_EVENT_LOOP = 300


def recommended_client_threads(target_concurrency: int) -> int:
    """Suggested ``num_client_threads`` to keep timings honest at a concurrency.

    Each client-worker asyncio loop stays accurate up to ~``STREAMS_PER_EVENT_LOOP``
    concurrent streams, so spread the target across ``ceil(target / that)`` loops.
    """
    if target_concurrency <= 0:
        return 1
    return max(1, math.ceil(target_concurrency / STREAMS_PER_EVENT_LOOP))


def maybe_warn_client_thread_sizing(benchmark_config: BenchmarkConfig) -> None:
    """Warn if num_client_threads is too low for a concurrent target to stay honest."""
    from veeksha.config.traffic import ConcurrentTrafficConfig

    scheduler = benchmark_config.traffic_scheduler
    if not isinstance(scheduler, ConcurrentTrafficConfig):
        return
    target = scheduler.target_concurrent_sessions
    n_threads = benchmark_config.runtime.num_client_threads
    recommended = recommended_client_threads(target)
    if n_threads < recommended:
        logger.warning(
            "num_client_threads=%d may be too low for target_concurrent_sessions=%d: "
            "each asyncio loop stays timing-accurate up to ~%d streams, so recommend "
            "num_client_threads >= %d. Validate first with "
            "`veeksha preflight --target_concurrency %d --num_client_threads %d`.",
            n_threads,
            target,
            STREAMS_PER_EVENT_LOOP,
            recommended,
            target,
            n_threads,
        )


def _persist_config_yaml(benchmark_config: BenchmarkConfig) -> str:
    """Write the resolved benchmark configuration to config.yml.

    Args:
        benchmark_config: The fully resolved benchmark configuration.

    Returns:
        Path to the persisted YAML file.
    """
    os.makedirs(benchmark_config.output_dir, exist_ok=True)
    config_dict = dataclass_to_dict(benchmark_config)
    config_path = os.path.join(benchmark_config.output_dir, "config.yml")
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(
            config_dict,
            config_file,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    logger.debug("Persisted benchmark config to %s", config_path)
    return config_path


def _init_output_dir(benchmark_config: BenchmarkConfig) -> str:
    """Resolve and prepare the final benchmark output directory.

    The function persists the config, computes its hash, and
    moves the config into a dated/hash-named subdirectory. The benchmark
    configuration's ``output_dir`` field is updated in-place to point to the
    resolved directory.

    Args:
        benchmark_config: Benchmark configuration to mutate.

    Returns:
        Path to the resolved output directory.
    """

    base_output_dir = benchmark_config.output_dir
    os.makedirs(base_output_dir, exist_ok=True)

    config_path = _persist_config_yaml(benchmark_config)
    with open(config_path, "rb") as config_file:
        config_bytes = config_file.read()
    config_hash = hashlib.sha1(config_bytes).hexdigest()[:8]

    timestamp_prefix = datetime.utcnow().strftime("%d_%m_%Y-%H_%M_%S")
    base_dir_name = f"{timestamp_prefix}-{config_hash}"
    resolved_output_dir = os.path.join(base_output_dir, base_dir_name)

    suffix = 1
    while os.path.exists(resolved_output_dir):
        suffix += 1
        resolved_output_dir = os.path.join(base_output_dir, f"{base_dir_name}-{suffix}")

    os.makedirs(resolved_output_dir, exist_ok=True)
    shutil.move(config_path, os.path.join(resolved_output_dir, "config.yml"))
    object.__setattr__(benchmark_config, "output_dir", resolved_output_dir)
    logger.info("Benchmark outputs will be stored in %s", resolved_output_dir)

    return resolved_output_dir


def maybe_run_warmup(session_generator, client) -> None:
    """Maybe run warmup sessions synchronously before benchmark.

    A warmup only runs the first request of each session specified.
    """

    import asyncio

    async def warmup_one(session):
        """Execute first request of a warmup session."""
        first_request = list(session.requests.values())[0]
        await client.send_request(first_request, session.id, 1)

    async def run_all(warmup_sessions):
        for session in tqdm(warmup_sessions, desc="Warmup", unit="sess"):
            await warmup_one(session)

    if hasattr(session_generator, "get_warmup_sessions"):
        warmup_sessions = session_generator.get_warmup_sessions()
        if warmup_sessions:
            logger.info(f"Running warmup with {len(warmup_sessions)} sessions")
            asyncio.run(run_all(warmup_sessions))
            logger.info("Warmup completed")


def build_evaluator(
    benchmark_config: BenchmarkConfig,
    *,
    seed_manager: SeedManager,
    session_generator: Any,
    benchmark_start_time: float,
) -> BaseEvaluator:
    """Build an evaluator instance (or composite evaluator) for a benchmark run.

    Notes:
    - Performance evaluator(s) are ordered first so that `CompositeEvaluator` uses
      performance for progress/timeout behavior.
    - Accuracy evaluation requires access to the session generator to map
      request IDs back to lm-eval instances.

    Args:
        benchmark_config: Benchmark configuration.
        seed_manager: Seed manager for reproducibility.
        session_generator: Session generator used for this run.
        benchmark_start_time: Run start time (monotonic), passed to evaluators for
            time-normalization and artifact timestamps.

    Returns:
        A `BaseEvaluator` (single evaluator or `CompositeEvaluator`).
    """
    evaluator_configs = sorted(
        benchmark_config.evaluators,
        key=lambda cfg: 0 if cfg.get_type() == EvaluationType.PERFORMANCE else 1,
    )

    evaluator_instances: list[BaseEvaluator] = []
    for cfg in evaluator_configs:
        kwargs: Dict[str, Any] = {
            "config": cfg,
            "seed_manager": seed_manager,
            "output_dir": f"{benchmark_config.output_dir}/metrics",
            "benchmark_start_time": benchmark_start_time,
        }
        if cfg.get_type() == EvaluationType.ACCURACY_LMEVAL:
            kwargs["session_generator"] = session_generator
        evaluator_instances.append(EvaluatorRegistry.get(cfg.get_type(), **kwargs))

    if not evaluator_instances:
        raise ValueError("BenchmarkConfig.evaluators must be non-empty.")

    return (
        evaluator_instances[0]
        if len(evaluator_instances) == 1
        else CompositeEvaluator(evaluator_instances)
    )


def _init_pbar(max_sessions: int, benchmark_timeout: float) -> Tuple[Any, bool]:
    """Initialize progress bar based on benchmark mode."""
    if max_sessions > 0:
        pbar = tqdm(
            total=max_sessions,
            desc="Sessions",
            unit="sess",
            dynamic_ncols=True,
            bar_format="{desc}: {n}/{total} [{percentage:3.0f}%] | {rate_fmt} | Elapsed: {elapsed}",
        )
        return pbar, False

    pbar = tqdm(
        total=int(benchmark_timeout),
        desc="Benchmark",
        unit="s",
        dynamic_ncols=True,
        bar_format="{desc}: {elapsed}/{total} s [{percentage:3.0f}%] | Sessions: {postfix}",
    )
    pbar.set_postfix_str("0")
    return pbar, True


def _update_pbar(
    pbar,
    time_based_progress: bool,
    elapsed: float,
    total_done: int,
    state: Dict[str, int],
) -> None:
    """Update progress bar with current state."""
    if time_based_progress:
        elapsed_int = int(elapsed)
        if elapsed_int > state["last_time_update"]:
            pbar.update(elapsed_int - state["last_time_update"])
            state["last_time_update"] = elapsed_int
        if total_done > state["last_completed"]:
            pbar.set_postfix_str(str(total_done))
            state["last_completed"] = total_done
        return

    if total_done > state["last_completed"]:
        pbar.update(total_done - state["last_completed"])
        state["last_completed"] = total_done


def _monitor_for_completion(
    traffic_scheduler,
    evaluator,
    pool_manager,
    benchmark_start: float,
    benchmark_timeout: float,
    timeout_triggered: bool,
    pre_timeout_request_ids: Set[str],
    max_sessions: int,
    post_timeout_grace_seconds: int = -1,
) -> Set[str]:
    """Observe worker progress and exit once requests settle.

    Returns:
        Set of request IDs that were still in-flight when monitoring stopped.
    """
    pbar, time_based_progress = _init_pbar(max_sessions, benchmark_timeout)
    pbar_state = {"last_completed": 0, "last_time_update": 0}
    timeout_start: float = 0.0
    in_flight_remaining: Set[str] = set()

    try:
        while True:
            time.sleep(0.1)

            completed, errored, _ = evaluator.get_session_counts()
            total_done = completed + errored
            elapsed = time.monotonic() - benchmark_start

            _update_pbar(pbar, time_based_progress, elapsed, total_done, pbar_state)

            if (
                not timeout_triggered
                and benchmark_timeout > 0
                and elapsed >= benchmark_timeout
            ):
                timeout_triggered = True
                timeout_start = time.monotonic()
                pre_timeout_request_ids = evaluator.get_registered_request_ids()
                in_flight = traffic_scheduler.get_in_flight_request_ids()
                pending = pre_timeout_request_ids & in_flight
                logger.info(
                    f"Benchmark timeout after {elapsed:.1f}s. "
                    f"Captured {len(pre_timeout_request_ids)} registered requests, "
                    f"{len(pending)} still in-flight."
                )

            prefetch_threads = pool_manager.thread_pools.get("prefetch", [])
            all_prefetch_done = all(not t.is_alive() for t in prefetch_threads)

            if timeout_triggered:
                current_in_flight = traffic_scheduler.get_in_flight_request_ids()
                remaining = pre_timeout_request_ids & current_in_flight

                # Check grace period if configured
                grace_elapsed = time.monotonic() - timeout_start
                if (
                    post_timeout_grace_seconds >= 0
                    and grace_elapsed >= post_timeout_grace_seconds
                ):
                    logger.warning(
                        f"Grace period of {post_timeout_grace_seconds}s expired. "
                        f"Force-exiting with {len(remaining)} requests still in-flight."
                    )
                    # Only include completed requests in metrics
                    completed_requests = pre_timeout_request_ids - remaining
                    evaluator.set_included_requests(completed_requests)
                    in_flight_remaining = remaining
                    break

                if not remaining:
                    logger.info("All pre-timeout requests completed")
                    evaluator.set_included_requests(pre_timeout_request_ids)
                    in_flight_remaining = set()
                    break
            elif all_prefetch_done and not traffic_scheduler.has_pending_work():
                logger.info("All sessions completed")
                in_flight_remaining = set()
                break
    finally:
        pbar.close()

    return in_flight_remaining
