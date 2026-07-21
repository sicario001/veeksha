import os
import sys
import sysconfig
import time
from dataclasses import replace
from typing import Optional, Set

from veeksha.benchmark_utils import (
    _init_output_dir,
    _monitor_for_completion,
    build_evaluator,
    maybe_run_warmup,
)
from veeksha.client.registry import ClientRegistry
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.endpoint import EndpointConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.trace_recorder import TraceRecorder
from veeksha.loop import (
    GeneratorSessionSource,
    MainLoopConfig,
    PregeneratedSessionSource,
    ResultDrain,
    create_main_loop,
)
from veeksha.generator.session.registry import SessionGeneratorRegistry
from veeksha.health import HealthChecker
from veeksha.logger import init_logger
from veeksha.orchestration.benchmark_orchestrator import managed_server
from veeksha.wandb_integration import (
    maybe_finish_wandb_run,
    maybe_init_wandb_run,
    maybe_log_benchmark_artifacts,
    maybe_log_benchmark_scalars,
)

logger = init_logger(__name__)


def _warn_if_gil_enabled(stage: str) -> None:
    """Warn when the GIL is active on a free-threaded build.

    A C extension that does not declare free-threading support re-enables the
    GIL at import time unless the process runs with ``-Xgil=0`` /
    ``PYTHON_GIL=0``. This can happen mid-run (e.g. the first
    ``librosa.load`` lazily imports ``msgpack``), silently serializing every
    client worker thread and invalidating high-concurrency measurements.
    """
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return
    if sys._is_gil_enabled():
        logger.warning(
            "The GIL is enabled at %s on a free-threaded Python build; "
            "client worker threads serialize on it. Launch with -Xgil=0 or "
            "PYTHON_GIL=0 to keep it disabled.",
            stage,
        )


def _maybe_pregenerate_sessions(benchmark_config, session_generator) -> Optional[list]:
    """Pre-generate sessions when enabled in runtime config."""
    if not (
        benchmark_config.runtime.pregenerate_sessions
        and benchmark_config.runtime.max_sessions > 0
    ):
        return None

    logger.info("Pre-generating %d sessions...", benchmark_config.runtime.max_sessions)
    pregenerated_sessions = []
    for _ in range(benchmark_config.runtime.max_sessions):
        try:
            session = session_generator.generate_session()
            pregenerated_sessions.append(session)
        except StopIteration:
            logger.warning(
                "Session generator exhausted at %d sessions",
                len(pregenerated_sessions),
            )
            break
    logger.info(
        "Pre-generation complete: %d sessions ready", len(pregenerated_sessions)
    )
    return pregenerated_sessions


def _run_main_loop(
    loop,
    result_drain: ResultDrain,
    source,
    evaluator,
    runtime_config,
    benchmark_start_time: float,
) -> None:
    """Run the main loop with the scoring drain and completion monitor."""
    logger.info("Starting main loop")
    _warn_if_gil_enabled("benchmark start")

    loop.start(source)
    result_drain.start()

    try:
        pending_in_flight: Set[int] = _monitor_for_completion(
            loop,
            evaluator,
            benchmark_start_time,
            runtime_config.benchmark_timeout,
            max_sessions=runtime_config.max_sessions,
            post_timeout_grace_seconds=runtime_config.post_timeout_grace_seconds,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted, stopping")
        pending_in_flight = set()

    # The GIL can flip on mid-run via lazy extension imports; re-check so a
    # serialized run is at least loudly reported.
    _warn_if_gil_enabled("benchmark end")

    # grace_s < 0: nothing pending, drain in-flight client work in join();
    # grace_s >= 0: the monitor already spent the grace budget on the
    # still-in-flight requests, do not wait for them again.
    loop.request_stop(grace_s=-1.0 if not pending_in_flight else 0.0)
    loop.join(timeout_s=1.0)
    result_drain.join()


def _run_benchmark(
    benchmark_config: BenchmarkConfig,
):
    """Run the benchmark and return evaluation results.

    Args:
        benchmark_config: The benchmark configuration.

    Returns:
        EvaluationResult from the evaluator.
    """

    seed_manager = SeedManager(benchmark_config.seed)

    # get session generator
    tokenizer_provider = benchmark_config.client.build_tokenizer_provider()

    append_min_tokens_instruction = False
    if (
        hasattr(benchmark_config.client, "use_min_tokens_prompt_fallback")
        and benchmark_config.client.use_min_tokens_prompt_fallback  # type: ignore
    ):
        append_min_tokens_instruction = True
        logger.info(
            "Min tokens prompt fallback enabled in config. "
            "Will append instructions to prompts for minimum token control."
        )

    session_generator_kwargs = {
        "config": benchmark_config.session_generator,
        "seed_manager": seed_manager,
        "tokenizer_provider": tokenizer_provider,
    }

    # lm-eval uses runtime.max_sessions as the only sample-size knob.
    if (
        benchmark_config.session_generator.get_type()
        == SessionGeneratorRegistry.get_key_from_str("lmeval")
    ):
        session_generator_kwargs["max_sessions"] = benchmark_config.runtime.max_sessions

    if (
        benchmark_config.session_generator.get_type()
        == SessionGeneratorRegistry.get_key_from_str("synthetic")
    ):
        session_generator_kwargs["append_min_tokens_instruction"] = (
            append_min_tokens_instruction
        )

    session_generator = SessionGeneratorRegistry.get(
        benchmark_config.session_generator.get_type(),
        **session_generator_kwargs,
    )

    # get client (the traffic scheduler is built inside the main loop from
    # the traffic config)
    client = ClientRegistry.get(
        benchmark_config.client.get_type(),
        config=benchmark_config.client,
        tokenizer_provider=tokenizer_provider,
    )

    # some session generators might define a warmup phase
    maybe_run_warmup(session_generator, client)

    # Pre-generate all sessions if requested (before starting timer)
    pregenerated_sessions = _maybe_pregenerate_sessions(
        benchmark_config, session_generator
    )

    benchmark_start_time = time.monotonic()

    # get evaluator
    evaluator = build_evaluator(
        benchmark_config,
        seed_manager=seed_manager,
        session_generator=session_generator,
        benchmark_start_time=benchmark_start_time,
    )

    # trace recorder
    trace_recorder = None
    if benchmark_config.trace_recorder.enabled:
        # ensure output dirs exists for traces
        os.makedirs(f"{benchmark_config.output_dir}/traces", exist_ok=True)
        trace_recorder = TraceRecorder(
            f"{benchmark_config.output_dir}/traces",
            benchmark_start_time,
            benchmark_config.trace_recorder,
        )
        trace_recorder.start()

    os.makedirs(f"{benchmark_config.output_dir}/metrics", exist_ok=True)

    # session intake, main loop, and scoring drain
    if pregenerated_sessions is not None:
        source = PregeneratedSessionSource(pregenerated_sessions)
    else:
        source = GeneratorSessionSource(
            session_generator, max_sessions=benchmark_config.runtime.max_sessions
        )

    loop = create_main_loop(
        "python",
        MainLoopConfig(
            runtime=benchmark_config.runtime,
            traffic=benchmark_config.traffic_scheduler,
            client=benchmark_config.client,
            monotonic_anchor=benchmark_start_time,
        ),
        seed_manager=seed_manager,
        tokenizer_provider=tokenizer_provider,
        client=client,
    )
    result_drain = ResultDrain(
        loop,
        evaluator,
        trace_recorder=trace_recorder,
        num_threads=benchmark_config.runtime.num_completion_threads,
    )

    try:
        _run_main_loop(
            loop=loop,
            result_drain=result_drain,
            source=source,
            evaluator=evaluator,
            runtime_config=benchmark_config.runtime,
            benchmark_start_time=benchmark_start_time,
        )
    finally:
        if trace_recorder:
            trace_recorder.stop()

    logger.info("Finalizing evaluator...")
    # finalize and save results
    finalize_started_at = time.monotonic()
    result = evaluator.finalize()
    logger.info(
        "Benchmark phase 'evaluator_finalize' took %.2fs",
        time.monotonic() - finalize_started_at,
    )

    save_started_at = time.monotonic()
    evaluator.save(f"{benchmark_config.output_dir}/metrics")
    logger.info(
        "Benchmark phase 'evaluator_save' took %.2fs",
        time.monotonic() - save_started_at,
    )

    # health checks
    logger.info("Running health checks...")
    health_started_at = time.monotonic()
    health_checker = HealthChecker(
        trace_file=f"{benchmark_config.output_dir}/traces/dispatch_trace.jsonl",
        metrics_file=f"{benchmark_config.output_dir}/metrics/request_level_metrics.jsonl",
        benchmark_config=benchmark_config,
    )
    health_checker.run_and_save(
        f"{benchmark_config.output_dir}/health_check_results.txt"
    )
    logger.info(
        "Benchmark phase 'health_checks' took %.2fs",
        time.monotonic() - health_started_at,
    )

    return result


def _with_endpoint(
    benchmark_config: BenchmarkConfig, endpoint: EndpointConfig
) -> BenchmarkConfig:
    return replace(
        benchmark_config,
        client=endpoint.apply_to_client_config(benchmark_config.client),
        endpoint=endpoint,
        server=None,
    )


def manage_benchmark_run(
    benchmark_config: BenchmarkConfig,
):
    """Run a benchmark, handling optional server orchestration.

    1. If server config exists: spin up server, update client config, run benchmark
    2. If no server config: run benchmark directly

    Args:
        benchmark_config: The benchmark configuration.

    Returns:
        EvaluationResult from the evaluator.
    """
    logger.info("Running benchmark with config:\n%s", benchmark_config)

    _init_output_dir(benchmark_config)

    if benchmark_config.server is not None:
        logger.info(f"Launching {benchmark_config.server.engine} server...")
        updated_benchmark_config = None
        result = None
        try:
            with managed_server(
                benchmark_config.server, output_dir=benchmark_config.output_dir
            ) as server_info:
                endpoint = server_info["endpoint"]
                logger.info(f"Server ready at {endpoint.api_base}")

                updated_benchmark_config = _with_endpoint(benchmark_config, endpoint)

                maybe_init_wandb_run(updated_benchmark_config, run_kind="benchmark")
                try:
                    result = _run_benchmark(updated_benchmark_config)
                finally:
                    logger.info("Server shutting down...")

            maybe_log_benchmark_scalars(updated_benchmark_config.output_dir)
            maybe_log_benchmark_artifacts(updated_benchmark_config)
            return result
        finally:
            if updated_benchmark_config is not None:
                maybe_finish_wandb_run(updated_benchmark_config.output_dir)
    else:
        if benchmark_config.endpoint is not None:
            benchmark_config = _with_endpoint(
                benchmark_config, benchmark_config.endpoint
            )
        maybe_init_wandb_run(benchmark_config, run_kind="benchmark")
        try:
            result = _run_benchmark(benchmark_config)
            maybe_log_benchmark_scalars(benchmark_config.output_dir)
            maybe_log_benchmark_artifacts(benchmark_config)
            return result
        finally:
            maybe_finish_wandb_run(benchmark_config.output_dir)


if __name__ == "__main__":
    from veeksha.cli.benchmarks import main

    main()
