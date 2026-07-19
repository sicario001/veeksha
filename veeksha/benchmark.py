import os
import threading
import time
from dataclasses import replace
from queue import Queue
from typing import Optional, Set

from veeksha.benchmark_utils import (
    _init_output_dir,
    _monitor_for_completion,
    build_evaluator,
    maybe_run_warmup,
    maybe_warn_client_thread_sizing,
)
from veeksha.client.registry import ClientRegistry
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.thread_pool import ThreadPoolManager
from veeksha.core.tokenizer import (
    TokenizerProvider,
    build_hf_tokenizer_handle_from_model,
)
from veeksha.core.trace_recorder import TraceRecorder
from veeksha.generator.session.registry import SessionGeneratorRegistry
from veeksha.health import HealthChecker
from veeksha.logger import init_logger
from veeksha.orchestration import managed_server
from veeksha.traffic.registry import TrafficSchedulerRegistry
from veeksha.types import ChannelModality
from veeksha.wandb_integration import (
    maybe_finish_wandb_run,
    maybe_init_wandb_run,
    maybe_log_benchmark_artifacts,
    maybe_log_benchmark_scalars,
)
from veeksha.workers import CompletionWorker, DispatchWorker, PrefetchWorker
from veeksha.workers.client_runner import ClientRunnerManager
from veeksha.workers.prefetch import SharedSessionCounter

logger = init_logger(__name__)


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


def _maybe_run_native(
    benchmark_config, evaluator, session_generator, pregenerated_sessions
):
    """Run the batch over the native (C++) transport when the client opts in.

    Returns the finalized EvaluationResult, or None to fall back to the Python
    worker pipeline. Native owns connection concurrency + kernel-time timing, so
    this path removes the Python per-event overhead. It needs a bounded run
    (max_sessions > 0) and handles independent (single-request) sessions; the
    Python pipeline still serves rate-based / multi-turn-dependent traffic.
    """
    from veeksha.native.runner import feed_native_results, should_use_native

    client_config = benchmark_config.client
    if not should_use_native(client_config):
        return None
    max_sessions = benchmark_config.runtime.max_sessions
    if max_sessions <= 0:
        logger.warning(
            "use_native_transport is set but max_sessions <= 0; the native path "
            "needs a bounded run. Falling back to the Python pipeline."
        )
        return None

    sessions = pregenerated_sessions
    if sessions is None:
        sessions = []
        for _ in range(max_sessions):
            try:
                sessions.append(session_generator.generate_session())
            except StopIteration:
                break
    requests = [req for s in sessions for req in s.requests.values()]
    concurrency = getattr(
        benchmark_config.traffic_scheduler, "target_concurrent_sessions", 0
    ) or min(len(requests), 64)
    logger.info(
        "Native transport: %d requests at concurrency %d (%s)",
        len(requests),
        concurrency,
        client_config.get_type(),
    )
    feed_native_results(requests, evaluator, client_config, concurrency)
    return evaluator.finalize()


def _run_main_loop(
    session_generator,
    traffic_scheduler,
    evaluator,
    client,
    runtime_config,
    trace_recorder=None,
    benchmark_start_time: Optional[float] = None,
    pregenerated_sessions: Optional[list] = None,
) -> None:
    """Run the main benchmark loop with all workers."""
    logger.info("Starting main loop")
    if benchmark_start_time is None:
        benchmark_start_time = time.monotonic()

    client_queues = [Queue() for _ in range(runtime_config.num_client_threads)]
    output_queue = Queue()
    stop_event = threading.Event()
    generator_lock = threading.Lock()

    session_counter = SharedSessionCounter(max_sessions=runtime_config.max_sessions)

    client_runner = ClientRunnerManager(
        client=client,
        input_queues=client_queues,
        output_queue=output_queue,
        stop_event=stop_event,
        traffic_scheduler=traffic_scheduler,
    )

    pool_manager = ThreadPoolManager(stop_event=stop_event)

    pool_manager.create_pool(
        name="prefetch",
        worker_class=PrefetchWorker,
        worker_kwargs={
            "traffic_scheduler": traffic_scheduler,
            "session_generator": session_generator,
            "generator_lock": generator_lock,
            "session_counter": session_counter,
            "pregenerated_sessions": pregenerated_sessions,
        },
        pool_size=1,
    )

    pool_manager.create_pool(
        name="dispatch",
        worker_class=DispatchWorker,
        worker_kwargs={
            "traffic_scheduler": traffic_scheduler,
            "client_queues": client_queues,
            "evaluator": evaluator,
            "trace_recorder": trace_recorder,
        },
        pool_size=runtime_config.num_dispatcher_threads,
    )

    pool_manager.create_pool(
        name="completion",
        worker_class=CompletionWorker,
        worker_kwargs={
            "output_queue": output_queue,
            "traffic_scheduler": traffic_scheduler,
            "evaluator": evaluator,
        },
        pool_size=runtime_config.num_completion_threads,
    )

    if trace_recorder:
        trace_recorder.start()

    client_runner.start()
    pool_manager.start_all()

    logger.info(
        f"Started {pool_manager.get_total_thread_count()} worker threads "
        f"and {client_runner.get_worker_count()} client workers"
    )

    benchmark_start = benchmark_start_time
    benchmark_timeout = runtime_config.benchmark_timeout
    timeout_triggered = False
    pre_timeout_request_ids: Set[str] = set()

    try:
        pending_in_flight = _monitor_for_completion(
            traffic_scheduler,
            evaluator,
            pool_manager,
            benchmark_start,
            benchmark_timeout,
            timeout_triggered,
            pre_timeout_request_ids,
            max_sessions=runtime_config.max_sessions,
            post_timeout_grace_seconds=runtime_config.post_timeout_grace_seconds,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted, stopping")
        pending_in_flight = set()

    stop_event.set()
    pool_manager.join_pool("prefetch", timeout=1.0)
    pool_manager.join_pool("dispatch", timeout=1.0)

    if trace_recorder:
        trace_recorder.stop()

    logger.info("Stopping client runner...")
    client_runner.stop()
    if not pending_in_flight:
        client_runner.wait()

    for _ in range(runtime_config.num_completion_threads):
        output_queue.put(None)
    pool_manager.join_pool("completion", timeout=1.0)


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

    # Warn early if the client-thread count is undersized for the target concurrency.
    maybe_warn_client_thread_sizing(benchmark_config)

    # get session generator
    model_name = benchmark_config.client.model
    # Audio clients (TTS/STT/realtime) supply their own tokenizer (a word-split
    # provider) since their models ship no HuggingFace tokenizer; text clients use
    # the HF tokenizer for the model.
    if hasattr(benchmark_config.client, "build_tokenizer_provider"):
        tokenizer_provider = benchmark_config.client.build_tokenizer_provider()
    else:
        tokenizer_provider = TokenizerProvider(
            {ChannelModality.TEXT: build_hf_tokenizer_handle_from_model(model_name)},
            model_name=model_name,
        )
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

    # get traffic scheduler, client
    traffic_scheduler = TrafficSchedulerRegistry.get(
        benchmark_config.traffic_scheduler.get_type(),
        config=benchmark_config.traffic_scheduler,
        seed_manager=seed_manager,
    )

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
    traffic_scheduler.reset_reference_time()

    # get evaluator
    evaluator = build_evaluator(
        benchmark_config,
        seed_manager=seed_manager,
        session_generator=session_generator,
        benchmark_start_time=benchmark_start_time,
    )

    # Native transport fast path: when the client opts in (and the endpoint is
    # plaintext + the run is bounded), run the batch over the C++ engine instead
    # of the Python worker pipeline, then finalize/save the same way.
    native_result = _maybe_run_native(
        benchmark_config, evaluator, session_generator, pregenerated_sessions
    )
    if native_result is not None:
        os.makedirs(f"{benchmark_config.output_dir}/metrics", exist_ok=True)
        evaluator.save(f"{benchmark_config.output_dir}/metrics")
        logger.info("Native transport run complete.")
        return native_result

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

    try:
        _run_main_loop(
            session_generator=session_generator,
            traffic_scheduler=traffic_scheduler,
            evaluator=evaluator,
            client=client,
            runtime_config=benchmark_config.runtime,
            trace_recorder=trace_recorder,
            benchmark_start_time=benchmark_start_time,
            pregenerated_sessions=pregenerated_sessions,
        )
    finally:
        if trace_recorder:
            trace_recorder.stop()

    logger.info("Finalizing evaluator...")
    # finalize and save results
    result = evaluator.finalize()

    evaluator.save(f"{benchmark_config.output_dir}/metrics")

    # health checks
    logger.info("Running health checks...")
    health_checker = HealthChecker(
        trace_file=f"{benchmark_config.output_dir}/traces/dispatch_trace.jsonl",
        metrics_file=f"{benchmark_config.output_dir}/metrics/request_level_metrics.jsonl",
        benchmark_config=benchmark_config,
    )
    health_checker.run_and_save(
        f"{benchmark_config.output_dir}/health_check_results.txt"
    )

    return result


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

        with managed_server(
            benchmark_config.server, output_dir=benchmark_config.output_dir
        ) as server_info:
            logger.info(f"Server ready at {server_info['api_base']}")

            # server dictates client
            updated_client_config = replace(
                benchmark_config.client,
                api_base=server_info["api_base"],
                api_key=server_info["api_key"],
                model=benchmark_config.server.model,
            )
            updated_benchmark_config = replace(
                benchmark_config,
                client=updated_client_config,
                server=None,
            )

            maybe_init_wandb_run(updated_benchmark_config, run_kind="benchmark")
            try:
                result = _run_benchmark(updated_benchmark_config)
                maybe_log_benchmark_scalars(updated_benchmark_config.output_dir)
                maybe_log_benchmark_artifacts(updated_benchmark_config)
                return result
            finally:
                maybe_finish_wandb_run(updated_benchmark_config.output_dir)
                logger.info("Server shutting down...")
    else:
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
