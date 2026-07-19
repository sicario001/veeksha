"""Microbenchmark configuration with inheritance."""

from enum import StrEnum

from vidhi import BasePolyConfig, field, frozen_dataclass

from veeksha.cli.base import VeekshaCommand
from veeksha.config.wandb import WandbConfig


class StressTrafficMode(StrEnum):
    """Traffic pattern for stress microbenchmarks."""

    FIXED_CLIENTS = "fixed-clients"
    FIXED_RATE = "fixed-rate"


class StressModeType(StrEnum):
    """Stress mode type discriminator."""

    MANUAL = "manual"
    RANGE = "range"
    AUTO = "auto"


@frozen_dataclass
class BaseMicrobenchmarkConfig:
    """Shared fields for all microbenchmark types."""

    model: str = field("meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
    api_base: str = field("http://localhost:8000/v1", help="API base URL")
    api_key: str = field("mock", help="API key")
    input_lengths: list[int] = field(
        default_factory=lambda: [128, 256, 512, 1024],
        help="Input lengths for benchmarks",
    )
    samples_per_length: int = field(10, help="Number of samples per input length")
    output_dir: str = field(
        "microbench_output", help="Output directory for benchmark results"
    )
    seed: int = field(42, help="Random seed")
    request_timeout: int = field(120, help="Request timeout in seconds")
    benchmark_timeout: int = field(600, help="Benchmark timeout in seconds")
    post_timeout_grace_seconds: int = field(
        30,
        help="Grace period (s) for in-flight requests after benchmark window ends. -1 waits for all, 0 exits immediately.",
    )
    max_tokens_param: str = field("max_tokens", help="Parameter name for max tokens")
    ignore_eos: bool = field(True, help="Ignore EOS token")
    validate_only: bool = field(
        False, help="Skip benchmark, only validate existing output"
    )
    skip_validation: bool = field(False, help="Skip post-run validation")
    wandb: WandbConfig = field(
        default_factory=WandbConfig,
        help="Weights & Biases logging configuration.",
    )


@frozen_dataclass
class PrefillMicrobenchmarkConfig(
    BaseMicrobenchmarkConfig, VeekshaCommand, name="prefill"
):
    """Run prefill (prompt processing) microbenchmark."""

    output_tokens: int = field(1, help="Output tokens per request")

    def __post_init__(self) -> None:
        if not self.input_lengths:
            raise ValueError("input_lengths must be non-empty")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if self.samples_per_length <= 0:
            raise ValueError("samples_per_length must be positive")


@frozen_dataclass
class DecodeMicrobenchmarkConfig(
    BaseMicrobenchmarkConfig, VeekshaCommand, name="decode"
):
    """Run decode (token generation) microbenchmark."""

    batch_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8],
        help="Batch sizes for decode benchmarks",
    )
    engine_chunk_size: int = field(512, help="Engine chunk size")

    def __post_init__(self) -> None:
        if not self.input_lengths:
            raise ValueError("input_lengths must be non-empty")
        if not self.batch_sizes:
            raise ValueError("batch_sizes must be non-empty")
        if self.samples_per_length <= 0:
            raise ValueError("samples_per_length must be positive")
        if self.engine_chunk_size <= 0:
            raise ValueError("engine_chunk_size must be positive")
        for bs in self.batch_sizes:
            if bs >= self.engine_chunk_size:
                raise ValueError(
                    f"batch_size {bs} must be less than engine_chunk_size {self.engine_chunk_size}"
                )


# ---------------------------------------------------------------------------
# Stress microbenchmark configs (polymorphic by mode)
# ---------------------------------------------------------------------------


def _validate_stress_base(cfg: "StressMicrobenchmarkConfig") -> None:
    """Shared validation for all stress config variants."""
    if cfg.input_length <= 0:
        raise ValueError("input_length must be positive")
    if cfg.output_length <= 0:
        raise ValueError("output_length must be positive")
    if cfg.point_duration <= cfg.warmup_duration:
        raise ValueError("point_duration must exceed warmup_duration")


@frozen_dataclass
class BaseStressModeConfig(BasePolyConfig):
    """Base class for stress mode variants."""


@frozen_dataclass
class ManualStressModeConfig(BaseStressModeConfig):
    """Explicit concurrency levels."""

    concurrency_levels: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32],
        help="Concurrency levels to test",
    )

    @classmethod
    def get_type(cls) -> StressModeType:
        return StressModeType.MANUAL


@frozen_dataclass
class RangeStressModeConfig(BaseStressModeConfig):
    """Log-spaced concurrency range."""

    concurrency_min: int = field(1, help="Minimum concurrency level")
    concurrency_max: int = field(64, help="Maximum concurrency level")
    concurrency_points: int = field(8, help="Number of log-spaced points to test")

    @classmethod
    def get_type(cls) -> StressModeType:
        return StressModeType.RANGE


@frozen_dataclass
class AutoStressModeConfig(BaseStressModeConfig):
    """Automatic concurrency discovery."""

    auto_throughput_threshold: float = field(
        0.05, help="Stop probing when throughput gain < this fraction"
    )
    auto_max_probes: int = field(20, help="Maximum number of probe points")
    auto_fill_points: int = field(
        8, help="Number of fill points between lower and upper bounds"
    )
    resume_dir: str = field("", help="Resume from a previous run directory")

    @classmethod
    def get_type(cls) -> StressModeType:
        return StressModeType.AUTO


@frozen_dataclass
class StressMicrobenchmarkConfig(
    BaseMicrobenchmarkConfig, VeekshaCommand, name="stress"
):
    """Run stress (throughput-vs-latency) microbenchmark."""

    input_length: int = field(512, help="Input token length (single value)")
    output_length: int = field(128, help="Output token length (single value)")
    point_duration: int = field(120, help="Seconds to run each concurrency point")
    warmup_duration: int = field(10, help="Warmup seconds to discard per point")
    traffic_mode: StressTrafficMode = field(
        StressTrafficMode.FIXED_CLIENTS,
        help="Traffic pattern: 'fixed-clients' or 'fixed-rate'",
    )
    max_tokens_per_second_estimate: float = field(
        500.0, help="Estimated max output tok/s (for session budget)"
    )
    num_gpus: int = field(
        0, help="Number of GPUs serving the model (0 = unknown, TPS/GPU not computed)"
    )
    mode: BaseStressModeConfig = field(
        default_factory=ManualStressModeConfig,
        help="Stress mode configuration",
    )

    def __post_init__(self) -> None:
        _validate_stress_base(self)
        if isinstance(self.mode, ManualStressModeConfig):
            if not self.mode.concurrency_levels:
                raise ValueError("concurrency_levels must be non-empty")
            if any(c <= 0 for c in self.mode.concurrency_levels):
                raise ValueError("all concurrency_levels must be positive")
        elif isinstance(self.mode, RangeStressModeConfig):
            if self.mode.concurrency_min >= self.mode.concurrency_max:
                raise ValueError("concurrency_min must be less than concurrency_max")
            if self.mode.concurrency_points <= 0:
                raise ValueError("concurrency_points must be positive")
        elif isinstance(self.mode, AutoStressModeConfig):
            if self.mode.auto_max_probes <= 0:
                raise ValueError("auto_max_probes must be positive")
            if self.mode.auto_fill_points <= 0:
                raise ValueError("auto_fill_points must be positive")


# Backwards compat aliases used by stress.py
ManualStressConfig = ManualStressModeConfig
RangeStressConfig = RangeStressModeConfig
AutoStressConfig = AutoStressModeConfig

STRESS_MODE_TO_CONFIG: dict[StressModeType, type[BaseStressModeConfig]] = {
    StressModeType.MANUAL: ManualStressModeConfig,
    StressModeType.RANGE: RangeStressModeConfig,
    StressModeType.AUTO: AutoStressModeConfig,
}
