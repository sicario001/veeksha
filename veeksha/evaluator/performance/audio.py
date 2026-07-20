"""Audio-channel performance evaluator (skeleton)."""

from typing import Optional

from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.evaluator.performance.channel_base import (
    NotImplementedChannelPerformanceEvaluator,
)
from veeksha.types import ChannelModality


class AudioPerformanceEvaluator(NotImplementedChannelPerformanceEvaluator):
    modality = ChannelModality.AUDIO
    evaluator_type = "audio_performance"

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: Optional[AudioChannelPerformanceConfig] = None,
        benchmark_start_time: float = 0.0,
    ):
        super().__init__(
            config,
            channel_config or AudioChannelPerformanceConfig(),
            benchmark_start_time=benchmark_start_time,
        )
