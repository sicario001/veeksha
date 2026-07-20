"""Image-channel performance evaluator (skeleton)."""

from typing import Optional

from veeksha.config.evaluator import (
    ImageChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.evaluator.performance.channel_base import (
    NotImplementedChannelPerformanceEvaluator,
)
from veeksha.types import ChannelModality


class ImagePerformanceEvaluator(NotImplementedChannelPerformanceEvaluator):
    modality = ChannelModality.IMAGE
    evaluator_type = "image_performance"

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: Optional[ImageChannelPerformanceConfig] = None,
        benchmark_start_time: float = 0.0,
    ):
        super().__init__(
            config,
            channel_config or ImageChannelPerformanceConfig(),
            benchmark_start_time=benchmark_start_time,
        )
