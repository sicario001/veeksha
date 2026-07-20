"""Video-channel performance evaluator (skeleton)."""

from typing import Optional

from veeksha.config.evaluator import (
    PerformanceEvaluatorConfig,
    VideoChannelPerformanceConfig,
)
from veeksha.evaluator.performance.channel_base import (
    NotImplementedChannelPerformanceEvaluator,
)
from veeksha.types import ChannelModality


class VideoPerformanceEvaluator(NotImplementedChannelPerformanceEvaluator):
    modality = ChannelModality.VIDEO
    evaluator_type = "video_performance"

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: Optional[VideoChannelPerformanceConfig] = None,
        benchmark_start_time: float = 0.0,
    ):
        super().__init__(
            config,
            channel_config or VideoChannelPerformanceConfig(),
            benchmark_start_time=benchmark_start_time,
        )
