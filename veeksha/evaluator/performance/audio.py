"""Audio-channel performance evaluator (skeleton).

The real audio metrics (TTFA / RTF / interactivity) live on the voice branches
(see analysis/02_asr_tts_branches.md) and will be ported onto
``BaseChannelPerformanceEvaluator``. Until then this is a shared no-op skeleton.
"""

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
    ):
        super().__init__(config, channel_config or AudioChannelPerformanceConfig())
