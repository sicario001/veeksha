"""Base class for per-modality (channel) performance evaluators.

`PerformanceEvaluator` (evaluator/performance/base.py) composes one channel
evaluator per target modality and fans out the request/session lifecycle to it.
Historically these channel evaluators shared no base class — `text.py` was a full
implementation and `audio/image/video.py` were near-verbatim copy-pasted stubs.
This ABC formalizes the protocol (doc `analysis/05_proposed_abstraction.md` §3) so
new modalities are additive, and `NotImplementedChannelPerformanceEvaluator`
collapses the three stubs into one place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from veeksha.evaluator.base import EvaluationResult
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

logger = init_logger(__name__)


class BaseChannelPerformanceEvaluator(ABC):
    """Protocol every per-modality performance evaluator implements."""

    @abstractmethod
    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        content: Any,
        requested_output: Any = None,
    ) -> None:
        """Record that a request for this channel was dispatched."""
        raise NotImplementedError

    @abstractmethod
    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
    ) -> None:
        """Record that a request for this channel completed."""
        raise NotImplementedError

    @abstractmethod
    def record_session_completed(
        self,
        session_id: int,
        session_size: int,
        first_dispatch_at: Optional[float],
        last_completion_at: Optional[float],
    ) -> None:
        """Record session-level metrics for this channel."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> EvaluationResult:
        """Finalize and return this channel's results."""
        raise NotImplementedError

    # --- optional hooks; most modalities can use these defaults --------------
    def get_streaming_metrics(self) -> Optional[Dict[str, Any]]:
        return None

    def save(self, output_dir: str) -> None:
        return None

    def flush_streaming_outputs(self, output_dir: str) -> None:
        return None


class NotImplementedChannelPerformanceEvaluator(BaseChannelPerformanceEvaluator):
    """Shared skeleton for modalities whose metrics are not implemented yet.

    Subclasses set ``modality`` and ``evaluator_type``. Collapses the previously
    duplicated audio/image/video stubs into one implementation.
    """

    modality: ChannelModality
    evaluator_type: str

    def __init__(self, config: Any, channel_config: Any = None) -> None:
        self.config = config
        self.channel_config = channel_config
        logger.warning(
            "%s is a skeleton implementation; %s metrics are not yet supported.",
            type(self).__name__,
            self.modality.name.lower(),
        )

    def register_request(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_request_completed(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_session_completed(self, *args: Any, **kwargs: Any) -> None:
        return None

    def finalize(self) -> EvaluationResult:
        name = self.modality.name.title()
        return EvaluationResult(
            evaluator_type=self.evaluator_type,
            channel=self.modality,
            metrics={
                "status": "not_implemented",
                "message": f"{name} performance evaluation not yet implemented",
            },
        )
