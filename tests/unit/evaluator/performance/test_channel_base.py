"""Tests for the channel-evaluator ABC and the collapsed skeleton stubs."""

from __future__ import annotations

import pytest

from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.evaluator.performance.audio import AudioPerformanceEvaluator
from veeksha.evaluator.performance.channel_base import (
    BaseChannelPerformanceEvaluator,
    NotImplementedChannelPerformanceEvaluator,
)
from veeksha.evaluator.performance.image import ImagePerformanceEvaluator
from veeksha.evaluator.performance.text import TextPerformanceEvaluator
from veeksha.evaluator.performance.video import VideoPerformanceEvaluator
from veeksha.types import ChannelModality

STUBS = [
    (AudioPerformanceEvaluator, ChannelModality.AUDIO),
    (ImagePerformanceEvaluator, ChannelModality.IMAGE),
    (VideoPerformanceEvaluator, ChannelModality.VIDEO),
]


def test_all_channel_evaluators_conform_to_base():
    for cls in (
        TextPerformanceEvaluator,
        AudioPerformanceEvaluator,
        ImagePerformanceEvaluator,
        VideoPerformanceEvaluator,
    ):
        assert issubclass(cls, BaseChannelPerformanceEvaluator)


def test_text_evaluator_is_instantiable_under_abc():
    # If any abstract method were missing, this would raise TypeError.
    ev = TextPerformanceEvaluator(PerformanceEvaluatorConfig())
    assert isinstance(ev, BaseChannelPerformanceEvaluator)


@pytest.mark.parametrize("cls,modality", STUBS)
def test_skeleton_stub_finalizes_not_implemented(cls, modality):
    ev = cls(PerformanceEvaluatorConfig())
    assert isinstance(ev, NotImplementedChannelPerformanceEvaluator)
    # no-ops must not raise
    ev.register_request(0, 0, 0.0, None, None)
    ev.record_request_completed(0, 0, 1.0, None)
    ev.record_session_completed(0, 1, 0.0, 1.0)
    result = ev.finalize()
    assert result.channel == modality
    assert result.metrics["status"] == "not_implemented"
    assert ev.get_streaming_metrics() is None


def test_base_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseChannelPerformanceEvaluator()  # abstract
