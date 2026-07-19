"""Vendored Open ASR Leaderboard text normalizer.

Adapted from https://github.com/huggingface/open_asr_leaderboard so transcript
WER is scored the same way the leaderboard scores it.
"""

from veeksha.evaluator.performance.asr_normalizer.normalizer import (
    EnglishTextNormalizer,
)

__all__ = ["EnglishTextNormalizer"]
