"""Vendored Open ASR Leaderboard text normalizer.

Adapted from https://github.com/huggingface/open_asr_leaderboard so transcript
WER is scored the same way the leaderboard scores it.
"""

from functools import lru_cache

from veeksha.evaluator.performance.asr_normalizer.normalizer import (
    EnglishTextNormalizer,
)


@lru_cache(maxsize=1)
def get_english_text_normalizer() -> EnglishTextNormalizer:
    """Shared `EnglishTextNormalizer` instance (construction is not free)."""
    return EnglishTextNormalizer()


__all__ = ["EnglishTextNormalizer", "get_english_text_normalizer"]
