"""Regression tests pinning EnglishTextNormalizer outputs.

The expected strings were produced by the original (pre-optimization)
open-asr-leaderboard normalizer implementation; the optimized version must
reproduce them exactly so WER and interactivity metrics stay numerically
identical across runs.
"""

import pytest

from veeksha.evaluator.performance.asr_normalizer import EnglishTextNormalizer
from veeksha.evaluator.performance.asr_normalizer.normalizer import (
    remove_symbols,
    remove_symbols_and_diacritics,
)

_PINNED_CASES = [
    (
        "Mr. O'Brien's café costs $20 million and ¢7, twenty-one and a half mhm",
        "mister 0 brien is cafe costs $20000000 and ¢721.5",
    ),
    (
        "one oh one double seven point five percent 1,234.56 3rd 1960s",
        "10177.5% 1234.56 3rd 1960s",
    ),
    (
        "won't can't y'all's it's don't we're I'ma [noise] (laughs) <unk>",
        "will not can not you all is it is do not we are i am going to",
    ),
    (
        "minus five degrees plus positive £3 euros œæßð ²",
        "-5 degrees +positive €3 oeaessd 2",
    ),
    (
        "one hundred and twenty three thousand four hundred fifty six",
        "123456",
    ),
    (
        "two and a half percent of $0.5 st dr colour flavour theatre",
        "2.5% of ¢5 saint doctor color flavor theater",
    ),
    (
        "The DOCTOR said:  it's   ten thirty--five, o'clock!",
        "the doctor said it is 10350 clock",
    ),
]


@pytest.mark.unit
class TestEnglishTextNormalizerRegression:
    @pytest.mark.parametrize("raw,expected", _PINNED_CASES)
    def test_pinned_outputs(self, raw: str, expected: str) -> None:
        assert EnglishTextNormalizer()(raw) == expected

    def test_empty_and_whitespace_inputs(self) -> None:
        normalizer = EnglishTextNormalizer()
        assert normalizer("") == ""
        assert normalizer("   ") == ""

    def test_normalizer_is_deterministic_across_calls(self) -> None:
        normalizer = EnglishTextNormalizer()
        raw = _PINNED_CASES[0][0]
        assert normalizer(raw) == normalizer(raw)


@pytest.mark.unit
class TestSymbolRemoval:
    def test_remove_symbols_and_diacritics_keep_set(self) -> None:
        assert (
            remove_symbols_and_diacritics("café $5.0, 100%!", keep=".%$¢€£")
            == "cafe $5.0  100% "
        )

    def test_remove_symbols_and_diacritics_default_keep(self) -> None:
        assert remove_symbols_and_diacritics("œÆß naïve!") == "oeAEss naive "

    def test_remove_symbols_keeps_diacritics(self) -> None:
        assert remove_symbols("naïve, test!") == "naïve  test "
