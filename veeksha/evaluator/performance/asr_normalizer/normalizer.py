# Adapted from https://github.com/huggingface/open_asr_leaderboard
# (normalizer/normalizer.py), the standard text normalizer used to score the
# Open ASR Leaderboard. Vendored verbatim so WER matches the leaderboard.
#
# Copyright 2022 The OpenAI team and The HuggingFace Team. All rights reserved.
# Most of the code is copy pasted from the original whisper repository
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union

import regex

from .english_abbreviations import english_spelling_normalizer

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


class _SymbolReplacementTable(dict):
    """Lazy per-codepoint replacement table for ``str.translate``.

    Computes the same per-character replacement as the original character
    loop and memoizes it, so repeated normalizer calls classify each distinct
    codepoint only once.
    """

    def __init__(self, keep: str):
        super().__init__()
        self._keep = keep

    def __missing__(self, codepoint: int) -> str:
        char = chr(codepoint)
        if char in self._keep:
            replacement = char
        elif char in ADDITIONAL_DIACRITICS:
            replacement = ADDITIONAL_DIACRITICS[char]
        elif unicodedata.category(char) == "Mn":
            replacement = ""
        elif unicodedata.category(char)[0] in "MSP":
            replacement = " "
        else:
            replacement = char
        self[codepoint] = replacement
        return replacement


_symbol_and_diacritic_tables: dict = {}


class _SymbolOnlyTable(dict):
    """Lazy per-codepoint table replacing 'MSP' categories with a space."""

    def __missing__(self, codepoint: int) -> str:
        char = chr(codepoint)
        replacement = " " if unicodedata.category(char)[0] in "MSP" else char
        self[codepoint] = replacement
        return replacement


_symbol_only_table = _SymbolOnlyTable()


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """
    table = _symbol_and_diacritic_tables.get(keep)
    if table is None:
        table = _symbol_and_diacritic_tables[keep] = _SymbolReplacementTable(keep)
    return unicodedata.normalize("NFKD", s).translate(table)


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return unicodedata.normalize("NFKC", s).translate(_symbol_only_table)


class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        return s


class BasicMultilingualTextNormalizer:
    def __init__(self, remove_diacritics: bool = True):
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        # Remove punctuations and extra spaces
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()

        return s


_NUMERIC_RE = re.compile(r"^\d+(\.\d+)?$")
_AND_A_HALF_RE = re.compile(r"\band\s+a\s+half\b")
_LETTER_DIGIT_RE = re.compile(r"([a-z])([0-9])")
_DIGIT_LETTER_RE = re.compile(r"([0-9])([a-z])")
_DIGIT_SUFFIX_SPACE_RE = re.compile(r"([0-9])\s+(st|nd|rd|th|s)\b")
_CURRENCY_CENTS_RE = re.compile(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b")
_ZERO_CENTS_RE = re.compile(r"[€£$]0.([0-9]{1,2})\b")
_ONE_READABILITY_RE = re.compile(r"\b1(s?)\b")
_BRACKET_CONTENT_RE = re.compile(r"[<\[][^>\]]*[>\]]")
_PAREN_CONTENT_RE = re.compile(r"\(([^)]+?)\)")
_SPACE_BEFORE_APOSTROPHE_RE = re.compile(r"\s+'")
_DIGIT_COMMA_RE = re.compile(r"(\d),(\d)")
_PERIOD_NON_DIGIT_RE = re.compile(r"\.([^0-9]|$)")
_SYMBOL_NON_DIGIT_RE = re.compile(r"[.$¢€£]([^0-9])")
_NON_DIGIT_PERCENT_RE = re.compile(r"([^0-9])%")
_WHITESPACE_RE = re.compile(r"\s+")


class EnglishNumberNormalizer:
    """
    Convert any spelled-out numbers into arabic numbers, while handling:

    - remove any commas
    - keep the suffixes such as: `1960s`, `274th`, `32nd`, etc.
    - spell out currency symbols after the number. e.g. `$20 million` -> `20000000 dollars`
    - spell out `one` and `ones`
    - interpret successive single-digit numbers as nominal: `one oh one` -> `101`
    """

    def __init__(self):
        super().__init__()

        self.zeros = {"o", "oh", "zero"}
        # fmt: off
        self.ones = {
            name: i
            for i, name in enumerate(
                ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"],
                start=1,
            )
        }
        # fmt: on
        self.ones_plural = {
            "sixes" if name == "six" else name + "s": (value, "s")
            for name, value in self.ones.items()
        }
        self.ones_ordinal = {
            "zeroth": (0, "th"),
            "first": (1, "st"),
            "second": (2, "nd"),
            "third": (3, "rd"),
            "fifth": (5, "th"),
            "twelfth": (12, "th"),
            **{
                name + ("h" if name.endswith("t") else "th"): (value, "th")
                for name, value in self.ones.items()
                if value > 3 and value != 5 and value != 12
            },
        }
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}

        self.tens = {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        self.tens_plural = {
            name.replace("y", "ies"): (value, "s") for name, value in self.tens.items()
        }
        self.tens_ordinal = {
            name.replace("y", "ieth"): (value, "th")
            for name, value in self.tens.items()
        }
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}

        self.multipliers = {
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
            "sextillion": 1_000_000_000_000_000_000_000,
            "septillion": 1_000_000_000_000_000_000_000_000,
            "octillion": 1_000_000_000_000_000_000_000_000_000,
            "nonillion": 1_000_000_000_000_000_000_000_000_000_000,
            "decillion": 1_000_000_000_000_000_000_000_000_000_000_000,
        }
        self.multipliers_plural = {
            name + "s": (value, "s") for name, value in self.multipliers.items()
        }
        self.multipliers_ordinal = {
            name + "th": (value, "th") for name, value in self.multipliers.items()
        }
        self.multipliers_suffixed = {
            **self.multipliers_plural,
            **self.multipliers_ordinal,
        }
        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "minus": "-",
            "negative": "-",
            "plus": "+",
            "positive": "+",
        }
        self.following_prefixers = {
            "pound": "£",
            "pounds": "£",
            "euro": "€",
            "euros": "€",
            "dollar": "$",
            "dollars": "$",
            "cent": "¢",
            "cents": "¢",
        }
        self.prefixes = set(
            list(self.preceding_prefixers.values())
            + list(self.following_prefixers.values())
        )
        self.suffixers = {
            "per": {"cent": "%"},
            "percent": "%",
        }
        self.specials = {"and", "double", "triple", "point"}

        self.words = {
            key
            for mapping in [
                self.zeros,
                self.ones,
                self.ones_suffixed,
                self.tens,
                self.tens_suffixed,
                self.multipliers,
                self.multipliers_suffixed,
                self.preceding_prefixers,
                self.following_prefixers,
                self.suffixers,
                self.specials,
            ]
            for key in mapping
        }
        self.literal_words = {"one", "ones"}

    def process_words(self, words: List[str]) -> Iterator[str]:
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        vocabulary = self.words
        prefixes = self.prefixes
        for i, current in enumerate(words):
            if skip:
                skip = False
                continue

            # Fast path: a word with no leading digit/currency sign that is not
            # part of the number vocabulary passes through unchanged whenever no
            # number is being accumulated. This mirrors the
            # `current not in self.words` branch below with `value`/`prefix`
            # both unset, where `output(current)` returns `current` verbatim.
            first_char = current[0]
            if (
                value is None
                and prefix is None
                and current not in vocabulary
                and not first_char.isdigit()
                and first_char not in prefixes
            ):
                yield current
                continue

            prev = words[i - 1] if i != 0 else None
            next = words[i + 1] if i != len(words) - 1 else None

            next_is_numeric = next is not None and _NUMERIC_RE.match(next)
            has_prefix = first_char in prefixes
            current_without_prefix = current[1:] if has_prefix else current
            if _NUMERIC_RE.match(current_without_prefix):
                # arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                if f is None:
                    raise ValueError("Converting the fraction failed")

                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        # concatenate decimals / ip address components
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator  # store integers as int
                else:
                    value = current_without_prefix
            elif current not in self.words:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if (
                        prev in self.tens and ones < 10
                    ):  # replace the last zero with the digit
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.ones_suffixed:
                # ordinal or cardinal; yield the number right away
                ones, suffix = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.tens_suffixed:
                # ordinal or cardinal; yield the number right away
                tens, suffix = self.tens_suffixed[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.multipliers_suffixed:
                multiplier, suffix = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:  # int
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                # apply prefix (positive, minus, etc.) if it precedes a number
                if value is not None:
                    yield output(value)

                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.following_prefixers:
                # apply prefix (dollars, cents, etc.) only after a number
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in self.suffixers:
                # apply suffix symbols (percent -> '%')
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if next not in self.words and not next_is_numeric:
                    # apply special handling only if the next word can be numeric
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "and":
                    # ignore "and" after hundreds, thousands, etc.
                    if prev not in self.multipliers:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "double" or current == "triple":
                    if next in self.ones or next in self.zeros:
                        repeats = 2 if current == "double" else 3
                        ones = self.ones.get(next, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "point":
                    if next in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    # should all have been covered at this point
                    raise ValueError(f"Unexpected token: {current}")
            else:
                # all should have been covered at this point
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        # replace "<number> and a half" with "<number> point five"
        results = []

        segments = _AND_A_HALF_RE.split(s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("point five")
                else:
                    results.append("and a half")

        s = " ".join(results)

        # put a space at number/letter boundary
        s = _LETTER_DIGIT_RE.sub(r"\1 \2", s)
        s = _DIGIT_LETTER_RE.sub(r"\1 \2", s)

        # but remove spaces which could be a suffix
        s = _DIGIT_SUFFIX_SPACE_RE.sub(r"\1\2", s)

        return s

    def postprocess(self, s: str):
        def combine_cents(m: Match):
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        # apply currency postprocessing; "$2 and ¢7" -> "$2.07"
        if "¢" in s:
            s = _CURRENCY_CENTS_RE.sub(combine_cents, s)
        if "€" in s or "£" in s or "$" in s:
            s = _ZERO_CENTS_RE.sub(extract_cents, s)

        # write "one(s)" instead of "1(s)", just for the readability
        if "1" in s:
            s = _ONE_READABILITY_RE.sub(r"one\1", s)

        return s

    def __call__(self, s: str):
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)

        return s


class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings as listed in [1].

    [1] https://www.tysto.com/uk-us-spelling-list.html
    """

    def __init__(self, english_spelling_mapping):
        self.mapping = english_spelling_mapping

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishTextNormalizer:
    def __init__(self, english_spelling_mapping=english_spelling_normalizer):
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            # common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # prefect tenses, ideally it should be any past participles, but it's harder..
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",  # "'s done" is ambiguous
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer(english_spelling_mapping)

        self._ignore_patterns_re = re.compile(self.ignore_patterns)
        # Every replacer pattern is a literal decorated only with zero-width
        # \b anchors, so any match must contain the bare literal as a
        # substring. A C-speed containment check therefore lets us skip the
        # regex pass entirely when it could not match (sub with no match is
        # the identity).
        self._compiled_replacers = [
            (pattern.replace("\\b", ""), re.compile(pattern), replacement)
            for pattern, replacement in self.replacers.items()
        ]

    def __call__(self, s: str):
        s = s.lower()

        if "[" in s or "<" in s:
            s = _BRACKET_CONTENT_RE.sub("", s)  # remove words between brackets
        if "(" in s:
            s = _PAREN_CONTENT_RE.sub("", s)  # remove words between parenthesis
        s = self._ignore_patterns_re.sub("", s)
        if "'" in s:
            s = _SPACE_BEFORE_APOSTROPHE_RE.sub(
                "'", s
            )  # standardize when there's a space before an apostrophe

        for literal, pattern, replacement in self._compiled_replacers:
            if literal in s:
                s = pattern.sub(replacement, s)

        if "," in s:
            s = _DIGIT_COMMA_RE.sub(r"\1\2", s)  # remove commas between digits
        if "." in s:
            s = _PERIOD_NON_DIGIT_RE.sub(
                r" \1", s
            )  # remove periods not followed by numbers
        s = remove_symbols_and_diacritics(
            s, keep=".%$¢€£"
        )  # keep some symbols for numerics

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = _SYMBOL_NON_DIGIT_RE.sub(r" \1", s)
        s = _NON_DIGIT_PERCENT_RE.sub(r"\1 ", s)

        s = _WHITESPACE_RE.sub(
            " ", s
        )  # replace any successive whitespace characters with a space

        return s
