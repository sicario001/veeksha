"""Tests for the Seed-TTS text trace flavor (TTS input generator).

The generator's dataset loader depends on the optional ``datasets`` package,
which has no free-threaded wheel; those paths are gated behind importorskip.
Config validation and the pure text helpers are always exercised.
"""

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from veeksha.config.generator.session import (
    SeedTTSTextTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.generator.session.trace.seed_tts_text import (
    SeedTTSTextTraceFlavorGenerator,
    _truncate_to_words,
    _word_count,
)


# ------------------------------------------------------------------ pure helpers
@pytest.mark.unit
def test_word_count_and_truncate():
    assert _word_count("alpha beta gamma") == 3
    assert _truncate_to_words("alpha beta gamma", 2) == "alpha beta"
    # truncation is a no-op when fewer words than target
    assert _truncate_to_words("alpha beta", 5) == "alpha beta"


# ------------------------------------------------------------------ config validation
@pytest.mark.unit
def test_seed_tts_config_type_and_defaults():
    cfg = SeedTTSTextTraceFlavorConfig()
    from veeksha.types import TraceFlavorType

    assert cfg.get_type() == TraceFlavorType.SEED_TTS_TEXT
    assert cfg.use_chars is False  # default: word-based length control


@pytest.mark.unit
def test_seed_tts_config_requires_both_char_bounds():
    with pytest.raises(ValueError, match="min_chars and max_chars"):
        SeedTTSTextTraceFlavorConfig(min_chars=10)  # max_chars left unset


@pytest.mark.unit
def test_seed_tts_config_char_min_le_max():
    with pytest.raises(ValueError, match="min_chars .* must be <= max_chars"):
        SeedTTSTextTraceFlavorConfig(min_chars=50, max_chars=10)


@pytest.mark.unit
def test_seed_tts_config_word_min_le_max():
    with pytest.raises(ValueError, match="min_tokens .* must be <= max_tokens"):
        SeedTTSTextTraceFlavorConfig(min_tokens=100, max_tokens=10)


@pytest.mark.unit
def test_seed_tts_config_requires_text_column():
    with pytest.raises(ValueError, match="text_column is required"):
        SeedTTSTextTraceFlavorConfig(text_column="")


# ------------------------------------------------------------------ trace prep (no datasets)
@pytest.mark.unit
def test_prepare_trace_df_filters_short_and_empty(monkeypatch):
    """_prepare_trace_df must skip empty/short rows and keep long enough ones.

    Exercised without the datasets dependency by patching the loader.
    """
    raw = pd.DataFrame(
        {
            "text": [
                "one two three four five six",  # 6 words, kept
                "",  # empty, skipped
                "too short",  # 2 words < min_tokens, skipped
                None,  # NaN, skipped
            ]
        }
    )
    monkeypatch.setattr(
        "veeksha.generator.session.trace.seed_tts_text._load_hf_dataset",
        lambda flavor_config: raw,
    )
    monkeypatch.setattr(
        "veeksha.generator.session.trace.seed_tts_text._dataset_to_dataframe",
        lambda dataset: raw,
    )

    flavor_config = SeedTTSTextTraceFlavorConfig(min_tokens=5, max_tokens=50)
    config = TraceSessionGeneratorConfig(flavor=flavor_config, wrap_mode=False)
    tokenizer_provider = MagicMock()
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tokenizer_provider.for_modality.return_value = tok

    gen = SeedTTSTextTraceFlavorGenerator(
        config, flavor_config, SeedManager(seed=7), tokenizer_provider
    )
    assert len(gen.trace_df) == 1
    assert gen.trace_df.iloc[0]["text"].startswith("one two three")

    session = gen.generate_session()
    request = session.requests[0]
    assert request.metadata["dataset_source"] == "huggingface"
    assert request.metadata["input_words"] >= 5


# ------------------------------------------------------------------ full loader (needs datasets)
@pytest.mark.unit
def test_seed_tts_generator_with_local_jsonl(tmp_path):
    pytest.importorskip("datasets")
    path = tmp_path / "seed.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"text": f"word {i} " * 10, "filename": f"row{i}"})
            for i in range(3)
        ),
        encoding="utf-8",
    )
    flavor_config = SeedTTSTextTraceFlavorConfig(
        local_path=str(path), min_tokens=5, max_tokens=50
    )
    config = TraceSessionGeneratorConfig(flavor=flavor_config, wrap_mode=False)
    tokenizer_provider = MagicMock()
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tokenizer_provider.for_modality.return_value = tok

    gen = SeedTTSTextTraceFlavorGenerator(
        config, flavor_config, SeedManager(seed=7), tokenizer_provider
    )
    assert len(gen.trace_df) == 3
    session = gen.generate_session()
    assert session.requests[0].metadata["source_id"] == "row0"
