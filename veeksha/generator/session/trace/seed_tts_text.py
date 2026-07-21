"""Seed TTS text dataset trace flavor for TTS benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pandas as pd

from veeksha.config.generator.session import (
    SeedTTSTextTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.generator.session.trace.base_flavor import TraceFlavorGeneratorBase
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

logger = init_logger(__name__)


def _load_hf_dataset(flavor_config: SeedTTSTextTraceFlavorConfig) -> Any:
    """Load the Seed TTS text dataset from HF or a local compatible copy."""
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(
            "SeedTTSTextTraceFlavorGenerator requires the 'datasets' package."
        ) from exc

    if flavor_config.local_path:
        path = Path(flavor_config.local_path)
        if not path.exists():
            raise FileNotFoundError(f"Seed TTS local_path not found: {path}")

        try:
            dataset = datasets.load_from_disk(str(path))
        except Exception:
            logger.debug(
                "Could not load %s with datasets.load_from_disk; trying load_dataset",
                path,
                exc_info=True,
            )
        else:
            return _select_split(dataset, flavor_config.split)

        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in {".json", ".jsonl"}:
                return datasets.load_dataset(
                    "json",
                    data_files=str(path),
                    split=flavor_config.split,
                )
            if suffix == ".csv":
                return datasets.load_dataset(
                    "csv",
                    data_files=str(path),
                    split=flavor_config.split,
                )
            if suffix == ".parquet":
                return datasets.load_dataset(
                    "parquet",
                    data_files=str(path),
                    split=flavor_config.split,
                )
            raise ValueError(
                f"Unsupported Seed TTS local file extension '{suffix}'. "
                "Use JSON/JSONL, CSV, Parquet, or a saved HF dataset directory."
            )

        args = [str(path)]
        if flavor_config.subset:
            args.append(flavor_config.subset)
        return datasets.load_dataset(*args, split=flavor_config.split)

    args = [flavor_config.dataset_name]
    if flavor_config.subset:
        args.append(flavor_config.subset)
    return datasets.load_dataset(*args, split=flavor_config.split)


def _select_split(dataset: Any, split: str) -> Any:
    """Select a split when a local saved dataset returns a DatasetDict."""
    if hasattr(dataset, "to_pandas"):
        return dataset
    if isinstance(dataset, dict):
        if split not in dataset:
            raise ValueError(
                f"Seed TTS dataset split '{split}' not found. "
                f"Available splits: {sorted(dataset.keys())}"
            )
        return dataset[split]
    return dataset


def _dataset_to_dataframe(dataset: Any) -> pd.DataFrame:
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()
    return pd.DataFrame(list(dataset))


def _word_count(text: str) -> int:
    return len(text.split())


def _truncate_to_words(text: str, target_words: int) -> str:
    words = text.split()
    if len(words) <= target_words:
        return text
    return " ".join(words[:target_words])


class SeedTTSTextTraceFlavorGenerator(TraceFlavorGeneratorBase):
    """Trace flavor that emits one text-only TTS session per Seed TTS row."""

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: SeedTTSTextTraceFlavorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        self.config = config
        self.flavor_config = flavor_config
        self.seed_manager = seed_manager
        self.tokenizer_provider = tokenizer_provider
        self.tokenizer = tokenizer_provider.for_modality(ChannelModality.TEXT)

        raw_dataset = _load_hf_dataset(flavor_config)
        raw_df = _dataset_to_dataframe(raw_dataset)
        self.trace_df = self._prepare_trace_df(raw_df)

        self._num_wraps = 0
        self._session_groups = None
        self._current_session_id = 0
        self._current_request_id = 0
        self._rng = seed_manager.random("seed_tts_text_shuffling")
        self._length_rng = seed_manager.random("seed_tts_text_input_length")

        logger.info(
            "Loaded %d Seed TTS text rows from %s/%s split=%s",
            len(self.trace_df),
            flavor_config.local_path or flavor_config.dataset_name,
            flavor_config.subset,
            flavor_config.split,
        )

    @property
    def required_columns(self) -> List[str]:
        return [self.flavor_config.text_column]

    def _prepare_trace_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        text_col = self.flavor_config.text_column
        id_col = self.flavor_config.id_column

        if text_col not in raw_df.columns:
            raise ValueError(
                f"Seed TTS dataset missing text column '{text_col}'. "
                f"Available columns: {list(raw_df.columns)}"
            )

        has_id_col = bool(id_col) and id_col in raw_df.columns
        rows = []
        skipped_empty = 0
        skipped_short = 0
        min_length = (
            self.flavor_config.min_chars
            if self.flavor_config.use_chars
            else self.flavor_config.min_tokens
        )
        unit = "chars" if self.flavor_config.use_chars else "words"

        for source_index, row in raw_df.iterrows():
            value = row[text_col]
            if pd.isna(value):
                skipped_empty += 1
                continue

            text = str(value).strip()
            if not text:
                skipped_empty += 1
                continue

            char_count = len(text)
            word_count = _word_count(text)
            current_length = char_count if self.flavor_config.use_chars else word_count
            if current_length < min_length:
                skipped_short += 1
                continue

            source_id = row[id_col] if has_id_col else source_index
            rows.append(
                {
                    "session_id": len(rows),
                    "text": text,
                    "source_id": str(source_id),
                    "source_index": (
                        int(source_index)
                        if isinstance(source_index, int)
                        else str(source_index)
                    ),
                    "word_count": word_count,
                    "char_count": char_count,
                }
            )

        if skipped_empty:
            logger.info("Skipped %d empty Seed TTS text rows", skipped_empty)
        if skipped_short:
            logger.info(
                "Skipped %d Seed TTS text rows shorter than %d %s",
                skipped_short,
                min_length,
                unit,
            )

        if not rows:
            raise ValueError(
                f"Seed TTS dataset contains no rows in column '{text_col}' with "
                f"at least {min_length} {unit}."
            )

        return pd.DataFrame(rows)

    def prepare_session(self, group: pd.DataFrame) -> Session:
        session_id = self._next_session_id()
        row = group.iloc[0]
        text = str(row["text"])

        if self.flavor_config.use_chars:
            target_chars = self._length_rng.randint(
                self.flavor_config.min_chars,
                self.flavor_config.max_chars,
            )
            if len(text) > target_chars:
                text = text[:target_chars]
        else:
            target_words = self._length_rng.randint(
                self.flavor_config.min_tokens,
                self.flavor_config.max_tokens,
            )
            text = _truncate_to_words(text, target_words)

        target_prompt_tokens = len(self.tokenizer.encode(text))

        request = self._create_text_request(
            node_id=0,
            prompt_text=text,
            target_output_tokens=1,
            wait_after_ready=0.0,
            parent_nodes=None,
            target_prompt_tokens=target_prompt_tokens,
        )
        request.metadata.update(
            {
                "dataset": self.flavor_config.dataset_name,
                "dataset_subset": self.flavor_config.subset,
                "dataset_split": self.flavor_config.split,
                "dataset_source": self.flavor_config.local_path or "huggingface",
                "source_id": row["source_id"],
                "source_index": row["source_index"],
                "input_words": _word_count(text),
                "input_chars": len(text),
            }
        )

        graph = self._build_linear_session_graph(1, [0.0])
        return Session(id=session_id, session_graph=graph, requests={0: request})

    def wrap(self) -> pd.DataFrame:
        return self._wrap_shuffled()
