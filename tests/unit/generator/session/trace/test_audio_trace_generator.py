import json
from unittest.mock import MagicMock

import pytest

from veeksha.config.generator.session import (
    AudioTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.generator.session.trace.audio import AudioTraceFlavorGenerator


@pytest.mark.unit
def test_audio_trace_metadata_preserves_reference_word_timestamp_list(tmp_path) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"")
    reference_word_timestamps = [
        {"word": "hello", "start_ms": 0.0, "end_ms": 100.0},
        {"word": "world", "start_ms": 120.0, "end_ms": 300.0},
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "session_id": 0,
                "audio_file": audio_path.name,
                "expected_transcript": "hello world",
                "reference_word_timestamps": reference_word_timestamps,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    flavor_config = AudioTraceFlavorConfig()
    config = TraceSessionGeneratorConfig(
        trace_file=str(manifest_path),
        flavor=flavor_config,
        wrap_mode=False,
    )
    tokenizer_provider = MagicMock()
    tokenizer_provider.for_modality.return_value = MagicMock()

    generator = AudioTraceFlavorGenerator(
        config,
        flavor_config,
        SeedManager(seed=42),
        tokenizer_provider,
    )
    session = generator.generate_session()

    assert session.requests[0].metadata["reference_word_timestamps"] == (
        reference_word_timestamps
    )


@pytest.mark.unit
def test_audio_trace_target_duration_trims_reference_and_adds_slice_metadata(
    tmp_path,
) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"")
    reference_word_timestamps = [
        {"word": "alpha", "start_ms": 0.0, "end_ms": 100.0},
        {"word": "beta", "start_ms": 150.0, "end_ms": 900.0},
        {"word": "gamma", "start_ms": 1200.0, "end_ms": 1400.0},
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "session_id": 0,
                "audio_file": audio_path.name,
                "expected_transcript": "alpha beta gamma",
                "reference_word_timestamps": reference_word_timestamps,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    flavor_config = AudioTraceFlavorConfig(target_duration_s=1.0)
    config = TraceSessionGeneratorConfig(
        trace_file=str(manifest_path),
        flavor=flavor_config,
        wrap_mode=False,
    )
    tokenizer_provider = MagicMock()
    tokenizer_provider.for_modality.return_value = MagicMock()

    generator = AudioTraceFlavorGenerator(
        config,
        flavor_config,
        SeedManager(seed=42),
        tokenizer_provider,
    )
    session = generator.generate_session()
    metadata = session.requests[0].metadata

    assert metadata["expected_transcript"] == "alpha beta"
    assert metadata["reference_word_timestamps"] == reference_word_timestamps[:2]
    assert metadata["input_audio_start_ms"] == 0.0
    assert metadata["input_audio_end_ms"] == 1000.0
    assert metadata["duration_s"] == 1.0


@pytest.mark.unit
def test_audio_trace_target_duration_requires_word_timestamps(tmp_path) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "session_id": 0,
                "audio_file": audio_path.name,
                "expected_transcript": "alpha beta",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    flavor_config = AudioTraceFlavorConfig(target_duration_s=1.0)
    config = TraceSessionGeneratorConfig(
        trace_file=str(manifest_path),
        flavor=flavor_config,
        wrap_mode=False,
    )
    tokenizer_provider = MagicMock()
    tokenizer_provider.for_modality.return_value = MagicMock()

    generator = AudioTraceFlavorGenerator(
        config,
        flavor_config,
        SeedManager(seed=42),
        tokenizer_provider,
    )

    with pytest.raises(ValueError, match="reference_word_timestamps"):
        generator.generate_session()


@pytest.mark.unit
def test_audio_trace_target_duration_rejects_empty_trimmed_transcript(
    tmp_path,
) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "session_id": 0,
                "audio_file": audio_path.name,
                "expected_transcript": "alpha",
                "reference_word_timestamps": [
                    {"word": "alpha", "start_ms": 1100.0, "end_ms": 1400.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    flavor_config = AudioTraceFlavorConfig(target_duration_s=1.0)
    config = TraceSessionGeneratorConfig(
        trace_file=str(manifest_path),
        flavor=flavor_config,
        wrap_mode=False,
    )
    tokenizer_provider = MagicMock()
    tokenizer_provider.for_modality.return_value = MagicMock()

    generator = AudioTraceFlavorGenerator(
        config,
        flavor_config,
        SeedManager(seed=42),
        tokenizer_provider,
    )

    with pytest.raises(ValueError, match="expected_transcript would"):
        generator.generate_session()


@pytest.mark.unit
def test_audio_trace_flavor_config_rejects_nonpositive_target_duration() -> None:
    with pytest.raises(ValueError, match="target_duration_s must be positive"):
        AudioTraceFlavorConfig(target_duration_s=0.0)
