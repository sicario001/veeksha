import os
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from veeksha.config.generator.session import (
    TraceSessionGeneratorConfig,
    BaseTraceFlavorConfig,
)
from veeksha.generator.session.trace.base import TraceSessionGenerator
from veeksha.generator.session.trace.base_flavor import TraceFlavorGeneratorBase
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session


# Setup Mock Implementation for Base Logic Testing
class DummyTraceFlavorGenerator(TraceFlavorGeneratorBase):
    @property
    def required_columns(self):
        return ["session_id", "prompt"]

    def prepare_session(self, group: pd.DataFrame) -> Session:
        session_id = group["session_id"].iloc[0]
        # minimal session creation
        return Session(id=session_id, session_graph=MagicMock(), requests={})

    def wrap(self) -> pd.DataFrame:
        self._wrap_called = True
        return self._shuffle_sessions(self.trace_df)


@pytest.fixture
def mock_tokenizer_provider():
    provider = MagicMock()
    provider.for_modality.return_value = MagicMock()
    return provider


@pytest.fixture
def mock_seed_manager():
    return SeedManager(seed=42)


@pytest.fixture
def trace_file(tmp_path):
    f = tmp_path / "trace.jsonl"
    data = [
        {"session_id": 0, "prompt": "foo"},
        {"session_id": 1, "prompt": "bar"},
        {"session_id": 2, "prompt": "baz"},
    ]
    with open(f, "w") as fd:
        for d in data:
            fd.write(json.dumps(d) + "\n")
    return str(f)


@pytest.fixture
def base_flavor_config():
    config = MagicMock(spec=BaseTraceFlavorConfig)
    config.get_type.return_value = "mock"
    return config


@pytest.fixture
def trace_config(trace_file, base_flavor_config):
    return TraceSessionGeneratorConfig(
        trace_file=trace_file, flavor=base_flavor_config, wrap_mode=False
    )


def test_trace_flavor_base_loading(
    trace_config, base_flavor_config, mock_seed_manager, mock_tokenizer_provider
):
    generator = DummyTraceFlavorGenerator(
        trace_config, base_flavor_config, mock_seed_manager, mock_tokenizer_provider
    )
    assert len(generator.trace_df) == 3
    assert generator.capacity() == 3


def test_trace_flavor_base_validation_missing_col(
    trace_file, base_flavor_config, mock_seed_manager, mock_tokenizer_provider, tmp_path
):
    # Create bad trace
    bad_f = tmp_path / "bad.jsonl"
    with open(bad_f, "w") as fd:
        fd.write('{"session_id": 0}\n')

    config = TraceSessionGeneratorConfig(
        trace_file=str(bad_f), flavor=base_flavor_config
    )

    with pytest.raises(ValueError, match="Trace missing required column"):
        DummyTraceFlavorGenerator(
            config, base_flavor_config, mock_seed_manager, mock_tokenizer_provider
        )


def test_trace_flavor_base_generation(
    trace_config, base_flavor_config, mock_seed_manager, mock_tokenizer_provider
):
    generator = DummyTraceFlavorGenerator(
        trace_config, base_flavor_config, mock_seed_manager, mock_tokenizer_provider
    )

    s1 = generator.generate_session()
    assert s1.id == 0
    s2 = generator.generate_session()
    assert s2.id == 1
    s3 = generator.generate_session()
    assert s3.id == 2

    with pytest.raises(StopIteration):
        generator.generate_session()


def test_trace_flavor_base_wrapping(
    trace_file, base_flavor_config, mock_seed_manager, mock_tokenizer_provider
):
    trace_config = TraceSessionGeneratorConfig(
        trace_file=trace_file, flavor=base_flavor_config, wrap_mode=True
    )
    generator = DummyTraceFlavorGenerator(
        trace_config, base_flavor_config, mock_seed_manager, mock_tokenizer_provider
    )

    assert generator.capacity() == -1

    # Consume first epoch
    ids_epoch1 = [generator.generate_session().id for _ in range(3)]
    assert sorted(ids_epoch1) == [0, 1, 2]

    # Should wrap now
    s4 = generator.generate_session()
    assert getattr(generator, "_wrap_called", False)
    assert s4.id in [0, 1, 2]


def test_trace_session_generator_delegation(
    trace_config, mock_seed_manager, mock_tokenizer_provider
):
    with patch(
        "veeksha.generator.session.trace.base.TraceFlavorGeneratorRegistry"
    ) as mock_registry:
        mock_impl = MagicMock()
        mock_registry.get.return_value = mock_impl

        generator = TraceSessionGenerator(
            trace_config, mock_seed_manager, mock_tokenizer_provider
        )

        # Test delegation
        generator.generate_session()
        mock_impl.generate_session.assert_called_once()

        generator.capacity()
        mock_impl.capacity.assert_called_once()

        generator.get_warmup_sessions()
        mock_impl.get_warmup_sessions.assert_called_once()
