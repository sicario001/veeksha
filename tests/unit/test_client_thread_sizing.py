"""Tests for client-thread sizing guidance."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from veeksha.benchmark_utils import (
    maybe_warn_client_thread_sizing,
    recommended_client_threads,
)
from veeksha.config.traffic import ConcurrentTrafficConfig, RateTrafficConfig


@pytest.mark.parametrize(
    "target,expected",
    [(0, 1), (-5, 1), (1, 1), (300, 1), (301, 2), (600, 2), (800, 3), (1500, 5)],
)
def test_recommended_client_threads(target, expected):
    assert recommended_client_threads(target) == expected


def _cfg(target, n_threads, scheduler_cls=ConcurrentTrafficConfig):
    if scheduler_cls is ConcurrentTrafficConfig:
        sched = ConcurrentTrafficConfig(target_concurrent_sessions=target)
    else:
        sched = scheduler_cls()
    return SimpleNamespace(
        traffic_scheduler=sched,
        runtime=SimpleNamespace(num_client_threads=n_threads),
    )


def test_warns_when_undersized(caplog):
    with caplog.at_level(logging.WARNING):
        maybe_warn_client_thread_sizing(_cfg(target=800, n_threads=1))
    assert any("num_client_threads=1" in r.message for r in caplog.records)


def test_no_warning_when_adequately_sized(caplog):
    with caplog.at_level(logging.WARNING):
        maybe_warn_client_thread_sizing(_cfg(target=800, n_threads=3))
    assert not caplog.records


def test_no_warning_for_non_concurrent_scheduler(caplog):
    with caplog.at_level(logging.WARNING):
        maybe_warn_client_thread_sizing(
            _cfg(target=800, n_threads=1, scheduler_cls=RateTrafficConfig)
        )
    assert not caplog.records
