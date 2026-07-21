"""Benchmark main-loop abstractions.

One protocol, multiple implementations (Python today, native C++ planned):
the loop takes only plain configs + a ``SessionSource`` and emits an ordered
``LoopEvent`` stream that a ``ResultDrain`` replays into the evaluator and
trace recorder. See ``docs/design/native_loop_prototype.md`` §7.
"""

from veeksha.loop.interface import (
    LoopCounters,
    LoopEvent,
    LoopEventKind,
    MainLoop,
    MainLoopConfig,
    SessionSource,
)
from veeksha.loop.python_loop import PythonMainLoop
from veeksha.loop.drain import ResultDrain
from veeksha.loop.source import GeneratorSessionSource, PregeneratedSessionSource
from veeksha.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "LoopCounters",
    "LoopEvent",
    "LoopEventKind",
    "MainLoop",
    "MainLoopConfig",
    "SessionSource",
    "PythonMainLoop",
    "ResultDrain",
    "GeneratorSessionSource",
    "PregeneratedSessionSource",
    "create_main_loop",
]


def create_main_loop(kind: str, config: MainLoopConfig, **kwargs) -> MainLoop:
    """Create a main loop implementation.

    Args:
        kind: Loop kind (``"python"``; the native C++ main loop lands in
            the next commit behind the same protocol).
        config: Minimal plain-data loop configuration.
        **kwargs: Implementation-specific construction dependencies (e.g. the
            Python main loop takes ``seed_manager``, ``tokenizer_provider`` and an
            optional pre-built ``client``).

    Returns:
        A ``MainLoop`` implementation.
    """
    if kind == "python":
        return PythonMainLoop(config, **kwargs)
    raise ValueError(f"Unknown main loop kind: {kind!r}")
