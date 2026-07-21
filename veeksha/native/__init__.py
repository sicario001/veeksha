"""Optional native (C++) benchmark main loop.

The ``veeksha_native`` extension (built by ``veeksha/native/build.sh`` or on
``pip install``) implements the NativeBenchmarkLoop of
``docs/design/native_loop_prototype.md``: native scheduler + dispatch +
reactor transport state machines + completion ack, with read-time
CLOCK_MONOTONIC stamps and the exact traffic semantics of the Python loop.
If it is not built, ``is_available()`` returns False and callers fall back to
the Python main loop.

Drift metrics are NOT computed here: the preflight scores the native and
Python paths with one shared harness so the two cannot be measured
differently.
"""

from __future__ import annotations

try:  # the compiled extension sits next to this file after build.sh
    from veeksha.native import veeksha_native as _ext  # type: ignore
except Exception:  # pragma: no cover - extension optional / not built
    _ext = None


def is_available() -> bool:
    return _ext is not None


def get_module():
    """Return the compiled extension module, or raise if unbuilt."""
    if _ext is None:
        raise RuntimeError(
            "veeksha_native is not built; run veeksha/native/build.sh "
            "or reinstall with a C++ toolchain available"
        )
    return _ext
