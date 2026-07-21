"""Optional native (C++) transport engine.

The `veeksha_native` extension (built by ``veeksha/native/build.sh`` or on
``pip install``) runs event-loop receive loops that timestamp SSE/WS events at
socket-read time. If it is not built, ``is_available()`` returns False and
callers fall back to Python.

Drift metrics are NOT computed here: the preflight scores the native and Python
paths with one shared harness (``veeksha.preflight.drivers``) so the two cannot
be measured differently.
"""

from __future__ import annotations

from typing import List

try:  # the compiled extension sits next to this file after build.sh
    from veeksha.native import veeksha_native as _ext  # type: ignore
except Exception:  # pragma: no cover - extension optional / not built
    _ext = None


def is_available() -> bool:
    return _ext is not None


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]
