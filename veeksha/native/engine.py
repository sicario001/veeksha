"""Python seam over the native (C++) receive engine.

The native ``run_batch`` owns the per-request I/O lifecycle end-to-end: it holds
connection concurrency in one poll() loop, sends caller-built HTTP requests, and
timestamps every SSE chunk at true socket-read time — so Python is never in the
per-event receive loop. This module builds the request bytes, invokes the engine,
and hands back a ``TimedEventStream`` per request (the same timing primitive the
Python pipeline produces), plus the parsed content and status.

If the extension is not built, ``native_available()`` is False and callers fall
back to the Python transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from veeksha.core.timed_event_stream import StreamEvent, TimedEventStream
from veeksha.types import ChannelModality

try:
    from veeksha.native import veeksha_native as _ext  # type: ignore
except Exception:  # pragma: no cover - extension optional / not built
    _ext = None


def native_available() -> bool:
    return _ext is not None


@dataclass
class NativeRequest:
    """One request the native engine will send (Python owns the *what*)."""

    path: str
    body: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    method: str = "POST"

    def to_wire(self, host: str) -> str:
        """Serialize to raw HTTP/1.1 request bytes (Connection: close)."""
        lines = [f"{self.method} {self.path} HTTP/1.1", f"Host: {host}"]
        headers = dict(self.headers)
        headers.setdefault("Content-Type", "application/json")
        headers["Content-Length"] = str(len(self.body.encode("utf-8")))
        headers["Connection"] = "close"
        for key, value in headers.items():
            lines.append(f"{key}: {value}")
        return "\r\n".join(lines) + "\r\n\r\n" + self.body


@dataclass
class NativeResult:
    """One request's outcome: status, parsed content, and the receive timeline."""

    index: int
    status: int
    content: str
    stream: TimedEventStream
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error and 200 <= self.status < 300


class NativeReceiveEngine:
    """Drives the native engine for a batch of requests against one endpoint."""

    def __init__(self, host: str, port: int):
        if _ext is None:
            raise RuntimeError(
                "veeksha_native not built; run veeksha/native/build.sh <python>"
            )
        self.host = host
        self.port = port

    def run(
        self,
        requests: List[NativeRequest],
        concurrency: int,
        sse: bool = True,
        timeout_s: float = 120.0,
        modality: ChannelModality = ChannelModality.TEXT,
    ) -> List[NativeResult]:
        """Send all requests (native owns concurrency); return per-request results.

        Each result's ``stream`` carries offsets in *seconds* from send-time with
        per-event sizes — ready for the shared per-stream derivations
        (time_to_first_event / inter_event_deltas / RTF).
        """
        wire = [r.to_wire(self.host) for r in requests]
        raw = _ext.run_batch(self.host, self.port, wire, concurrency, timeout_s, sse)
        results: List[NativeResult] = []
        for item in raw:
            events = [
                StreamEvent(offset_s=off / 1000.0, size=size, kind="output")
                for off, size in zip(item.offsets_ms, item.sizes)
            ]
            stream = TimedEventStream(modality, events)
            results.append(
                NativeResult(
                    index=item.index,
                    status=item.status,
                    content=item.content,
                    stream=stream,
                    error=item.error,
                )
            )
        return results
