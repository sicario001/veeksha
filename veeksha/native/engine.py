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


def native_can_handle(url_or_scheme: str) -> bool:
    """Whether the native transport can serve this endpoint.

    The native engine owns the plaintext hot path (``http://`` / ``ws://``) — the
    common case for inference servers inside a trusted network (vLLM, datacenter
    gateways), where microsecond receive/pacing precision matters most. TLS
    endpoints (``https://`` / ``wss://``) route to the Python transport, whose
    per-request overhead is negligible relative to the remote-network latency a
    TLS endpoint implies. Callers use this to decide native-vs-Python per endpoint.
    """
    if not native_available():
        return False
    scheme = (
        url_or_scheme.split("://", 1)[0].lower()
        if "://" in url_or_scheme
        else (url_or_scheme.lower())
    )
    return scheme in ("http", "ws", "")


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
    dispatch_offset_s: float = 0.0  # actual launch time (open-loop arrival dispatch)
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
        dispatch_offsets_s: Optional[List[float]] = None,
        num_threads: int = 1,
    ) -> List[NativeResult]:
        """Send all requests (native owns concurrency); return per-request results.

        Each result's ``stream`` carries offsets in *seconds* from send-time with
        per-event sizes — ready for the shared per-stream derivations
        (time_to_first_event / inter_event_deltas / RTF).

        When ``dispatch_offsets_s`` is given the engine runs OPEN-LOOP: request i
        is launched on its arrival deadline (rate-based traffic), with
        ``concurrency`` as a max-in-flight safety cap — so native owns the
        arrival-dispatch timing, not just the receive timing.

        ``num_threads`` > 1 shards the connections across that many native
        poll-loop threads (true parallel on free-threaded CPython), splitting the
        in-flight budget across shards. Only helps when a single loop is
        CPU-bound; give the server its own host first (co-located, extra threads
        just steal cores from the server).
        """
        wire = [r.to_wire(self.host) for r in requests]
        offsets_ms = (
            [s * 1000.0 for s in dispatch_offsets_s] if dispatch_offsets_s else []
        )
        raw = _ext.run_batch(
            self.host,
            self.port,
            wire,
            concurrency,
            timeout_s,
            sse,
            offsets_ms,
            num_threads,
        )
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
                    dispatch_offset_s=item.dispatch_offset_ms / 1000.0,
                    error=item.error,
                )
            )
        return results

    def run_chains(
        self,
        chains: List[List[NativeRequest]],
        concurrency: int,
        timeout_s: float = 120.0,
        modality: ChannelModality = ChannelModality.TEXT,
    ) -> List["NativeChainResult"]:
        """Closed-loop / dependent workloads: each chain is a sequence of turns
        where turn N+1 is dispatched only after turn N completes.

        Native fires each next turn the instant the prior completes (no Python on
        the receive->dispatch path) and reports per-turn TimedEventStreams plus
        ``handoff_s`` — the native-measured inter-turn coupling latency, which is
        the number that would otherwise be corrupted by Python queue jitter.
        """
        wire = [[req.to_wire(self.host) for req in chain] for chain in chains]
        raw = _ext.run_chains(self.host, self.port, wire, concurrency, timeout_s)
        results: List[NativeChainResult] = []
        for item in raw:
            turns = []
            for turn in item.turns:
                events = [
                    StreamEvent(offset_s=off / 1000.0, size=1, kind="output")
                    for off in turn.offsets_ms
                ]
                turns.append(
                    NativeTurn(
                        status=turn.status,
                        content=turn.content,
                        stream=TimedEventStream(modality, events),
                    )
                )
            results.append(
                NativeChainResult(
                    index=item.index,
                    turns=turns,
                    handoff_s=[h / 1000.0 for h in item.handoff_ms],
                    error=item.error,
                )
            )
        return results


@dataclass
class NativeTurn:
    """One turn's outcome within a dependent chain."""

    status: int
    content: str
    stream: TimedEventStream


@dataclass
class NativeChainResult:
    """A dependent chain's per-turn results + native inter-turn handoff latency."""

    index: int
    turns: List[NativeTurn]
    handoff_s: List[float] = field(default_factory=list)
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error and len(self.turns) > 0


@dataclass
class NativeWsResult:
    """One WebSocket connection's receive timeline + paced-send record."""

    index: int
    stream: TimedEventStream  # server frames, offsets in seconds from handshake
    content: str
    frames: List[str] = field(default_factory=list)  # per-frame payloads
    sent_offsets_s: List[float] = field(default_factory=list)
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error and len(self.stream) > 0

    def timed_frames(self):
        """Yield (offset_s, payload) per received data frame, in arrival order."""
        for event, payload in zip(self.stream.events, self.frames):
            yield event.offset_s, payload


class NativeWsEngine:
    """Native WebSocket transport: per-frame receive timing + timer-wheel sends.

    The interactivity-critical path (per-token receive + per-chunk paced send)
    owned entirely in C++: one poll() loop over all connections timestamps each
    server frame at socket-read time and dispatches each client message on its
    absolute deadline — Python is never in the per-event loop.
    """

    def __init__(self, host: str, port: int):
        if _ext is None:
            raise RuntimeError(
                "veeksha_native not built; run veeksha/native/build.sh <python>"
            )
        self.host = host
        self.port = port

    def stream(
        self,
        path: str,
        init_messages: List[str],
        concurrency: int,
        send_offsets_s: Optional[List[float]] = None,
        timeout_s: float = 30.0,
        modality: ChannelModality = ChannelModality.AUDIO,
    ) -> List[NativeWsResult]:
        """Open ``concurrency`` WS connections; send init_messages (paced by
        ``send_offsets_s``, seconds from handshake, or all immediately) and
        collect the server-frame receive timeline per connection."""
        send_offsets_ms = [s * 1000.0 for s in send_offsets_s] if send_offsets_s else []
        raw = _ext.ws_stream(
            self.host,
            self.port,
            path,
            init_messages,
            concurrency,
            timeout_s,
            send_offsets_ms,
        )
        results: List[NativeWsResult] = []
        for item in raw:
            events = [
                StreamEvent(offset_s=off / 1000.0, size=size, kind="output")
                for off, size in zip(item.offsets_ms, item.sizes)
            ]
            results.append(
                NativeWsResult(
                    index=item.index,
                    stream=TimedEventStream(modality, events),
                    content=item.content,
                    frames=list(item.frames),
                    sent_offsets_s=[o / 1000.0 for o in item.sent_offsets_ms],
                    error=item.error,
                )
            )
        return results

    def stream_batch(
        self,
        path: str,
        req_messages: List[List[str]],
        concurrency: int,
        req_offsets_s: Optional[List[List[float]]] = None,
        timeout_s: float = 30.0,
        modality: ChannelModality = ChannelModality.AUDIO,
    ) -> List[NativeWsResult]:
        """Run N distinct WS requests concurrently (per-connection messages).

        Each request in ``req_messages`` gets its own message sequence + paced
        send schedule; native caps simultaneous connections at ``concurrency``
        and refills as they complete — so audio requests scale like the SSE path
        instead of running one at a time.
        """
        if req_offsets_s is None:
            offsets_ms = [[] for _ in req_messages]
        else:
            offsets_ms = [[s * 1000.0 for s in off] for off in req_offsets_s]
        raw = _ext.ws_run_batch(
            self.host,
            self.port,
            path,
            req_messages,
            offsets_ms,
            concurrency,
            timeout_s,
        )
        results: List[NativeWsResult] = []
        for item in raw:
            events = [
                StreamEvent(offset_s=off / 1000.0, size=size, kind="output")
                for off, size in zip(item.offsets_ms, item.sizes)
            ]
            results.append(
                NativeWsResult(
                    index=item.index,
                    stream=TimedEventStream(modality, events),
                    content=item.content,
                    frames=list(item.frames),
                    sent_offsets_s=[o / 1000.0 for o in item.sent_offsets_ms],
                    error=item.error,
                )
            )
        return results
