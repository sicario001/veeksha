"""Shared infrastructure for the preflight mock servers.

``ShardedLoopServer`` runs N asyncio event loops (one per thread) that all
accept from ONE shared listening socket. Binding N listeners with SO_REUSEPORT
does NOT distribute TCP accepts on macOS — the kernel delivers essentially every
connection to the last-bound listener (verified empirically: 40/40 connections
landed on the last-bound listener), silently collapsing "8 accept loops" into
one saturated loop. Sharing a single listening socket (each loop waits on its
own dup of the same fd; whichever loop accepts first wins the connection)
distributes on Linux and Darwin alike, and the single bind also removes the
probe-then-rebind port race.

``PhaseSpreader`` staggers each connection's emit phase across one cadence
window. Connections opened by a benchmark start together, so identical emit
schedules put every deadline in the same instant; one event loop must then walk
the whole burst and whoever it serves last records that walk as *client* jitter.
Spreading empirically drops the mock's own self-jitter from 2-6 ms to <1 ms.

``ShardedTelemetry`` gives each loop thread its own sample list so the emit hot
path never takes a cross-thread lock (on free-threaded CPython a shared lock on
every chunk is real contention that both delays the send and inflates the
recorded jitter).

``ServerRecordBook`` holds the paired-timestamp ground truth: per request, the
instant the mock RECEIVED it and the instant it sent each response event. Those
stamps are the reference the preflight scores the client's own stamps against,
so the recording path is built to stay off the critical section — a connection
takes the book's lock exactly once (when it opens its record) and appends
lock-free to its own lists thereafter.
"""

from __future__ import annotations

import asyncio
import itertools
import re
import socket
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ACCEPT_BACKLOG",
    "PFID_PREFIX",
    "PhaseSpreader",
    "ServerRecord",
    "ServerRecordBook",
    "ShardedLoopServer",
    "ShardedTelemetry",
    "format_pfid",
    "parse_pfid",
    "percentile",
]

# Accept-queue depth. Every turn of a multi-turn conversation opens a fresh
# connection, so N conversations resuming together arrive as an N-connection
# burst; if the queue is full the kernel silently drops the handshake and the
# client waits a full retransmit timeout — a second on Linux — which lands in
# the measurement as a stall the client did not cause.
#
# This MUST be passed to asyncio.start_server / websockets.serve, not just to
# socket.listen(): both call listen() again on the socket they are handed, with
# a default backlog of 100, silently overriding it. The kernel then caps
# whatever is asked for at net.core.somaxconn.
ACCEPT_BACKLOG = 65535

# Connections that all emit on the same schedule come due in the same instant,
# and one event loop must then walk the whole burst — so whoever it serves last
# records that walk as "jitter". Spreading each connection's phase across one
# cadence window removes an artifact of the test rig (real sessions do not start
# in lockstep) and is what lets a mock server act as a punctual reference.
_PHASE_SLOTS = 50


def percentile(values: List[float], q: float) -> float:
    """Nearest-rank percentile of ``values`` (``q`` in [0, 1]); 0.0 if empty."""
    if not values:
        return 0.0
    xs = sorted(values)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


class PhaseSpreader:
    """Hands out per-connection phase offsets spread over one cadence window.

    The first connection always gets 0, so low-concurrency tests are unaffected.
    """

    def __init__(self) -> None:
        self._seq = itertools.count()
        self._lock = threading.Lock()

    def next_phase(self, cadence_s: float) -> float:
        with self._lock:
            i = next(self._seq)
        return (i % _PHASE_SLOTS) * (cadence_s / _PHASE_SLOTS)

    def reset(self) -> None:
        with self._lock:
            self._seq = itertools.count()


class ShardedTelemetry:
    """Per-thread sample shards, merged at read time. Record path is lock-free."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._shards: List[List[float]] = []
        self._lock = threading.Lock()

    def record(self, value: float) -> None:
        shard = getattr(self._local, "shard", None)
        if shard is None:
            shard = []
            self._local.shard = shard
            with self._lock:
                self._shards.append(shard)
        shard.append(value)

    def merged(self) -> List[float]:
        with self._lock:
            shards = list(self._shards)
        out: List[float] = []
        for shard in shards:
            out.extend(shard)
        return out

    def clear(self) -> None:
        with self._lock:
            for shard in self._shards:
                shard.clear()

    def count(self) -> int:
        return len(self.merged())

    def p99(self) -> float:
        return percentile(self.merged(), 0.99)

    def percentile(self, q: float) -> float:
        return percentile(self.merged(), q)


# ---------------------------------------------------------------- correlation
# The preflight id that lets a server record be matched to the client request
# that produced it. For every text-carrying modality the preflight builds the
# request text itself, so it embeds ``PFID:<n>`` there and the mock reads it
# back off the wire. STT has no text on the way in, so its client echoes the id
# into the session.update handshake as ``veeksha_request_id`` instead (the audio
# cannot carry a marker -- the clip decode round-trips through float).
PFID_PREFIX = "PFID:"
_PFID_RE = re.compile(r"PFID:(\d+)")


def format_pfid(request_id: int) -> str:
    """The marker the preflight embeds in prompt / input text."""
    return f"{PFID_PREFIX}{int(request_id)}"


def parse_pfid(text: str) -> Optional[int]:
    """First ``PFID:<n>`` in ``text``, or ``None`` if there is none."""
    match = _PFID_RE.search(text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # pragma: no cover - the regex only matches digits
        return None


@dataclass(frozen=True)
class ServerRecord:
    """What one mock server observed for one request, in monotonic time.

    Both sides of the pairing read ``time.monotonic()`` on the same host, so
    these stamps are directly comparable with the client's — no clock
    synchronisation and no cadence reference in between.

    * ``received_at`` — stamped immediately after the read that delivered the
      request (HTTP: the request head; WS: the connection's first frame) and
      BEFORE any parsing, so it never carries the mock's own decode cost.
    * ``emitted_at`` — one stamp per emitted event, taken immediately before
      the write call.
    * ``first_append_at`` / ``append_at`` — STT only: arrival of the client's
      paced audio appends, same pre-parse convention.
    """

    request_id: int
    received_at: float
    emitted_at: List[float] = field(default_factory=list)
    first_append_at: Optional[float] = None
    append_at: List[float] = field(default_factory=list)


class _LiveRecord:
    """Mutable per-connection accumulator behind a :class:`ServerRecord`.

    Kept separate from the frozen record so the stamped path is a plain list
    append on a list only one event loop touches: no lock, no dataclass
    rebuild, nothing that could delay the very send it is timing. The id is
    filled in after the stamp (it is only knowable after parsing).
    """

    __slots__ = (
        "request_id",
        "received_at",
        "emitted_at",
        "first_append_at",
        "append_at",
    )

    def __init__(self, received_at: float) -> None:
        self.request_id: Optional[int] = None
        self.received_at = received_at
        self.emitted_at: List[float] = []
        self.first_append_at: Optional[float] = None
        self.append_at: List[float] = []


class ServerRecordBook:
    """Collects one :class:`_LiveRecord` per connection, keyed at read time.

    A connection that never yielded a usable id (no ``PFID`` marker, or an id
    another connection already claimed) is still kept — under a negative key —
    and counted by :meth:`unidentified_connections`. Dropping it silently would
    turn "the harness lost the correlation" into "there was less data", which
    reads as a healthy run; the scorer must be able to see it as a measurement
    error instead.
    """

    def __init__(self) -> None:
        self._live: List[_LiveRecord] = []
        self._lock = threading.Lock()

    def open(self, received_at: float) -> _LiveRecord:
        """Register a connection's record, stamped at ``received_at``."""
        rec = _LiveRecord(received_at)
        with self._lock:
            self._live.append(rec)
        return rec

    def _snapshot(self) -> List[_LiveRecord]:
        with self._lock:
            return list(self._live)

    def records(self) -> Dict[int, ServerRecord]:
        """Frozen snapshot of every connection's record, keyed by request id."""
        out: Dict[int, ServerRecord] = {}
        unidentified = 0
        for live in self._snapshot():
            rid = live.request_id
            if rid is None or rid < 0 or rid in out:
                unidentified += 1
                rid = -unidentified
            out[rid] = ServerRecord(
                request_id=rid,
                received_at=live.received_at,
                emitted_at=list(live.emitted_at),
                first_append_at=live.first_append_at,
                append_at=list(live.append_at),
            )
        return out

    def unidentified_connections(self) -> int:
        """Connections whose record could not be keyed by a unique preflight id."""
        seen = set()
        unidentified = 0
        for live in self._snapshot():
            rid = live.request_id
            if rid is None or rid < 0 or rid in seen:
                unidentified += 1
            else:
                seen.add(rid)
        return unidentified

    def clear(self) -> None:
        with self._lock:
            self._live = []


class _ConnectionCounter:
    """Increments the server's live-connection count for the life of a handler."""

    __slots__ = ("_server",)

    def __init__(self, server: "ShardedLoopServer") -> None:
        self._server = server

    def __enter__(self) -> "_ConnectionCounter":
        s = self._server
        with s._conn_lock:
            s._conns += 1
            s.total_conns += 1
            if s._conns > s.max_active_conns:
                s.max_active_conns = s._conns
        return self

    def __exit__(self, *exc) -> None:
        s = self._server
        with s._conn_lock:
            s._conns -= 1
        return None


class ShardedLoopServer:
    """Base for mock servers: one listening socket, ``num_loops`` accept loops.

    Subclasses implement ``_start_server(sock)`` — an async factory that starts
    serving on the given (already bound + listening) socket and returns an
    object with ``close()`` / ``wait_closed()`` (an asyncio ``Server`` or a
    websockets ``Server``).
    """

    def __init__(self, host: str = "127.0.0.1", num_loops: int = 8):
        self.host = host
        self.num_loops = max(1, num_loops)
        self.port: int = 0
        self._listen_sock: Optional[socket.socket] = None
        self._threads: List[threading.Thread] = []
        self._stoppers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._started = False
        self._stopped = False
        # Peak simultaneous connections, observed server-side. This is the same
        # ground truth every mock server reports, and it is what "achieved
        # concurrency" must mean everywhere: a COUNT of completed requests is
        # not concurrency, and reporting one as the other hides a client that
        # never applied the load it was asked to.
        self._conns = 0
        self.max_active_conns = 0
        self.total_conns = 0
        self._conn_lock = threading.Lock()
        # Which loop shard served each connection — the cheap way to verify
        # that accept sharding actually distributes.
        self._shard_of_thread = threading.local()
        self.shard_conn_counts: List[int] = []

    # ------------------------------------------------------------ connections
    def track_connection(self):
        """Context manager counting one live connection."""
        idx = getattr(self._shard_of_thread, "idx", None)
        if idx is not None and 0 <= idx < len(self.shard_conn_counts):
            self.shard_conn_counts[idx] += 1
        return _ConnectionCounter(self)

    @property
    def active_conns(self) -> int:
        with self._conn_lock:
            return self._conns

    def reset_connection_peak(self) -> None:
        with self._conn_lock:
            self._conns = 0
            self.max_active_conns = 0
            self.total_conns = 0
        self.shard_conn_counts = [0] * self.num_loops

    def shards_used(self) -> int:
        """Number of accept loops that have served at least one connection."""
        return sum(1 for c in self.shard_conn_counts if c > 0)

    # ---------------------------------------------------------------- lifecycle
    async def _start_server(self, sock: socket.socket):
        raise NotImplementedError

    def start(self):
        if self._started:
            return self
        self._started = True
        self.shard_conn_counts = [0] * self.num_loops

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, 0))
        # Must be listening before the loops dup it, so they share one accept
        # queue. The depth asked for here does not survive: each loop's
        # start_server/serve calls listen() again — see ACCEPT_BACKLOG.
        s.listen(ACCEPT_BACKLOG)
        self.port = s.getsockname()[1]
        self._listen_sock = s

        readies = [threading.Event() for _ in range(self.num_loops)]
        name = type(self).__name__

        def _run(idx: int):
            self._shard_of_thread.idx = idx
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            stop_ev = asyncio.Event()
            self._stoppers.append((loop, stop_ev))
            # Each loop serves its own dup of the shared listening fd; the
            # server takes ownership of (and closes) the dup on shutdown.
            assert self._listen_sock is not None
            shard_sock = self._listen_sock.dup()

            async def _main():
                server = await self._start_server(shard_sock)
                readies[idx].set()
                await stop_ev.wait()  # graceful shutdown signal
                server.close()
                try:
                    await server.wait_closed()
                except Exception:
                    pass

            try:
                loop.run_until_complete(_main())
            except Exception:
                readies[idx].set()
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        for i in range(self.num_loops):
            t = threading.Thread(
                target=_run, args=(i,), daemon=True, name=f"{name}-{i}"
            )
            t.start()
            self._threads.append(t)
        for r in readies:
            r.wait(timeout=5.0)
        return self

    def stop(self) -> None:
        """Shut every accept loop down. Safe to call more than once."""
        if self._stopped:
            return
        self._stopped = True
        for loop, ev in list(self._stoppers):
            try:
                loop.call_soon_threadsafe(ev.set)
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        self._stoppers = []
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except Exception:
                pass
            self._listen_sock = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---------------------------------------------------------------- addresses
    @property
    def http_base(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_base(self) -> str:
        return f"ws://{self.host}:{self.port}"
