"""Shared infrastructure for the preflight mock servers.

``ShardedLoopServer`` runs N asyncio event loops (one per thread) that all
accept from ONE shared listening socket. Binding N listeners with SO_REUSEPORT
does NOT distribute TCP accepts on macOS — the kernel delivers essentially every
connection to the last-bound listener (verified empirically), silently collapsing
"8 accept loops" into one saturated loop. Sharing a single listening socket
(each loop waits on its own dup of the same fd; whichever loop accepts first
wins the connection) distributes on Linux and Darwin alike, and the single bind
also removes the probe-then-rebind port race.

``ShardedTelemetry`` gives each loop thread its own sample list so the emit hot
path never takes a cross-thread lock (on free-threaded CPython a shared lock on
every chunk is real contention that both delays the send and inflates the
recorded jitter).
"""

from __future__ import annotations

import asyncio
import itertools
import socket
import threading
from typing import List, Tuple

__all__ = ["ShardedLoopServer", "ShardedTelemetry", "PhaseSpreader"]

# Connections that all emit on the same schedule come due in the same instant,
# and one event loop must then walk the whole burst — so whoever it serves last
# records that walk as "jitter". Spreading each connection's phase across one
# cadence window removes an artifact of the test rig (real sessions do not start
# in lockstep) and is what lets a mock server act as a punctual reference.
_PHASE_SLOTS = 50


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

    def p99(self) -> float:
        xs = sorted(self.merged())
        if not xs:
            return 0.0
        return xs[min(len(xs) - 1, int(round(0.99 * (len(xs) - 1))))]


class _ConnectionCounter:
    """Increments the server's live-connection count for the life of a handler."""

    __slots__ = ("_server",)

    def __init__(self, server: "ShardedLoopServer") -> None:
        self._server = server

    def __enter__(self) -> "_ConnectionCounter":
        s = self._server
        with s._conn_lock:
            s._conns += 1
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
    websockets ``WebSocketServer``).
    """

    def __init__(self, host: str = "127.0.0.1", num_loops: int = 8):
        self.host = host
        self.num_loops = max(1, num_loops)
        self.port: int = 0
        self._listen_sock: socket.socket | None = None
        self._threads: List[threading.Thread] = []
        self._stoppers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        # Peak simultaneous connections, observed server-side. This is the same
        # ground truth MockStreamingEngine reports, and it is what "achieved
        # concurrency" must mean everywhere: a COUNT of completed requests is
        # not concurrency, and reporting one as the other hides a client that
        # never applied the load it was asked to.
        self._conns = 0
        self.max_active_conns = 0
        self._conn_lock = threading.Lock()

    def track_connection(self):
        """Context manager counting one live connection."""
        return _ConnectionCounter(self)

    def reset_connection_peak(self) -> None:
        with self._conn_lock:
            self._conns = 0
            self.max_active_conns = 0

    async def _start_server(self, sock: socket.socket):
        raise NotImplementedError

    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, 0))
        s.listen(2048)
        self.port = s.getsockname()[1]
        self._listen_sock = s

        readies = [threading.Event() for _ in range(self.num_loops)]
        name = type(self).__name__

        def _run(idx: int):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            stop_ev = asyncio.Event()
            self._stoppers.append((loop, stop_ev))
            # Each loop serves its own dup of the shared listening fd; the
            # server takes ownership of (and closes) the dup on shutdown.
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
        for loop, ev in list(self._stoppers):
            try:
                loop.call_soon_threadsafe(ev.set)
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except Exception:
                pass
            self._listen_sock = None
