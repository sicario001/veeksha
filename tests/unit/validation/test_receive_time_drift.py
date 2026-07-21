"""Validation tests for streamed-chunk receive-time drift.

A mock streaming "engine" emits
chunks on an ABSOLUTE schedule (ground truth). Client coroutines share a single
asyncio event loop and timestamp each chunk on receipt — mirroring
veeksha/client/openai_chat.py:400. We measure drift = recorded_receive_time -
known_emit_time.

  * At low concurrency, drift is small (hard assertion).
  * Sharding the same load across more loops restores low drift (hard assertion)
    — this is the "raise num_client_threads / go native" fix from the doc.
  * The saturation knee is reported (not asserted) to avoid CI flakiness.

Uses only stdlib asyncio so it runs without the full client/tokenizer stack.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Dict, List, Tuple

import pytest

HOST = "127.0.0.1"
N_CHUNKS = 30
CHUNK_INTERVAL = 0.02
PER_CHUNK_CPU = 1500

_emit: Dict[Tuple[int, int], float] = {}
_emit_lock = threading.Lock()


def _start_server():
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    port_holder: List[int] = []
    conn_counter = [0]

    async def handle(reader, writer):
        try:
            await asyncio.wait_for(reader.readline(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        with _emit_lock:
            conn_id = conn_counter[0]
            conn_counter[0] += 1
        start = time.monotonic()
        for i in range(N_CHUNKS):
            target = start + i * CHUNK_INTERVAL
            now = time.monotonic()
            if target > now:
                await asyncio.sleep(target - now)
            with _emit_lock:
                _emit[(conn_id, i)] = time.monotonic()
            writer.write(f"data: {json.dumps({'i': i, 'c': conn_id})}\n".encode())
            try:
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                return
        writer.write(b"data: [DONE]\n")
        try:
            await writer.drain()
            writer.close()
        except Exception:
            pass

    async def main():
        server = await asyncio.start_server(handle, HOST, 0)
        port_holder.append(server.sockets[0].getsockname()[1])
        ready.set()
        async with server:
            await server.serve_forever()

    t = threading.Thread(
        target=lambda: (asyncio.set_event_loop(loop), loop.run_until_complete(main())),
        daemon=True,
    )
    t.start()
    ready.wait(timeout=5.0)
    return port_holder[0]


async def _one_stream(port: int, drifts: List[float]):
    reader, writer = await asyncio.open_connection(HOST, port)
    writer.write(b"GET /s\n")
    await writer.drain()
    while True:
        line = await reader.readline()
        if not line:
            break
        receive_time = time.monotonic()  # analogue of openai_chat.py:400
        s = line.decode().strip()
        if not s.startswith("data:"):
            continue
        body = s[5:].strip()
        if body == "[DONE]":
            break
        data = json.loads(body)
        acc = 0
        for k in range(PER_CHUNK_CPU):
            acc += k * k
        emit = _emit.get((data["c"], data["i"]))
        if emit is not None:
            drifts.append((receive_time - emit) * 1000.0)  # ms
    writer.close()


async def _run_on_one_loop(port: int, concurrency: int) -> List[float]:
    drifts: List[float] = []
    await asyncio.gather(*[_one_stream(port, drifts) for _ in range(concurrency)])
    return drifts


def _run_sharded(port: int, total: int, loops: int) -> List[float]:
    per = total // loops
    out: List[float] = []
    lock = threading.Lock()

    def worker():
        d = asyncio.run(_run_on_one_loop(port, per))
        with lock:
            out.extend(d)

    threads = [threading.Thread(target=worker) for _ in range(loops)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def _p99(xs: List[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.99 * (len(xs) - 1)))]


@pytest.fixture(scope="module")
def port():
    return _start_server()


def test_low_concurrency_drift_is_small(port):
    """At low concurrency the recorded chunk time tracks the true emit time."""
    _emit.clear()
    drifts = asyncio.run(_run_on_one_loop(port, 8))
    assert drifts, "no chunks recorded"
    p99 = _p99(drifts)
    assert p99 < 25.0, (
        f"receive-time drift too high at low concurrency: p99={p99:.1f}ms "
        f"(chunk interval is {CHUNK_INTERVAL*1000:.0f}ms)"
    )


def test_sharding_across_loops_reduces_drift(port):
    """The documented fix: spreading load across loops keeps drift low.

    Same total concurrency, 1 loop vs 4 loops. 4 loops must be clearly better —
    this is the 'raise num_client_threads / native receive path' remedy.
    """
    total = 200
    _emit.clear()
    one = _p99(_run_sharded(port, total, 1))
    _emit.clear()
    four = _p99(_run_sharded(port, total, 4))
    assert (
        four < one
    ), f"sharding did not help: 1-loop p99={one:.1f}ms 4-loop p99={four:.1f}ms"


@pytest.mark.slow
def test_report_receive_drift_knee(port, capsys):
    """Informational: report drift vs concurrency to locate the saturation knee."""
    lines = ["", "receive-time drift (single loop) vs concurrency:"]
    for c in (1, 25, 100, 200, 400):
        _emit.clear()
        drifts = asyncio.run(_run_on_one_loop(port, c))
        lines.append(
            f"  concurrency={c:>4}  p99={_p99(drifts):8.2f}ms  "
            f"max={max(drifts):8.2f}ms"
        )
    with capsys.disabled():
        print("\n".join(lines))
