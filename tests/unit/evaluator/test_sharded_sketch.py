"""Tests for the lock-free thread-sharded CDF sketch."""

from __future__ import annotations

import threading
import time

from veeksha.evaluator.cdf_sketch import CDFSketch
from veeksha.evaluator.sharded_sketch import ShardedCDFSketch


def test_sharded_matches_unsharded_single_thread():
    ref = CDFSketch("m")
    sharded = ShardedCDFSketch("m")
    data = [float(i % 137) for i in range(5000)]
    ref.extend(data)
    sharded.extend(data)

    assert len(sharded) == len(ref) == 5000
    assert abs(sharded.sum - ref.sum) < 1e-6
    rs, ss = ref.get_summary(), sharded.get_summary()
    for k in rs:
        assert abs((rs[k] or 0) - (ss[k] or 0)) < 1e-6, k


def test_sharded_merge_is_correct_across_threads():
    sharded = ShardedCDFSketch("m")
    ref = CDFSketch("m")
    n_threads, per = 8, 2000
    all_values = []
    lock = threading.Lock()

    def worker(seed: int):
        vals = [float((seed * 31 + i) % 997) for i in range(per)]
        with lock:
            all_values.extend(vals)
        for v in vals:  # each thread writes its own shard, no lock
            sharded.put(v)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ref.extend(all_values)
    assert len(sharded) == n_threads * per
    # DDSketch.merge is exact, so quantiles match the single-sketch reference
    rs, ss = ref.get_summary(), sharded.get_summary()
    for k in rs:
        assert abs((rs[k] or 0) - (ss[k] or 0)) < 1e-6, k


def test_concurrent_puts_scale_without_a_global_lock():
    """Many threads hammering the sketch must not serialize on one lock.

    We assert only that concurrent writes complete quickly and correctly (a
    global-lock implementation would still be correct, so this is a smoke test
    for the lock-free design rather than a strict timing gate).
    """
    sharded = ShardedCDFSketch("m")
    n_threads, per = 8, 20000

    def worker():
        for i in range(per):
            sharded.put(float(i % 251))

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0

    assert len(sharded) == n_threads * per
    # generous ceiling; purely to catch pathological serialization/deadlock
    assert elapsed < 30.0
