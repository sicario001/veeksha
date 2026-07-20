"""Lock-free-across-threads, thread-sharded CDF sketch for free-threaded Python.

Drift benchmarking showed the evaluator's single global
lock — held across per-chunk DDSketch inserts in
`PerformanceEvaluator.record_request_completed` — serializes all completion
processing and cancels the free-threading payoff. `DDSketch` is not thread-safe,
so the fix is not a finer lock but *sharding*: each thread accumulates into its
own DDSketch shard, and shards are merged on read.

Each shard carries its own lock, taken once per `put`/`extend` *batch*. In steady
state a shard has a single writer (its owning thread), so that lock is
uncontended — there is no cross-thread contention on the hot path, which is the
whole point. The per-shard lock exists only so a concurrent reader (e.g. the
streaming-metrics flush) can `merge` a shard safely while its owner may be
writing — DDSketch would otherwise corrupt/raise if its bins resized mid-read.
`DDSketch.merge` is exact, so the merged result equals a single unsharded sketch.

Drop-in compatible with `CDFSketch` for the write path (`put`/`extend`) and the
read path used by the evaluator (`get_summary`, `_to_df`, `plot_cdf`, `.sketch`,
`sum`, `len`) — the latter delegate to a freshly merged view.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from ddsketch import DDSketch

from veeksha.evaluator.cdf_sketch import CDFSketch

_RELATIVE_ACCURACY = 0.001


class _Shard:
    __slots__ = ("sketch", "lock")

    def __init__(self) -> None:
        self.sketch = DDSketch(relative_accuracy=_RELATIVE_ACCURACY)
        self.lock = threading.Lock()


class ShardedCDFSketch:
    """A CDF sketch whose write path has no cross-thread contention."""

    def __init__(
        self,
        metric_name: str,
        should_write_to_wandb: bool = True,
        unit: Optional[str] = None,
    ) -> None:
        self.metric_name = metric_name
        self.should_write_to_wandb = should_write_to_wandb
        self.unit = unit
        self._local = threading.local()
        self._shards: List[_Shard] = []
        self._shards_lock = threading.Lock()  # guards shard *registration* only

    def _shard(self) -> _Shard:
        shard = getattr(self._local, "shard", None)
        if shard is None:
            shard = _Shard()
            with self._shards_lock:  # once per thread, essentially uncontended
                self._shards.append(shard)
            self._local.shard = shard
        return shard

    # ---- hot path: per-thread shard, no cross-thread contention --------------
    def put(self, data: float) -> None:
        shard = self._shard()
        with shard.lock:
            shard.sketch.add(data)

    def extend(self, values: List[float]) -> None:
        shard = self._shard()
        with shard.lock:  # one acquire per batch, not per value
            for v in values:
                shard.sketch.add(v)

    # ---- read path: merge shards (safe vs concurrent writes) -----------------
    def _snapshot(self) -> List[_Shard]:
        with self._shards_lock:
            return list(self._shards)

    def merged_sketch(self) -> DDSketch:
        merged = DDSketch(relative_accuracy=_RELATIVE_ACCURACY)
        for shard in self._snapshot():
            with shard.lock:
                if shard.sketch.count:
                    merged.merge(shard.sketch)
        return merged

    def __len__(self) -> int:
        total = 0
        for shard in self._snapshot():
            with shard.lock:
                total += int(shard.sketch.count)
        return total

    @property
    def sum(self) -> float:
        total = 0.0
        for shard in self._snapshot():
            with shard.lock:
                total += float(shard.sketch.sum)
        return total

    @property
    def sketch(self) -> DDSketch:
        """Merged DDSketch view (for code that reads the raw sketch)."""
        return self.merged_sketch()

    def to_cdf_sketch(self) -> CDFSketch:
        """A `CDFSketch` backed by the merged data, for output/plotting."""
        cdf = CDFSketch(
            self.metric_name,
            should_write_to_wandb=self.should_write_to_wandb,
            unit=self.unit,
        )
        cdf.sketch = self.merged_sketch()
        return cdf

    # ---- read drop-ins that mirror CDFSketch --------------------------------
    def get_summary(self):
        return self.to_cdf_sketch().get_summary()

    def _to_df(self):
        return self.to_cdf_sketch()._to_df()

    def plot_cdf(self, *args, **kwargs) -> None:
        self.to_cdf_sketch().plot_cdf(*args, **kwargs)
