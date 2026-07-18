# veeksha_native — native (C++) receive path

An optional free-threaded C++ extension that runs a single-thread `poll()` loop over many
sockets, parses SSE, and timestamps each chunk at true socket-read time. It exists to answer,
with numbers, whether a native dispatcher is worth building (see `analysis/07_native_dispatcher_decision.md`).

## Build
```bash
veeksha/native/build.sh /path/to/free-threaded-python   # e.g. .venv/bin/python
```
Requires a C++17 compiler and `pybind11`. Produces `veeksha_native.<ext-suffix>.so` next to
this file. The `.so` is git-ignored (build artifact). `veeksha.native.is_available()` reports
whether it's built; everything degrades gracefully to Python if not.

The module declares `py::mod_gil_not_used()`, so importing it does **not** re-enable the GIL on
free-threaded CPython (verified: `sys._is_gil_enabled()` stays False).

## Use
```bash
# compare native vs the Python pipeline on this box, same engine + same metric code:
veeksha preflight --target_concurrency 800 --compare_native true
```

## Measured (this laptop, through veeksha's own drift metrics)
Native (single thread), isolated runs with a well-provisioned engine (24 loops):

| C | native ivl-p99 | native stretch | engine's own jitter (measurement floor) |
|---:|:--|:--|:--|
| 400 | ~4 ms | 1.01 | ~1 ms |
| 800 | ~5–6 ms | 1.01 | ~2 ms |
| 1600 | ~7–9 ms | 1.02 | ~5–10 ms |

For comparison the Python pipeline (tuned to `ceil(C/300)` threads, pool uncapped) is honest
only to ~200–300 on this box, then drift climbs to 17–41 ms.

**Two caveats, both important:**
- At C ≥ ~800 the native ivl-p99 *tracks the engine's own jitter* — i.e. native's true drift is
  at or below what our Python test-engine can resolve, so those ms figures are a measurement
  floor, not a native limit. Numbers vary run-to-run with box load (a value near 16 ms appears
  only when the Python pipeline is hammering the same laptop alongside). A dedicated host / the
  engine on a separate box is needed to measure native cleanly above ~800.
- The probe does less per-chunk work than the full client (no json/token bookkeeping), so a
  production native client lands between these and Python.

**Bottom line:** native reaches ~1600 concurrency on **one** thread at ~1–2 % stretch; Python
needs 3–6 threads and still drifts more. The exact high-C ms can't be pinned down on one laptop.

## Scope / next
This wraps the **receive** path (the drift-dominant part). A full `TransportTimer` would also
own send-pacing (ASR) and hand Python a `TimedEventStream` (doc 05 §4). Packaging (build on
`pip install` via a setuptools/CMake ext) is a follow-up; today it's an opt-in local build.
