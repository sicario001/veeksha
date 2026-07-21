# veeksha_native — the native (C++) transport + timing engine

An optional free-threaded pybind11 extension that owns the timing-critical
path of a benchmark run: firing requests on their arrival deadlines, holding
connection concurrency in event-driven reactor loops, timestamping every
received SSE chunk / WebSocket frame at socket-read time (CLOCK_MONOTONIC),
pacing outgoing audio/text on absolute deadlines, and firing dependent
multi-turn requests — with prior turns' outputs spliced into the next prompt —
the instant their think-time deadline arrives. No Python runs anywhere on the
per-event path.

Python keeps everything outside the measurement window: it compiles sessions
into request bytes/templates and traffic schedules before the run, and parses,
scores and persists results after it. The seam is
`veeksha/native/engine.py` (typed wrappers over the extension) →
`veeksha/native/transport.py` (per-modality wire building / result parsing) →
`veeksha/native/runner.py` + `benchmark.py` (schedule compilation and routing
behind `client.use_native_transport`).

## Build
```bash
veeksha/native/build.sh /path/to/free-threaded-python   # e.g. .venv/bin/python
```
Requires a C++17 compiler and `pybind11`; also builds automatically on
`pip install` (with graceful fallback when no compiler is available). The
`.so` is git-ignored. `veeksha.native.is_available()` reports whether it is
built; every caller degrades to the Python transport if not.

The module declares `py::mod_gil_not_used()`, so importing it does **not**
re-enable the GIL on free-threaded CPython.

## Engine entry points
- `run_batch` — per-request HTTP/SSE engine (chunked transfer-encoding
  decoded), closed-loop or open-loop via per-request arrival offsets;
  shardable across N reactor threads.
- `run_chains` — multi-turn conversations: per-turn request templates whose
  holes native fills with the JSON-escaped output of prior turns
  (Content-Length recomputed), fired at completion + think time.
- `ws_stream` / `ws_run_batch` — WebSocket receive + absolute-deadline paced
  sends (realtime TTS/STT), per-connection schedules, shardable.
- `NativeSseServer` — a sub-ms reference SSE server for validating the native
  client without a Python mock's punctuality floor.

Readiness notification uses kqueue (darwin/BSD) or epoll (Linux) behind one
`Reactor` interface with a portable `poll()` fallback
(`VEEKSHA_NATIVE_REACTOR=poll` selects it at runtime for A/B measurement).

## Scope boundaries (deliberate)
- Plaintext `http://` / `ws://` only; TLS endpoints route to the Python
  transport (remote TLS calls are network-bound, not drift-bound).
- Linear multi-turn conversations run natively; DAG-shaped sessions keep the
  Python history path.
- Bounded runs (`max_sessions > 0`); unbounded runs stay on Python.

## Validate on your box
```bash
veeksha preflight --target_concurrency 800 --compare_native true
```
runs the Python pipeline and this engine on the same concurrency ladder
against the same known-cadence mock server, with the same metric definitions.
