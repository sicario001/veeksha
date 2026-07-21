# Prototype: native main loop structures, pybind surface, and the Python↔C++ boundary

Companion to `preflight_and_native_loop.md` §3. Status: PROPOSAL / prototype.

## 0. The boundary problem, stated precisely

Today's loop is:

```python
_run_main_loop(session_generator, traffic_scheduler, evaluator, client,
               runtime_config, trace_recorder, benchmark_start_time,
               pregenerated_sessions)
```

Every argument is a live Python object. If any of them is *called* from a
native thread during the run, we pay: GIL/refcount traffic on the free-threaded
build, allocation on the hot path, and — worst — Python scheduling jitter
re-entering the exact path we went native to protect. So the rule the
prototype enforces:

> **Native threads never call Python. Ever.** Every argument is either
> (a) converted to a plain-data struct once at loop start, (b) compiled into
> plain data continuously by a Python *feeder* thread, or (c) served by a
> Python *drainer* thread that pulls plain data out. The only crossings are
> three well-defined pull/push APIs, all made *from* Python threads.

Per-argument fate:

| `_run_main_loop` argument | Fate | Residual interaction during run |
|---|---|---|
| `runtime_config` | → `NativeRuntimeConfig` struct at start | none |
| `traffic_scheduler` | → `TrafficPlanConfig` struct (native scheduler is built from it; Python scheduler object is not used in native mode) | none (RATE draws streamed as plain floats by the feeder) |
| `session_generator` / `pregenerated_sessions` | Python **feeder thread**: generate → `PlanCompiler` → `SessionPlan` structs → `feed_sessions()` (bounded ring) | Python→C++ push, GIL-released enqueue, off the dispatch path (native holds look-ahead; mirrors PrefetchWorker's burst-then-throttle) |
| `client` | **not passed at all.** Its *config* becomes `EndpointConfig` + per-request `TransportPlan`s built by `PlanCompiler` (which reuses the Python clients' body-building code paths) | none |
| `evaluator` | Python **drainer thread**: `drain_events()` → replay `register_request` / `record_request_completed` in order | C++→Python pull, batched, off the critical path by construction |
| `trace_recorder` | same drainer (DISPATCHED events carry what `record_dispatch` needs) | same |
| `benchmark_start_time` | clock handshake at construction (§5) | none |
| callbacks (`on_request_sent`/`on_request_dispatched`, `DispatchTracker`) | fully internal native events (the tracker gate is native) | none |
| monitor loop | stays Python; polls `counters()` (atomics snapshot) every 0.1 s | C++→Python pull, trivially cheap |

So the run-time interaction surface is exactly three calls, all initiated by
Python threads: `feed_sessions` / `feed_intervals`, `drain_events`, and
`counters` (+ `in_flight_request_ids` once, at timeout).

## 1. Input structs (converted once, or streamed as plain data)

```cpp
// ---------- static configuration (converted once at loop start) ----------

struct NativeRuntimeConfig {
  int    max_sessions;              // runtime.max_sessions; -1 = unlimited
  double benchmark_timeout_s;       // runtime.benchmark_timeout
  double post_timeout_grace_s;      // runtime.post_timeout_grace_seconds; -1 = wait all
  int    num_dispatcher_threads;    // runtime.num_dispatcher_threads (default 2)
  int    num_completion_threads;    // runtime.num_completion_threads (default 8) -> ack threads
  int    num_client_threads;        // resolved in Python with the SAME formula
                                    //   max(3, ceil(target/8)) (benchmark.py:102-113)
};

enum class TrafficKind : int { RATE = 0, CONCURRENT = 1, SEQUENTIAL_LAUNCH = 2 };
enum class TicketOrdering : int { DISPATCH = 0, PREFILL = 1, REQUEST = 2 };

struct TrafficPlanConfig {
  TrafficKind kind;
  // CONCURRENT (concurrent.py:29-48):
  int    target_concurrent_sessions = 0;
  double rampup_seconds = 0.0;      // native reproduces int(target * t/rampup)
  // SEQUENTIAL_LAUNCH:
  TicketOrdering ordering = TicketOrdering::DISPATCH;
  bool   cancel_session_on_failure = true;   // BaseTrafficConfig
  // RATE: interarrival draws are NOT generated natively. Python draws them
  // with the exact seeded chain (derive_seed(root,"interval","0") ->
  // RandomState -> poisson/gamma/fixed) and streams them via feed_intervals().
  // Bit-identical schedules across loop implementations by construction.
};

struct EndpointConfig {
  std::string host;
  int         port;
  std::string base_path;            // api_base path prefix ("" if none)
  std::vector<std::pair<std::string, std::string>> headers;  // incl. auth
  double      request_timeout_s;    // per-request budget (client config)
};

// ---------- per-session plan (compiled by Python, consumed by native) ----------

enum class TransportKind : int { TEXT_SSE = 0, TTS_HTTP = 1,
                                 TTS_REALTIME_WS = 2, STT_WS = 3 };

// Large payloads (STT base64 audio frames) are NOT inlined per session:
// many sessions share one clip (the Python client has a per-clip cache for
// the same reason, stt.py:118-121). Python registers a blob once and frames
// reference (blob_id, offset, len) slices. Keeps feed cost and memory at
// clip-cache parity with the Python main loop.
struct BlobRef { int blob_id = -1; long long offset = 0; long long len = 0; };

struct WsFrame {
  std::string payload;          // inline payload (small frames), OR:
  BlobRef     blob;             // slice of a registered blob (audio appends)
  double      send_offset_ms;   // absolute offset from pacing anchor; <0 = "asap, in order"
};

struct TransportPlan {
  TransportKind kind;

  // -- TEXT_SSE / TTS_HTTP (templated HTTP/1.1 request) --
  // wire = header_prefix + itoa(body_len) + header_suffix + body
  // body = seg[0] + escape(extract(history_refs[0])) + seg[1] + ...
  std::string header_prefix;            // through "Content-Length: "
  std::string header_suffix;            // remaining headers + CRLFCRLF
  std::vector<std::string> body_segments;
  std::vector<int> history_refs;        // node_ids whose extracted output fills holes
                                        // (empty for single-turn / pre-baked history)

  // -- TTS_REALTIME_WS / STT_WS --
  std::string ws_path;                  // "/v1/realtime", provider-specific
  std::vector<WsFrame> setup_frames;    // session.update etc. (ordered, asap)
  std::vector<WsFrame> paced_frames;    // text deltas / audio appends on schedule
  std::vector<WsFrame> finish_frames;   // commit / response.create (after paced)
  std::vector<std::string> done_markers;// terminal event substrings ("response.done",
                                        // "transcription.done", provider-specific)
  // pacing anchor semantics: TTS_REALTIME_WS -> offsets from post-handshake
  // (matches realtime_tts.py t_start-relative deltas); STT_WS -> offsets from
  // first paced send (matches stt.py audio_started_at).
};

struct RequestPlan {
  long long request_id;
  int       node_id;
  double    wait_after_ready_s;                    // think time (SessionNode)
  std::vector<std::pair<int, bool>> parents;       // (node_id, is_history_parent)
  TransportPlan transport;
};

struct SessionPlan {
  long long session_id;
  std::vector<RequestPlan> requests;    // topological order (Python validates)
  int dispatch_ticket_base = -1;        // SEQUENTIAL_LAUNCH: first root ticket
};
```

Notes on fidelity to today's semantics:

- `history_refs` gives the native loop the same *dynamic* history splicing the
  schedulers do (`_record_history`/`_populate_history`): the hole is filled
  with the accumulated conversation of the referenced parent turn (native
  keeps per-node extracted output + the pre-serialized user turn from the
  plan). Dataset-prepopulated history (conversation flavor,
  `is_history_parent=False`) arrives **already baked into `body_segments`**
  by the compiler — zero native work, exactly like the Python scheduler
  returns `[]`.
- Non-text history parents (image/audio content blocks in `_record_history`)
  are not supported natively: the compiler detects them and the run falls
  back to the Python main loop, loudly.

## 2. Output structs (drained by Python)

```cpp
struct ChunkStamp { double offset_ms; int size; };   // read-time, from request send

struct RawRequestResult {
  long long request_id;
  long long session_id;
  int       session_total_requests;
  int       status = 0;                // HTTP status / WS close mapping
  std::string error;                   // "" = success

  // The five lifecycle stamps, ms from the native-loop epoch (§5), SAME anchor
  // points as the Python loop:
  double scheduler_ready_ms;           // ready pop (dispatch.py:77 analogue)
  double scheduler_dispatched_ms;      // back-to-back with ready (dispatch.py:78)
  double client_picked_up_ms;          // after ticket gate (client_runner.py:116)
  double client_completed_ms;          // stream terminal event
  double result_processed_ms;          // native completion-ack (completion.py:50)

  std::vector<ChunkStamp> recv_stamps;   // per SSE event / audio delta / transcript delta
  std::vector<double>     send_offsets_ms; // per paced send, stamped BEFORE send()
  std::string             content;       // extracted assistant text / final transcript
  long long               recv_bytes = 0;// total payload bytes (audio duration source)
  // realtime event offsets: ("session.updated", t), ("response.done", t), ...
  std::vector<std::pair<std::string, double>> event_offsets_ms;
};

enum class EventKind : int { DISPATCHED = 0, COMPLETED = 1 };

// One ordered event log. DISPATCHED must be drained before that request's
// COMPLETED (the native loop guarantees emission order; the drainer preserves it),
// so evaluator.register_request/record_request_completed replay is valid.
struct NativeLoopEvent {
  EventKind kind;
  long long request_id;
  long long session_id;
  int       session_total_requests;
  double    ready_ms, dispatched_ms;   // DISPATCHED payload (register + trace)
  RawRequestResult result;             // COMPLETED payload (empty for DISPATCHED)
};

struct NativeLoopCounters {                 // lock-free snapshot for the monitor
  long long sessions_completed;         // maps to evaluator.get_session_counts()
  long long sessions_errored;
  long long sessions_seen;
  long long requests_dispatched;
  long long requests_completed;
  long long in_flight;
  bool      intake_exhausted;           // feeder called close_intake()
  bool      idle;                       // no pending work (has_pending_work == False)
};
```

## 3. pybind11 surface (module `veeksha_native`, `py::mod_gil_not_used()`)

```cpp
PYBIND11_MODULE(veeksha_native, m, py::mod_gil_not_used()) {
  // --- plain-data classes: py::class_ with def_readwrite on every field ---
  // NativeRuntimeConfig, TrafficPlanConfig, EndpointConfig, BlobRef, WsFrame,
  // TransportPlan, RequestPlan, SessionPlan, ChunkStamp, RawRequestResult,
  // NativeLoopEvent, NativeLoopCounters
  // + enums: TrafficKind, TicketOrdering, TransportKind, EventKind

  py::class_<NativeBenchmarkLoop>(m, "NativeBenchmarkLoop")
    .def(py::init<NativeRuntimeConfig, TrafficPlanConfig, EndpointConfig,
                  double /*py_monotonic_anchor_s*/>())
    // ---- intake (called by the Python feeder thread) ----
    .def("register_blob", &NativeBenchmarkLoop::register_blob,
         py::arg("data"),                    // py::bytes -> copied once
         "Register shared payload bytes (e.g. one audio clip's pre-encoded "
         "append frames); returns blob_id for WsFrame.blob references.")
    .def("feed_sessions", &NativeBenchmarkLoop::feed_sessions,
         py::arg("plans"), py::call_guard<py::gil_scoped_release>(),
         "Enqueue compiled SessionPlans. Blocks (GIL released) if the intake "
         "ring is full — natural backpressure replacing PrefetchWorker's "
         "throttle. Returns number accepted.")
    .def("feed_intervals", &NativeBenchmarkLoop::feed_intervals,
         py::arg("intervals_s"), py::call_guard<py::gil_scoped_release>(),
         "RATE only: stream the next batch of seeded interarrival draws.")
    .def("close_intake", &NativeBenchmarkLoop::close_intake,
         "Generator exhausted / max_sessions reached; loop may finish.")
    // ---- drain (called by the Python drainer thread) ----
    .def("drain_events", &NativeBenchmarkLoop::drain_events,
         py::arg("max_items") = 256, py::arg("timeout_s") = 0.1,
         "Pop up to max_items ordered NativeLoopEvents; blocks up to timeout_s "
         "(GIL released while waiting, re-acquired to build the returned "
         "list). Empty list = nothing pending.")
    // ---- monitor / control (called by the Python monitor thread) ----
    .def("counters", &NativeBenchmarkLoop::counters)          // atomics snapshot
    .def("in_flight_request_ids", &NativeBenchmarkLoop::in_flight_request_ids)
    .def("request_stop", &NativeBenchmarkLoop::request_stop,
         py::arg("grace_s") = 0.0,
         "Benchmark-timeout path: stop dispatching new work; after grace_s, "
         "fail remaining in-flight as timed out (keeping partial streams).")
    .def("join", &NativeBenchmarkLoop::join, py::arg("timeout_s"),
         py::call_guard<py::gil_scoped_release>());
}
```

Binding-level decisions worth calling out:

- **Everything crossing the boundary is copyable plain data** (strings,
  doubles, vectors). No `py::object` is stored anywhere in the native loop; no
  keep-alive relationships; nothing to refcount from native threads.
- `feed_sessions` copies the plans into native memory under
  `gil_scoped_release` (pybind converts before release via the arg — in
  practice: accept `std::vector<SessionPlan>` by value; conversion happens on
  the feeder thread, which is Python and *supposed* to spend its time here).
  Conversion cost sits on the feeder thread exactly like generation cost sits
  on PrefetchWorker today.
- `drain_events` is the *only* API that builds Python objects, and it runs on
  the drainer thread. Batched (default 256) so per-event overhead amortizes.
- `register_blob` keeps STT feed costs sane: one 5 s clip ≈ 160 KB PCM ≈
  213 KB base64; 500 sessions sharing 20 clips = 20 blobs, not 500 copies.

## 4. Python-side shim: `_run_native_main_loop`

```python
def _run_native_main_loop(session_generator, traffic_config, evaluator, client_config,
                          runtime_config, trace_recorder, benchmark_start_time,
                          pregenerated_sessions):
    compiler = PlanCompiler(client_config)      # per-modality wire builders; reuses
                                                # the Python clients' body-building code
    loop = veeksha_native.NativeBenchmarkLoop(
        to_native_runtime(runtime_config, resolved_client_threads),
        to_native_traffic(traffic_config),
        to_native_endpoint(client_config),
        py_monotonic_anchor_s=time.monotonic(),  # §5 clock handshake
    )

    # (1) FEEDER — replaces PrefetchWorker's producer half (still one thread,
    #     still guarded by generator_lock semantics, still counts max_sessions)
    def feed():
        for session in iter_sessions(session_generator, pregenerated_sessions,
                                     runtime_config.max_sessions):
            plans, intervals = compiler.compile(session)   # blobs registered lazily
            loop.feed_intervals(intervals)                 # RATE draws, seeded in Python
            loop.feed_sessions([plans])
        loop.close_intake()

    # (2) DRAINER — replaces CompletionWorker's *evaluator* half (the scheduler
    #     half — notify_completion — is native). Order-preserving replay.
    def drain():
        while True:
            for ev in loop.drain_events(max_items=256, timeout_s=0.1):
                if ev.kind == DISPATCHED:
                    evaluator.register_request(ev.request_id, ev.session_id,
                                               anchor + ev.dispatched_ms/1e3,
                                               channels_of(ev), requested_output_of(ev))
                    if trace_recorder: trace_recorder.record_dispatch(...)
                else:
                    evaluator.record_request_completed(
                        ev.request_id, ev.session_id,
                        completed_at=anchor + ev.result.client_completed_ms/1e3,
                        response=to_request_result(ev.result))   # exact metrics keys of §1(C)
            if done_draining(loop): return

    # (3) MONITOR — unchanged algorithm (benchmark_utils._monitor_for_completion),
    #     reading loop.counters() instead of evaluator/scheduler queries; the
    #     timeout path calls loop.request_stop(grace) + loop.in_flight_request_ids().
```

The feeder needs `channels_of(ev)`/`requested_output_of(ev)` — i.e. the
drainer must map `request_id` back to the original Python `Request`. The
compiler keeps a `request_id -> (channels, requested_output, metadata)` dict
(bounded by in-flight + drain lag; entries dropped after COMPLETED). This dict
is touched only by Python threads.

## 5. Clock handshake (`benchmark_start_time` correspondence)

Python's `time.monotonic()` and C++ `clock_gettime(CLOCK_MONOTONIC)` are not
guaranteed to share an epoch on every platform (macOS in particular). The
constructor takes the caller's `time.monotonic()` reading; native immediately
pairs it with its own `now_ms()` and stores the offset. Every stamp is kept
native-side as ms-from-native-loop-epoch and converted by the *drainer* to Python
monotonic floats (`anchor + offset`), so evaluator/report code sees the same
clock domain as today, and cross-implementation comparisons share one basis. (The
handshake costs one crossing at construction; measured skew on the dev box is
sub-microsecond, and stays constant for the run.)

## 6. What happens if we *don't* convert an argument (the failure modes)

For the record — the designs this prototype deliberately rejects:

- **Passing `client` in and calling `send_request` from native dispatch**
  (Python-in-the-loop): every dispatch re-enters asyncio; the native scheduler
  then measures Python's scheduling noise. Whole exercise defeated.
- **Passing `evaluator` in and calling it per completion from a native
  thread**: requires acquiring Python thread state per event on the
  free-threaded build, allocating dicts on the ack path — completion drift
  (the metric preflight gates!) would inherit Python jitter. The pull-drain
  keeps the ack path pure and moves the Python cost to a thread whose lag
  cannot delay refill.
- **Keeping the Python `traffic_scheduler` and asking it for readiness**: the
  single-Condition lock *is* the measured 175-stream ceiling; any design that
  keeps it in the dispatch path keeps the ceiling.
- **Generating RATE draws natively with a ported RNG**: silently divergent
  schedules the day numpy changes anything; precomputed draws are bit-exact
  and testably identical (diff the dispatch logs).

## 7. Refactor: `_run_main_loop` takes ONLY the minimal struct interface
## (applies to BOTH loop implementations — this is a prerequisite PR, pure Python)

Decision: rather than keeping today's signature for the Python main loop and a
narrow one for native, we refactor the Python loop to the *same* minimal
interface. Anything that is a full Python class not needed inside the loop is
removed from the signature. After the refactor there is one protocol, two
implementations:

```python
# veeksha/loop/interface.py (new)

@frozen_dataclass
class MainLoopConfig:
    runtime: RuntimeConfig        # existing frozen dataclass — already plain data
    traffic: BaseTrafficConfig    # existing frozen dataclass (poly: rate/concurrent/seq)
    client: BaseClientConfig      # existing frozen dataclass (endpoint, timeouts, pacing)
    monotonic_anchor: float       # replaces benchmark_start_time argument
    # NOTE: these vidhi configs are plain, immutable, and mirror 1:1 onto the
    # C++ structs of §1 (NativeRuntimeConfig / TrafficPlanConfig /
    # EndpointConfig). The native adapter converts them once at start.

class SessionSource(Protocol):
    """Pull-based session intake. Replaces: session_generator +
    pregenerated_sessions + generator_lock + SharedSessionCounter."""
    def next_session(self) -> Optional[Session]   # None = exhausted; thread-safe;
                                                  # max_sessions enforced HERE

class MainLoop(Protocol):
    """One control surface for both loop implementations. Replaces _run_main_loop."""
    def start(self, source: SessionSource) -> None
    def drain_events(self, max_items: int = 256, timeout_s: float = 0.1) -> list[LoopEvent]
    def counters(self) -> LoopCounters            # sessions done/errored/seen,
                                                  # dispatched, completed, in_flight,
                                                  # intake_exhausted, idle
    def in_flight_request_ids(self) -> set[int]
    def dispatched_request_ids(self) -> set[int]  # timeout bookkeeping (see below)
    def request_stop(self, grace_s: float) -> None
    def join(self, timeout_s: float) -> bool
```

What left the signature, and where it went:

| Was an argument | Now | Why |
|---|---|---|
| `traffic_scheduler` (live object) | `MainLoopConfig.traffic` (config); each loop implementation builds its own scheduler internally | the loop needs scheduling *semantics*, not a shared mutable object |
| `client` (live object) | `MainLoopConfig.client` (config); PythonMainLoop constructs the client via `ClientRegistry` internally; NativeMainLoop compiles plans + `EndpointConfig` | same |
| `evaluator` | **gone from the loop entirely** — the loop emits an ordered `LoopEvent` stream (`DISPATCHED`/`COMPLETED`), consumed outside | the evaluator is scoring, not scheduling; identical drain consumption for both loop implementations |
| `trace_recorder` | gone — `DISPATCHED` events carry what `record_dispatch` needs | same |
| `session_generator` / `pregenerated_sessions` | `SessionSource` | one intake abstraction; pregeneration = a list-backed source |
| `benchmark_start_time` | `MainLoopConfig.monotonic_anchor` | plain float |

**PythonMainLoop** = today's workers with exactly two internal changes:

1. `DispatchWorker` emits `LoopEvent.DISPATCHED` into an internal event queue
   instead of calling `evaluator.register_request`/`trace_recorder` directly;
   `CompletionWorker` still calls `scheduler.notify_completion` first (that is
   loop-internal and timing-critical), then emits `LoopEvent.COMPLETED`
   instead of calling `evaluator.record_request_completed`.
2. Scheduler/client are constructed inside `start()` from the configs.

`drain_events()` on the Python main loop just pops that internal queue — so the
**consumption model is pull on both loop implementations**, and the code that replays
events into the evaluator is shared verbatim:

```python
# veeksha/loop/scoring.py — shared by both loop implementations
class ResultDrain:
    """Pool of threads consuming loop.drain_events() and replaying into the
    evaluator + trace recorder. Events are routed to workers by
    hash(session_id): per-session order (DISPATCHED before COMPLETED, turn
    order) is preserved; cross-session order is irrelevant to the evaluator.
    Pool size defaults to runtime.num_completion_threads so ASR scoring keeps
    the same parallelism it has today on the Python main loop."""
```

**`_run_benchmark` after the refactor:**

```python
source   = build_session_source(session_generator, pregenerated, runtime)
loop     = create_main_loop(kind, MainLoopConfig(...))   # python | native
drain    = ResultDrain(loop, evaluator, trace_recorder, pool_size=...)
loop.start(source); drain.start()
_monitor_for_completion(loop, evaluator, runtime)   # polls loop.counters()
loop.request_stop(grace)/join(); drain.join()       # drain fully flushes first
evaluator.finalize(); evaluator.save(...)
```

Two behavior notes (both deliberate, both apply to the Python main loop too):

- **Timeout bookkeeping moves to the loop handle.** Today the monitor
  snapshots `evaluator.get_registered_request_ids()`; with drain lag that set
  would trail reality. The monitor now uses `loop.dispatched_request_ids()` ∩
  `loop.in_flight_request_ids()` (both loop-side, exact);
  `evaluator.set_included_requests(...)` is still applied at grace end, with
  ids from the loop.
- **Evaluator scoring moves off the completion threads** (to the drain pool)
  on the Python main loop as well. Completion workers get cheaper (ack + emit),
  which if anything *reduces* completion-queue drift; the preflight measures
  it before/after the refactor as part of PR-1's baseline.

**Level 2 (explicitly deferred):** making the Python main loop consume compiled
`SessionPlan`s and execute wire bytes through thin per-transport executors —
i.e. one plan compiler feeding both loop implementations. Maximum parity, but it rewrites
the Python clients' execution path; not needed to reach the minimal
interface, and the schedule/wire parity tests (§8) already pin cross-implementation
equivalence. Revisit only if maintaining body-building in two places (clients
vs compiler) proves error-prone.

## 8. Prototype validation plan (before real implementation)

1. **Struct round-trip test**: build a `SessionPlan` in Python for each of the
   four transports from real generator output; assert compiled wire bytes ==
   the bytes the Python clients would send (capture via mock server).
2. **Order guarantee test**: high-rate mock run; assert every DISPATCHED
   precedes its COMPLETED in drain order and session finalize counts match.
3. **Schedule parity test**: same seed, RATE + CONCURRENT(rampup) + SEQUENTIAL
   (3 orderings); diff native dispatch log vs Python main loop dispatch trace.
4. **Boundary micro-bench**: feed/drain throughput with realistic plan sizes
   (incl. STT blobs) to confirm feeder/drainer keep up at target rates with
   headroom (they are off-path, but starvation would idle the native loop).
