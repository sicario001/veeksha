# Design: Measurement-Fidelity Preflight + Native C++ NativeLoop

Branch: `veeksha-native-preflight` (based on `tts_merge`). Status: PROPOSAL.

Scope: two work items, deliberately without any cross-modality abstraction
refactor —

1. **Preflight**: a `veeksha preflight` gate that measures, on this machine and
   before a real run, the timing drift of the benchmark harness itself at the
   concurrency you intend to benchmark, using mock servers with deterministic
   emit schedules — through the *real, complete* pipeline.
2. **Native main loop**: C++ implementations of the benchmark main loop's workers
   (`PrefetchWorker`, `DispatchWorker`, `CompletionWorker`, `ClientWorker`,
   `ThreadPoolManager`) — i.e. `_run_main_loop` runs natively with the same
   logic, config parameters, queues and lifecycle timestamps as the Python
   loop, so metrics and checks stay directly comparable.

Everything below references the code as it stands on this branch.

---

## 1. The benchmark pathways today: what is common, where modalities diverge

Every benchmark — text, TTS, STT — runs **one identical pipeline**
(`veeksha/benchmark.py:86` `_run_main_loop`):

```
SessionGenerator ──PrefetchWorker(×1, generator_lock)──▶ scheduler.schedule_session
TrafficScheduler ──DispatchWorker(×num_dispatcher_threads): wait_for_ready──▶
   evaluator.register_request + (request, session_id, session_size,
   scheduler_ready_at, dispatched_at) ──▶ client_queues[power-of-two pick]
ClientWorker(×num_client_threads, one asyncio loop each) ──▶
   client.send_request(...) ──▶ output_queue
CompletionWorker(×num_completion_threads) ──▶
   scheduler.notify_completion  →  evaluator.record_request_completed
```

**Common to all modalities** (single implementation, no per-modality code):

- The session model: `Session{id, session_graph, requests}`
  (`core/session.py:8`), nodes carrying `wait_after_ready` (think time), edges
  carrying `is_history_parent` (`core/session_graph.py:5-15`). Multi-turn
  release (`ready_at = max(parent completions) + wait_after_ready`,
  `session_graph.py:58-75`) and history splicing
  (`_record_history`/`_populate_history`) live **in the scheduler**, never in
  dispatch/client code.
- The scheduler interface (`traffic/base.py`): `schedule_session`,
  `wait_for_ready`, `pop_ready`, `notify_completion`,
  `get_in_flight_request_ids`, `has_pending_work` — and the locking model: in
  all three schedulers, **one `threading.Condition` guards everything**
  (`concurrent.py:32`, `rate.py:33`, `sequential_launch.py:41`). Every
  dispatch decision and every completion notification serializes on it.
- The dispatch handoff: the 5-tuple
  `(request, session_id, session_size, scheduler_ready_at, dispatched_at)`
  produced at `workers/dispatch.py:97`, consumed at
  `workers/client_runner.py:99-105`; power-of-two queue selection
  (`dispatch.py:53-64`).
- The client contract (`client/base.py:52-76`): one async
  `send_request(request, session_id, session_total_requests,
  on_request_sent, on_request_dispatched) -> RequestResult`, identical for all
  six clients; the two-phase callbacks feed `DispatchTracker` orderings
  (`dispatch`/`prefill`/`request`, `client_runner.py:118-139`).
- The five lifecycle timestamps, same for every modality, all
  `time.monotonic()`:
  `scheduler_ready_at`/`scheduler_dispatched_at` (DispatchWorker,
  `dispatch.py:77-78`), `client_picked_up_at` (ClientWorker,
  `client_runner.py:116`), `client_completed_at` (inside the client),
  `result_processed_at` (CompletionWorker, `completion.py:50`).
- Completion ordering: `notify_completion` (scheduler refill / next-turn
  release) is called **before** `evaluator.record_request_completed` on the
  same CompletionWorker thread (`completion.py:48-66`).
- Termination monitor, timeout/grace drain (`benchmark_utils.py:253-342`),
  seeding (`core/seeding.py` — blake2b-derived hierarchical streams), and the
  runtime config knobs (`config/runtime.py`).

**Where the modalities actually diverge** — exactly three places:

**(A) Request-content preparation** — *before* the timing window, Python-only:

| Modality | Preparation | Where |
|---|---|---|
| text | message building + HF tokenization (transformers) | `openai_chat.py:83-130`, `core/tokenizer.py:14` |
| TTS (HTTP) | passthrough `input_text` | `tts.py:122` |
| TTS (realtime) | whitespace segmentation into paced deltas (no tokenizer) | `realtime_tts.py:238`, `utils.py:47-80` |
| STT | librosa decode → PCM16 → slice → pre-encoded base64 frames, per-clip cache, in executor, before `t_start` | `stt.py:63-121, 284-318, 560-577` |

**(B) The transport exchange inside `send_request`** — *the* timing window:

| Modality | Transport | Direction | Client-side pacing | Timing anchors |
|---|---|---|---|---|
| text chat | httpx SSE POST `/v1/chat/completions` | receive-only stream | none (server-paced) | `t_start` just before stream open (`openai_chat.py:392`); per-SSE-event stamps |
| text completions | httpx non-streaming POST | single response | none | start/complete only |
| TTS HTTP | httpx byte-stream POST `/v1/audio/speech` | receive-only stream | none | `t_start` pre-stream (`tts.py:139`); per-chunk reads |
| TTS realtime | websockets | **bidirectional**: paced text-delta sends + audio-delta receives | absolute-deadline `TextDeltaPacer` (`realtime_tts.py:284-289`) | `t_start` pre-connect (`:276`); all event offsets relative to it |
| STT | websockets | **bidirectional, send-dominated**: 1×-realtime paced audio appends + transcript-delta receives | absolute-deadline audio schedule = `audio_started_at + bytes/rate` (`stt.py:369-383`) | `audio_started_at` (first send) and `audio_end_at` (EOF) (`stt.py:365-385`) |

Provider dialects (chat vs completions; `vllm_realtime` vs
`vajra_openai_realtime`, `stt.py:682-776`) change only message shapes, never
the pathway.

**(C) Metric extraction in the evaluator** — *after* the timing window:

| Modality | Keys consumed | Metrics derived | Where |
|---|---|---|---|
| text | `inter_chunk_times`, `num_output_tokens`, `num_{total,delta}_prompt_tokens`, `is_stream` | TTFC=`ict[0]`, E2E=`sum(ict)`, TPOT=(E2E−TTFC)/(n−1), TBC=mean `ict[1:]` | `text.py:43-82, 258-265` |
| TTS | `ttfc`, `end_to_end_latency`, `audio_chunk_timestamps`, `text_delta_timestamps`, `sample_rate`, `*_offset_ms` events | TTFA, RTF, streaming-RTF, interactivity family (fixed-delay/buffer-target playback sims, `required_startup_delay_ms`, `done_after_last_audio_ms`, …) | `audio.py:194-267`, `audio_interactivity.py:48-329` |
| STT | `final/partial_transcript`, `transcript_snapshots`, `reference_word_timestamps`, `time_to_*` scalars, `pcm_byte_count` | WER (jiwer + vendored normalizer), word-level visibility interactivity, latency scalars | `asr.py:219-297`, `asr_interactivity.py:36-126` |

**The consequence that shapes both work items:** (A) and (C) sit outside the
timing window; (B) differs only in *what bytes go out/come in and when*. So:

- the **preflight** must certify the common pipeline once and the three (B)
  transports each;
- the **native main loop** must replicate the common pipeline once (workers,
  queues, scheduler, stamps) and implement (B) as three transport state
  machines fed with **pre-built wire content** — (A) stays Python (pre-run),
  (C) stays Python (post-run / off the critical path).

---

## 2. Part 1 — Preflight

### 2.1 What "supposed to receive" and "supposed to send" mean (req. 1.4)

Two definitions, used consistently by every check, chosen so that *constants
we don't want to measure cancel out* and *variable harness error does not*:

**RECEIVE drift — cadence-relative, per chunk.** The mock emits chunk *i* of a
stream at absolute deadline `T_conn + prefill + i·dt`. We do **not** score
absolute arrival (that would include connect latency, TTFB, network transit
and per-connection phase — all legitimate, none of them harness error). We
score the *gap*: `err(i) = |recorded_gap(i) − dt|`. Every constant offset
cancels; what remains is exactly the error that corrupts TPOT/TBC/TTFA-family
metrics: harness receive-loop scheduling delay, stamp-placement error (parse
cost charged to arrival), and (localhost: negligible) network jitter. Gate =
p99 of `err` across all chunks of all streams at a rung, plus **stream
stretch** (last−first arrival over ideal span) as the aggregate check — the
two are complementary: stretch ≈ 1.0 can hide alternating early/late gaps;
per-gap error alone can hide slow cumulative drift.
What is deliberately *excluded*: server lateness. The mock records its **own**
emit lateness against its own deadlines; if the mock was late, the rung is
**native-loop-limited** (see 2.4), not a client failure.

**SEND drift — absolute-schedule-anchored, per chunk (ASR pacing, req. 1.3).**
The 1×-realtime schedule for audio chunk *i* is
`scheduled(i) = audio_started_at + chunk_bytes·i / (bytes_per_sample ·
sample_rate)` — exactly the schedule the STT client already paces against
(`stt.py:369-372`). Drift(i) = `max(0, actual_send_stamp(i) − scheduled(i))`
(late-only: an absolute-deadline pacer is never meaningfully early; earliness
within jitter is not an error). The send stamp is taken immediately **before**
the socket send call — the moment the harness decided to send; stamping after
the call would charge kernel/syscall time to pacing.
Two vantages, reported separately, never mixed:
- **client-side** (gated): the client's own per-chunk send stamps vs the
  schedule → pure harness pacing fidelity;
- **server-side ground truth** (reported): the mock server stamps every
  `input_audio_buffer.append` on **raw frame receipt, before any JSON/base64
  parse**, and scores those arrivals against the same schedule. The
  server-minus-client gap *is* the network + server-accept floor,
  disaggregated instead of silently misattributed to the client.

**Completion-queue drift (pipeline-health, not wire timing).**
`result_processed_at − client_completed_at` = output-queue wait +
CompletionWorker dequeue latency. This lag delays
`scheduler.notify_completion` (`completion.py:52`) and therefore biases
closed-loop refill and multi-turn `ready_at` computation. Gated on the Python
path; it is the number that tells you your `num_completion_threads` is
undersized before your session metrics silently stretch.

**Included/excluded summary** (printed in the report legend):

| Reported number | includes | excludes |
|---|---|---|
| receive drift p99 | harness receive scheduling + stamp placement + network *jitter* | connect, TTFB/prefill, constant network latency, server lateness (separate column) |
| send drift p99 (client) | harness pacing scheduling + stamp placement | network, server |
| send drift p99 (server-observed) | client drift + network transit + server accept | server JSON/base64 parse (stamped before it) |
| completion drift | queue wait + dequeue scheduling | evaluator scoring time (stamped before scoring) |

### 2.2 Where the drift computations live (req. 1, placement)

New package **`veeksha/preflight/`** + a `veeksha preflight` CLI subcommand:

```
veeksha/preflight/
  __init__.py
  mock_native loop.py      # text SSE mock server (known cadence, self-lateness telemetry)
  audio_server.py     # realtime-TTS + STT WS mock servers (same discipline)
  sharded_server.py   # shared accept-sharding + phase-spreading infrastructure
  drivers.py          # per-modality workloads: build sessions/config, run the REAL pipeline
  scorer.py           # pure drift math over recorded per-chunk data (no I/O, unit-testable)
  validator.py        # rungs, gates, tri-state verdicts, ladder
  report.py           # human-readable report + column legend + exit code
  runner.py           # CLI entry (config: veeksha/config/preflight.py)
```

The split that answers "where do the computations live":

- **Recording happens in the clients, scoring happens in `preflight/scorer.py`
  after the run.** The hot path only ever *records* timestamps it already
  takes (or should take — §2.5); no drift arithmetic, comparison, or
  aggregation runs inside the timing window. The preflight then reads the
  same `RequestResult.channels[..].metrics` the evaluator reads — plus the
  mock servers' own telemetry — and does all math offline. This guarantees
  the preflight measures the pipeline *as it ships*, not an instrumented
  variant (req. 1.1), and it means zero new overhead in real benchmark runs.
- The **drivers run `_run_main_loop` itself** — real `PrefetchWorker`,
  real scheduler, real `DispatchWorker`/`ClientWorker`/`CompletionWorker`,
  real clients, with a thin collecting evaluator (the evaluator API's
  `register_request`/`record_request_completed` surface, accumulating raw
  results; scoring-side metric math is not what preflight certifies). The
  workload is expressed as a normal `BenchmarkConfig`: synthetic sessions
  whose text/audio content targets the mock server. Nothing dummy about the
  pipeline; only the *server* is mock (req. 1.1).

### 2.3 Mock servers (deterministic "supposed to" times)

- **Text SSE mock** (`mock_native loop.py`): asyncio server speaking
  `POST /v1/chat/completions` + SSE; per stream: sleep `prefill_ms`, then emit
  `num_chunks` chat-delta events on **absolute deadlines**
  (`T_conn + prefill + i·chunk_ms`), recording its own per-emit lateness.
  Defaults model a real model: `chunk_ms=20` (~50 tok/s), `prefill_ms=200`,
  `num_chunks=240` (≈5 s request lifetime — a mock that answers in 0.4 s
  reports a concurrency ceiling no real 5 s-response benchmark ever reaches,
  because harness load scales with requests/second = N/lifetime).
- **Realtime-TTS WS mock**: accepts the real `RealtimeTTSClient` protocol
  (`session.update`/`conversation.item.create`/`response.create` in;
  `session.updated`, `response.output_audio.delta`×N on absolute deadlines,
  `response.done` out).
- **STT WS mock**: accepts `input_audio_buffer.append` frames — **stamping
  each on raw receipt before parsing** (the server-side pacing ground truth) —
  then emits transcript deltas on a fixed cadence after final commit.
- Shared infrastructure (`sharded_server.py`):
  - **accept sharding**: N asyncio loops accepting from ONE listening socket
    (dup'd fd). `SO_REUSEPORT` does **not** distribute TCP accepts on macOS —
    verified empirically: 40/40 connections landed on the last-bound listener.
  - **PhaseSpreader**: all connections start together, so identical emit
    schedules put every deadline in the same instant and one loop then walks
    the burst — the *mock's* walk shows up as client drift. Spreading each
    connection's phase across one cadence window (what real staggered sessions
    do anyway) empirically drops mock self-jitter from 2–6 ms to <1 ms.
  - per-emit **self-lateness telemetry** (lock-free, per-thread), the input to
    the native loop-limited verdict.

### 2.4 Checks, gates, verdicts

Per concurrency rung (default: **only the target concurrency** — the question
is "can I trust *my* run"; `--scan_ladder` walks 10→target to locate the knee):

| Check | Pipeline | Gates |
|---|---|---|
| text receive drift | full `_run_main_loop` + `OpenAIChatCompletionsClient` vs SSE mock | per-chunk p99 err, stretch, achieved concurrency, completion drift |
| TTS receive drift | full loop + `RealtimeTTSClient` vs WS mock | per-chunk p99 err on `audio_chunk_timestamps` gaps, stretch |
| ASR send pacing | full loop + `STTClient` (1× pacing on) vs STT mock | client-side per-chunk send-drift p99 (gated); server-observed drift (reported); transcript receive-gap p99 (the receive half of word interactivity) |
| dispatch accuracy | rate-scheduled arrivals vs seeded schedule | per-request dispatch lateness p99 (`max(0, actual−scheduled)`) |

Verdict per rung is **tri-state**: `honest` / `dishonest` /
`native-loop-limited` — the last when the mock's own recorded lateness exceeds its
threshold: client fidelity is then *unmeasured* and reported as inconclusive,
never as a pass and never as a client failure. NaN gates fail safe. Exit code
non-zero iff any gated check is dishonest at the target. Every check carries
`notes` disambiguating its columns; the report prints a legend.

### 2.5 Small client changes required on this branch (record, don't compute)

The preflight needs per-chunk raw data that the tts_merge clients partly
discard today. Four contained changes (all are "record what you already have"
— no behavior change, no refactor):

1. **`openai_chat.py` stamp placement**: the per-chunk `receive_time`
   (`:408`) is taken after `_process_stream` has already line-split and
   `json.loads`-ed the event (`:294-313`). Stamp at `aiter_text` read time and
   carry it alongside the yielded event. On the prior branch this exact fix
   cut self-reported inter-token error at c=400 from ~21 ms to ~4 ms — the
   parse cost was being charged to the server.
2. **`stt.py`: record per-chunk send stamps.** The send loop paces every
   chunk (`:362-385`) but records only `audio_started_at`/`audio_end_at`.
   Append `(monotonic − audio_started_at)·1000` **before each `ws.send`** into
   a `send_offsets_ms` list and emit it in metrics. This is the single change
   that makes req. 1.3 measurable at all.
3. **`stt.py` receive stamps**: `now` is taken after
   `json.loads(await ws.recv())` (`:390-391`) — same defect as (1); stamp
   before parsing. Also record the raw per-delta arrival offsets (the current
   `transcript_snapshots` are deduplicated by content, so they under-count
   wire events).
4. **`tts.py` (HTTP)**: per-chunk `receive_time` (`:158`) is discarded after
   TTFC. Record `[offset_ms, n_bytes]` per chunk (mirroring realtime's
   `audio_chunk_timestamps` key) so HTTP-TTS receive drift is measurable too.

(`realtime_tts.py` already records everything needed: send-side
`text_delta_timestamps` at `:291`, receive-side `audio_chunk_timestamps`
stamped before decode at `:300-314`.)

### 2.6 Reuse from the prior branch

The `veeksha-modality-abstraction-v3` WIP (commit `59846fbc`) contains a
working, audited preflight package with exactly this design (tri-state
verdicts, PhaseSpreader, sharded accepts, single-rung default, report legend).
Porting = retargeting `drivers.py` at this branch's config/client/evaluator
shapes; the mock servers, scorer math, validator and report carry over nearly
unchanged. Its measured findings (knee locations, jitter numbers) were
obtained on the same machine and inform the default thresholds.

---

## 3. Part 2 — Native C++ main loop (worker-level port)

### 3.1 Principle: 1:1 correspondence, same seams, same numbers

The native main loop is a **drop-in `_run_main_loop`**: same worker roles, same
queue topology, same config parameters, same five lifecycle timestamps with
the same anchor points, same result shapes. Python keeps everything outside
the loop: config, session generation and content preparation before; metric
scoring, files, plots after. A single flag (`runtime.use_native_engine`)
selects the loop; the preflight gates **both** loops with the same scorer, so
"native is more honest at concurrency C" is a measured, like-for-like claim.

### 3.2 Component-by-component correspondence

| Python (today) | Native (proposed) | Correspondence contract |
|---|---|---|
| `_run_main_loop` (`benchmark.py:86`) | `veeksha_native.run_benchmark_loop(plan, config)` | same construction order, same start/drain/join sequence incl. timeout+grace semantics (`benchmark_utils.py:291-334`) |
| `ThreadPoolManager` (`core/thread_pool.py:13`) | `std::thread` pools, names preserved (`prefetch-0`, `dispatch-i`, …) | same pool sizes from the same config fields; join semantics with per-pool timeouts |
| `PrefetchWorker` ×1 (`workers/prefetch.py:37`) | native intake thread consuming a **session plan ring buffer** fed by Python | same burst-then-throttle behavior (`_BURST_DURATION_S=5.0`, `_MAX_POLL_INTERVAL_S=0.05`); `max_sessions` counter semantics identical |
| `TrafficScheduler` (rate/concurrent/sequential) | native scheduler: deadline min-heap over `ready_at` + session-graph state, **fine-grained locking or lock-free MPSC** | identical semantics: rampup formula (`concurrent.py:42-48`), `wait_after_ready`, cancel-on-failure, ticket orderings; identical *schedules* (see 3.4) |
| `DispatchWorker` ×N (`dispatch.py:19`) | native dispatch threads | stamps `scheduler_ready_at`/`scheduler_dispatched_at` at the same point in the sequence; same power-of-two queue choice; register_request events logged to the result drain (3.5) |
| `ClientWorker` ×N, 1 asyncio loop each (`client_runner.py`) | native reactor threads (kqueue/epoll/poll) running the three transport state machines | `client_picked_up_at` after ticket gate; `on_request_sent`/`on_request_dispatched` become internal events driving `DispatchTracker` + `notify_request_sent` equivalents at the same protocol points (HTTP 200 / first content chunk / request end) |
| client transports (httpx SSE / httpx bytes / websockets ×2) | HTTP/1.1+SSE state machine; WS client codec; absolute-deadline send pacer (per-loop deadline heap) | wire-identical requests built by **Python** pre-dispatch (bytes/templates); per-chunk stamps at socket-read time (`CLOCK_MONOTONIC`); pacing schedules identical to `stt.py:369-372` / `TextDeltaPacer` |
| `CompletionWorker` ×N (`completion.py:19`) | **split** (3.5): native completion-ack thread(s) call scheduler `notify_completion` immediately; results appended to a drain queue | `result_processed_at` stamped at native ack (same meaning: "the run acknowledged this result"); ordering `notify_completion` → evaluator preserved |
| `evaluator.register_request` / `record_request_completed` | **stays Python**, fed by a background drain thread replaying the native event log in order | same call signatures, same data; runs concurrently with the benchmark but off the critical path (bounded only by memory; it cannot delay dispatch/refill by construction) |
| `DispatchTracker` (`dispatch_tracker.py`) | native ticket gate (atomic counter + futex/condvar) | same three orderings; advance-on-error preserved |
| monitor loop (`benchmark_utils.py:253`) | Python, polling native counters (`get_session_counts`-equivalent atomics) | same 0.1 s cadence, same timeout/grace algorithm, progress file/pbar unchanged |

**What deliberately stays Python (answering "does X need a native port?"):**

- **`session_generator` — no.** Generation (tokenization, trace/dataset
  loading, librosa decode) is pre-benchmark work; on the native path Python
  *pre-compiles* each session into a wire-ready plan (3.3) either fully
  upfront (`pregenerate_sessions` semantics) or streamed into the native ring
  during the run by a Python producer thread — the analogue of today's
  single PrefetchWorker feeding the scheduler, and just as off-path.
- **`evaluator` — scoring no, acknowledgement yes.** The timing-critical part
  of "completion" is `notify_completion` (refill + next-turn release); that
  goes native. WER/jiwer/difflib/sketches are scoring; they stay Python on a
  drain thread. This split is already half-present in the Python code
  (`completion.py:52` notifies *before* evaluating; `base.py:244-254` pushes
  scoring outside the aggregate lock).
- **`traffic_scheduler` — yes, native.** It is the measured bottleneck (see
  3.7) and every decision it makes is a WHEN-decision.

### 3.3 The Python↔native boundary: the session plan

Python compiles, per session, a **plan** — everything the native loop needs,
nothing it must call back for:

```
SessionPlan {
  session_id
  nodes: [ { node_id, wait_after_ready,
             parents: [(node_id, is_history_parent)],
             transport: TEXT_SSE | TTS_HTTP | TTS_REALTIME_WS | STT_WS,
             wire: TransportPlan } ]
}
TransportPlan (per kind):
  TEXT_SSE:      header template + body segments with history/holes
                 (multi-turn splicing done natively from extracted
                 delta.content of parent turns)
  TTS_HTTP:      full request bytes
  TTS_REALTIME_WS: pre-serialized session.update / item.create frames +
                 text-delta segments + send-offset schedule + response.create
  STT_WS:        pre-serialized handshake frames + base64 append frames +
                 1×-realtime send schedule + commit frame
```

Native returns, per request, a **RawResult**: status/close info, per-chunk
`(recv_offset_ms, size)` stamps, per-send `(send_offset_ms)` stamps, extracted
content (assistant text / transcript deltas / audio byte counts), the five
lifecycle stamps, error string. A Python translator maps RawResult → the
exact `RequestResult` + per-modality `metrics` keys of §1(C) — so evaluators,
request-level JSONL, SLOs, and the preflight scorer are byte-compatible
across loop implementations.

### 3.4 Determinism and schedule parity

The rate scheduler's arrival schedule must be **bit-identical** across
engines. Rather than porting NumPy's MT19937 + the Poisson/Gamma transforms
to C++ (replicable but fragile), Python **precomputes the interarrival draws**
with the exact seeded generators (`seeding.py` chain:
`derive_seed(root, "interval", "0")` → `RandomState`) and ships them in the
plan stream; the native scheduler consumes draws in order. Unbounded runs
stream draws in batches. Same for per-node `wait_after_ready` (already static
in the plan). Result: `python main loop ≡ native main loop` schedules for the same
seed, by construction — testable by diffing dispatch logs.

### 3.5 Result drain and GIL discipline

- The extension is `py::mod_gil_not_used()` (free-threaded CPython); native
  threads never touch Python objects. Crossing points: (a) plan intake
  (Python producer → native ring, GIL released), (b) result drain (native →
  Python consumer thread pulling batches), (c) monitor counter reads
  (atomics).
- The drain preserves per-request event order (`register` before `completed`)
  and total order per session, so evaluator session logic
  (`session_total_requests`, `base.py:299`) behaves identically.
- Nothing on the native hot path allocates Python objects, takes Python
  locks, or imports anything (notably: transformers/tokenizers never load —
  on this branch importing the chat client's tokenizer path re-enables the
  GIL silently; the native path is immune by construction, and the Python
  path should adopt the lazy-import fix regardless).

### 3.6 Config parity

Same fields, same meanings, no new knobs beyond the native loop flag:

| Config field | Python consumer today | Native consumer |
|---|---|---|
| `runtime.num_client_threads` (default `max(3, ceil(target/8))`, `benchmark.py:102-113`) | ClientWorker count | reactor thread count (same default formula) |
| `runtime.num_dispatcher_threads` (2) | DispatchWorker pool | native dispatch threads |
| `runtime.num_completion_threads` (8) | CompletionWorker pool | native ack threads + drain sizing |
| `runtime.max_sessions`, `benchmark_timeout`, `post_timeout_grace_seconds` | counter/monitor | identical semantics (monitor stays Python) |
| `traffic.*` (target_concurrent_sessions, rampup_seconds, arrival rate/CV, ordering) | schedulers | native scheduler, same formulas |
| client config (request_timeout, ws chunk sizes, pacing flags, provider) | clients | transport state machines |

### 3.7 Why this is worth doing (measured, same machine)

- The single scheduler `Condition` caps the Python pipeline at ~175 concurrent
  streams / ~250 req/s regardless of thread counts (swept 2/2/2 → 16/8/8 on a
  27-CPU box); bare asyncio+httpx against the same mock reaches 400+ — the
  lock, not the transport, is the ceiling.
- One asyncio receive loop stays timing-honest to ~300 concurrent streams;
  beyond that recorded inter-chunk error climbs into tens of ms — the tool
  reports its own scheduling lag as server latency.
- Post-parse stamping alone contributed ~17 ms of phantom inter-token error at
  c=400.
- A prior-generation native transport native loop (same box, C++ reactor loops,
  read-time stamping) held 3–6 ms p99 inter-chunk error at c=800 where the
  Python pipeline recorded ~29 ms and dropped requests — and moved the
  preflight-certified ceiling ~2.7×. The worker-level port targets the same
  wins while keeping the *entire* loop semantics identical rather than
  bypassing the scheduler.

### 3.8 Reuse from the prior branch

`veeksha-modality-abstraction-v3` (WIP `59846fbc`) contains audited,
adversarially-tested C++ building blocks that transplant directly into the
worker-level design: the reactor abstraction (kqueue/epoll/poll,
runtime-selectable), the HTTP/SSE state machine with chunked decoding and
read-time stamps, delta-content extraction with full JSON unescape, the WS
client codec (fragmentation, ping/pong, Upgrade validation), absolute-deadline
send pacing via per-loop deadline heaps, deadline-safe fd lifecycle, QoS
pinning for Apple efficiency-core avoidance, and strided sharding with
disjoint result slots. What is *new* in this design: the native scheduler,
dispatch/ack workers, ticket gate, plan/drain ring buffers, and the
session-graph state machine (the prior engine's "chains" covered only linear
uniform-history sessions; this design implements the full `ready_at`/
history-splice semantics of `session_graph.py`).

Protocol-layer boundary (per prior decision): parsing stays hand-rolled and
narrowly scoped; adopt sans-io libraries (llhttp, wslay) the day the native
path needs TLS/HTTP2/proxies/another dialect. The reactor/pacing/stamping
loop stays ours regardless.

### 3.9 Phasing (each phase lands green and preflight-certified)

1. **PR-1 — Preflight package + client recording fixes (§2.5).** Certifies the
   Python baseline; establishes the gate all later PRs must improve or hold.
2. **PR-2 — Minimal-interface refactor (Python-only; prototype doc §7).**
   `_run_main_loop` is replaced by the `MainLoop` protocol taking only plain
   structs (`MainLoopConfig`), a `SessionSource`, and emitting a drained
   `LoopEvent` stream; evaluator/trace_recorder leave the loop signature; the
   shared `ResultDrain` replays events. Behavior-preserving for benchmarks
   (preflight re-run pins completion drift before/after).
3. **PR-3 — Native transports + ClientWorker** behind the same interface.
   Python scheduler/dispatch still drive; native reactor threads own
   send/pace/receive/stamp. Biggest fidelity win, smallest semantic surface.
4. **PR-4 — Native scheduler + DispatchWorker + completion-ack + ticket gate**
   (full `NativeMainLoop`). Removes the throughput ceiling; schedule parity
   tests + preflight ladder both loop implementations.
5. **PR-5 — Benchmarks & docs.** Text/TTS/STT ladders, Python vs native, via
   `veeksha preflight --scan_ladder --compare_native`; results into docs.

### 3.10 Risks / open questions

- **Session-graph generality in C++**: DAG sessions with mixed
  `is_history_parent` edges and per-node channel mixes are fully supported in
  the plan format, but history splicing for *non-text* parents (image/audio
  content blocks in `_record_history`, `concurrent.py:250-292`) has no native
  consumer today — propose: plans with non-text history fall back to the
  Python main loop, logged loudly (same guard discipline as the prior engine).
- **Evaluator drain backpressure**: unbounded drain memory under extreme rates
  vs bounded-with-blocking (which could stall the ack path). Proposal:
  unbounded with high-water warning; results are small (stamps + text).
- **`openai_completions` (non-streaming, logprobs for lm-eval)**: timing is a
  single span; native support is trivial (plain HTTP) but lm-eval accuracy
  flows pull tokenizers — keep on Python main loop initially, guard + fall back.
- **Windows**: poll() backend exists; not a target platform for timing runs.
- **Two engines to maintain**: mitigated by the parity test (same seed, mock
  server → identical metric keys, counts, schedules; timings within tolerance)
  run in CI on every PR touching either loop.
