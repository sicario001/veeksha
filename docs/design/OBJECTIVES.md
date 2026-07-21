# Veeksha native main loop — objectives brief

Self-contained statement of what this effort must achieve, for independent
review or implementation. Repo: veeksha, branch `veeksha-native-preflight`
(base: tts_merge). Full designs: `docs/design/preflight_and_native_loop.md`
and `docs/design/native_loop_prototype.md`.

## Problem (measured, not hypothetical)

Veeksha benchmarks LLM inference servers (text SSE, TTS over HTTP/WebSocket,
STT/ASR over WebSocket) by recording per-chunk timing. Two harness defects
corrupt results at scale on the pure-Python pipeline:

1. **Timing fidelity.** One Python asyncio receive loop stays honest only to
   ~300 concurrent streams; beyond that, recorded inter-chunk timings absorb
   the harness's own event-loop scheduling lag and report it as server
   latency (tens of ms of phantom TPOT). Timestamping after JSON parsing
   alone added ~17 ms of false inter-token error at 400 streams.
2. **Throughput ceiling.** Every dispatch decision and completion
   notification serializes on a single scheduler lock; the pipeline plateaus
   at ~175 concurrent sessions / ~250 req/s regardless of thread counts. A
   run configured at concurrency 400 actually exercises the server at ~175 —
   silently.

## Objective

Make benchmark timing trustworthy at high concurrency by moving ONLY the
timing-critical path into a native C++ main loop — dispatch scheduling, request
send/pacing, chunk receive + timestamping, completion acknowledgement —
while Python keeps everything outside the measurement window: session and
content generation before the run, metric scoring and reporting after.
Results from the two loop implementations must be indistinguishable in shape, semantics
and (for deterministic schedules) content.

## Non-negotiable invariants

1. **Native threads never call Python.** The only run-time crossings are:
   Python pushes precompiled session plans (bounded ring), Python pulls an
   ordered event stream (batched), Python polls atomic counters. Evaluator,
   trace recorder, and generators are never invoked from native code.
2. **Same loop semantics and config.** The native loop is a drop-in for the
   Python loop behind one shared protocol: same thread-count config fields
   and defaults, same traffic-scheduler semantics (rate / concurrent with
   linear ramp-up / sequential-launch with dispatch tickets and all three
   orderings), same session-graph behavior (think time, parent-completion
   gating, transitive history accumulation with reset on non-history edges,
   cancel-on-failure), same stop/timeout/grace lifecycle.
3. **Wire fidelity.** Native requests must be byte-identical (HTTP) or
   frame-identical (WebSocket) to what the Python clients send for the same
   session — bodies, headers, pacing schedules, provider dialects. Content
   preparation (tokenization, audio decode/encode) stays in Python; native
   receives prebuilt bytes/templates.
4. **Deterministic schedule parity.** Seeded arrival schedules must be
   bit-identical across loop implementations (draws made once in Python and streamed to
   native; no RNG reimplementation).
5. **Same result vocabulary.** Five lifecycle timestamps with the same anchor
   points (scheduler-ready, dispatched, client-picked-up, client-completed,
   result-processed), and per-modality metric keys exactly as the existing
   evaluators consume them — one scoring/reporting path for both loop implementations.
6. **Timestamp discipline.** Receive stamps at socket-read time, before any
   parsing; send stamps immediately before the send call; pacing on absolute
   deadlines. This discipline is the product; any library or abstraction
   that sits between the socket event and the stamp is disqualified (hence:
   sans-io parsers like llhttp are acceptable, loop-owning frameworks like
   ASIO/Beast-streams are not).
7. **Honest failure discipline.** Anything the native path cannot serve
   faithfully falls back to the Python main loop with the reason logged — never
   a silently different workload. Malformed protocol input fails the request
   with a reason — never a silent completion.
8. **Independent verification.** A preflight gate (separate work item)
   measures BOTH loop implementations' timing drift with one scorer against
   deterministic mock servers; cross-implementation parity tests pin metric-key
   sets, request/session/token counts, and schedules. Claims of native
   superiority must come from these, not from construction.

## Explicitly out of scope

Metric computation in C++ (WER, interactivity, sketches — stays Python);
tokenization in C++; TLS/HTTP2/proxies on the native path (fall back);
new metrics or client abstractions; Windows as a timing platform.

## Acceptance criteria

- Full existing test suite green with the Python main loop (behavior preserved).
- Native main loop selected by one config field; every ineligibility falls back
  loudly; the Python main loop remains fully supported.
- Wire-fidelity and parity tests pass; seeded schedules diff clean.
- Preflight (when it lands) shows the native main loop honest at concurrencies
  where the Python main loop is not, on the same machine, same scorer — the
  quantitative justification for the whole effort.
