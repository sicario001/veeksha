"""Preflight scoring math — pure functions over paired stamps.

Nothing here does I/O, starts a server, or touches the benchmark pipeline. The
clients and the mock servers *record* stamps inside the timing window; every
comparison, subtraction and aggregation happens here, after the run. That split
is what lets the preflight certify the harness *as it ships* rather than an
instrumented variant of it, and it is why this module is exhaustively unit
testable with hand-computed numbers.

The preflight asks three questions, and each one is a DIRECT PAIRED comparison
of two stamps taken around one event. There is no cadence reference and no gap
differencing anywhere in this module: client and mock run on one host and both
read ``time.monotonic()``, so the two stamps are directly comparable and the
quantity of interest can be read off without introducing a third.

**C1 — send schedule (client-side only).** Did the harness START each request
when the traffic pattern said it should? Under rate-based (open-loop) traffic
every session has a deterministic seeded arrival offset, so ``drift(i) =
max(0, actual(i) - scheduled(i))`` with ``actual(i) = scheduler_dispatched_at -
epoch``. Late-only for a structural reason: the scheduler pops from a deadline
heap and cannot dispatch before ``ready_at``, so earliness does not exist and
must not be averaged against lateness. Needs no server, so this is the one
check that also works on a real production run. See
:func:`rate_schedule_offsets_ms` and :func:`dispatch_drift_ms`.

The **multi-turn think-time** check is the same C1 idea one level up: instead of
"did request *i* start on its seeded arrival", it asks "did the harness release
the NEXT turn of a session on time". The scheduler releases a child node at
``parent_completion + wait_after_ready``, so with ``wait_after_ready`` set to a
think time the intended release of turn 2 is ``client_completed_at[turn1] +
think_time`` and ``think_time_drift = max(0, actual_sent[turn2] - intended)`` is
late-only for the same structural reason. It folds in whatever completion-notify
lag the harness adds (the scheduler cannot learn turn 1 finished until the
completion worker dequeues it), so a late release means the server sees a
multi-turn conversation at the wrong cadence. See :func:`think_time_drift_ms`.

**C2 — request delivery.** Client ``request_sent_monotonic`` -> server
``received_at``: one delivery lag per request (per audio chunk for ASR, where
the client's paced send series pairs against the mock's per-append arrivals).

**C3 — response delivery.** Server ``emitted_at[i]`` -> client absolute arrival
of event *i*: one delivery lag per streamed event, over every request.

For C2 and C3 the measured quantity is the DELIVERY LAG

    L = (later stamp) - (earlier stamp)        always > 0

which decomposes as a constant floor (the loopback path: the write syscall, the
kernel, the read wake) plus a variable part (the harness waking late because
its loop is saturated). p99(L) is GATED; p50(L) is REPORTED as the observed
floor, so a large constant is visible rather than hidden inside a passing gate.
Differences of two L values are never gated: that would reintroduce exactly the
derived quantity this model exists to avoid.

The clients publish offsets, not absolute instants, so the absolute client
times are reconstructed HERE from the anchors the clients now expose
(:func:`text_arrival_times`, :func:`audio_arrival_times`,
:func:`stt_send_times`). Reconstruction in the scorer, not in the client, keeps
the client's recording path unchanged and therefore keeps the thing being
certified identical to the thing that ships.

Included / excluded, per reported number:

===========================  ==========================================  =========================================
number                       includes                                    excludes
===========================  ==========================================  =========================================
C2 lag p99 / p50             loopback delivery cost (write syscall,      parse cost on BOTH sides: the client
(request delivery)           kernel, read wake) + harness scheduling     stamps immediately before the syscall
                             delay on the send side + the mock's         and the mock immediately after recv(),
                             receive-loop wake                           before it looks at the bytes
C3 lag p99 / p50             loopback delivery cost + the mock's send    the mock's own emit LATENESS: the
(response delivery)          wake + the client's receive-loop            reference is the stamp the mock took
                             scheduling delay                            when it actually sent, not its deadline
C1 dispatch drift p99        scheduler wake, ready-heap pop, Condition   everything client-side: connect, send,
                             contention, DispatchWorker scheduling       network, server
think-time drift p99         completion-notify lag + scheduler release   everything up to turn 1 finishing; the
(multi-turn C1)              of turn 2 + client pickup/send of turn 2    server (it is a client-side send stamp)
===========================  ==========================================  =========================================
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from veeksha.core.audio_contract import BYTES_PER_SAMPLE
from veeksha.preflight.sharded_server import percentile

__all__ = [
    "BYTES_PER_SAMPLE",
    "absolute_times_from_offsets_ms",
    "asr_send_schedule_ms",
    "audio_arrival_times",
    "delivery_lag_ms",
    "delivery_lags_ms",
    "dispatch_drift_ms",
    "p50",
    "p99",
    "percentile",
    "percentile_or_nan",
    "rate_schedule_offsets_ms",
    "send_drift",
    "stt_send_times",
    "text_arrival_times",
    "text_pacing_schedule_ms",
    "think_time_drift_ms",
]

NAN = float("nan")


# --------------------------------------------------------------------- helpers
def percentile_or_nan(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, or NaN when there is nothing to score.

    ``sharded_server.percentile`` returns 0.0 for an empty sample — the right
    answer for "the server emitted nothing, so it was never late", and exactly
    the wrong answer for a gate ("no samples" would read as perfect fidelity).
    Gates use this wrapper: NaN compares False against every threshold, so an
    unmeasured measurement fails safe instead of passing silently.
    """
    if not values:
        return NAN
    return percentile(list(values), q)


def p99(values: Sequence[float]) -> float:
    """p99 of ``values``; NaN when empty (see :func:`percentile_or_nan`).

    The GATED statistic for a delivery lag: it is the tail that lands in a
    reported latency, not the median.
    """
    return percentile_or_nan(values, 0.99)


def p50(values: Sequence[float]) -> float:
    """p50 of ``values``; NaN when empty.

    The REPORTED statistic for a delivery lag: with the variable part gone,
    what remains is the loopback floor of this machine. Printing it next to the
    p99 is what stops a large constant transit cost from hiding inside a
    passing gate — a floor of 3 ms under a 5 ms gate is a very different
    machine from a floor of 0.05 ms under the same gate.
    """
    return percentile_or_nan(values, 0.50)


def _is_stamp(value) -> bool:
    """True when ``value`` is a usable absolute monotonic stamp."""
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number)


# ------------------------------------------------------------- delivery lag (C2/C3)
def delivery_lag_ms(earlier: float, later: float) -> float:
    """``(later - earlier)`` in ms, for two monotonic stamps of ONE event.

    The whole of C2 and C3 is this subtraction. Both stamps are taken on the
    same host from the same clock source and both sit adjacent to their
    syscall, so the result is a physical transport quantity: the loopback path
    cost plus whatever scheduling delay the receiving side added by waking
    late.

    It is deliberately NOT clamped at zero. A negative lag cannot happen for a
    correctly paired event, so a negative value is evidence that the pairing
    (or the anchor it was reconstructed from) is wrong — clamping would convert
    that evidence into a healthy-looking 0.0. Callers surface negatives as a
    measurement error.
    """
    return (float(later) - float(earlier)) * 1000.0


def delivery_lags_ms(earlier: Sequence[float], later: Sequence[float]) -> List[float]:
    """Per-event delivery lags in ms, paired POSITIONALLY, in ms.

    Event *i* on the sending side is event *i* on the receiving side: one
    connection carries one request, and neither side reorders. Pairing stops at
    the shorter sequence, so a stream that was cut short scores the events that
    actually completed the round trip rather than inventing lags for events
    only one side saw.
    """
    return [delivery_lag_ms(sent, received) for sent, received in zip(earlier, later)]


# ------------------------------------------------- absolute-time reconstruction
def absolute_times_from_offsets_ms(
    anchor_monotonic: Optional[float], offsets_ms: Sequence[float]
) -> List[float]:
    """``anchor + offset/1000`` per offset — offsets made absolute.

    Every client records event times as milliseconds from an anchor it takes
    once per request. Pairing against a server stamp needs the absolute instant,
    and the anchor is the only thing the client had to add for that. Returns
    ``[]`` for a missing or NaN anchor: an unanchored series cannot be paired,
    and must vanish rather than be paired against the wrong origin.
    """
    if not _is_stamp(anchor_monotonic):
        return []
    anchor = float(anchor_monotonic)
    return [anchor + float(offset) / 1000.0 for offset in offsets_ms]


def text_arrival_times(
    request_start_monotonic: Optional[float], inter_chunk_times_s: Sequence[float]
) -> List[float]:
    """Absolute monotonic arrival of each text chunk.

    ``arrival[i] = request_start_monotonic + sum(inter_chunk_times[0..i])``.
    The text client records *deltas* in seconds, with element 0 being
    ``t_start -> first chunk``, so the running sum against the anchor is the
    arrival instant. Unlike the cadence model this scoring replaced, element 0
    is NOT dropped: it is the arrival of chunk 0, and chunk 0 has a server send
    stamp to be paired with like every other chunk.
    """
    if not _is_stamp(request_start_monotonic):
        return []
    out: List[float] = []
    running = float(request_start_monotonic)
    for delta_s in inter_chunk_times_s:
        running += float(delta_s)
        out.append(running)
    return out


def audio_arrival_times(
    request_start_monotonic: Optional[float], chunk_offsets_ms: Sequence[float]
) -> List[float]:
    """Absolute monotonic arrival of each audio chunk.

    ``arrival[i] = request_start_monotonic + audio_chunk_timestamps[i][0]/1000``
    — the audio clients record absolute offsets from ``t_start`` rather than
    deltas, so pass the first column of ``audio_chunk_timestamps``.
    """
    return absolute_times_from_offsets_ms(request_start_monotonic, chunk_offsets_ms)


def stt_send_times(
    audio_started_monotonic: Optional[float], send_offsets_ms: Sequence[float]
) -> List[float]:
    """Absolute monotonic instant each paced audio chunk went on the wire.

    ``send[i] = audio_started_monotonic + send_offsets_ms[i]/1000``, anchored on
    the first audio byte on the wire (``send_offsets_ms[0]`` is 0.0 by
    construction). Paired against the mock's per-append arrivals this is the
    ASR pacing ground truth, stated as a delivery lag: the client's schedule
    adherence and the delivery cost are then one measured quantity instead of
    two schedules compared to a third, nominal one.
    """
    return absolute_times_from_offsets_ms(audio_started_monotonic, send_offsets_ms)


# ----------------------------------------- C1: ASR chunk-pacing adherence
# The client-side counterpart of the request-dispatch C1, one level finer: it
# asks "when was audio chunk i SUPPOSED to go out, and when did it" against the
# 1x-realtime schedule the STT client paces on. Client-only (no server stamp),
# so — like request dispatch — it also holds on a real production run, and it is
# the number that says whether veeksha kept up with generating realtime audio:
# a saturated client sends chunks late and silently benchmarks slower-than-1x
# audio. This is distinct from the C2 delivery lag (send -> append arrival),
# which is a transport quantity; pacing adherence is a scheduling quantity.
def asr_send_schedule_ms(
    n_chunks: int, chunk_bytes: int, sample_rate: int
) -> List[float]:
    """The 1x-realtime schedule (ms offsets) the STT client paces against.

    Chunk *i* is due at ``i * chunk_bytes / (BYTES_PER_SAMPLE * sample_rate)``
    seconds after the first audio byte hit the wire — byte-for-byte the schedule
    in ``stt.py`` (``audio_started_at + byte_offset / BYTES_PER_SAMPLE /
    sample_rate``). Offset 0 is 0.0: the first send stamp is the anchor, so it
    defines the origin and cannot itself be late.
    """
    if sample_rate <= 0 or chunk_bytes <= 0:
        return []
    period_ms = chunk_bytes / BYTES_PER_SAMPLE / sample_rate * 1000.0
    return [i * period_ms for i in range(max(0, n_chunks))]


def send_drift(
    send_offsets_ms: Sequence[float], schedule_ms: Sequence[float]
) -> List[float]:
    """Per-chunk ``max(0, actual - scheduled)`` — LATE-ONLY, in ms.

    Late-only is a definition, not a convenience. The client paces on absolute
    deadlines: it sleeps until ``scheduled(i)`` and then sends, so it can only
    be early by the resolution of the sleep, and averaging that earliness
    against real lateness would cancel exactly the error we are hunting. A pacer
    that fell 40 ms behind and then "caught up" 40 ms early has damaged the
    measurement twice, not zero times.

    ``send_offsets_ms`` is already the actual offset from the audio anchor, so
    it pairs positionally with :func:`asr_send_schedule_ms` and stops at the
    shorter of the two — a truncated stream scores the chunks it actually sent.
    """
    return [
        max(0.0, actual - scheduled)
        for actual, scheduled in zip(send_offsets_ms, schedule_ms)
    ]


# ------------------------------------- C1: realtime/Vajra TTS text-delta pacing
# The realtime-TTS and Vajra clients emulate an upstream LLM's decode rate by
# PACING their input text deltas: each delta is sent on an absolute deadline
# ``deadline += pacer.next_gap()`` (realtime_tts.py / vajra_tts_stream.py
# send_loop). This is the client-side C1 twin of the STT audio pacing — a
# saturated client streams the emulated input late/jittery and the TTS server
# then sees input at the wrong rate — so it is scored the same way: late-only,
# against the schedule the client actually paced to.
def text_pacing_schedule_ms(pacing, request_id: int, n_segments: int) -> List[float]:
    """The intended per-delta send offsets (ms) the client's pacer paced to.

    Reproduces the SAME gap sequence the client drew — ``TextDeltaPacer(pacing,
    seed=pacing.seed + request_id)``, one ``next_gap()`` per segment — so it is
    exact for both the ``fixed`` cadence (every gap = ``tokens_per_delta /
    tokens_per_second``) and the seeded ``poisson`` jitter, rather than an
    approximation of either.

    Offsets are RELATIVE TO THE FIRST delta (offset 0 is 0.0). The client stamps
    deltas from ``t_start`` but paces on a deadline anchor taken a scheduling hop
    later inside the send task, so the absolute origins differ by an unknowable
    wake delay; anchoring both the schedule and the actual sends on the first
    delta (exactly as :func:`asr_send_schedule_ms` does) removes that delay —
    the first send defines the origin and cannot itself be late, and delta *i* is
    then due ``gap[1] + ... + gap[i]`` after it. Pair with :func:`send_drift`
    over the first-delta-relative actual offsets.
    """
    if n_segments <= 0:
        return []
    from veeksha.client.utils import TextDeltaPacer

    pacer = TextDeltaPacer(pacing, seed=pacing.seed + int(request_id))
    absolute: List[float] = []
    running = 0.0
    for _ in range(n_segments):
        running += pacer.next_gap()
        absolute.append(running)  # sum(gap[0..i]); the client's send is at t0+this
    first = absolute[0]
    return [(value - first) * 1000.0 for value in absolute]


# ------------------------------------------------------- C1: dispatch fidelity
def rate_schedule_offsets_ms(
    seed: int, interval_generator_config, n_sessions: int
) -> List[float]:
    """The arrival offsets ``RateTrafficScheduler`` will assign, in ms.

    This is the "supposed to send a request" schedule, and it is reproduced
    rather than observed, so it must mirror ``traffic/rate.py`` exactly:

    * at construction the scheduler builds its interval generator from
      ``seed_manager.numpy_factory("interval")()`` — the *first* draw of that
      factory (the factory is a counter, so calling it twice would give a
      different stream), and
    * ``_next_start_time`` starts at ``0.0``; ``schedule_session`` assigns
      ``start_time = _next_start_time`` and only *then* advances it by
      ``get_next_interval()``.

    So ``scheduled[0] = 0.0`` and ``scheduled[i] = sum of the first i draws``.
    Root nodes are queued at ``start_time + wait_after_ready``, which is 0.0
    for the preflight's single-node sessions, so the session's start time IS
    its request's deadline.

    Session *i* here means the i-th session handed to ``schedule_session``.
    The pipeline runs exactly one ``PrefetchWorker`` (``benchmark.py``,
    ``pool_size=1``), so that is the order the preflight's session source
    generated them in, i.e. session id order.

    ``tests/unit/preflight/test_dispatch.py`` schedules real sessions into a
    real scheduler and asserts these offsets equal the ``session_start_time``
    values it assigned: a subtly wrong reproduction would report phantom drift
    that no other test could catch.
    """
    from veeksha.core.seeding import SeedManager
    from veeksha.generator.interval.registry import IntervalGeneratorRegistry

    if n_sessions <= 0:
        return []
    seed_manager = SeedManager(seed)
    generator = IntervalGeneratorRegistry.get(
        interval_generator_config.get_type(),
        interval_generator_config,
        rng=seed_manager.numpy_factory("interval")(),
    )
    offsets: List[float] = []
    next_start_s = 0.0
    for _ in range(n_sessions):
        offsets.append(next_start_s * 1000.0)
        next_start_s += generator.get_next_interval()
    return offsets


def dispatch_drift_ms(
    actual_offsets_ms: Sequence[float], scheduled_offsets_ms: Sequence[float]
) -> List[float]:
    """Per-request ``max(0, actual - scheduled)`` — LATE-ONLY, in ms. This is C1.

    ``actual`` is ``scheduler_dispatched_at - epoch``, where the epoch is the
    instant ``reset_reference_time()`` set the scheduler's zero;
    ``scheduled`` comes from :func:`rate_schedule_offsets_ms`.

    Late-only is structural, not a convention: ``_try_pop_ready_locked`` pops an
    item only once ``ready_at <= now``, so the harness *cannot* dispatch a
    request early. A negative value would mean the reproduction or the epoch is
    wrong, not that the harness was fast — clamping to zero is therefore the
    honest reading of a schedule that has no early side.

    Includes: scheduler wake latency, ready-heap pop, contention on the single
    ``threading.Condition`` that guards every scheduler operation, and
    ``DispatchWorker`` thread scheduling delay.
    Excludes: everything client-side — connect, send, network, server. This is
    purely "did the harness start the request when it said it would", which is
    why it needs no mock server and works unchanged on a production run.

    Pairs positionally; the caller aligns the two lists (by session id) and
    drops requests whose schedule slot is unknown.
    """
    return [
        max(0.0, actual - scheduled)
        for actual, scheduled in zip(actual_offsets_ms, scheduled_offsets_ms)
    ]


# -------------------------------------------------- C1: multi-turn think time
def think_time_drift_ms(
    prev_turn_completed_at: Optional[float],
    next_turn_sent_at: Optional[float],
    think_time_s: float,
) -> float:
    """Late-only think-time drift for one 2-turn session, in ms. This is C1.

    For a session whose turn 2 waits ``think_time_s`` after turn 1 finishes, the
    scheduler releases turn 2 at ``ready_at = client_completed_at[turn1] +
    think_time_s`` (``notify_completion`` is called with the client-completion
    stamp, so that IS the intended release), and

        intended          = prev_turn_completed_at + think_time_s
        think_time_drift  = max(0, (next_turn_sent_at - intended) * 1000)   ms

    where ``next_turn_sent_at`` is turn 2's client ``request_sent_monotonic``.
    Late-only for the same structural reason as dispatch C1: the scheduler
    cannot release turn 2 before ``ready_at``, so earliness does not exist and
    must not be averaged against lateness.

    It INCLUDES the completion-notify lag (the scheduler only learns turn 1
    finished when the completion worker dequeues it), the scheduler's release of
    the child node, and the client pickup/send of turn 2 — the whole path
    between "turn 1's response is in" and "turn 2 is on the wire". It EXCLUDES
    everything up to turn 1 finishing, and the server (it is a client-side send
    stamp, like dispatch C1, so it also holds on a real production run).

    Returns NaN when either stamp is missing — an unmeasured session must fail
    the gate safe (``NaN < threshold`` is False), never read as punctual. When
    the reported *server-arrival* variant is wanted, pass turn 2's server
    ``received_at`` as ``next_turn_sent_at``: the result then additionally
    carries the C2 delivery lag (did the next turn ARRIVE at the server on time).
    """
    if not _is_stamp(prev_turn_completed_at) or not _is_stamp(next_turn_sent_at):
        return NAN
    intended = float(prev_turn_completed_at) + float(think_time_s)
    return max(0.0, (float(next_turn_sent_at) - intended) * 1000.0)
