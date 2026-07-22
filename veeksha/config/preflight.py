"""Config for ``veeksha preflight`` — the measurement-fidelity gate.

The preflight answers one question before you spend GPU hours: *at the
concurrency I intend to benchmark, on this machine, does the harness still
record timings faithfully?* It runs the real pipeline (real scheduler, real
workers, real clients) against mock servers whose emit schedule is known
exactly, and scores the difference.
"""

from typing import List, Optional

from vidhi import field, frozen_dataclass, parse_cli_sweep

from veeksha.cli.base import VeekshaCommand


@frozen_dataclass
class PreflightCheckConfig(VeekshaCommand, name="preflight"):
    """Certify this machine's measurement fidelity at a target concurrency.

    No GPU, no model, no network: built-in mock servers with deterministic
    emit schedules stand in for the inference server, and the *entire*
    benchmark pipeline runs against them.
    """

    # ------------------------------------------------------------ what to measure
    target_concurrency: int = field(
        100,
        help="The concurrency you intend to benchmark at. Every check reports "
        "whether timings stay faithful at this level; the exit code is "
        "non-zero if any gated check is dishonest here.",
    )
    check_text: bool = field(
        True,
        help="Measure text response delivery (C3): the real chat client "
        "streaming SSE from the mock, scored as the per-chunk lag from the "
        "mock's send stamp to the client's arrival stamp. This is the check "
        "that certifies TPOT/TBC.",
    )
    check_tts: bool = field(
        True,
        help="Measure realtime-TTS audio response delivery (C3) over a real "
        "WebSocket: the per-delta lag from the mock's send to the client's "
        "arrival. Certifies the TTFA/interactivity family.",
    )
    check_vajra_tts: bool = field(
        True,
        help="Measure Vajra streaming-TTS response delivery (C3) over a real "
        "WebSocket. Unlike check_tts, audio arrives WHILE the paced text input "
        "is still being sent (that is what Vajra's protocol does), so this is "
        "the only check that exercises interleaved paced-send and "
        "timestamped-receive on one event loop — where delivery lag appears "
        "first under concurrency.",
    )
    check_stt: bool = field(
        True,
        help="Measure ASR request delivery (C2): the client's 1x-realtime paced "
        "audio sends paired against the mock's per-append arrival stamps, one "
        "lag per audio chunk. Certifies that the harness delivers input audio "
        "on time. Pulls librosa via the STT client.",
    )
    check_dispatch: bool = field(
        True,
        help="Measure the send schedule (C1): with rate-based (open-loop) "
        "traffic "
        "every session has a deterministic seeded arrival time, so this check "
        "asks whether the harness actually STARTED each request then. The other "
        "checks certify the receive and send paths; this one certifies the "
        "traffic scheduler itself — all three schedulers guard every operation "
        "with a single threading.Condition, so this is where scheduler "
        "contention shows up, and it matters most at high concurrency.",
    )
    check_multiturn: bool = field(
        True,
        help="Measure multi-turn think time (C1): build 2-turn TEXT sessions "
        "where turn 2 waits think_time_s after turn 1 completes, and score how "
        "late the harness actually RELEASES turn 2 against that intended "
        "schedule (client_completed_at[turn1] + think_time_s). It certifies the "
        "harness's ability to release multi-turn follow-ups on schedule — a late "
        "release means the server sees the conversation's turns at the wrong "
        "cadence — and folds in the completion-notify path (the scheduler only "
        "learns turn 1 finished when the completion worker dequeues it). This is "
        "the multi-turn analog of the dispatch C1 check.",
    )

    # --------------------------------------------------------------- thresholds
    delivery_lag_threshold_ms: float = field(
        5.0,
        help="Gate on p99 DELIVERY LAG (ms) for both paired checks: C2 = the "
        "client's send stamp -> the mock's receive stamp, C3 = the mock's send "
        "stamp -> the client's arrival stamp, one lag per request and per "
        "streamed event respectively. Physically the lag is the loopback path "
        "(write syscall, kernel, read wake) plus however late the receiving "
        "side's loop woke up; both stamps sit next to the syscall, so no "
        "parsing cost is inside it. The p50 of the same samples is REPORTED as "
        "this machine's delivery floor: if p50 is already large, every latency "
        "the benchmark reports carries that constant, even when the p99 gate "
        "passes.",
    )
    max_unpaired_fraction: float = field(
        0.02,
        help="A check is honest only if the fraction of requests that could NOT "
        "be paired with a server record stays below this. Unpaired means the "
        "mock never saw the request's PFID marker (or reported a connection it "
        "could not identify): the events those requests produced are absent "
        "from C2/C3 entirely, so a silently-dropping preflight would score only "
        "the healthy subset and read as honest. Kept small on purpose — this is "
        "a measurement error, not a performance number.",
    )
    think_time_drift_threshold_ms: float = field(
        10.0,
        help="The multi-turn think-time check (C1) passes only if p99 late-only "
        "think-time drift stays below this (ms): per 2-turn session, max(0, "
        "actual turn-2 send - (client_completed_at[turn1] + think_time_s)). A "
        "SCHEDULING quantity like dispatch drift, not a transport lag, so it "
        "keeps its own knob (default looser than delivery_lag_threshold_ms — a "
        "turn released a few ms late still measures its own latency correctly). "
        "A late release silently benchmarks multi-turn conversations at the "
        "wrong turn cadence, which no per-request metric reveals.",
    )
    pacing_drift_threshold_ms: float = field(
        10.0,
        help="The ASR check's C1 CHUNK-PACING term passes only if p99 late-only "
        "pacing drift (actual audio-chunk send offset minus its intended "
        "1x-realtime schedule offset) stays below this (ms). This is a "
        "SCHEDULING quantity — did the client keep up with generating realtime "
        "audio — physically distinct from the delivery lags, so it has its own "
        "knob. Client-side only: it also holds on a real production run. Late "
        "sends silently benchmark slower-than-1x audio, which no per-request "
        "metric reveals.",
    )
    dispatch_drift_threshold_ms: float = field(
        10.0,
        help="The dispatch-accuracy check passes only if p99 request "
        "dispatch lateness (actual dispatch - the seeded arrival time it was "
        "scheduled for) stays below this (ms). Lateness here is offered load "
        "the server never saw when it was supposed to: it silently reshapes "
        "the arrival process you configured, and no per-request metric shows "
        "it. Looser than delivery_lag_threshold_ms on purpose — a request that "
        "starts a few ms late still measures its own latency correctly, whereas "
        "a chunk stamped a few ms late corrupts TPOT directly.",
    )
    min_served_fraction: float = field(
        0.95,
        help="A check is INCONCLUSIVE (engine-limited) if the mock answered "
        "less than this fraction of the requests offered to it — the one "
        "surviving engine-limited case: the mock could not serve the offered "
        "connections at all, so the load under measurement was never applied "
        "and nothing was proven about the client either way.",
    )
    achieved_concurrency_fraction: float = field(
        0.95,
        help="A check is honest only if the peak SIMULTANEOUS connections the "
        "mock observed reach this fraction of the requested concurrency. A "
        "harness that quietly applied less load than asked measured a "
        "different experiment than the one you configured.",
    )

    # --------------------------------------------------- text workload (SSE mock)
    num_chunks: int = field(
        240,
        help="Chunks (tokens) per synthetic TEXT response. Request LIFETIME is "
        "the number that matters: holding N streams open costs the harness "
        "N/lifetime request hand-offs per second, so a mock that answers in "
        "0.4 s reports a ceiling no real 5 s-response benchmark ever reaches. "
        "With the defaults a request lives ~5 s (200 ms prefill + 240 x 20 ms).",
    )
    chunk_ms: float = field(
        20.0,
        help="Inter-chunk cadence of the text SSE mock (ms). 20 ms models a "
        "model streaming ~50 tokens/second. Keep it well above the mock's own "
        "emit jitter so measured drift is attributable to the client.",
    )
    prefill_ms: float = field(
        200.0,
        help="Synthetic prefill delay before the first chunk (ms) — the mock's "
        "time-to-first-token. It is not part of the delivery lag (which pairs "
        "each chunk's own send and arrival stamps) but it is part of the "
        "request lifetime, and thus of the load the harness carries.",
    )

    # ------------------------------------------------ dispatch workload (rate SSE)
    dispatch_response_chunks: int = field(
        10,
        help="Chunks per response for the DISPATCH check only (the other text "
        "knobs still set prefill and cadence). The response content is "
        "irrelevant here — only offered load is — so it is deliberately short: "
        "arrival rate = target_concurrency / request lifetime, and the whole "
        "check must finish inside PrefetchWorker's 5 s unthrottled burst window "
        "(after which it emits one session per 50 ms and you would be "
        "measuring the prefetch throttle instead). A short lifetime buys a high "
        "rate, and a high rate is what reaches the target concurrency inside "
        "that window.",
    )

    # ------------------------------------------------- realtime TTS workload (WS)
    audio_num_chunks: int = field(
        250,
        help="Audio deltas per synthetic TTS response — independent of "
        "num_chunks, which sizes text. With the default cadence this is ~5 s of "
        "speech, i.e. a realistic request lifetime.",
    )
    audio_chunk_ms: float = field(
        20.0,
        help="Cadence the realtime-TTS mock emits audio deltas at (ms). Keep it "
        "comfortably above the mock's own emit jitter (1-5 ms) so the measured "
        "drift is the client's, not the server's.",
    )
    audio_first_delta_ms: float = field(
        20.0,
        help="Delay before the first audio delta (ms) — the mock's TTFA. Like "
        "prefill_ms it is not part of the per-delta delivery lag but counts "
        "towards request lifetime.",
    )
    audio_chunk_bytes: int = field(
        960,
        help="PCM16 bytes per audio delta. 960 bytes at 24 kHz is 20 ms of "
        "audio, so the mock streams at ~1x realtime like a real TTS engine.",
    )

    # --------------------------------------------------------- ASR workload (WS)
    pacing_clip_s: float = field(
        5.0,
        help="Duration of the synthetic input clip for the ASR check (s). This "
        "IS the ASR request lifetime, because the client streams the clip at 1x "
        "real time. Set it near the utterance length you expect.",
    )
    transcript_delta_ms: float = field(
        30.0,
        help="Cadence the STT mock emits transcript deltas at (ms) — a stand-in "
        "for how fast an ASR model streams decoded text. Same guidance as "
        "audio_chunk_ms: stay well above the mock's own jitter.",
    )
    stt_sample_rate: int = field(
        16000, help="Sample rate of the synthetic ASR clip (Hz)."
    )
    stt_ws_chunk_size: int = field(
        4096,
        help="Bytes of PCM per WebSocket append (mirror client.ws_chunk_size). "
        "It sets the pacing period — 4096 bytes at 16 kHz is one send every "
        "128 ms — and therefore how finely send drift is resolved.",
    )
    stt_transcript: str = field(
        "the quick brown fox jumps over the lazy dog",
        help="Transcript the STT mock streams back, one delta per word. More "
        "words means more transcript-cadence samples per request.",
    )

    # ---------------------------------------------- multi-turn workload (SSE mock)
    think_time_s: float = field(
        0.3,
        help="Think time between the two turns of the multi-turn check's "
        "sessions (s) — the wait_after_ready on turn 2. It is the intended "
        "release delay the check scores turn 2's actual send against; keep it "
        "well above the harness's own release jitter so the measured drift is "
        "attributable to the harness. Turn responses are kept short so the run "
        "measures turn CADENCE, not stream length.",
    )

    # ------------------------------------------------------------ harness shape
    num_client_threads: Optional[int] = field(
        None,
        help="Client worker threads — mirror your benchmark's runtime setting. "
        "None uses the benchmark's own default (one thread per eight target "
        "sessions, minimum three). One asyncio loop stays timing-honest to only "
        "a few hundred streams, so this knob moves the knee directly.",
    )
    num_dispatcher_threads: int = field(
        2, help="Dispatcher threads (mirror runtime.num_dispatcher_threads)."
    )
    num_completion_threads: int = field(
        8,
        help="Completion threads (mirror runtime.num_completion_threads). "
        "Under-provisioning delays scheduler.notify_completion, which shows up "
        "as multi-turn think-time drift (the next turn is released late).",
    )
    request_timeout_s: int = field(
        120, help="Per-request client timeout (s) for every preflight client."
    )
    server_loops: int = field(
        0,
        help="Accept loops per mock server. 0 auto-sizes from "
        "target_concurrency. Raise it if the mock cannot serve the offered "
        "connections (served_fraction drops) — a saturated mock makes a check "
        "inconclusive (engine-limited) rather than failed.",
    )
    budget_s: float = field(
        10.0,
        help="Approximate wall-time target per check (s), used to size the "
        "request count. It is a floor-bounded target, not a cap: at least "
        "ceil(1.2 x concurrency) requests always run so the check can actually "
        "reach its concurrency, and with long request lifetimes that floor — "
        "not this budget — sets the check's duration.",
    )
    output_dir: Optional[str] = field(
        None,
        help="If set, write the report text here as preflight_report.txt.",
    )

    def __post_init__(self) -> None:
        if self.target_concurrency < 1:
            raise ValueError("target_concurrency must be >= 1")
        if not (
            self.check_text
            or self.check_tts
            or self.check_vajra_tts
            or self.check_stt
            or self.check_dispatch
            or self.check_multiturn
        ):
            raise ValueError(
                "At least one of check_text/check_tts/check_vajra_tts/"
                "check_stt/check_dispatch/check_multiturn"
            )
        if self.dispatch_response_chunks < 1:
            raise ValueError("dispatch_response_chunks must be >= 1")

    @classmethod
    def create_from_cli_args(cls) -> List["PreflightCheckConfig"]:
        return parse_cli_sweep(cls)
