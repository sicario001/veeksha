"""Config for `veeksha validate-preflight` — the preflight measurement check."""

from typing import List, Optional

from vidhi import field, frozen_dataclass, parse_cli_sweep

from veeksha.cli.base import VeekshaCommand


@frozen_dataclass
class PreflightCheckConfig(VeekshaCommand, name="preflight"):
    """Validate that this system can measure timings faithfully at a target
    concurrency, BEFORE running a real benchmark. Uses a built-in streaming mock
    engine (no GPU, no model, no network)."""

    target_concurrency: int = field(
        100,
        help="The concurrency you intend to benchmark at. The check reports "
        "whether this system's timing stays faithful up to it.",
    )
    num_client_threads: int = field(
        3, help="Client worker threads (mirror your benchmark's runtime setting)."
    )
    num_dispatcher_threads: int = field(2, help="Dispatcher threads.")
    num_completion_threads: int = field(2, help="Completion threads.")
    max_connections: Optional[int] = field(
        None,
        help="httpx pool cap per client worker (mirror client.max_connections). "
        "None = unlimited.",
    )
    drift_threshold_ms: float = field(
        5.0,
        help="A concurrency counts as 'honest' only if p99 inter-chunk timing "
        "error stays below this (ms).",
    )
    stretch_threshold: float = field(
        1.05,
        help="A concurrency is 'honest' only if p99 recorded/ideal stream duration "
        "stays below this.",
    )
    num_chunks: int = field(20, help="Chunks per synthetic stream.")
    chunk_ms: float = field(
        20.0,
        help="Inter-token cadence the text (SSE) mock engine emits at (ms). "
        "20 ms models a model streaming ~50 tokens/second.",
    )
    audio_chunk_ms: float = field(
        50.0,
        help="Inter-chunk cadence the realtime-TTS mock server emits audio at "
        "(ms). Keep it comfortably above the mock's own emit jitter (1-5 ms) "
        "so the measured drift is attributable to the client, not the server.",
    )
    transcript_delta_ms: float = field(
        50.0,
        help="Cadence the STT mock server emits transcript deltas at (ms) — a "
        "stand-in for how fast an ASR model streams out decoded text. Same "
        "guidance as audio_chunk_ms: stay well above the mock's own jitter.",
    )
    prefill_ms: float = field(
        50.0, help="Synthetic prefill delay before first chunk (ms)."
    )
    engine_loops: int = field(
        0, help="Mock-engine accept loops. 0 = auto-size from target_concurrency."
    )
    budget_s: float = field(
        4.0, help="Approx wall-time budget per concurrency point (s)."
    )
    check_pacing: bool = field(
        True, help="Also validate real-time send-pacing preflight (for audio/ASR)."
    )
    pacing_clip_s: float = field(
        2.0, help="Synthetic clip duration for the pacing check (s)."
    )
    check_audio_transport: bool = field(
        False,
        help="Also measure per-chunk receive drift on the REAL realtime-audio "
        "WebSocket transport (drives the actual RealtimeTTSClient against a "
        "fixed-cadence mock server). Off by default: pulls the client package.",
    )
    compare_native: bool = field(
        False,
        help="Also measure the native (C++) receive path for comparison, if the "
        "veeksha_native extension is built (veeksha/native/build.sh). Shows how far "
        "the honest-concurrency knee moves with the native dispatcher.",
    )
    output_dir: Optional[str] = field(
        None, help="If set, write the report text here (preflight_report.txt)."
    )

    @classmethod
    def create_from_cli_args(cls) -> List["PreflightCheckConfig"]:
        return parse_cli_sweep(cls)
