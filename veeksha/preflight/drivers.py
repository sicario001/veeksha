"""One measurement harness; the dispatch path is the only thing that varies.

The preflight's job is to certify the HARNESS, so a Python rung and a native
rung must differ in exactly one respect: who moves the bytes. Everything else —
which mock server answers, what gets sent, how the recorded timings are scored,
and the terms a rung must pass to count as honest — is defined once, here, and
shared by both.

That is a correctness property, not tidiness. When each path carried its own
copy of the request payload and its own honesty gate, the two drifted: the
native rung was sending a smaller request than the Python rung and was scored
against one fewer gate term, which flattered it. Sharing the definitions makes
that class of bug unrepresentable — there is no second copy to fall out of step.

A workload owns its mock server, its request payloads, its scorer and its gate
terms, and exposes the two dispatch paths. A new transport becomes one more
``dispatch_*`` method, never a parallel copy of the harness.
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
import wave
from abc import ABC, abstractmethod
from queue import Queue
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from veeksha.core.request import Request
from veeksha.core.response import RequestResult
from veeksha.types import ChannelModality

PYTHON = "python"
NATIVE = "native"


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


@dataclass(frozen=True)
class MetricView:
    """One reported check derived from a workload's metrics.

    A single dispatch can answer more than one question — ASR measures the send
    and receive halves of interactivity from the same run — so a view names the
    check, selects its columns and declares its gate terms. Two views over one
    run beat two runs, because the two halves then describe the same traffic.
    """

    name: str
    notes: str
    select: Callable[[Dict[str, float]], Dict[str, float]]
    terms: Callable[[Dict[str, float], float, float], Dict[str, bool]]


def mock_text_client(engine, config):
    """The real chat client, pointed at the preflight's mock engine.

    Shared by the workloads that speak HTTP to it, so there is one definition of
    what the Python path sends.
    """
    from veeksha.client.openai_chat import OpenAIChatCompletionsClient
    from veeksha.config.client import OpenAIChatCompletionsClientConfig
    from veeksha.preflight.probe import _whitespace_tokenizer_provider

    return OpenAIChatCompletionsClient(
        OpenAIChatCompletionsClientConfig(
            api_base=f"http://127.0.0.1:{engine.port}/v1",
            api_key="none",
            model="mock",
            request_timeout=120,
            max_connections=config.max_connections,
        ),
        _whitespace_tokenizer_provider(),
    )


class PreflightWorkload(ABC):
    """A thing the preflight can measure, over either dispatch path.

    Subclasses supply the mock server, the requests, the scorer and the gate
    terms. The base class supplies nothing but the contract — which is the
    point: the contract is what both paths are held to.
    """

    #: Name of the check, as it appears in the report.
    name: str = ""
    #: Column legend appended to the check's report block.
    notes: str = ""
    #: Whether a native rung can be measured for this workload.
    supports_native: bool = True
    #: Native reactor threads. One by default: the comparisons deliberately give
    #: Python at least as many workers, so the asymmetry runs against native.
    native_threads: int = 1

    # -- mock server lifecycle (one server per check, reset between rungs) --
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def reset_telemetry(self) -> None: ...

    @abstractmethod
    def server_jitter_p99_ms(self) -> float:
        """The server's own emit lateness — the control that separates a
        saturated mock from a genuinely drifting client."""

    # -- what gets sent (identical for both paths, by construction) --
    @abstractmethod
    def build_requests(self, n_requests: int) -> List[Request]: ...

    @abstractmethod
    def request_count(self, concurrency: int) -> int:
        """How many requests this rung needs at ``concurrency``."""

    # -- who sends it (the ONLY axis that varies) --
    @abstractmethod
    def dispatch_python(
        self, requests: List[Request], concurrency: int
    ) -> List[RequestResult]: ...

    def dispatch_native(
        self, requests: List[Request], concurrency: int
    ) -> List[RequestResult]:
        raise NotImplementedError(f"{self.name} has no native path")

    def dispatch(
        self, path: str, requests: List[Request], concurrency: int
    ) -> List[RequestResult]:
        if path == NATIVE:
            return self.dispatch_native(requests, concurrency)
        return self.dispatch_python(requests, concurrency)

    # -- how it is scored (one scorer, both paths) --
    @abstractmethod
    def score(self, results: List[RequestResult], path: str) -> Dict[str, float]:
        """Reduce results to the reported metrics. Must key ``ivl_err_p99_ms``
        (the gated drift) and ``achieved``."""

    @property
    def views(self) -> List[MetricView]:
        """Checks derived from one dispatch. One by default."""
        return [
            MetricView(
                name=self.name,
                notes=self.notes,
                select=lambda m: m,
                terms=lambda m, d, s: self.gate_terms(m, d, s),
            )
        ]

    # -- what makes a rung honest (one gate, both paths) --
    @abstractmethod
    def gate_terms(
        self,
        metrics: Dict[str, float],
        drift_threshold_ms: float,
        stretch_threshold: float,
    ) -> Dict[str, bool]:
        """Named honesty terms. A rung is honest when every term holds (plus
        the shared achieved-concurrency and server-not-saturated terms).

        Terms may legitimately differ BETWEEN modalities. Where a term exists on
        only one dispatch path it must be justified by that path genuinely
        lacking the mechanism — not by convenience — and the difference is
        reported rather than hidden.
        """


# --------------------------------------------------------------------- text
class TextWorkload(PreflightWorkload):
    """SSE chat-completions streaming at a known inter-token cadence."""

    name = "receive-drift accuracy"

    def __init__(self, engine, config, chunk_dt: float, prefill_s: float):
        self._engine = engine
        self._config = config
        self.chunk_dt = chunk_dt
        self.prefill_s = prefill_s
        self._wall = 0.0
        self._completed = 0

    # the engine is started by the validator and shared with every check
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def reset_telemetry(self) -> None:
        self._engine.reset_telemetry()

    def server_jitter_p99_ms(self) -> float:
        return self._engine.server_jitter_p99_ms()

    def request_count(self, concurrency: int) -> int:
        # honesty needs achieved >= 0.95*c, which is impossible if the request
        # count barely covers c.
        cfg = self._config
        return max(
            60,
            math.ceil(1.2 * concurrency),
            min(
                2000,
                int(
                    cfg.budget_s
                    * concurrency
                    / (self.prefill_s + cfg.num_chunks * self.chunk_dt)
                ),
            ),
        )

    def build_requests(self, n_requests: int) -> List[Request]:
        from veeksha.core.request_content import TextChannelRequestContent
        from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec

        return [
            Request(
                id=i,
                channels={
                    ChannelModality.TEXT: TextChannelRequestContent(
                        input_text="hi", target_prompt_tokens=1
                    )
                },
                requested_output=RequestedOutputSpec(
                    text=TextOutputSpec(target_tokens=self._config.num_chunks)
                ),
            )
            for i in range(n_requests)
        ]

    def _client(self):
        return mock_text_client(self._engine, self._config)

    def dispatch_python(self, requests, concurrency):
        from veeksha.preflight.probe import run_pipeline

        results, wall = run_pipeline(
            self._client(),
            requests,
            concurrency,
            self._config,
            request_duration_s=self.prefill_s + self._config.num_chunks * self.chunk_dt,
        )
        self._wall = wall
        return results

    def dispatch_native(self, requests, concurrency):
        import time

        from veeksha.native.transport import NativeTransport

        t0 = time.monotonic()
        results = NativeTransport("127.0.0.1", self._engine.port).run_text(
            requests,
            concurrency=concurrency,
            model="mock",
            timeout_s=120.0,
            num_threads=self.native_threads,
        )
        self._wall = time.monotonic() - t0
        return results

    def score(self, results, path):
        from veeksha.preflight.probe import score_receive_results

        scored = score_receive_results(results, self._config.num_chunks, self.chunk_dt)
        completed = sum(1 for r in results if getattr(r, "success", False))
        out = {
            "achieved": float(self._engine.max_active_conns),
            "completed": float(completed),
            "ivl_err_p99_ms": _pct(scored["ivl_err"], 99),
            "stretch_p99": _pct(scored["stretch"], 99),
            "ttfc_p99_ms": _pct(scored["ttfc"], 99),
            "throughput": completed / self._wall if self._wall else 0.0,
        }
        # Completion-queue drift exists only where there IS a completion queue:
        # the Python pipeline hands results to a completion worker, native scores
        # straight off the socket. Reported (and gated) only where it is real.
        if path == PYTHON:
            drift = [
                (r.result_processed_at - r.client_completed_at) * 1000.0
                for r in scored["window"]
                if r.success
                and r.client_completed_at is not None
                and r.result_processed_at is not None
            ]
            out["completion_drift_p99_ms"] = _pct(drift, 99)
        return out

    def gate_terms(self, m, drift_threshold_ms, stretch_threshold):
        terms = {
            "inter-chunk drift": m["ivl_err_p99_ms"] < drift_threshold_ms,
            "stream stretch": m["stretch_p99"] < stretch_threshold,
        }
        if "completion_drift_p99_ms" in m:
            terms["completion-queue drift"] = (
                m["completion_drift_p99_ms"] < drift_threshold_ms
            )
        return terms


# ---------------------------------------------------------------- realtime TTS
class TtsWorkload(PreflightWorkload):
    """Realtime audio deltas arriving over WS on a known cadence."""

    name = "audio-transport receive drift (realtime TTS, WS)"
    notes = (
        "ivlP99 = per-chunk audio RECEIVE drift against the server's known "
        "emit cadence."
    )

    def __init__(self, config):
        self._config = config
        self._server = None

    def start(self) -> None:
        from veeksha.preflight.audio_server import MockRealtimeAudioServer

        self._server = MockRealtimeAudioServer(
            num_chunks=self._config.num_chunks,
            chunk_dt=self._config.audio_chunk_ms / 1000.0,
        ).start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None

    def reset_telemetry(self) -> None:
        self._server.reset_telemetry()

    def server_jitter_p99_ms(self) -> float:
        return self._server.server_jitter_p99_ms()

    def request_count(self, concurrency: int) -> int:
        # one long-lived WS stream per unit of concurrency
        return concurrency

    def build_requests(self, n_requests: int) -> List[Request]:
        from veeksha.core.request_content import TextChannelRequestContent

        return [
            Request(
                id=i,
                channels={
                    ChannelModality.TEXT: TextChannelRequestContent(
                        input_text="preflight realtime audio transport probe request"
                    )
                },
            )
            for i in range(n_requests)
        ]

    def dispatch_python(self, requests, concurrency):
        from veeksha.client.realtime_tts import RealtimeTTSClient
        from veeksha.config.client import RealtimeTTSClientConfig

        from veeksha.preflight.probe import run_pipeline

        client = RealtimeTTSClient(
            RealtimeTTSClientConfig(
                api_base=f"http://127.0.0.1:{self._server.port}",
                model="preflight-tts",
                request_timeout=30.0,
            )
        )
        results, _ = run_pipeline(
            client,
            requests,
            concurrency,
            self._config,
            request_duration_s=self._config.num_chunks
            * self._config.audio_chunk_ms
            / 1000.0,
        )
        return results

    def dispatch_native(self, requests, concurrency):
        from veeksha.native.transport import NativeTransport

        return NativeTransport("127.0.0.1", self._server.port).run_realtime_tts(
            requests,
            concurrency=concurrency,
            timeout_s=30.0,
            num_threads=self.native_threads,
        )

    def score(self, results, path):
        from veeksha.core.audio_contract import AudioMetricKey

        cadence = self._config.audio_chunk_ms
        drift: List[float] = []
        achieved = 0
        for result in results:
            if not getattr(result, "success", False):
                continue
            channel = result.channels.get(ChannelModality.AUDIO)
            if channel is None:
                continue
            timeline = channel.metrics.get(
                AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value, []
            )
            if len(timeline) < 2:
                continue
            achieved += 1
            offsets = [row[0] for row in timeline]
            for a, b in zip(offsets, offsets[1:]):
                drift.append(abs((b - a) - cadence))
        return {
            # peak SIMULTANEOUS connections the server saw — the same ground
            # truth the text engine reports. A count of completed requests is
            # not concurrency and must not be reported as it.
            "achieved": float(self._server.max_active_conns),
            "completed": float(achieved),
            "ivl_err_p99_ms": _pct(drift, 99),
            "stretch_p99": float("nan"),
            "ttfc_p99_ms": float("nan"),
            "throughput": float("nan"),
        }

    def gate_terms(self, m, drift_threshold_ms, stretch_threshold):
        return {"audio receive drift": m["ivl_err_p99_ms"] < drift_threshold_ms}


# ------------------------------------------------------------------ realtime STT
class SttWorkload(PreflightWorkload):
    """Realtime ASR: paced audio out, transcript deltas back.

    ASR word interactivity is a subtraction across BOTH streams, so each side
    gets its own gate — but from one dispatch per rung, via two views, so the
    two checks describe the same run rather than two runs that merely ran near
    each other.
    """

    name = "audio-transport (realtime STT/ASR, WS)"
    SAMPLE_RATE = 16000

    def __init__(self, config):
        self._config = config
        self._server = None
        self._tmpdir: Optional[str] = None
        self._wav: Optional[str] = None
        self._chunk_period_ms = 0.0

    @property
    def views(self) -> List["MetricView"]:
        return [
            MetricView(
                name="audio-transport send pacing (realtime STT/ASR, WS)",
                notes=(
                    "columns: ivlP99 = CLIENT-side per-chunk send drift (the "
                    "gated property); ttfcP99 = server-observed ARRIVAL drift "
                    "(client + network + server-receive floor — their gap IS "
                    "that floor)."
                ),
                select=lambda m: m,
                # gated on the CLIENT's own dispatch stamps — the property the
                # preflight certifies. Server-observed arrival carries the
                # delivery floor too, so it is reported, not gated.
                terms=lambda m, d, s: {
                    "client send drift": m["ivl_err_p99_ms"] < d,
                    "send stretch": m["stretch_p99"] < s,
                },
            ),
            MetricView(
                name="audio-transport transcript receive (realtime STT/ASR, WS)",
                notes=(
                    "ASR word interactivity subtracts two streams (when a "
                    "word's audio went out, when its transcript came back), so "
                    "error on EITHER side lands in it — this is the half the "
                    "send check cannot see."
                ),
                select=lambda m: {
                    **m,
                    "ivl_err_p99_ms": m["recv_drift_p99_ms"],
                    "stretch_p99": float("nan"),
                    "ttfc_p99_ms": float("nan"),
                },
                terms=lambda m, d, s: {
                    "transcript receive drift": m["ivl_err_p99_ms"] < d
                },
            ),
        ]

    def start(self) -> None:
        from veeksha.preflight.audio_server import MockSTTPreflightServer

        self._server = MockSTTPreflightServer(
            delta_dt=self._config.transcript_delta_ms / 1000.0
        ).start()
        self._tmpdir = tempfile.mkdtemp()
        self._wav = os.path.join(self._tmpdir, "clip.wav")
        with wave.open(self._wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.SAMPLE_RATE)
            w.writeframes(
                b"\x00\x00" * int(self.SAMPLE_RATE * self._config.pacing_clip_s)
            )

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
        try:
            if self._wav:
                os.remove(self._wav)
            if self._tmpdir:
                os.rmdir(self._tmpdir)
        except OSError:
            pass

    def reset_telemetry(self) -> None:
        self._server.reset_telemetry()

    def server_jitter_p99_ms(self) -> float:
        return self._server.server_jitter_p99_ms()

    def request_count(self, concurrency: int) -> int:
        return concurrency

    def build_requests(self, n_requests: int) -> List[Request]:
        from veeksha.core.request_content import AudioChannelRequestContent

        return [
            Request(
                id=i,
                channels={
                    ChannelModality.AUDIO: AudioChannelRequestContent(
                        input_audio=self._wav
                    )
                },
                metadata={"expected_transcript": "x", "dataset": "preflight"},
            )
            for i in range(n_requests)
        ]

    def _client_config(self):
        from veeksha.config.client import STTClientConfig

        return STTClientConfig(
            api_base=f"http://127.0.0.1:{self._server.port}",
            model="preflight-stt",
            provider="vllm_realtime",
            sample_rate=self.SAMPLE_RATE,
            ws_realtime_pacing=True,
            request_timeout=30.0,
        )

    def dispatch_python(self, requests, concurrency):
        from veeksha.client.stt import STTClient

        from veeksha.preflight.probe import run_pipeline

        config = self._client_config()
        client = STTClient(config)
        self._chunk_period_ms = config.ws_chunk_size / 2.0 / self.SAMPLE_RATE * 1000.0
        results, _ = run_pipeline(
            client,
            requests,
            concurrency,
            self._config,
            request_duration_s=self._config.pacing_clip_s,
        )
        return results

    def dispatch_native(self, requests, concurrency):
        from veeksha.native.transport import NativeTransport

        # Frame size matches the Python client's exactly, so both paths pace to
        # the same schedule and their drift numbers mean the same thing.
        chunk_ms = self._client_config().ws_chunk_size / 2.0 / self.SAMPLE_RATE * 1000.0
        self._chunk_period_ms = chunk_ms
        return NativeTransport("127.0.0.1", self._server.port).run_stt(
            requests,
            concurrency=concurrency,
            sample_rate=self.SAMPLE_RATE,
            chunk_ms=chunk_ms,
            timeout_s=30.0,
            num_threads=self.native_threads,
        )

    def score(self, results, path):
        period = self._chunk_period_ms
        cadence = self._config.transcript_delta_ms
        client_drift: List[float] = []
        recv_drift: List[float] = []
        for result in results:
            if not getattr(result, "success", False):
                continue
            channel = result.channels.get(ChannelModality.AUDIO)
            if channel is None:
                continue
            for i, offset in enumerate(channel.metrics.get("send_offsets_ms") or []):
                if i:  # chunk 0 defines the pacing origin
                    client_drift.append(abs(offset - i * period))
            arrivals = [
                float(row["elapsed_ms"])
                for row in (channel.metrics.get("transcript_snapshots") or [])
            ]
            for a, b in zip(arrivals, arrivals[1:]):
                recv_drift.append(abs((b - a) - cadence))

        # server-observed arrival: client pacing + delivery + server receive cost
        send_drift: List[float] = []
        stretches: List[float] = []
        for arrivals in self._server.append_arrivals:
            if len(arrivals) < 2:
                continue
            for i, arrival in enumerate(arrivals):
                send_drift.append(abs(arrival - i * period))
            ideal = (len(arrivals) - 1) * period
            if ideal > 0:
                stretches.append((arrivals[-1] - arrivals[0]) / ideal)
        return {
            "achieved": float(self._server.max_active_conns),
            "completed": float(len(self._server.append_arrivals)),
            "ivl_err_p99_ms": _pct(client_drift, 99),
            "stretch_p99": _pct(stretches, 99),
            "ttfc_p99_ms": _pct(send_drift, 99),
            "throughput": float("nan"),
            "recv_drift_p99_ms": _pct(recv_drift, 99),
        }

    def gate_terms(self, m, drift_threshold_ms, stretch_threshold):
        raise AssertionError("SttWorkload gates through its views")


# ------------------------------------------------------------- the shared rung
def measure_rung(
    workload: PreflightWorkload,
    path: str,
    concurrency: int,
    drift_threshold_ms: float,
    stretch_threshold: float,
    n_requests: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure one workload at one concurrency over one dispatch path.

    This is the whole harness. Both paths reach the wire through it, so the
    request payloads, the sample window, the scorer and the honesty gate are the
    same objects for each — not two implementations kept in agreement by
    review. The dispatch call is the only branch.
    """
    from veeksha.preflight.report import ConcurrencyPoint

    workload.reset_telemetry()
    if n_requests is None:
        n_requests = workload.request_count(concurrency)
    requests = workload.build_requests(n_requests)
    results = workload.dispatch(path, requests, concurrency)
    metrics = workload.score(results, path)
    server_jitter = workload.server_jitter_p99_ms()

    points: Dict[str, ConcurrencyPoint] = {}
    for view in workload.views:
        m = view.select(metrics)
        # A rung only counts if the mock server kept its own schedule: a
        # saturated server is late for both paths, and calling that a client
        # failure would misattribute the machine's limit to the harness.
        engine_ok = server_jitter < max(drift_threshold_ms, m["ivl_err_p99_ms"] * 0.5)
        terms = view.terms(m, drift_threshold_ms, stretch_threshold)
        honest = (
            m["achieved"] >= 0.95 * concurrency and all(terms.values()) and engine_ok
        )
        points[view.name] = ConcurrencyPoint(
            concurrency=concurrency,
            achieved=int(m["achieved"]),
            ivl_err_p99_ms=m["ivl_err_p99_ms"],
            stretch_p99=m.get("stretch_p99", float("nan")),
            ttfc_p99_ms=m.get("ttfc_p99_ms", float("nan")),
            throughput=m.get("throughput", float("nan")),
            server_jitter_p99_ms=server_jitter,
            honest=honest,
            engine_limited=not honest and not engine_ok,
            completion_drift_p99_ms=m.get("completion_drift_p99_ms", float("nan")),
        )
    return points


# ------------------------------------------------------------ dispatch timing
class DispatchWorkload(PreflightWorkload):
    """When work was scheduled to START, versus when it actually started.

    The receive checks measure timing INSIDE a request that is already running.
    This one measures the decision that shapes the offered load: a request due
    at t went out when? Get it wrong and the experiment that ran is not the
    experiment that was configured — the harness quietly applies less load than
    asked, and every throughput number inherits the error.

    Both paths are given the same arrival schedule and stamped at the same
    point, the moment the request is about to go on the wire. Lateness is
    anchored on the first dispatch, so only the SPREAD of the schedule is
    measured and no clock epoch can leak in.
    """

    name = "dispatch precision (arrival schedule)"
    notes = (
        "ivlP99 = p99 lateness against each request's scheduled arrival. "
        "In-flight is set by the arrival rate: period = stream duration / "
        "concurrency, so concurrency C means C requests are live at once."
    )

    def __init__(self, engine, config, chunk_dt: float, prefill_s: float):
        self._engine = engine
        self._config = config
        self.chunk_dt = chunk_dt
        self.prefill_s = prefill_s
        self._offsets_ms: List[float] = []

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def reset_telemetry(self) -> None:
        self._engine.reset_telemetry()

    def server_jitter_p99_ms(self) -> float:
        return self._engine.server_jitter_p99_ms()

    def request_count(self, concurrency: int) -> int:
        # Enough requests that the schedule runs for a few stream durations,
        # so the measurement covers steady state rather than just the ramp.
        return min(2400, max(300, concurrency * 3))

    def _period_ms(self, concurrency: int) -> float:
        # In flight = arrival rate x how long a request lives, and a request
        # lives for prefill + the whole stream. Leaving prefill out here would
        # overshoot the requested concurrency by that fraction.
        duration_s = self.prefill_s + self._config.num_chunks * self.chunk_dt
        return duration_s * 1000.0 / max(1, concurrency)

    def build_requests(self, n_requests: int) -> List[Request]:
        from veeksha.core.request_content import TextChannelRequestContent
        from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec

        return [
            Request(
                id=i,
                channels={
                    ChannelModality.TEXT: TextChannelRequestContent(
                        input_text="hi", target_prompt_tokens=1
                    )
                },
                requested_output=RequestedOutputSpec(
                    text=TextOutputSpec(target_tokens=self._config.num_chunks)
                ),
            )
            for i in range(n_requests)
        ]

    def _client(self):
        return mock_text_client(self._engine, self._config)

    def _schedule(self, n_requests: int, concurrency: int) -> List[float]:
        period = self._period_ms(concurrency)
        self._offsets_ms = [i * period for i in range(n_requests)]
        return [o / 1000.0 for o in self._offsets_ms]

    def dispatch_python(self, requests, concurrency):
        from veeksha.preflight.probe import run_pipeline

        offsets_s = self._schedule(len(requests), concurrency)
        # Open loop: the ARRIVAL SCHEDULE sets how many are in flight, so the
        # concurrency cap must not bind — a cap below the steady-state in-flight
        # would queue requests behind it and report the backlog as lateness.
        results, _ = run_pipeline(
            self._client(),
            requests,
            len(requests),
            self._config,
            request_duration_s=self.prefill_s + self._config.num_chunks * self.chunk_dt,
            dispatch_offsets_s=offsets_s,
        )
        return results

    def dispatch_native(self, requests, concurrency):
        from veeksha.native.transport import NativeTransport

        offsets_s = self._schedule(len(requests), concurrency)
        return NativeTransport("127.0.0.1", self._engine.port).run_text(
            requests,
            concurrency=len(requests),  # open loop: the schedule sets in-flight
            model="mock",
            timeout_s=120.0,
            dispatch_offsets_s=offsets_s,
        )

    def score(self, results, path):
        # client_picked_up_at is the same stamp on both paths: the instant the
        # request is about to go on the wire. Native sets it from the engine's
        # own dispatch offset; Python from the client worker's pickup.
        stamps = {
            r.request_id: r.client_picked_up_at
            for r in results
            if getattr(r, "success", False) and r.client_picked_up_at is not None
        }
        lateness: List[float] = []
        if 0 in stamps:
            anchor = stamps[0]
            for rid, ts in stamps.items():
                if rid == 0 or rid >= len(self._offsets_ms):
                    continue
                lateness.append(abs((ts - anchor) * 1000.0 - self._offsets_ms[rid]))
        return {
            "achieved": float(self._engine.max_active_conns),
            "ivl_err_p99_ms": _pct(lateness, 99),
            "stretch_p99": float("nan"),
            "ttfc_p99_ms": _pct(lateness, 50),  # median lateness, for context
            "throughput": float(len(stamps)),
        }

    def gate_terms(self, m, drift_threshold_ms, stretch_threshold):
        return {"arrival lateness": m["ivl_err_p99_ms"] < drift_threshold_ms}


# ------------------------------------------------------- multi-turn dispatch
class ChainDispatchWorkload(PreflightWorkload):
    """Turn K+1 was due at (turn K completed + think time). When did it go out?

    The other decision that shapes offered load, and the one a conversational
    benchmark depends on: if the next turn is late, the think time you configured
    is not the think time that ran. Both paths carry the same conversations with
    the same think time, and lateness is measured against each turn's own due
    time, so no clock epoch enters.
    """

    name = "dispatch precision (multi-turn think time)"
    notes = (
        "ivlP99 = p99 lateness of turn K+1 against (turn K completed + think "
        "time). Concurrency is the number of simultaneous conversations."
    )

    TURNS = 3
    THINK_S = 0.1

    def __init__(self, engine, config, chunk_dt: float, prefill_s: float):
        self._engine = engine
        self._config = config
        self.chunk_dt = chunk_dt
        self.prefill_s = prefill_s

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def reset_telemetry(self) -> None:
        self._engine.reset_telemetry()

    def server_jitter_p99_ms(self) -> float:
        return self._engine.server_jitter_p99_ms()

    def request_count(self, concurrency: int) -> int:
        return concurrency  # one conversation per unit of concurrency

    def build_requests(self, n_requests: int) -> List[Request]:
        from veeksha.core.request_content import TextChannelRequestContent
        from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec

        # Flat list of every turn of every conversation, ids grouped by chain so
        # both paths can reconstruct the same conversations.
        return [
            Request(
                id=chain * self.TURNS + turn,
                channels={
                    ChannelModality.TEXT: TextChannelRequestContent(
                        input_text="hi", target_prompt_tokens=1
                    )
                },
                requested_output=RequestedOutputSpec(
                    text=TextOutputSpec(target_tokens=self._config.num_chunks)
                ),
            )
            for chain in range(n_requests)
            for turn in range(self.TURNS)
        ]

    def dispatch_python(self, requests, concurrency):
        from veeksha.preflight.probe import run_chain_pipeline

        return run_chain_pipeline(
            mock_text_client(self._engine, self._config),
            requests,
            turns=self.TURNS,
            think_s=self.THINK_S,
            concurrency=concurrency,
            config=self._config,
            request_duration_s=self.prefill_s + self._config.num_chunks * self.chunk_dt,
        )

    def dispatch_native(self, requests, concurrency):
        from veeksha.native.transport import NativeTransport

        chains = [
            requests[i : i + self.TURNS] for i in range(0, len(requests), self.TURNS)
        ]
        delays = [
            [0.0 if k == 0 else self.THINK_S for k in range(self.TURNS)] for _ in chains
        ]
        history = [[k > 0 for k in range(self.TURNS)] for _ in chains]
        nested = NativeTransport("127.0.0.1", self._engine.port).run_text_chains(
            chains,
            chain_delays_s=delays,
            chain_history=history,
            concurrency=len(chains),
            model="mock",
            timeout_s=120.0,
            num_threads=self.native_threads,
        )
        return [r for chain in nested for r in chain]

    def score(self, results, path):
        by_id = {r.request_id: r for r in results if getattr(r, "success", False)}
        lateness: List[float] = []
        for rid, result in by_id.items():
            turn = rid % self.TURNS
            if turn == 0:
                continue
            prev = by_id.get(rid - 1)
            if prev is None or prev.client_completed_at is None:
                continue
            if result.client_picked_up_at is None:
                continue
            due = prev.client_completed_at + self.THINK_S
            lateness.append(max(0.0, (result.client_picked_up_at - due) * 1000.0))
        return {
            "achieved": float(self._engine.max_active_conns),
            "ivl_err_p99_ms": _pct(lateness, 99),
            "stretch_p99": float("nan"),
            "ttfc_p99_ms": _pct(lateness, 50),
            "throughput": float(len(by_id)),
        }

    def gate_terms(self, m, drift_threshold_ms, stretch_threshold):
        return {"turn hand-off lateness": m["ivl_err_p99_ms"] < drift_threshold_ms}
