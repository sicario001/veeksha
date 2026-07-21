"""PlanCompiler: compiles Sessions into native SessionPlans with wire fidelity.

The native main loop executes pre-compiled plain-data plans
(``docs/design/native_loop_prototype.md`` §1); this module is the Python
side that turns a ``Session`` into those plans. Wire fidelity is THE
requirement: for each transport, the compiled bytes/frames must equal what
the Python clients (``veeksha/client/*``) send —

- TEXT_SSE mirrors ``openai_chat.py`` body/header building. The HTTP bytes
  are produced by ``httpx``'s own request builder (the exact library the
  Python client sends through), so headers, ordering, and JSON encoding are
  identical by construction rather than by imitation.
- TTS_HTTP mirrors ``tts.py`` (``OpenAISpeechRequest``).
- TTS_REALTIME_WS reuses ``realtime_tts.RealtimeTTSProtocol`` /
  ``segment_text`` / ``TextDeltaPacer`` for frames and pacing offsets.
- STT_WS (provider ``vllm_realtime`` only) reuses ``stt.py``'s decode/slice
  helpers and append-frame encoding; append frames for one clip+slice are
  registered as ONE shared blob and referenced by slices.

Dynamic multi-turn history (``is_history_parent=True``) follows the
scheduler's exact semantics (``traffic/rate.py`` ``_record_history`` /
``_populate_history``): accumulation is transitive through contiguous
history edges and RESETS on a non-history edge. User turns are static text
and are baked into ``body_segments``; assistant turns are holes
(``history_refs``) that native fills with the referenced turn's extracted
output.

Sessions the native path cannot serve faithfully raise ``NativeIneligible``
with a reason; ``create_main_loop`` catches config-level ineligibility at
start and falls back to the Python main loop with a warning.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from veeksha.config.traffic import (
    ConcurrentTrafficConfig,
    RateTrafficConfig,
    SequentialLaunchTrafficConfig,
)
from veeksha.core.request import Request
from veeksha.core.session import Session
from veeksha.core.session_graph import parents, topological_order
from veeksha.logger import init_logger
from veeksha.types import ChannelModality, ClientType
from veeksha.generator.interval.registry import IntervalGeneratorRegistry

logger = init_logger(__name__)

# Transport tags (mirrors veeksha_native.TransportKind without importing it).
KIND_TEXT_SSE = "text_sse"
KIND_TTS_HTTP = "tts_http"
KIND_TTS_REALTIME_WS = "tts_realtime_ws"
KIND_STT_WS = "stt_ws"

_SUPPORTED_STT_PROVIDER = "vllm_realtime"


class NativeIneligible(Exception):
    """The native main loop cannot serve this config/session faithfully."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


# ---------------------------------------------------------------------------
# Config-level eligibility / endpoint resolution
# ---------------------------------------------------------------------------


def _resolve_api_base(client_config: Any) -> str:
    """The endpoint URL, resolved exactly like ``BaseLLMClient`` (config
    first, then OPENAI_API_BASE)."""
    api_base = getattr(client_config, "api_base", None)
    if not api_base:
        api_base = os.environ.get("OPENAI_API_BASE")
    if not api_base:
        raise NativeIneligible(
            "api_base is not set in config and OPENAI_API_BASE is unset"
        )
    return str(api_base)


def _resolve_api_key(client_config: Any) -> Optional[str]:
    """API key resolution matching ``BaseLLMClient`` (config, then env)."""
    api_key = getattr(client_config, "api_key", None)
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    return api_key


def classify_client(client_config: Any) -> str:
    """Map a client config onto a native transport kind, or raise.

    Raises:
        NativeIneligible: client type / provider / endpoint scheme is not
            natively serviceable.
    """
    ctype = client_config.get_type()
    if ctype == ClientType.OPENAI_CHAT_COMPLETIONS:
        kind = KIND_TEXT_SSE
    elif ctype == ClientType.TTS:
        kind = KIND_TTS_HTTP
    elif ctype == ClientType.REALTIME_TTS:
        kind = KIND_TTS_REALTIME_WS
    elif ctype == ClientType.STT:
        provider = getattr(client_config, "provider", None)
        if provider != _SUPPORTED_STT_PROVIDER:
            raise NativeIneligible(
                f"STT provider {provider!r} is not supported natively "
                f"(only {_SUPPORTED_STT_PROVIDER!r}; its WS dialect is the one "
                "the native state machine speaks)"
            )
        kind = KIND_STT_WS
    else:
        raise NativeIneligible(
            f"client type {ctype!r} has no native transport (supported: "
            "openai_chat_completions, tts, realtime_tts, stt/vllm_realtime)"
        )

    api_base = _resolve_api_base(client_config)
    scheme = urlparse(api_base).scheme.lower()
    if scheme not in ("http", "ws", ""):
        raise NativeIneligible(
            f"endpoint {api_base} uses TLS ({scheme}); the native main loop "
            "owns plaintext only"
        )
    return kind


def check_native_eligibility(client_config: Any) -> str:
    """Config-level eligibility check used by ``create_main_loop``."""
    return classify_client(client_config)


# ---------------------------------------------------------------------------
# Config -> native struct conversions
# ---------------------------------------------------------------------------


def to_native_runtime(vn: Any, runtime_config: Any, traffic_config: Any) -> Any:
    """RuntimeConfig -> veeksha_native.NativeRuntimeConfig.

    ``num_client_threads`` is resolved with the SAME formula as
    ``PythonMainLoop.start`` (max(3, ceil(target/8)))."""
    rt = vn.NativeRuntimeConfig()
    rt.max_sessions = runtime_config.max_sessions
    rt.benchmark_timeout_s = float(runtime_config.benchmark_timeout)
    rt.post_timeout_grace_s = float(runtime_config.post_timeout_grace_seconds)
    rt.num_dispatcher_threads = runtime_config.num_dispatcher_threads
    rt.num_completion_threads = runtime_config.num_completion_threads

    num_client_threads = runtime_config.num_client_threads
    if num_client_threads is None:
        target = getattr(traffic_config, "target_concurrent_sessions", None)
        num_client_threads = max(3, -(-int(target) // 8)) if target else 3
    rt.num_client_threads = int(num_client_threads)
    return rt


_TICKET_ORDERINGS = {"dispatch": 0, "prefill": 1, "request": 2}


def to_native_traffic(vn: Any, traffic_config: Any) -> Any:
    """BaseTrafficConfig -> veeksha_native.TrafficPlanConfig."""
    tp = vn.TrafficPlanConfig()
    tp.cancel_session_on_failure = traffic_config.cancel_session_on_failure
    if isinstance(traffic_config, RateTrafficConfig):
        tp.kind = vn.TrafficKind.RATE
    elif isinstance(traffic_config, ConcurrentTrafficConfig):
        tp.kind = vn.TrafficKind.CONCURRENT
        tp.target_concurrent_sessions = traffic_config.target_concurrent_sessions
        tp.rampup_seconds = float(traffic_config.rampup_seconds)
    elif isinstance(traffic_config, SequentialLaunchTrafficConfig):
        tp.kind = vn.TrafficKind.SEQUENTIAL_LAUNCH
        tp.ordering = vn.TicketOrdering(_TICKET_ORDERINGS[traffic_config.ordering])
    else:
        raise NativeIneligible(
            f"traffic config {type(traffic_config).__name__} has no native scheduler"
        )
    return tp


def to_native_endpoint(vn: Any, client_config: Any) -> Any:
    """Client config -> veeksha_native.EndpointConfig.

    ``headers`` apply to the WS handshake only (HTTP kinds carry fully baked
    headers in the plan): the realtime-TTS Python client sends Authorization
    on the handshake when an api_key is set; the STT client sends none.
    """
    api_base = _resolve_api_base(client_config)
    parsed = urlparse(api_base)
    ep = vn.EndpointConfig()
    ep.host = parsed.hostname or "127.0.0.1"
    ep.port = parsed.port or 80
    ep.base_path = ""  # paths are fully baked into plans
    headers: List[Tuple[str, str]] = []
    if client_config.get_type() == ClientType.REALTIME_TTS:
        api_key = _resolve_api_key(client_config)
        if api_key:
            headers.append(("Authorization", f"Bearer {api_key}"))
    ep.headers = headers
    ep.request_timeout_s = float(client_config.request_timeout)
    return ep


def build_interval_generator(traffic_config: Any, seed_manager: Any) -> Any:
    """The RATE interarrival generator with the EXACT seeded chain the
    ``RateTrafficScheduler`` uses (``seed_manager.numpy_factory("interval")``
    -> RandomState -> interval generator), so schedules are bit-identical
    across loop implementations by construction."""
    return IntervalGeneratorRegistry.get(
        traffic_config.interval_generator.get_type(),
        traffic_config.interval_generator,
        rng=seed_manager.numpy_factory("interval")(),
    )


# ---------------------------------------------------------------------------
# Sidecar: per-request Python-side context the drainer needs
# ---------------------------------------------------------------------------


@dataclass
class RequestSidecar:
    """Python-side context for one compiled request.

    Kept in a ``request_id -> sidecar`` dict touched only by Python threads;
    dropped after the request's COMPLETED event is translated.
    """

    request: Request
    session_id: int
    session_size: int
    node_id: int
    kind: str

    # TEXT_SSE: ordered message contents for the exact token accounting the
    # Python chat client performs (per-message token counts, summed).
    # Entries are ("text", str) for static contents and ("hole", node_id)
    # for spliced assistant turns resolved at translation time.
    message_parts: List[Tuple[str, Any]] = field(default_factory=list)
    delta_text: str = ""

    # TTS (both transports)
    input_text: str = ""
    input_tokens: int = 0
    segment_chars: List[int] = field(default_factory=list)

    # STT
    pcm_byte_count: int = 0
    input_audio_duration_ms: float = 0.0

    @property
    def channels(self) -> Dict[ChannelModality, Any]:
        return self.request.channels

    @property
    def requested_output(self) -> Any:
        return self.request.requested_output

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.request.metadata or {}


# ---------------------------------------------------------------------------
# STT blob cache entry
# ---------------------------------------------------------------------------


@dataclass
class _SttClipPlan:
    blob_id: int
    frame_slices: List[Tuple[int, int]]  # (offset, len) into the blob
    schedule_ms: List[float]  # 1x-realtime send offsets (from first send)
    pcm_byte_count: int
    duration_ms: float


# ---------------------------------------------------------------------------
# The compiler
# ---------------------------------------------------------------------------


class PlanCompiler:
    """Compiles ``Session`` objects into native ``SessionPlan`` structs.

    Args:
        client_config: The benchmark's client config (wire ground truth).
        seed_manager: SeedManager (per-request realtime-TTS pacing seeds come
            from the client config; the seed manager is used for the RATE
            interval chain via :func:`build_interval_generator`).
        vn: The ``veeksha_native`` extension module (defaults to the built
            one). Injectable for tests.

    Raises:
        NativeIneligible: the client config is not natively serviceable.
    """

    def __init__(self, client_config: Any, seed_manager: Any, vn: Any = None):
        self._config = client_config
        self._seed_manager = seed_manager
        self._kind = classify_client(client_config)
        self._api_base = _resolve_api_base(client_config)
        self._api_key = _resolve_api_key(client_config)

        if vn is None:
            from veeksha.native import get_module

            vn = get_module()
        self._vn = vn

        # Unique per-compiler hole sentinel: ASCII-only so JSON encoding
        # passes it through verbatim; uniqueness makes prompt-text collisions
        # practically impossible (and splits are verified).
        self._hole_token = f"@@VEEKSHA-HOLE-{uuid.uuid4().hex}"

        # httpx builds the HTTP requests: same library, same defaults, same
        # header order and JSON encoding as the Python clients' AsyncClient.
        import httpx

        self._httpx = httpx.Client()

        # STT: per-(clip, slice) compiled frames sharing one registered blob.
        self._stt_clip_plans: Dict[
            Tuple[str, Optional[float], Optional[float]], _SttClipPlan
        ] = {}

    @property
    def kind(self) -> str:
        return self._kind

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def compile(
        self,
        session: Session,
        blob_registrar: Optional[Callable[[bytes], int]] = None,
    ) -> Tuple[Any, Dict[int, RequestSidecar]]:
        """Compile one session.

        Args:
            session: The session to compile.
            blob_registrar: ``bytes -> blob_id`` (``loop.register_blob``).
                Required for STT sessions.

        Returns:
            (native SessionPlan, request_id -> RequestSidecar dict).

        Raises:
            NativeIneligible: the session cannot be served faithfully.
        """
        vn = self._vn
        graph = session.session_graph
        try:
            order = topological_order(graph)
        except ValueError as exc:
            raise NativeIneligible(f"session {session.id}: {exc}") from exc

        sidecars: Dict[int, RequestSidecar] = {}
        request_plans = []
        session_size = len(session.requests)

        history_cache: Dict[int, List[Tuple[str, Any]]] = {}

        for node_id in order:
            request = session.requests.get(node_id)
            if request is None:
                raise NativeIneligible(
                    f"session {session.id}: node {node_id} has no request"
                )

            if self._kind == KIND_TEXT_SSE:
                transport, sidecar = self._compile_text(
                    session, node_id, request, history_cache
                )
            elif self._kind == KIND_TTS_HTTP:
                transport, sidecar = self._compile_tts_http(session, node_id, request)
            elif self._kind == KIND_TTS_REALTIME_WS:
                transport, sidecar = self._compile_tts_realtime(
                    session, node_id, request
                )
            else:
                transport, sidecar = self._compile_stt(
                    session, node_id, request, blob_registrar
                )

            sidecar.session_id = session.id
            sidecar.session_size = session_size

            rp = vn.RequestPlan()
            rp.request_id = request.id
            rp.node_id = node_id
            rp.wait_after_ready_s = float(graph.nodes[node_id].wait_after_ready)
            rp.parents = [
                (e.src, bool(e.is_history_parent)) for e in parents(graph, node_id)
            ]
            rp.transport = transport
            request_plans.append(rp)
            sidecars[request.id] = sidecar

        plan = vn.SessionPlan()
        plan.session_id = session.id
        plan.requests = request_plans
        plan.dispatch_ticket_base = -1
        return plan, sidecars

    # ------------------------------------------------------------------
    # TEXT_SSE
    # ------------------------------------------------------------------

    def _accumulated_history(
        self,
        session: Session,
        node_id: int,
        cache: Dict[int, List[Tuple[str, Any]]],
    ) -> List[Tuple[str, Any]]:
        """The conversation ``node_id`` would RECORD for its history children
        (the scheduler's ``node_histories[node_id]``).

        Entries are ("user", text) / ("assistant", node_id-hole). The
        accumulation is transitive through contiguous history edges and
        resets on a non-history edge — exactly ``_populate_history`` +
        ``_record_history``.
        """
        if node_id in cache:
            return cache[node_id]
        graph = session.session_graph
        incoming = [e for e in parents(graph, node_id) if e.is_history_parent]
        if len(incoming) > 1:
            raise NativeIneligible(
                f"session {session.id}: ambiguous history inheritance for "
                f"node {node_id}"
            )
        inherited: List[Tuple[str, Any]] = (
            list(self._accumulated_history(session, incoming[0].src, cache))
            if incoming
            else []
        )
        request = session.requests[node_id]
        self._require_text_only(session, node_id, request)
        text_content = request.channels[ChannelModality.TEXT]
        inherited.append(("user", text_content.input_text))
        inherited.append(("assistant", node_id))
        cache[node_id] = inherited
        return inherited

    def _require_text_only(
        self, session: Session, node_id: int, request: Request
    ) -> None:
        modalities = set(request.channels.keys())
        if modalities != {ChannelModality.TEXT}:
            extra = sorted(str(m) for m in modalities - {ChannelModality.TEXT})
            raise NativeIneligible(
                f"session {session.id} node {node_id}: non-text channels "
                f"{extra} are not supported by the native text transport"
            )

    def _sampling_params(self, request: Request) -> Dict[str, Any]:
        """Global + per-request sampling params, same merge order and lm-eval
        key mapping as ``OpenAIBaseClient._get_sampling_params``."""
        params: Dict[str, Any] = {}
        global_params = getattr(self._config, "additional_sampling_params_dict", None)
        if isinstance(global_params, dict):
            params.update(global_params)
        if isinstance(request.metadata, dict):
            per_request = request.metadata.get("sampling_params")
            if isinstance(per_request, dict):
                params.update(per_request)
        if "until" in params:
            if "stop" not in params:
                params["stop"] = params["until"]
            params.pop("until", None)
        if "max_gen_toks" in params:
            max_gen_toks = params.pop("max_gen_toks", None)
            max_tokens_param = getattr(self._config, "max_tokens_param", None)
            if (
                max_tokens_param
                and max_gen_toks is not None
                and max_tokens_param not in params
            ):
                params[max_tokens_param] = max_gen_toks
        return params

    def _chat_address(self) -> str:
        api_base = self._api_base
        if not api_base.endswith("/"):
            api_base = api_base + "/"
        return api_base + str(self._config.address_append_value)

    def _compile_text(
        self,
        session: Session,
        node_id: int,
        request: Request,
        history_cache: Dict[int, List[Tuple[str, Any]]],
    ) -> Tuple[Any, RequestSidecar]:
        vn = self._vn
        graph = session.session_graph
        self._require_text_only(session, node_id, request)

        incoming = [e for e in parents(graph, node_id) if e.is_history_parent]
        if len(incoming) > 1:
            raise NativeIneligible(
                f"session {session.id}: ambiguous history inheritance for "
                f"node {node_id}"
            )
        history_entries: List[Tuple[str, Any]] = (
            list(self._accumulated_history(session, incoming[0].src, history_cache))
            if incoming
            else []
        )

        text_content = request.channels[ChannelModality.TEXT]
        current_text = text_content.input_text

        # messages exactly as openai_chat._build_message_content builds them:
        # history first, then the current single-text-block user message as a
        # plain string.
        messages: List[Dict[str, Any]] = []
        message_parts: List[Tuple[str, Any]] = []
        hole_nodes: List[int] = []
        for role, payload in history_entries:
            if role == "user":
                messages.append({"role": "user", "content": payload})
                message_parts.append(("text", payload))
            else:
                sentinel = f"{self._hole_token}-{len(hole_nodes)}@@"
                messages.append({"role": "assistant", "content": sentinel})
                message_parts.append(("hole", payload))
                hole_nodes.append(payload)
        messages.append({"role": "user", "content": current_text})
        message_parts.append(("text", current_text))

        # body exactly as openai_chat.send_request builds it.
        body: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "stream": True,
            "ignore_eos": self._config.ignore_eos,
        }
        body.update(self._sampling_params(request))

        max_tokens_limit = None
        if request.requested_output is not None and request.requested_output.text:
            max_tokens_limit = request.requested_output.text.target_tokens
        max_tokens_param = getattr(self._config, "max_tokens_param", None)
        if (
            max_tokens_limit is not None
            and int(max_tokens_limit) > 0
            and max_tokens_param
            and max_tokens_param not in body
        ):
            body[max_tokens_param] = max_tokens_limit
        min_tokens_param = getattr(self._config, "min_tokens_param", None)
        if (
            min_tokens_param
            and max_tokens_limit is not None
            and int(max_tokens_limit) > 0
            and min_tokens_param not in body
        ):
            body[min_tokens_param] = max_tokens_limit

        # NOTE: the Python client formats config.api_key directly (even when
        # None); replicate verbatim for wire fidelity.
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        transport = self._build_http_transport(
            vn.TransportKind.TEXT_SSE,
            self._chat_address(),
            body,
            headers,
            hole_nodes,
        )
        sidecar = RequestSidecar(
            request=request,
            session_id=session.id,
            session_size=0,  # set by compile()
            node_id=node_id,
            kind=KIND_TEXT_SSE,
            message_parts=message_parts,
            delta_text=current_text,
        )
        return transport, sidecar

    def _build_http_transport(
        self,
        kind: Any,
        url: str,
        json_body: Dict[str, Any],
        headers: Dict[str, str],
        hole_nodes: List[int],
    ) -> Any:
        """Render an HTTP plan through httpx's own request builder.

        The returned TransportPlan reproduces the request byte-for-byte:
        ``header_prefix + str(len(body)) + header_suffix + body`` with
        native recomputing Content-Length after splicing holes.
        """
        vn = self._vn
        req = self._httpx.build_request("POST", url, json=json_body, headers=headers)
        body_bytes = req.read()
        target = req.url.raw_path.decode("ascii")

        lines: List[Tuple[str, str]] = [
            (k.decode("latin-1"), v.decode("latin-1")) for k, v in req.headers.raw
        ]
        cl_index = next(
            i for i, (k, _) in enumerate(lines) if k.lower() == "content-length"
        )
        prefix = f"POST {target} HTTP/1.1\r\n"
        for k, v in lines[:cl_index]:
            prefix += f"{k}: {v}\r\n"
        prefix += f"{lines[cl_index][0]}: "
        suffix = "\r\n"
        for k, v in lines[cl_index + 1 :]:
            suffix += f"{k}: {v}\r\n"
        suffix += "\r\n"

        body_str = body_bytes.decode("utf-8")
        segments: List[str] = []
        rest = body_str
        for i in range(len(hole_nodes)):
            token = f"{self._hole_token}-{i}@@"
            idx = rest.find(token)
            if idx < 0:
                raise NativeIneligible(
                    "internal: history hole sentinel not found in encoded body"
                )
            segments.append(rest[:idx])
            rest = rest[idx + len(token) :]
        segments.append(rest)

        t = vn.TransportPlan()
        t.kind = kind
        t.header_prefix = prefix
        t.header_suffix = suffix
        t.body_segments = segments
        t.history_refs = list(hole_nodes)
        return t

    # ------------------------------------------------------------------
    # TTS_HTTP
    # ------------------------------------------------------------------

    def _compile_tts_http(
        self, session: Session, node_id: int, request: Request
    ) -> Tuple[Any, RequestSidecar]:
        from veeksha.client.tts import _build_audio_speech_url
        from veeksha.core.request_content import TextChannelRequestContent

        vn = self._vn
        text_content = request.channels.get(ChannelModality.TEXT)
        if not isinstance(text_content, TextChannelRequestContent):
            raise NativeIneligible(
                f"session {session.id} node {node_id}: TTS request has no "
                "TEXT channel"
            )
        input_text = text_content.input_text

        # Exactly TTSClient._build_request.
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: Dict[str, Any] = {
            "model": self._config.model,
            "input": input_text,
            "voice": self._config.voice_id,
            "response_format": "pcm" if self._config.raw_pcm else "wav",
            "stream": True,
            "stream_format": "audio",
        }
        url = _build_audio_speech_url(self._api_base)

        transport = self._build_http_transport(
            vn.TransportKind.TTS_HTTP, url, payload, headers, []
        )
        sidecar = RequestSidecar(
            request=request,
            session_id=session.id,
            session_size=0,
            node_id=node_id,
            kind=KIND_TTS_HTTP,
            input_text=input_text,
            input_tokens=text_content.target_prompt_tokens or 0,
        )
        return transport, sidecar

    # ------------------------------------------------------------------
    # TTS_REALTIME_WS
    # ------------------------------------------------------------------

    def _compile_tts_realtime(
        self, session: Session, node_id: int, request: Request
    ) -> Tuple[Any, RequestSidecar]:
        from veeksha.client.realtime_tts import RealtimeTTSProtocol
        from veeksha.client.utils import TextDeltaPacer, segment_text
        from veeksha.core.request_content import TextChannelRequestContent

        vn = self._vn
        text_content = request.channels.get(ChannelModality.TEXT)
        if not isinstance(text_content, TextChannelRequestContent):
            raise NativeIneligible(
                f"session {session.id} node {node_id}: realtime TTS request "
                "has no TEXT channel"
            )
        input_text = text_content.input_text

        protocol = RealtimeTTSProtocol(self._config, api_key=self._api_key)
        ws_url = protocol.build_ws_url(self._api_base)
        parsed = urlparse(ws_url)
        ws_path = parsed.path + (f"?{parsed.query}" if parsed.query else "")

        pacing = self._config.pacing
        segments = segment_text(input_text, pacing.tokens_per_delta)
        pacer = TextDeltaPacer(pacing, seed=pacing.seed + request.id)

        t = vn.TransportPlan()
        t.kind = vn.TransportKind.TTS_REALTIME_WS
        t.ws_path = ws_path
        t.setup_frames = [self._ws_frame(protocol.session_update_json())]
        paced = []
        offset_s = pacer.initial_delay_s
        for seg in segments:
            offset_s += pacer.next_gap()
            paced.append(
                self._ws_frame(
                    protocol.conversation_item_create_json(seg.text),
                    send_offset_ms=offset_s * 1000.0,
                )
            )
        t.paced_frames = paced
        t.finish_frames = [self._ws_frame(protocol.response_create_json())]
        t.done_markers = ["response.done"]

        sidecar = RequestSidecar(
            request=request,
            session_id=session.id,
            session_size=0,
            node_id=node_id,
            kind=KIND_TTS_REALTIME_WS,
            input_text=input_text,
            input_tokens=text_content.target_prompt_tokens
            or sum(seg.n_tokens for seg in segments),
            segment_chars=[seg.n_chars for seg in segments],
        )
        return t, sidecar

    def _ws_frame(
        self, payload: str, send_offset_ms: float = -1.0, blob: Any = None
    ) -> Any:
        vn = self._vn
        f = vn.WsFrame()
        if payload is not None:
            f.payload = payload
        if blob is not None:
            f.blob = blob
        f.send_offset_ms = send_offset_ms
        return f

    # ------------------------------------------------------------------
    # STT_WS (vllm_realtime)
    # ------------------------------------------------------------------

    def _stt_clip_plan(
        self,
        audio_path: str,
        start_ms: Optional[float],
        end_ms: Optional[float],
        blob_registrar: Callable[[bytes], int],
    ) -> _SttClipPlan:
        """One blob per distinct clip+slice: the concatenated pre-encoded
        append frames, with per-frame (offset, len) slices and the 1x
        schedule. Reuses stt.py's decode/slice helpers."""
        key = (audio_path, start_ms, end_ms)
        cached = self._stt_clip_plans.get(key)
        if cached is not None:
            return cached

        from veeksha.client.stt import (
            BYTES_PER_SAMPLE,
            _audio_to_pcm16_bytes,
            _pcm_duration_ms,
            _slice_pcm16_bytes,
        )

        sample_rate = self._config.sample_rate
        chunk_size = self._config.ws_chunk_size
        pcm = _audio_to_pcm16_bytes(audio_path, sample_rate)
        sliced = bytes(
            _slice_pcm16_bytes(
                memoryview(pcm), sample_rate, start_ms=start_ms, end_ms=end_ms
            )
        )

        frames: List[bytes] = []
        schedule: List[float] = []
        for byte_offset in range(0, len(sliced), chunk_size):
            chunk = sliced[byte_offset : byte_offset + chunk_size]
            # Exactly VllmRealtimeSTTClient._encode_chunk.
            message = json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }
            )
            frames.append(message.encode("utf-8"))
            # stt.py paces chunk i at audio_start + byte_offset/(2*sr) s.
            schedule.append(byte_offset / BYTES_PER_SAMPLE / sample_rate * 1000.0)

        blob = b"".join(frames)
        blob_id = blob_registrar(blob)
        slices: List[Tuple[int, int]] = []
        off = 0
        for frame in frames:
            slices.append((off, len(frame)))
            off += len(frame)

        plan = _SttClipPlan(
            blob_id=blob_id,
            frame_slices=slices,
            schedule_ms=schedule,
            pcm_byte_count=len(sliced),
            duration_ms=_pcm_duration_ms(len(sliced), sample_rate),
        )
        self._stt_clip_plans[key] = plan
        return plan

    def _compile_stt(
        self,
        session: Session,
        node_id: int,
        request: Request,
        blob_registrar: Optional[Callable[[bytes], int]],
    ) -> Tuple[Any, RequestSidecar]:
        from veeksha.client.stt import _metadata_ms
        from veeksha.core.request_content import AudioChannelRequestContent

        vn = self._vn
        if blob_registrar is None:
            raise NativeIneligible("STT compilation requires a blob registrar")
        audio_content = request.channels.get(ChannelModality.AUDIO)
        if not isinstance(audio_content, AudioChannelRequestContent):
            raise NativeIneligible(
                f"session {session.id} node {node_id}: STT request has no "
                "AUDIO channel"
            )
        audio_path = audio_content.input_audio
        metadata = request.metadata or {}
        try:
            start_ms = _metadata_ms(metadata, "input_audio_start_ms")
            end_ms = _metadata_ms(metadata, "input_audio_end_ms")
            clip = self._stt_clip_plan(audio_path, start_ms, end_ms, blob_registrar)
        except NativeIneligible:
            raise
        except Exception as exc:
            raise NativeIneligible(
                f"session {session.id} node {node_id}: failed to decode/slice "
                f"audio clip {audio_path!r}: {exc}"
            ) from exc

        pacing = bool(self._config.ws_realtime_pacing)

        t = vn.TransportPlan()
        t.kind = vn.TransportKind.STT_WS
        # Exactly _STTClientBase._http_to_ws: api_base path prefix + ws_path.
        t.ws_path = urlparse(self._api_base).path + "/v1/realtime"
        # Exactly VllmRealtimeSTTClient._open_session's two sends. (The
        # Python client first WAITS for session.created; the native main loop
        # sends setup frames immediately after the handshake — see the
        # deviations note in native_loop.py.)
        t.setup_frames = [
            self._ws_frame(
                json.dumps({"type": "session.update", "model": self._config.model})
            ),
            self._ws_frame(json.dumps({"type": "input_audio_buffer.commit"})),
        ]
        paced = []
        for (offset, length), when_ms in zip(clip.frame_slices, clip.schedule_ms):
            b = vn.BlobRef()
            b.blob_id = clip.blob_id
            b.offset = offset
            b.len = length
            paced.append(
                self._ws_frame(None, send_offset_ms=when_ms if pacing else -1.0, blob=b)
            )
        t.paced_frames = paced
        # Exactly VllmRealtimeSTTClient._eof().
        t.finish_frames = [
            self._ws_frame(
                json.dumps({"type": "input_audio_buffer.commit", "final": True})
            )
        ]
        t.done_markers = ["transcription.done"]

        sidecar = RequestSidecar(
            request=request,
            session_id=session.id,
            session_size=0,
            node_id=node_id,
            kind=KIND_STT_WS,
            pcm_byte_count=clip.pcm_byte_count,
            input_audio_duration_ms=clip.duration_ms,
        )
        return t, sidecar
