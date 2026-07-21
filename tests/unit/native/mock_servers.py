"""Self-contained asyncio mock servers + plan builders for the native main loop
tests.

Deliberately no dependency on tests/helpers or any veeksha Python module: the
native main loop's pybind surface IS the contract under test, and the servers
speak raw HTTP/1.1 + SSE (chunked) and raw RFC 6455 WebSocket so the native
state machines are exercised over real sockets.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
import time

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class AsyncServerThread:
    """asyncio TCP server running on its own thread + event loop."""

    def __init__(self):
        self.loop = None
        self.port = None
        self._thread = None
        self._started = threading.Event()
        self._stop_ev = None

    async def _handle(self, reader, writer):  # pragma: no cover - abstract
        raise NotImplementedError

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._started.wait(5):
            raise RuntimeError("mock server failed to start")
        return self

    def _run(self):
        asyncio.run(self._main())

    async def _main(self):
        self.loop = asyncio.get_running_loop()
        # clients (the native main loop) legitimately hang up mid-stream in the
        # stop/grace tests; suppress the unretrieved BrokenPipeError noise
        self.loop.set_exception_handler(lambda loop, ctx: None)
        self._stop_ev = asyncio.Event()
        server = await asyncio.start_server(self._safe_handle, "127.0.0.1", 0)
        self.port = server.sockets[0].getsockname()[1]
        self._started.set()
        await self._stop_ev.wait()
        server.close()
        await server.wait_closed()

    async def _safe_handle(self, reader, writer):
        try:
            await self._handle(reader, writer)
        except (
            asyncio.IncompleteReadError,
            ConnectionResetError,
            BrokenPipeError,
        ):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    def stop(self):
        if self.loop is not None and self._started.is_set():
            self.loop.call_soon_threadsafe(self._stop_ev.set)
        if self._thread is not None:
            self._thread.join(5)


# ---------------------------------------------------------------------------
# HTTP / SSE
# ---------------------------------------------------------------------------


async def _read_http_request(reader):
    """Read one HTTP request; returns (headers_bytes, body_str)."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = await reader.read(65536)
        if not chunk:
            return None, None
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    clen = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            clen = int(line.split(b":", 1)[1])
    while len(body) < clen:
        chunk = await reader.read(65536)
        if not chunk:
            break
        body += chunk
    return head, body.decode("utf-8")


def _write_http_chunk(writer, text):
    payload = text.encode("utf-8")
    writer.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")


class SseChatServer(AsyncServerThread):
    """Mock OpenAI chat-completions SSE endpoint with a known emit cadence.

    Emits `num_chunks` delta events on absolute deadlines spaced
    `chunk_gap_s` apart after a `prefill_s` sleep, over chunked
    transfer-encoding. Captures every request body (the wire-parity /
    history-splice proof). A request whose body contains ``fail_marker``
    receives `fail_status` with no stream.
    """

    def __init__(
        self,
        num_chunks=6,
        chunk_gap_s=0.03,
        prefill_s=0.05,
        token_text="tok{i} ",
        fail_marker=None,
        fail_status=500,
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.chunk_gap_s = chunk_gap_s
        self.prefill_s = prefill_s
        self.token_text = token_text
        self.fail_marker = fail_marker
        self.fail_status = fail_status
        self.bodies = []
        self._lock = threading.Lock()

    def token(self, i):
        return self.token_text.format(i=i)

    def full_reply(self):
        return "".join(self.token(i) for i in range(self.num_chunks))

    async def _handle(self, reader, writer):
        _, body = await _read_http_request(reader)
        if body is None:
            return
        with self._lock:
            self.bodies.append(body)
        if self.fail_marker is not None and self.fail_marker in body:
            writer.write(
                f"HTTP/1.1 {self.fail_status} ERR\r\n"
                "Content-Length: 0\r\nConnection: close\r\n\r\n".encode()
            )
            await writer.drain()
            return
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )
        await writer.drain()
        await asyncio.sleep(self.prefill_s)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for i in range(self.num_chunks):
            delay = t0 + i * self.chunk_gap_s - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            payload = json.dumps({"choices": [{"delta": {"content": self.token(i)}}]})
            _write_http_chunk(writer, f"data: {payload}\n\n")
            await writer.drain()
        _write_http_chunk(writer, "data: [DONE]\n\n")
        writer.write(b"0\r\n\r\n")
        await writer.drain()


class MalformedChunkSseServer(AsyncServerThread):
    """SSE endpoint that violates HTTP chunked framing for requests whose body
    contains `bad_marker`: after one valid data chunk it emits a chunk-size
    line that is not hex, then closes. Other requests stream a normal short
    SSE reply. Proves malformed framing fails the request (llhttp reason)
    while the native loop keeps serving."""

    def __init__(self, bad_marker="BAD-ME", num_chunks=3, chunk_gap_s=0.01):
        super().__init__()
        self.bad_marker = bad_marker
        self.num_chunks = num_chunks
        self.chunk_gap_s = chunk_gap_s

    async def _handle(self, reader, writer):
        _, body = await _read_http_request(reader)
        if body is None:
            return
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )
        if self.bad_marker in body:
            payload = json.dumps({"choices": [{"delta": {"content": "x"}}]})
            _write_http_chunk(writer, f"data: {payload}\n\n")
            writer.write(b"zz\r\n")  # chunk size must be hex -> framing broken
            await writer.drain()
            return
        for i in range(self.num_chunks):
            await asyncio.sleep(self.chunk_gap_s)
            payload = json.dumps({"choices": [{"delta": {"content": f"t{i} "}}]})
            _write_http_chunk(writer, f"data: {payload}\n\n")
            await writer.drain()
        _write_http_chunk(writer, "data: [DONE]\n\n")
        writer.write(b"0\r\n\r\n")
        await writer.drain()


# ---------------------------------------------------------------------------
# WebSocket framing
# ---------------------------------------------------------------------------


async def _ws_handshake(reader, writer):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = await reader.read(65536)
        if not chunk:
            return False
        data += chunk
    head = data.split(b"\r\n\r\n", 1)[0]
    key = ""
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"sec-websocket-key:"):
            key = line.split(b":", 1)[1].strip().decode()
    accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
    writer.write(
        (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode()
    )
    await writer.drain()
    return True


async def _ws_read_frame(reader):
    hdr = await reader.readexactly(2)
    b0, b1 = hdr[0], hdr[1]
    opcode = b0 & 0x0F
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F
    if length == 126:
        length = int.from_bytes(await reader.readexactly(2), "big")
    elif length == 127:
        length = int.from_bytes(await reader.readexactly(8), "big")
    mask = await reader.readexactly(4) if masked else None
    payload = await reader.readexactly(length)
    if mask:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def _ws_send_text(writer, text):
    data = text.encode("utf-8")
    hdr = bytearray([0x81])
    n = len(data)
    if n < 126:
        hdr.append(n)
    elif n <= 0xFFFF:
        hdr.append(126)
        hdr += n.to_bytes(2, "big")
    else:
        hdr.append(127)
        hdr += n.to_bytes(8, "big")
    writer.write(bytes(hdr) + data)


class RealtimeTtsServer(AsyncServerThread):
    """Mock realtime-TTS WS endpoint: session.updated on connect; on a frame
    containing "response.create" emits `num_chunks` audio deltas on a fixed
    cadence, then response.done. Records every received frame with a
    time.monotonic() stamp."""

    def __init__(self, num_chunks=5, chunk_gap_s=0.03, first_delay_s=0.02):
        super().__init__()
        self.num_chunks = num_chunks
        self.chunk_gap_s = chunk_gap_s
        self.first_delay_s = first_delay_s
        self.received = []  # (monotonic, payload_str)
        self._lock = threading.Lock()

    async def _handle(self, reader, writer):
        if not await _ws_handshake(reader, writer):
            return
        _ws_send_text(writer, json.dumps({"type": "session.updated"}))
        await writer.drain()
        while True:
            opcode, payload = await _ws_read_frame(reader)
            if opcode == 0x8:
                return
            if opcode not in (0x1, 0x2):
                continue
            text = payload.decode("utf-8", errors="replace")
            with self._lock:
                self.received.append((time.monotonic(), text))
            if "response.create" in text:
                break
        await asyncio.sleep(self.first_delay_s)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for i in range(self.num_chunks):
            delay = t0 + i * self.chunk_gap_s - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            _ws_send_text(
                writer,
                json.dumps(
                    {"type": "response.output_audio.delta", "delta": "QUJDRA=="}
                ),
            )
            await writer.drain()
        _ws_send_text(writer, json.dumps({"type": "response.done"}))
        await writer.drain()
        # real realtime servers keep the session open; the client finishes on
        # the done marker. Wait for the client to close.
        try:
            await reader.read(65536)
        except Exception:
            pass


class SttServer(AsyncServerThread):
    """Mock STT WS endpoint: records input_audio_buffer.append payloads
    (stamped on receipt), and on commit emits transcript deltas on a fixed
    cadence followed by transcription.done."""

    def __init__(self, num_deltas=3, delta_gap_s=0.03, word="w{i} "):
        super().__init__()
        self.num_deltas = num_deltas
        self.delta_gap_s = delta_gap_s
        self.word = word
        self.appends = []  # (monotonic, payload_str)
        self._lock = threading.Lock()

    def transcript(self):
        return "".join(self.word.format(i=i) for i in range(self.num_deltas))

    async def _handle(self, reader, writer):
        if not await _ws_handshake(reader, writer):
            return
        while True:
            opcode, payload = await _ws_read_frame(reader)
            if opcode == 0x8:
                return
            if opcode not in (0x1, 0x2):
                continue
            text = payload.decode("utf-8", errors="replace")
            if "input_audio_buffer.append" in text:
                with self._lock:
                    self.appends.append((time.monotonic(), text))
            elif "input_audio_buffer.commit" in text:
                break
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for i in range(self.num_deltas):
            delay = t0 + i * self.delta_gap_s - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            _ws_send_text(
                writer,
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "delta": self.word.format(i=i),
                    }
                ),
            )
            await writer.drain()
        _ws_send_text(writer, json.dumps({"type": "transcription.done"}))
        await writer.drain()
        try:
            await reader.read(65536)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# plan / loop builders (thin wrappers over the pybind structs)
# ---------------------------------------------------------------------------


def make_loop(
    vn,
    kind,
    port,
    *,
    target=4,
    rampup=0.0,
    ordering=None,
    request_timeout_s=15.0,
    clients=2,
    dispatchers=2,
    acks=2,
    cancel_on_failure=True,
):
    rt = vn.NativeRuntimeConfig()
    rt.num_dispatcher_threads = dispatchers
    rt.num_completion_threads = acks
    rt.num_client_threads = clients
    tp = vn.TrafficPlanConfig()
    tp.kind = kind
    tp.target_concurrent_sessions = target
    tp.rampup_seconds = rampup
    tp.cancel_session_on_failure = cancel_on_failure
    if ordering is not None:
        tp.ordering = ordering
    ep = vn.EndpointConfig()
    ep.host = "127.0.0.1"
    ep.port = port
    ep.request_timeout_s = request_timeout_s
    return vn.NativeBenchmarkLoop(rt, tp, ep, time.monotonic())


def text_transport(vn, port, body_segments, history_refs=()):
    t = vn.TransportPlan()
    t.kind = vn.TransportKind.TEXT_SSE
    t.header_prefix = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: "
    )
    t.header_suffix = "\r\nConnection: close\r\n\r\n"
    t.body_segments = list(body_segments)
    t.history_refs = list(history_refs)
    return t


def request_plan(vn, request_id, node_id, transport, parents=(), wait=0.0):
    r = vn.RequestPlan()
    r.request_id = request_id
    r.node_id = node_id
    r.wait_after_ready_s = wait
    r.parents = list(parents)
    r.transport = transport
    return r


def session_plan(vn, session_id, requests, ticket_base=-1):
    s = vn.SessionPlan()
    s.session_id = session_id
    s.requests = list(requests)
    s.dispatch_ticket_base = ticket_base
    return s


def ws_frame(vn, payload=None, blob=None, send_offset_ms=-1.0):
    f = vn.WsFrame()
    if payload is not None:
        f.payload = payload
    if blob is not None:
        f.blob = blob
    f.send_offset_ms = send_offset_ms
    return f


def drain_until_done(loop, expect_completed, timeout_s=20.0):
    """Drain events until `expect_completed` COMPLETED events arrived and the
    loop reports idle; join and return the ordered event list."""
    events = []
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        events.extend(loop.drain_events(max_items=256, timeout_s=0.1))
        c = loop.counters()
        completed = sum(1 for e in events if int(e.kind) == 1)
        if completed >= expect_completed and c.idle:
            break
    else:
        raise AssertionError(
            f"loop did not finish in {timeout_s}s: counters="
            f"{loop.counters().requests_completed} events={len(events)}"
        )
    assert loop.join(10.0), "native loop threads did not join"
    events.extend(loop.drain_events(max_items=8192, timeout_s=0.0))
    return events


def split_events(vn, events):
    dispatched = [e for e in events if e.kind == vn.EventKind.DISPATCHED]
    completed = [e for e in events if e.kind == vn.EventKind.COMPLETED]
    return dispatched, completed
