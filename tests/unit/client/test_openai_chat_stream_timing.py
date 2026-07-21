"""Read-time stamping in the chat streaming client.

The per-chunk stamp must be taken when bytes arrive from the transport, not
after SSE framing + ``json.loads`` -- otherwise the client's own parse cost is
recorded as server latency. Two SSE events framed out of one read share that
read's stamp, so their inter-chunk delta is exactly 0.0.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from veeksha.client.openai_chat import OpenAIChatCompletionsClient
from veeksha.config.client import OpenAIChatCompletionsClientConfig
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.tokenizer import TokenizerHandle, TokenizerProvider
from veeksha.types import ChannelModality


def _chat_client(api_base: str) -> OpenAIChatCompletionsClient:
    tokenizer_handle = TokenizerHandle(
        count_tokens=lambda text: len(str(text).split()),
        decode=lambda token_ids: "",
        encode=lambda text: [0] * len(str(text).split()),
    )
    config = OpenAIChatCompletionsClientConfig(
        api_base=api_base,
        api_key="test-key",
        model="dummy",
    )
    return OpenAIChatCompletionsClient(
        config=config,
        tokenizer_provider=TokenizerProvider({ChannelModality.TEXT: tokenizer_handle}),
    )


def _sse_event(text: str) -> str:
    payload = {"choices": [{"delta": {"content": text}}]}
    return f"data: {json.dumps(payload)}\n\n"


class _FakeResponse:
    """Minimal httpx.Response stand-in yielding pre-baked transport reads."""

    def __init__(self, reads: list[str]) -> None:
        self._reads = reads

    def raise_for_status(self) -> None:
        return None

    async def aiter_text(self):
        for read in self._reads:
            yield read


@pytest.mark.unit
def test_process_stream_shares_one_read_stamp_across_coalesced_events() -> None:
    """Events framed out of a single read carry that read's stamp."""
    client = _chat_client("http://127.0.0.1:1/v1")
    response = _FakeResponse(
        [
            _sse_event("a") + _sse_event("b"),  # two events in one read
            _sse_event("c"),
            "data: [DONE]\n\n",
        ]
    )

    async def _collect() -> list[tuple[dict, float]]:
        return [item async for item in client._process_stream(response)]

    events = asyncio.run(_collect())

    assert [e["choices"][0]["delta"]["content"] for e, _ in events] == ["a", "b", "c"]
    assert events[0][1] == events[1][1]  # same read -> same stamp
    assert events[2][1] >= events[1][1]


@pytest.mark.unit
def test_process_stream_stamps_split_event_at_completing_read() -> None:
    """An event split across two reads is stamped at the read that completed it."""
    client = _chat_client("http://127.0.0.1:1/v1")
    whole = _sse_event("split")
    head, tail = whole[:12], whole[12:]
    response = _FakeResponse([head, tail + _sse_event("next"), "data: [DONE]\n\n"])

    async def _collect() -> list[tuple[dict, float]]:
        return [item async for item in client._process_stream(response)]

    events = asyncio.run(_collect())

    assert [e["choices"][0]["delta"]["content"] for e, _ in events] == ["split", "next"]
    # Both were completed by the second read.
    assert events[0][1] == events[1][1]


class _SSEServer:
    """Localhost SSE server writing a scripted sequence of TCP writes."""

    def __init__(self, writes: list[tuple[str, float]]) -> None:
        self._writes = writes
        self._server: asyncio.Server | None = None
        self.port = 0

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader, writer) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Connection: close\r\n\r\n"
        )
        await writer.drain()
        for payload, delay_s in self._writes:
            if delay_s:
                await asyncio.sleep(delay_s)
            writer.write(payload.encode())
            await writer.drain()
        writer.close()


@pytest.mark.unit
def test_send_request_charges_no_parse_cost_to_coalesced_events() -> None:
    """End-to-end: two events in one TCP write yield a 0.0 inter-chunk delta."""
    gap_s = 0.08
    server = _SSEServer(
        [
            (_sse_event("a") + _sse_event("b"), 0.05),
            (_sse_event("c"), gap_s),
            ("data: [DONE]\n\n", 0.0),
        ]
    )

    async def _run():
        await server.start()
        try:
            client = _chat_client(f"http://127.0.0.1:{server.port}/v1")
            request = Request(
                id=1,
                channels={
                    ChannelModality.TEXT: TextChannelRequestContent(input_text="hi")
                },
            )
            return await client.send_request(request, session_id=0)
        finally:
            await server.stop()

    result = asyncio.run(_run())

    assert result.success is True
    metrics = result.channels[ChannelModality.TEXT].metrics
    inter_chunk_times = metrics["inter_chunk_times"]

    # Semantics unchanged: one entry per text delta, element 0 = send -> first chunk.
    assert len(inter_chunk_times) == 3
    assert inter_chunk_times[0] > 0.0
    # Coalesced pair: identical read stamp, so an exact zero gap.
    assert inter_chunk_times[1] == 0.0
    # The separately written event keeps its real gap.
    assert inter_chunk_times[2] >= gap_s * 0.5
    assert result.channels[ChannelModality.TEXT].content == "abc"

    # Absolute anchors are recorded additively and reconstruct chunk arrivals:
    # chunk i arrived at request_start_monotonic + sum(inter_chunk_times[:i+1]).
    start = metrics["request_start_monotonic"]
    sent = metrics["request_sent_monotonic"]
    assert isinstance(start, float)
    assert isinstance(sent, float)
    assert sent >= start  # POST leaves after t_start...
    running = 0.0
    prev = sent
    for gap in inter_chunk_times:
        running += gap
        arrival = start + running
        assert arrival >= sent  # ...and every chunk arrives after it.
        assert arrival >= prev - 1e-9  # reconstruction is monotonic
        prev = arrival
