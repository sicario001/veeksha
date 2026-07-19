import asyncio

import pytest

from veeksha.client import ClientRegistry
from veeksha.config.client import OpenAICompletionsClientConfig
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import RequestResult
from veeksha.core.tokenizer import TokenizerHandle, TokenizerProvider
from veeksha.types import ChannelModality


@pytest.mark.unit
def test_openai_completions_client_sends_all_requests_to_completions() -> None:
    tokenizer_handle = TokenizerHandle(
        count_tokens=lambda text: len(str(text).split()),
        decode=lambda token_ids: "",
        encode=lambda text: [0] * len(str(text).split()),
    )
    tokenizer_provider = TokenizerProvider({ChannelModality.TEXT: tokenizer_handle})

    config = OpenAICompletionsClientConfig(
        api_base="http://example.com/v1",
        api_key="",
        model="mock",
    )
    client = ClientRegistry.get(
        config.get_type(),
        config=config,
        tokenizer_provider=tokenizer_provider,
    )

    called = {"count": 0}

    async def _fake_send_completions_request(
        *,
        request: Request,
        session_id: int,
        session_total_requests: int,
        on_request_sent=None,
        on_request_dispatched=None,
    ) -> RequestResult:
        called["count"] += 1
        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            success=True,
            client_completed_at=0.0,
        )

    client._send_completions_request = _fake_send_completions_request  # type: ignore[attr-defined]

    from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec

    req = Request(
        id=123,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="hello",
            )
        },
        requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=1)),
    )

    result = asyncio.run(
        client.send_request(req, session_id=7, session_total_requests=1)
    )
    assert called["count"] == 1
    assert result.success is True
    assert result.request_id == 123
    assert result.session_id == 7
