import asyncio

import pytest

from veeksha.client import ClientRegistry
from veeksha.config.client import OpenAIRouterClientConfig
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import RequestResult
from veeksha.core.tokenizer import TokenizerHandle, TokenizerProvider
from veeksha.types import ChannelModality


@pytest.mark.unit
def test_openai_router_client_routes_per_request_api_mode() -> None:
    tokenizer_handle = TokenizerHandle(
        count_tokens=lambda text: len(str(text).split()),
        decode=lambda token_ids: "",
        encode=lambda text: [0] * len(str(text).split()),
    )
    tokenizer_provider = TokenizerProvider({ChannelModality.TEXT: tokenizer_handle})

    config = OpenAIRouterClientConfig(
        api_base="http://example.com/v1",
        api_key="",
        model="mock",
    )
    client = ClientRegistry.get(
        config.get_type(),
        config=config,
        tokenizer_provider=tokenizer_provider,
    )

    called = {"chat": 0, "completions": 0}

    async def _fake_chat_send_request(
        *,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent=None,
        on_request_dispatched=None,
    ) -> RequestResult:
        called["chat"] += 1
        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            success=True,
            client_completed_at=0.0,
        )

    async def _fake_completions_send_request(
        *,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent=None,
        on_request_dispatched=None,
    ) -> RequestResult:
        called["completions"] += 1
        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            success=True,
            client_completed_at=0.0,
        )

    client._chat_client.send_request = _fake_chat_send_request  # type: ignore[attr-defined]
    client._completions_client.send_request = (  # type: ignore[attr-defined]
        _fake_completions_send_request
    )

    from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec

    chat_req = Request(
        id=1,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="hello",
            )
        },
        metadata={"api_mode": "chat"},
        requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=1)),
    )
    completions_req = Request(
        id=2,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="hello",
            )
        },
        metadata={"api_mode": "completions"},
        requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=1)),
    )

    asyncio.run(client.send_request(chat_req, session_id=0, session_total_requests=1))
    asyncio.run(
        client.send_request(completions_req, session_id=0, session_total_requests=1)
    )

    assert called["chat"] == 1
    assert called["completions"] == 1


@pytest.mark.unit
def test_openai_router_client_uses_separate_max_tokens_params() -> None:
    """Verify the router uses different max_tokens_param for each endpoint.

    The /chat/completions endpoint uses max_tokens_param (default: max_completion_tokens),
    while /completions uses completions_max_tokens_param (default: max_tokens).
    """
    tokenizer_handle = TokenizerHandle(
        count_tokens=lambda text: len(str(text).split()),
        decode=lambda token_ids: "",
        encode=lambda text: [0] * len(str(text).split()),
    )
    tokenizer_provider = TokenizerProvider({ChannelModality.TEXT: tokenizer_handle})

    config = OpenAIRouterClientConfig(
        api_base="http://example.com/v1",
        api_key="",
        model="mock",
        max_tokens_param="max_completion_tokens",  # for chat
        completions_max_tokens_param="max_tokens",  # for completions
    )
    client = ClientRegistry.get(
        config.get_type(),
        config=config,
        tokenizer_provider=tokenizer_provider,
    )

    # Verify the chat client uses the chat-specific max_tokens_param
    assert client._chat_client.config.max_tokens_param == "max_completion_tokens"

    # Verify the completions client uses the completions-specific max_tokens_param
    assert client._completions_client.config.max_tokens_param == "max_tokens"
