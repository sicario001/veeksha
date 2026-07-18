import json
from typing import Optional

from vidhi import BasePolyConfig, field, frozen_dataclass

from veeksha.logger import init_logger
from veeksha.types import ClientType

logger = init_logger(__name__)


@frozen_dataclass
class BaseClientConfig(BasePolyConfig):
    """LLM client configuration (OpenAI-compatible API)."""

    api_base: Optional[str] = field(
        None, help="API base URL. Defaults to OPENAI_API_BASE env var."
    )
    api_key: Optional[str] = field(
        None, help="API key. Defaults to OPENAI_API_KEY env var."
    )
    model: str = field(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        help="The model to use for this load test.",
    )
    address_append_value: str = field(
        "chat/completions", help="The address append value for the LLM API."
    )
    request_timeout: int = field(
        300, help="The timeout for each request to the LLM API (in seconds)."
    )
    max_connections: Optional[int] = field(
        None,
        help="Max concurrent HTTP connections in each client worker's httpx pool. "
        "None = unlimited (correct for load generation). httpx's own default is 100, "
        "which silently caps in-flight requests at num_client_threads*100 and corrupts "
        "high-concurrency timings — see analysis/bench/RESULTS.md.",
    )
    additional_sampling_params: str = field(
        "{}",
        help="Additional sampling params to send with each request to the LLM API.",
    )

    def __post_init__(self):
        self.additional_sampling_params_dict = {}
        if self.additional_sampling_params:
            self.additional_sampling_params_dict = json.loads(
                self.additional_sampling_params
            )


@frozen_dataclass
class OpenAIChatCompletionsClientConfig(BaseClientConfig):
    """OpenAI-compatible Chat Completions client configuration.

    `client.type: openai_chat_completions` uses `/chat/completions` (streaming).
    For per-request routing between chat + completions endpoints (e.g. lm-eval),
    use `client.type: openai_router`.
    """

    max_tokens_param: Optional[str] = field(
        "max_completion_tokens", help="Server parameter name for maximum tokens."
    )
    ignore_eos: bool = field(
        True,
        help="Sets the sampling param ignore_eos for requests to reach the desired max_tokens.",
    )
    min_tokens_param: Optional[str] = field(
        None,
        help="Server parameter name for minimum tokens, usually set if ignore_eos is not available or does no offer enough control over output tokens (see health_check_results.txt). Note: a wrong value might cause requests to fail.",
    )
    use_min_tokens_prompt_fallback: bool = field(
        False,
        help="If True, appends instructions to the prompt to generate at least N tokens (e.g. 'Generate at least 20 tokens'). Useful if the server does not support ignore_eos or min_tokens. Only available on synthetic content generation.",
    )

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.OPENAI_CHAT_COMPLETIONS

    def __post_init__(self):
        super().__post_init__()
        if self.use_min_tokens_prompt_fallback and self.min_tokens_param is None:
            logger.warning(
                "use_min_tokens_prompt_fallback is True but min_tokens_param is None. This will result in no min tokens control."
            )


@frozen_dataclass
class OpenAICompletionsClientConfig(OpenAIChatCompletionsClientConfig):
    """OpenAI Completions client configuration."""

    address_append_value: str = field(
        "completions", help="The address append value for the LLM API."
    )

    max_tokens_param: Optional[str] = field(
        "max_tokens", help="Server parameter name for maximum tokens."
    )

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.OPENAI_COMPLETIONS


@frozen_dataclass
class OpenAIRouterClientConfig(OpenAIChatCompletionsClientConfig):
    """OpenAI-compatible router client configuration.

    This config has the same surface area as `OpenAIChatCompletionsClientConfig`, but the
    corresponding client (`client.type: openai_router`) can route *per request*
    between:
    - `/chat/completions` (streaming)
    - `/completions` (non-stream)

    Routing is controlled by `request.metadata["api_mode"]` (e.g. set by the session generator).

    Note: The two endpoints have different parameter conventions. Use
    `completions_max_tokens_param` to override max tokens for the completions
    endpoint (defaults to "max_tokens"). The chat endpoint uses `max_tokens_param`
    (defaults to "max_completion_tokens").
    """

    completions_max_tokens_param: Optional[str] = field(
        "max_tokens",
        help="Server parameter name for maximum tokens on /completions endpoint. "
        "Defaults to 'max_tokens'. The /chat/completions endpoint uses max_tokens_param instead.",
    )

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.OPENAI_ROUTER
