from veeksha.core.lazy_loader import _LazyLoader
from veeksha.types import ClientType
from veeksha.types.base_registry import BaseRegistry


class ClientRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ClientType:
        return ClientType.from_str(key_str)  # type: ignore


ClientRegistry.register(
    ClientType.OPENAI_CHAT_COMPLETIONS,
    _LazyLoader(
        "veeksha.client.openai_chat",
        "OpenAIChatCompletionsClient",
    ),
)

ClientRegistry.register(
    ClientType.OPENAI_ROUTER,
    _LazyLoader(
        "veeksha.client.openai_router",
        "OpenAIRouterClient",
    ),
)

ClientRegistry.register(
    ClientType.OPENAI_COMPLETIONS,
    _LazyLoader(
        "veeksha.client.openai_completions",
        "OpenAICompletionsClient",
    ),
)

ClientRegistry.register(
    ClientType.TTS,
    _LazyLoader(
        "veeksha.client.tts",
        "TTSClient",
    ),
)
