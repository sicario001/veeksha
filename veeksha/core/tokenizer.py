from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from transformers import AutoTokenizer

from veeksha.types import ChannelModality

RawContent = TypeVar("RawContent")
TokenIds = Sequence[int]

TokenCounter = Callable[[RawContent], int]
TokenDecoder = Callable[[TokenIds], RawContent]
TokenEncoder = Callable[[RawContent], TokenIds]


@dataclass
class TokenizerHandle(Generic[RawContent]):
    """Minimal tokenizer abstraction used by channel generators."""

    count_tokens: TokenCounter
    decode: TokenDecoder
    encode: TokenEncoder
    get_vocab: Optional[Callable[[], List[int]]] = None


class TokenizerProvider:
    """Lightweight provider that returns a tokenizer handle per modality."""

    def __init__(
        self,
        tokenizers: Dict[ChannelModality, TokenizerHandle[Any]],
        model_name: Optional[str] = None,
    ):
        self._tokenizers = tokenizers
        self._model_name = model_name

    def for_modality(self, modality: ChannelModality) -> TokenizerHandle[Any]:
        return self._tokenizers[modality]

    @property
    def model_name(self) -> Optional[str]:
        """Return the model name for loading raw tokenizers."""
        return self._model_name


def build_hf_tokenizer_handle(tokenizer) -> TokenizerHandle[str]:
    """Wrap a Hugging Face tokenizer into a TokenizerHandle."""

    # cache vocab
    vocab = sorted(tokenizer.vocab.values())[: tokenizer.vocab_size]

    return TokenizerHandle(
        count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        decode=lambda token_ids: tokenizer.decode(token_ids, skip_special_tokens=False),
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
        get_vocab=lambda: vocab,
    )


def build_hf_tokenizer_handle_from_model(model: str) -> TokenizerHandle[str]:
    """Instantiate a Hugging Face tokenizer from a model name and wrap it."""

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return build_hf_tokenizer_handle(tokenizer)


def build_word_split_tokenizer_provider(model_name: str) -> "TokenizerProvider":
    """Build a TokenizerProvider backed by a simple whitespace word-split tokenizer.

    Used by TTS/STT-style clients whose models do not ship a HuggingFace
    tokenizer; tokens are approximated as whitespace-delimited words. Needs no
    model download and no ``transformers`` at call time.
    """
    handle: TokenizerHandle[str] = TokenizerHandle(
        count_tokens=lambda text: len(text.split()),
        decode=lambda ids: " ".join(str(i) for i in ids),
        encode=lambda text: list(range(len(text.split()))),
    )
    return TokenizerProvider(
        {ChannelModality.TEXT: handle},
        model_name=model_name,
    )


def gen_prompt_from_corpus(
    num_tokens: int,
    pretokenized_lines: Iterable[Sequence[int]],
    tokenizer_handle: TokenizerHandle[RawContent],
    rng,
    suffix: Optional[RawContent] = None,
) -> RawContent:
    """Assemble ~num_tokens tokens from pre-tokenized corpus via a single decode.

    Takes exactly num_tokens token IDs from the (shuffled, tiled) corpus and
    decodes them once.  The resulting text will re-tokenize to approximately
    num_tokens tokens (within ±a few at chunk boundaries), which is accurate
    enough for throughput benchmarks.

    The previous implementation binary-searched over decode+encode round-trips
    to hit an exact count.  With the Sarvam tokenizer that cost ~38ms per
    step × 16 steps × 1172 sessions = 12 minutes of stall before the
    benchmark started for 28k-token inputs.
    """
    empty_content = tokenizer_handle.decode([])
    effective_suffix = suffix if suffix is not None else empty_content

    if num_tokens <= 0:
        return empty_content

    token_lines = [line for line in pretokenized_lines if line]
    if not token_lines:
        return effective_suffix

    rng.shuffle(token_lines)
    base_tokens = [tok for line in token_lines for tok in line]

    if len(base_tokens) < num_tokens and base_tokens:
        repeats = (num_tokens // len(base_tokens)) + 2
        body_ids = (base_tokens * repeats)[:num_tokens]
    else:
        body_ids = base_tokens[:num_tokens]

    return tokenizer_handle.decode(body_ids) + effective_suffix
