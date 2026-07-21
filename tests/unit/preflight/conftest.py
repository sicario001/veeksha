"""Make the preflight tests runnable without the (heavy, Rust-built) tokenizers
package: inject a whitespace `transformers.AutoTokenizer` stand-in if the real
one is not importable. Token counts are irrelevant to what these tests check.
"""

import sys
import types

try:  # pragma: no cover - exercised only when transformers is installed
    import transformers  # noqa: F401
except Exception:  # pragma: no cover - the shim path
    _m = types.ModuleType("transformers")

    class _Tok:
        vocab = {"x": 0}
        vocab_size = 1

        def encode(self, text, add_special_tokens=False):
            return list(range(len(str(text).split())))

        def decode(self, ids, skip_special_tokens=False):
            return " ".join("x" for _ in ids)

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _Tok()

    _m.AutoTokenizer = AutoTokenizer
    sys.modules["transformers"] = _m
