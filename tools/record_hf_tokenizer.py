#!/usr/bin/env python3
"""record_hf_tokenizer.py — record the exact token ids for a corpus, as the tokenizer's target.

The engine's tokenizer is GPT-2 byte-level BPE. Qwen2's is different in three ways that all
matter:
  1. its pre-tokenizer is a UNICODE regex (`\p{L}`, `\p{N}`), so it needs category tables, not
     the byte classes GPT-2's adapted regex uses;
  2. it applies NFC normalization first;
  3. it has 22 added/special tokens (`<|im_start|>`, `<|im_end|>`, ...) that must not be split.

A tokenizer that is close but not exact does not fail loudly. It changes the token ids, and the
model then answers the wrong question fluently. So the target is recorded here, exactly, and the
C implementation must reproduce these ids with ZERO mismatches — the same rule as the logits.

The corpus deliberately includes the cases that break naive implementations: contractions,
leading/trailing whitespace, runs of newlines, tabs, digits, punctuation clusters, accented
Latin, CJK, Cyrillic and emoji (multi-byte, and some outside the BMP).
"""
import json
from pathlib import Path
import numpy as np

TEXTS = [
    "The capital of France is",
    "Hello, world!",
    "don't won't it's I'd we'll they've",
    "  leading and trailing whitespace  ",
    "line one\nline two\n\n\nline four",
    "tabs\tand\tspaces",
    "1234567890 and 3.14159 and -42",
    "punctuation: ;.,!?()[]{}<>@#$%^&*_+=|\\/\"'`~",
    "café naïve résumé über",
    "日本語のテキストです",
    "Привет, мир",
    "emoji: 😀🎉🚀 and flags 🇫🇷🇩🇪",
    "mixed: 42 apples, 3.5 kg — done!",
    "def foo(x): return x + 1  # comment",
    "<|im_start|>user\nhi<|im_end|>",
    "",
]

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("weights/qwen2.5-0.5b-instruct")
    out = {}
    lens = []
    for i, text in enumerate(TEXTS):
        ids = tok(text, add_special_tokens=False).input_ids
        out[f"text{i}"] = np.array(list(text.encode("utf-8")), dtype=np.uint8)
        out[f"ids{i}"] = np.array(ids, dtype=np.int64)
        lens.append((i, len(ids)))
        print(f"  {i:2} {len(ids):3} ids  {text[:44]!r}")
    chat = tok.apply_chat_template([{"role": "user", "content": TEXTS[0]}], tokenize=True,
                                   add_generation_prompt=True)
    # transformers 5.x returns a BatchEncoding, not a list; taking it as a list of ids failed
    # with "invalid literal for int(): 'input_ids'".
    if isinstance(chat, dict) or hasattr(chat, "keys"):
        chat = chat["input_ids"]
    out["chat_ids"] = np.array(chat, dtype=np.int64)
    out["n_texts"] = np.array([len(TEXTS)], dtype=np.int64)
    out["meta"] = np.array(json.dumps({
        "tokenizer": type(tok).__name__, "vocab": len(tok),
        "pre_tokenizer": "unicode-regex + NFC (see tokenizer.json)"}))
    p = Path("data/traces/qwen25_05b_tokenizer.npz")
    np.savez_compressed(p, **out)
    print(f"\nchat template -> {len(chat)} ids, starts {chat[:3]}")
    print(f"wrote {p} ({p.stat().st_size/1e3:.0f} KB)")
    print("  A C tokenizer must reproduce these ids EXACTLY — zero mismatches.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
