"""verify_tokenizer.py — Phase 2 gate: round-trip text->tokens->text,
plus an HF-id comparison (reported, not hard-gated — our chunker is a
simplified version of GPT-2's regex).

Run from repo root:  python3 tools/verify_tokenizer.py
"""

import os
import subprocess
import sys

from transformers import GPT2TokenizerFast

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

PROMPTS = [
    "The capital of France is",
    "Hello, world!",
    "I don't think it's 100% right, but we'll see.",
    "In 1969, Apollo 11 landed on the Moon.",
    "def f(x): return x * 2",
]


def main():
    hf = GPT2TokenizerFast.from_pretrained("gpt2")
    all_ok = True
    for prompt in PROMPTS:
        out = subprocess.run(["./build/cachepotato", "weights/gpt2.bin", "-p", prompt],
                             capture_output=True, text=True, cwd=REPO).stdout
        lines = {k: v for k, v in
                 (l.split(": ", 1) for l in out.splitlines() if ": " in l)}
        decoded = lines.get("decoded", "")[1:-1]  # strip surrounding quotes
        ours = [int(x) for x in lines.get("tokens", "").split()]
        ref = hf.encode(prompt)

        rt_ok = decoded == prompt
        ids_ok = ours == ref
        all_ok &= rt_ok
        print(f"{prompt!r}")
        print(f"  round-trip: {'OK' if rt_ok else f'MISMATCH -> {decoded!r}'}")
        print(f"  ids vs HF:  {'match' if ids_ok else f'diverge (ours {ours} vs hf {ref})'}")
    print("PASS — round-trip works" if all_ok else "FAIL — round-trip broken")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
