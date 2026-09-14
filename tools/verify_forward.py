"""verify_forward.py — Phase 1 milestone gate: C engine vs. HuggingFace GPT-2.

Tokenizes a prompt with GPT-2's tokenizer, runs HF's model (fp32, CPU) for
reference logits, runs ./build/cachepotato on the same token ids, and
compares. PASS = max abs diff under tolerance AND same argmax next token.

Run from repo root:  python3 tools/verify_forward.py [prompt]
"""

import os
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
TOL = 5e-2  # fp32 CPU vs naive-C accumulation order; argmax is the real check

prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
ids = tok.encode(prompt)
print(f"prompt: {prompt!r} -> {len(ids)} tokens: {ids}")

with torch.no_grad():
    ref = model(torch.tensor([ids])).logits[0].numpy()  # (T, V)

out = subprocess.run(["./build/cachepotato", "weights/gpt2.bin"] + [str(i) for i in ids],
                     capture_output=True, text=True, cwd=REPO).stdout
block = out.split("logits:\n")[1].strip().split("\n")
got = np.array([np.fromstring(line, sep=" ") for line in block[:len(ids)]])

diff = np.abs(got - ref).max()
same_argmax = got[-1].argmax() == ref[-1].argmax()
print(f"max abs diff vs HF: {diff:.2e}  (tol {TOL:.0e})")
print(f"next-token argmax — HF: {ref[-1].argmax()} ({tok.decode([ref[-1].argmax()])!r})"
      f" | ours: {got[-1].argmax()} ({tok.decode([got[-1].argmax()])!r})"
      f" | match: {same_argmax}")

if diff < TOL and same_argmax:
    print("PASS — C engine reproduces HF GPT-2 logits")
    sys.exit(0)
print("FAIL")
sys.exit(1)
