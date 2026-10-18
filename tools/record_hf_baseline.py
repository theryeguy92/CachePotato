#!/usr/bin/env python3
"""record_hf_baseline.py — record a FIXED numeric target from HuggingFace transformers.

Why record instead of re-running: the port's reference and the engine must both be checked
against the same numbers. If each run recomputes them, a change in the library, the dtype or
the thread count silently moves the target, and a port that drifts passes. So the numbers are
captured once, with provenance, and everything downstream compares against the file.

What is recorded (all fp32):
  input_ids          the exact token sequence
  logits             (T, vocab) full logits — the primary target
  greedy_ids         a 16-token greedy continuation from those logits (end-to-end target)
  top_ids/top_vals   top-32 per position, so a reader can eyeball agreement without numpy
  meta               model, revision, dtype, torch version, thread count

Two prompts are recorded: raw text (no template) and the chat-template form. The raw one is the
numeric target (no template dependency); the chat one exists because the REPL must eventually
produce it, and it is better to have the target before writing the code that consumes it.
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

MODEL = "weights/qwen2.5-0.5b-instruct"
OUT = "data/traces/qwen25_05b_hf.npz"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=32, help="truncate the prompt to this many")
    ap.add_argument("--greedy", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    chat = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)

    rec = {}
    with torch.no_grad():
        for tag, text in (("raw", args.prompt), ("chat", chat)):
            ids = tok(text, return_tensors="pt").input_ids
            # Truncate ONLY the raw case. The chat prompt is a template that ends in a
            # generation marker; cutting it mid-template records a target for a prompt nobody
            # would send (the first recording did exactly that and "continued" a truncated
            # user turn).
            if tag == "raw":
                ids = ids[:, :args.tokens]
            out = model(ids)
            logits = out.logits[0].float().numpy()
            # greedy continuation from the recorded logits
            gen = ids.clone()
            for _ in range(args.greedy):
                nxt = int(logits[-1].argmax())
                gen = torch.cat([gen, torch.tensor([[nxt]])], dim=1)
                logits = model(gen).logits[0, -1:].float().numpy()
            top = np.argsort(-logits[-1])[:32]
            rec[f"{tag}_ids"] = ids[0].numpy().astype(np.int64)
            rec[f"{tag}_logits"] = out.logits[0].float().numpy()
            rec[f"{tag}_greedy"] = gen[0].numpy().astype(np.int64)
            rec[f"{tag}_top_ids"] = top.astype(np.int64)
            rec[f"{tag}_top_vals"] = logits[-1][top].astype(np.float32)
            print(f"{tag:5} ids {ids.shape[1]:3} tokens | next argmax {int(logits[-1].argmax())} "
                  f"| greedy {gen.shape[1] - ids.shape[1]} tokens")
            print(f"      decode: {tok.decode(gen[0][ids.shape[1]:].tolist())!r}"[:200])

    rec["meta"] = np.array(json.dumps({
        "model": args.model, "dtype": "float32", "threads": args.threads,
        "torch": torch.__version__,
        "weights_sha256_16": sha256_file(Path(args.model) / "model.safetensors"),
        "config_sha256_16": sha256_file(Path(args.model) / "config.json"),
    }))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **rec)
    sz = Path(args.out).stat().st_size
    print(f"\nwrote {args.out} ({sz/1e6:.1f} MB)")
    print(f"  weights sha256[0:16] {sha256_file(Path(args.model) / 'model.safetensors')}")
    print("  This file IS the target. Do not regenerate it to make a failing test pass.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
