#!/usr/bin/env python3
"""record_hf_intermediates.py — capture HF's per-layer activations, so each port block can be
verified on its own.

WHY THIS IS WORTH A SECOND FILE
  A 24-layer transformer with four new blocks (RMSNorm, RoPE, GQA, SwiGLU) is an integration
  problem. Implement all four, diff the final logits, and a wrong RoPE appears as "something is
  off, 24 layers downstream" — indistinguishable from a wrong GQA or a wrong SwiGLU. Capture
  HF's own intermediates instead, and each block gets its own target: a failure names the block.

  Cost is a few MB. The alternative is what this project has already paid for three times: a
  silent bug that produced a model which ran and was wrong.

WHAT IS CAPTURED (per layer i), all fp32:
  norm1_out     input_layernorm output          -> gates RMSNorm
  q_proj/k_proj/v_proj                          -> gates the projections and GQA head split
  cos/sin       rotary_emb output               -> gates RoPE's frequencies and positions
  attn_out      self_attn output                -> gates the whole attention block
  norm2_out     post_attention_layernorm output -> gates RMSNorm again on the 2nd site
  mlp_out       mlp output                      -> gates SwiGLU
plus embed_out and final_norm_out.

The hooks are looked up by module NAME and their absence is reported rather than silently
skipped: a missing target must fail loudly, not quietly shrink the gate set.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/qwen2.5-0.5b-instruct")
    ap.add_argument("--out", default="data/traces/qwen25_05b_intermediates.npz")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids
    T = ids.shape[1]

    rec, missing, seen = {}, [], []
    base = model.model
    def hook(name):
        def fn(mod, inp, out):
            # Do NOT collapse tuples blindly: rotary_emb returns (cos, sin) and taking out[0]
            # silently records only cos, which would under-verify RoPE by half.
            if isinstance(out, (tuple, list)):
                for j, o in enumerate(out):
                    if torch.is_tensor(o) and o.dtype == torch.float32:
                        rec[f"{name}.{j}"] = o.detach().float().numpy().copy()
                return
            if torch.is_tensor(out):
                rec[name] = out.detach().float().numpy().copy()
        return fn

    targets = {}
    for i in range(len(base.layers)):
        # NOTE: rotary_emb is NOT per-attention in transformers 5.x — it is shared at
        # `model.rotary_emb`, so it is hooked once below rather than per layer.
        for sub in ("input_layernorm", "self_attn.q_proj", "self_attn.k_proj",
                    "self_attn.v_proj", "self_attn", "post_attention_layernorm", "mlp"):
            targets[f"L{i}.{sub}"] = f"model.layers.{i}.{sub}"
    targets["norm"] = "model.norm"
    targets["embed"] = "model.embed_tokens"
    targets["rotary"] = "model.rotary_emb"

    handles = []
    for label, path in targets.items():
        mod = model          # full paths start with "model."
        try:
            for part in path.split("."):
                mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
        except AttributeError:
            missing.append(label); continue
        handles.append(mod.register_forward_hook(hook(label)))
        seen.append(label)

    with torch.no_grad():
        out = model(ids)
    for h in handles:
        h.remove()

    print(f"prompt {T} tokens, {len(seen)} hook sites, {len(missing)} missing")
    if missing:
        print("MISSING (gates that would silently not run):", missing[:8])
    for k in sorted(rec)[:6]:
        print(f"   {k:22} {rec[k].shape}")
    rec["input_ids"] = ids[0].numpy().astype(np.int64)
    rec["logits"] = out.logits[0].float().numpy()
    rec["meta"] = np.array(json.dumps({
        "model": args.model, "dtype": "float32", "threads": args.threads,
        "torch": torch.__version__, "sites": len(seen), "missing": missing}))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **rec)
    print(f"wrote {args.out} ({(Path(args.out).stat().st_size)/1e6:.1f} MB, {len(rec)} arrays)")
    return 1 if missing else 0

if __name__ == "__main__":
    raise SystemExit(main())
