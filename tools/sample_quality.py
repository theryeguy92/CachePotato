#!/usr/bin/env python3
"""sample_quality.py — Q2: is the λ=0.5 model usable under sampled decoding?

Protocol is pre-registered in the vault (Phase-7-Integration § Q2). This script
generates and reports raw metrics; the PASS/FAIL rule is applied separately.

Saves prompt + generated token ids to JSON so decoding/eyeballing can happen
wherever the GPT-2 tokenizer lives.

Usage (from repo root, on the training server):
  python tools/sample_quality.py --ckpt ckpt/l05/model_only.pt \
      --data data/fineweb_edu_valid.bin --out q2/l05.json \
      --n-prompts 24 --prompt-len 48 --max-new 128 --device cuda:0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_moe as tm  # noqa: E402


def extract_prompts(ids, eos, n, plen):
    """Natural document starts: plen tokens following an EOS, from documents
    long enough that the prefix is real text rather than a fragment."""
    pos = np.nonzero(ids == eos)[0]
    out = []
    for p in pos:
        if p + plen + 2 >= len(ids):
            break
        seg = ids[p + 1:p + 1 + plen]
        if len(seg) < plen:
            continue
        nxt = np.nonzero(ids[p + 1:p + 1 + plen] == eos)[0]
        if len(nxt):
            continue                      # document shorter than the prefix
        out.append(seg.astype(np.int64))
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def generate(model, prompt, max_new, temperature, top_p, device, eos, n_ctx):
    ids = torch.from_numpy(prompt)[None, :].to(device)
    gen = []
    for _ in range(max_new):
        logits, _, _ = model(ids[:, -n_ctx:], None)
        logits = logits[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            keep = (cum - sp) < top_p
            sp, si = sp[keep], si[keep]
            sp = sp / sp.sum()
            nxt = int(si[torch.multinomial(sp, 1)])
        if nxt == eos:
            break
        gen.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    return gen


def metrics(gen):
    if len(gen) < 8:
        return {"n": len(gen), "distinct4": None, "rep": None, "uniq": None}
    g = np.asarray(gen)
    grams = [tuple(g[i:i + 4]) for i in range(len(g) - 3)]
    d4 = len(set(grams)) / len(grams)
    return {"n": int(len(g)), "distinct4": d4, "rep": 1.0 - d4,
            "uniq": len(set(gen)) / len(g)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--prompt-len", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = tm.Config(**ck["config"])
    model = tm.MoEModel(cfg)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(args.device)
    print("loaded %s step=%s lam=%s layout MoE %s  device=%s"
          % (args.ckpt, ck.get("step"), cfg.lam, cfg.moe_layers(), args.device),
          flush=True)

    ids = np.memmap(args.data, dtype=np.uint16, mode="r")
    prompts = extract_prompts(ids, tm.EOS, args.n_prompts, args.prompt_len)
    print("prompts: %d x %d tokens (after EOS, full-document prefixes)"
          % (len(prompts), args.prompt_len), flush=True)

    torch.manual_seed(args.seed)
    result = {"ckpt": args.ckpt, "lam": cfg.lam, "step": int(ck.get("step", -1)),
              "prompt_len": args.prompt_len, "max_new": args.max_new,
              "temperature": args.temperature, "top_p": args.top_p,
              "seed": args.seed, "modes": {}}

    for mode, temp in (("sampled", args.temperature), ("greedy", 0.0)):
        t0, gens, ms = time.time(), [], []
        for i, p in enumerate(prompts):
            g = generate(model, p, args.max_new, temp, args.top_p,
                         args.device, tm.EOS, cfg.n_ctx)
            gens.append(g)
            ms.append(metrics(g))
            if (i + 1) % 4 == 0:
                print("  %s %d/%d  %.1fs" % (mode, i + 1, len(prompts),
                                             time.time() - t0), flush=True)
        agg = {k: float(np.nanmean([m[k] for m in ms if m[k] is not None]))
               for k in ("distinct4", "rep", "uniq")}
        result["modes"][mode] = {"prompts": [p.tolist() for p in prompts],
                                 "generations": gens, "per_prompt": ms,
                                 "aggregate": agg}
        print("  %s aggregate: distinct4=%.4f rep=%.4f uniq=%.4f (%.0fs)"
              % (mode, agg["distinct4"], agg["rep"], agg["uniq"], time.time() - t0),
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    print("wrote %s (%.1f KB)" % (out, out.stat().st_size / 1e3))


if __name__ == "__main__":
    main()
