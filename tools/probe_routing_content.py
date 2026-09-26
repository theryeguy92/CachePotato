#!/usr/bin/env python3
"""probe_routing_content.py — does the router actually dispatch by CONTENT?

The question: when the model produces fluent-but-off-topic text, is it because
the router "called on the wrong experts" — i.e. it discriminates between inputs
but picked badly? Or because the router barely discriminates at all?

Method. For several deliberately different prompts, collect the per-layer top-2
selections and the full router softmax. Then compare, per layer:

  * cross-prompt divergence — mean total-variation distance between the
    expert-usage histograms of DIFFERENT prompts.
  * within-prompt noise floor — the same statistic between the two halves of the
    SAME prompt. This is what "no content signal" looks like.

If cross ≈ within, the router is not using content to choose experts.
Also reported: how peaked the histogram is at all (uniform = 0.031), and how
much probability mass and margin the top-2 actually has.

Usage:
  python tools/probe_routing_content.py --ckpt weights/moe-scale-run/l05/model_only.pt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_moe as tm  # noqa: E402

PROMPTS = [
    ("question",  "what is the capital of france"),
    ("greeting",  "Hello how are you doing today"),
    ("technical", "The RAID 5 volume must be configured with a"),
    ("narrative", "Once upon a time there was a small girl who"),
    ("code",      "def compute_loss(logits, targets):"),
    ("list",      "Ingredients: two cups of flour, one egg, and"),
]


def run(model, ids):
    saved, savedp = tm.TRACE, tm.TRACE_P
    tm.TRACE, tm.TRACE_P = {}, {}
    with torch.no_grad():
        model(torch.tensor([ids]), None)
    tr, trp = tm.TRACE, tm.TRACE_P
    tm.TRACE, tm.TRACE_P = saved, savedp
    layers = model.cfg.moe_layers()
    sel = np.stack([torch.stack(tr[l]).squeeze(1).numpy() for l in layers])
    probs = np.stack([torch.stack(trp[l]).squeeze(1).numpy() for l in layers])
    return sel, probs                       # (L,T,2), (L,T,N)


def hist(sel_layer, N):
    h = np.bincount(sel_layer.ravel(), minlength=N).astype(float)
    return h / h.sum()


def tvd(a, b):
    return 0.5 * np.abs(a - b).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = tm.Config(**ck["config"])
    model = tm.MoEModel(cfg)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    L, N = len(cfg.moe_layers()), cfg.n_exp

    data = {}
    for name, text in PROMPTS:
        ids = tok.encode(text)
        sel, probs = run(model, ids)
        data[name] = (sel, probs)
        print("  %-10s %2d tokens  %s" % (name, len(ids), text[:44]))

    print("\nper-layer: is cross-prompt divergence above the within-prompt floor?")
    print("  layer | hist max (unif .031) | top-2 mass | top1-top3 margin |"
          " cross-TVD | within-TVD | ratio")
    names = [n for n, _ in PROMPTS]
    for li, l in enumerate(cfg.moe_layers()):
        hs = [hist(data[n][0][li], N) for n in names]
        cross = np.mean([tvd(hs[i], hs[j])
                         for i in range(len(hs)) for j in range(i + 1, len(hs))])
        withins = []
        for n in names:
            s = data[n][0][li]
            if s.shape[0] < 4:
                continue
            mid = s.shape[0] // 2
            withins.append(tvd(hist(s[:mid], N), hist(s[mid:], N)))
        within = float(np.mean(withins))

        P = np.concatenate([data[n][1][li] for n in names], axis=0)   # (sumT,N)
        sp = np.sort(P, axis=-1)
        mass = float(np.mean(sp[:, -1] + sp[:, -2]))
        margin = float(np.mean(sp[:, -1] - sp[:, -3]))
        maxshare = float(np.mean([h.max() for h in hs]))
        ratio = cross / within if within > 1e-9 else float("inf")
        print("   %4d | %19.3f | %10.3f | %16.4f | %9.4f | %10.4f | %5.2fx"
              % (l, maxshare, mass, margin, cross, within, ratio))

    print("""
reading: a router that dispatches by content shows cross-TVD well above the
within-prompt floor (ratio >> 1) and a top-2 mass well above 2/N (= %.3f).
A ratio near 1 means the expert choice is not tracking the input.""" % (2 / N))


if __name__ == "__main__":
    main()
