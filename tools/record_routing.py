#!/usr/bin/env python3
"""record_routing.py — record per-token top-2 expert selections (Phase C).

Runs a trained CachePotato-MoE checkpoint over held-out token windows and dumps
the router's selections plus the token ids, so the locality mechanism can be
measured offline (temporal reuse rate, transition matrices, LRU replay) without
needing the C engine or a CPW1 v3 export.

The routing it records is the *inference-time* routing: the λ locality bias is
active and the previous-token locality set resets at window start and at EOS,
because it goes through the same moe_ffn used in training (instrumented by the
opt-in TRACE hook, which is verified bit-identical by tools/_trace_equiv.py).

Usage (from repo root):
  python tools/record_routing.py --ckpt ckpt/l05/model_only.pt \
      --data data/fineweb_edu_valid.bin --out traces/l05_fp32.npz \
      --tokens 500000 --precision fp32
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_moe as tm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=500_000)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--start", type=int, default=0, help="token offset into the bin")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = tm.Config(**ck["config"])
    model = tm.MoEModel(cfg)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(args.device)
    print("loaded %s  step=%s  lam=%s  device=%s  precision=%s"
          % (args.ckpt, ck["step"], cfg.lam, args.device, args.precision), flush=True)

    ids = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_win = args.tokens // args.T
    moe_layers = [l for l in range(cfg.n_layer) if 0 < l < cfg.n_layer - 1]
    K = cfg.topk

    toks = np.zeros((n_win, args.T), dtype=np.uint16)
    sel = np.zeros((len(moe_layers), n_win, args.T, K), dtype=np.int16)

    amp = (args.precision == "bf16" and args.device == "cuda")
    t0 = time.time()
    with torch.no_grad():
        for w in range(n_win):
            s = args.start + w * args.T
            win = ids[s:s + args.T].astype(np.int64)
            if win.shape[0] != args.T:
                n_win = w
                toks = toks[:n_win]
                sel = sel[:, :n_win]
                break
            toks[w] = win
            x = torch.from_numpy(win)[None, :].to(args.device)
            tm.TRACE = {}
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                model(x, None)
            for li, l in enumerate(moe_layers):
                sel[li, w] = torch.stack(tm.TRACE[l]).squeeze(1).numpy()
            tm.TRACE = None
            if (w + 1) % 100 == 0 or w + 1 == n_win:
                el = time.time() - t0
                print("  window %d/%d  %.0f tok/s"
                      % (w + 1, n_win, args.tokens * (w + 1) / n_win / el), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tokens=toks,
        sel=sel,
        moe_layers=np.array(moe_layers, dtype=np.int16),
        lam=np.float32(cfg.lam),
        topk=np.int16(K),
        T=np.int32(args.T),
        precision=np.array(args.precision),
        step=np.int64(ck["step"]),
        start=np.int64(args.start),
        eos=np.int64(tm.EOS),
    )
    print("wrote %s (%.1f MB)  %d windows x %d tokens = %s tokens, %d MoE layers"
          % (out, out.stat().st_size / 1e6, n_win, args.T, f"{n_win * args.T:,}",
             len(moe_layers)))


if __name__ == "__main__":
    main()
