#!/usr/bin/env python3
"""verify_cpw1_v3.py — acceptance gate for a CPW1 v3 export.

Three checks, strongest last:

  A. STRUCTURE  header fields match the config, and the file size is exactly what
                the config implies (the loud-failure guarantee the format exists
                for — a layout or version skew cannot pass silently).
  B. EXACT      the model rebuilt FROM THE FILE reproduces the checkpoint's
                logits, loss and per-token top-2 routing bit-for-bit, on a fixed
                fixture that contains an EOS (so the locality reset is exercised).
  C. TRACES     the model rebuilt from the file reproduces the ROUTING RECORDED
                from the trained model on held-out corpus windows.

Check C is the one that matters: it is the gate that says the exported bytes
carry the mechanism, not just plausible weights.

Usage:
  python tools/verify_cpw1_v3.py --ckpt weights/moe-scale-run/l05/model_only.pt \
      --cpw weights/moe-scale-run/l05.cpw1v3 \
      --trace data/traces/l05_fp32.npz --windows 2
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import cpw1_v3 as v3      # noqa: E402
import train_moe as tm    # noqa: E402

# Same fixture as the C bridge: real corpus text, EOS at index 8.
FIXTURE = np.array([262, 2071, 318, 17764, 351, 6868, 18366, 737, 50256, 14906,
                    314, 25, 1041, 3145, 896, 2422, 3842, 11, 355, 564, 250,
                    13217, 283, 28914], dtype=np.int64)


def selections(model, tokens):
    """Run a window and return (n_moe, T, K) top-2 selections via the TRACE hook."""
    saved, tm.TRACE = tm.TRACE, {}
    with torch.no_grad():
        model(torch.from_numpy(tokens)[None, :], None)
    trace = tm.TRACE
    tm.TRACE = saved
    return np.stack([torch.stack(trace[l]).squeeze(1).numpy()
                     for l in model.cfg.moe_layers()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cpw", required=True)
    ap.add_argument("--trace", default="")
    ap.add_argument("--windows", type=int, default=2)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = tm.Config(**ck["config"])
    ref = tm.MoEModel(cfg)
    ref.load_state_dict(ck["model"], strict=True)
    ref.eval()

    print("=" * 78)
    print("CPW1 v3 verification — %s" % args.cpw)
    print("=" * 78)
    fails = []

    # ---- A. structure
    hdr = v3.read_header(args.cpw)
    exp_ints = dict(zip(v3.HDR_INT_FIELDS, v3.header_ints(cfg)))
    hdr_ok = all(hdr[k] == exp_ints[k] for k in v3.HDR_INT_FIELDS)
    live = v3.payload_floats(cfg)
    exp_size = v3.HDR_SIZE + live * 4
    got_size = os.path.getsize(args.cpw)
    print("A. STRUCTURE")
    print("   header     : %s" % ("matches config" if hdr_ok else "MISMATCH"))
    if not hdr_ok:
        for k in v3.HDR_INT_FIELDS:
            if hdr[k] != exp_ints[k]:
                print("     %-14s file=%s config=%s" % (k, hdr[k], exp_ints[k]))
        fails.append("header")
    print("   size       : %s bytes (%s)" % (f"{got_size:,}",
          "exact" if got_size == exp_size else "MISMATCH, expected %s" % f"{exp_size:,}"))
    if got_size != exp_size:
        fails.append("size")
    print("   live params: %s   dead stripped: %s"
          % (f"{live:,}", f"{sum(p.numel() for p in ref.parameters()) - live:,}"))

    # ---- B. exact round-trip
    _, arrays = v3.read_arrays(args.cpw, cfg)
    rebuilt = tm.MoEModel(cfg)
    rebuilt.load_state_dict(v3.arrays_to_state(arrays, cfg, rebuilt), strict=True)
    rebuilt.eval()

    toks = torch.from_numpy(FIXTURE)
    inp, tgt = toks[:-1].unsqueeze(0), toks[1:].unsqueeze(0)
    with torch.no_grad():
        l_ref, loss_ref, _ = ref(inp, tgt)
        l_new, loss_new, _ = rebuilt(inp, tgt)
    dlog = (l_ref - l_new).abs().max().item()
    dloss = abs(loss_ref.item() - loss_new.item())
    sel_ref = selections(ref, FIXTURE[:-1])
    sel_new = selections(rebuilt, FIXTURE[:-1])
    sel_eq = bool((sel_ref == sel_new).all())
    print("\nB. EXACT ROUND-TRIP (fixture: %d tokens, EOS@8)" % len(FIXTURE))
    print("   logits     : max|Δ| %.3e" % dlog)
    print("   loss       : %.6f vs %.6f (Δ %.3e)" % (loss_ref.item(), loss_new.item(), dloss))
    print("   routing    : %s" % ("IDENTICAL" if sel_eq else
          "%d MISMATCHES" % int((sel_ref != sel_new).sum())))
    if dlog != 0.0 or sel_eq is False:
        fails.append("exact-roundtrip")

    # ---- C. against the recorded traces
    if args.trace:
        z = np.load(args.trace, allow_pickle=False)
        toks_all, sel_all = z["tokens"], z["sel"]
        n = min(args.windows, toks_all.shape[0])
        tot = mism = 0
        for w in range(n):
            got = selections(rebuilt, toks_all[w].astype(np.int64))
            exp = sel_all[:, w, :, :]
            mism += int((got != exp).sum())
            tot += exp.size
        rate = 100.0 * (1 - mism / tot)
        print("\nC. AGAINST RECORDED TRACES (%d windows x %d tokens = %s positions)"
              % (n, toks_all.shape[1], f"{n * toks_all.shape[1]:,}"))
        print("   recorded λ  : %.2f   agreement: %.4f%%  (%d/%d selections differ)"
              % (float(z["lam"]), rate, mism, tot))
        if rate < 99.9:
            fails.append("traces %.2f%%" % rate)

    print("\n" + "=" * 78)
    if fails:
        print("CPW1 v3 GATE: FAIL — %s" % ", ".join(fails))
        return 1
    print("CPW1 v3 GATE: PASS (structure + exact round-trip%s)"
          % (" + recorded traces" if args.trace else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
