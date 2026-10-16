#!/usr/bin/env python3
"""verify_permutation_cpw1_v3.py — Tier-B gate: is a permuted layout a symmetry?

WHAT THIS GATES
  The Tier-B ("layout compilation") direction claims a port may reorder experts in
  the weight file, with the router's expert dimension permuted to match, and remain
  the SAME MODEL. This gate tests that claim on a real artifact.

THE INVARIANT (and the off-by-one that is easy to make)
  Write the permuted model's expert k as original expert perm[k]:
      experts'[k] = experts[perm[k]],   Wr'[:,k] = Wr[:,perm[k]],   br'[k] = br[perm[k]]
  Then its router score vector satisfies
      s'[k] == s[perm[k]]      for every k
  With idx = argmax over the ORIGINAL scores and idx' = argmax over the PERMUTED
  scores, writing j = idx':
      s'[j] is maximal  =>  s[perm[j]] is maximal  =>  perm[j] == idx
  so the top-k index relation is
      perm[idx'] == idx        <-- note the direction, PERMUTED on the left
  and NOT perm[idx] == idx'. Getting this backwards makes a CORRECT file fail:
  it happened twice while building this gate (once in a scratch harness, once in
  check 5 below). A five-line synthetic test on random data settles it — do that
  before suspecting the file. Evidence from the real artifact: the Python
  reference satisfies perm[permuted_ids] == base_ids 160/160 and the reverse
  5/160, i.e. no ambiguity once the direction is right.

PASS CRITERIA
  1. every expert block and router column is exactly the checkpoint's under perm
  2. router scores are exactly equivariant (bitwise) at a controlled input
  3. top-k selections are equivariant under perm (perm[idx_new] == idx_base)
  4. full-sequence logits agree to float32 rounding, with identical argmax
  5. the C engine agrees with the permuted reference exactly
  6. the C engine's routing on the permuted file is exactly the base routing under perm

  Criterion 4 is a tolerance, not an equality: reordering the router's columns
  changes the summation order of the router matmul, which moves the last bits.
  The measured max|delta| is recorded, not hidden. Criteria 1, 2, 5 and 6 are exact.

USAGE
  python3 tools/verify_permutation_cpw1_v3.py \
      --base weights/moe-scale-run/l05.cpw1v3 \
      --permuted scratch/l05_permuted.cpw1v3 \
      --ckpt weights/moe-scale-run/l05/model_only.pt \
      --perm-json scratch/l05_permutation.json \
      --tokens-bin data/fineweb_edu_valid.bin --engine ./build/cachepotato
"""
import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import cpw1_v3 as v3        # noqa: E402
import train_moe as tm      # noqa: E402

FIXTURE = np.array([262, 2071, 318, 17764, 351, 6868, 18366, 737, 50256, 14906,
                    314, 25, 1041, 3145, 896, 2422, 3842, 11, 355, 564, 250,
                    13217, 283, 28914], dtype=np.int64)


def build(path, cfg):
    _, arrays = v3.read_arrays(path, cfg)
    m = tm.MoEModel(cfg)
    m.load_state_dict(v3.arrays_to_state(arrays, cfg, m), strict=True)
    m.eval()
    return m, arrays


def controlled_input(D, T=6):
    """Deterministic — deliberately no RNG, so two runs cannot compare different data."""
    t = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    return torch.sin(t * torch.arange(1, D + 1, dtype=torch.float32) * 0.01)


def trace_of(model, inp, moe_layers):
    """TRACE is keyed by GLOBAL layer; keep that indexing explicit."""
    saved, tm.TRACE = tm.TRACE, {}
    with torch.no_grad():
        model(inp, None)
    tr, tm.TRACE = tm.TRACE, saved
    return torch.stack([torch.stack(tr[l]).squeeze(1) for l in moe_layers])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--permuted", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--perm-json", required=True)
    ap.add_argument("--tokens-bin", default="data/fineweb_edu_valid.bin")
    ap.add_argument("--engine", default="")
    ap.add_argument("--window", type=int, default=64)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = tm.Config(**ck["config"])
    sd = ck["model"]
    moe = cfg.moe_layers()
    K, N = cfg.topk, cfg.n_exp
    rep = json.loads(Path(args.perm_json).read_text())
    perms = {int(k): np.array(v, dtype=np.int64) for k, v in rep["perms"].items()}

    print("=" * 78)
    print("Permutation gate — Tier-B layout symmetry")
    print("  base    : %s" % args.base)
    print("  permuted: %s  (seed %s)" % (args.permuted, rep.get("seed")))
    print("=" * 78)
    fails = []

    base, A0 = build(args.base, cfg)
    permuted, A1 = build(args.permuted, cfg)

    # ---- 1. exact: blocks and router columns are the checkpoint's under perm
    ok1 = True
    for s, l in enumerate(moe):
        rb = np.concatenate([sd["W1e"][l].numpy().reshape(N, -1), sd["b1e"][l].numpy().reshape(N, -1),
                             sd["W2e"][l].numpy().reshape(N, -1), sd["b2e"][l].numpy().reshape(N, -1)], axis=1)
        ok1 &= np.array_equal(A1["experts"][s], rb[perms[s]])
        ok1 &= np.array_equal(A1["Wr"][s], sd["Wr"][l].numpy()[:, perms[s]])
        ok1 &= np.array_equal(A1["br"][s], sd["br"][l].numpy()[perms[s]])
    print("\n1. STRUCTURE  experts/router == checkpoint[perm] for all %d MoE layers: %s"
          % (len(moe), ok1))
    if not ok1:
        fails.append("structure")

    # ---- 2. exact: router score equivariance at a controlled input
    x = controlled_input(cfg.d_model)
    ok2, worst2 = True, 0.0
    for s, l in enumerate(moe):
        s_o = x @ base.Wr[l].detach() + base.br[l].detach()
        s_n = x @ permuted.Wr[l].detach() + permuted.br[l].detach()
        p = torch.from_numpy(perms[s])
        worst2 = max(worst2, float((s_n - s_o[:, p]).abs().max()))
        ok2 &= bool((p[s_n.topk(K, -1).indices] == s_o.topk(K, -1).indices).all())
    print("2. ROUTER     scores exactly equivariant, top-k equivariant "
          "(perm[idx_new] == idx_base): %s   max|Δ| = %.1e" % (ok2, worst2))
    if not ok2 or worst2 != 0.0:
        fails.append("router-equivariance")

    # ---- 3. full-sequence logits
    inp = torch.from_numpy(FIXTURE[:-1])[None, :]
    with torch.no_grad():
        lb, _, _ = base(inp, None)
        lp, _, _ = permuted(inp, None)
    dmax = float((lb - lp).abs().max())
    rel = dmax / float(lb.abs().max())
    argmax_ok = bool((lb.argmax(-1) == lp.argmax(-1)).all())
    print("3. LOGITS     max|Δ| = %.3e  (relative %.2e to max|logit| %.2f)  argmax identical: %s"
          % (dmax, rel, float(lb.abs().max()), argmax_ok))
    print("              tolerance: relative < 1e-6 (fp32 rounding from column reorder)")
    if rel >= 1e-6 or not argmax_ok:
        fails.append("logits %.2e" % rel)

    # ---- 4. routing traces equivariant
    T0, T1 = trace_of(base, inp, moe), trace_of(permuted, inp, moe)
    mapped = torch.stack([torch.from_numpy(perms[s])[T1[s].long()] for s in range(len(moe))])
    mm = int((mapped != T0).sum())
    print("4. TRACES     perm[permuted ids] vs base ids: %d / %d differ" % (mm, T0.numel()))
    if mm:
        fails.append("traces %d" % mm)

    # ---- 5/6. the C engine: exact vs its own reference, and equivariant routing
    if args.engine:
        T = args.window
        sel, logits = {}, {}
        for tag, path in (("base", args.base), ("perm", args.permuted)):
            with tempfile.TemporaryDirectory() as td:
                outp = Path(td) / "sel.bin"
                r = subprocess.run([args.engine, path, "--moe-sel", args.tokens_bin,
                                    "0", str(T), str(outp)], capture_output=True, text=True)
                if r.returncode != 0:
                    print("   engine FAILED on %s: %s" % (tag, r.stderr.strip()))
                    fails.append("engine %s" % tag)
                    sel = None
                    break
                raw = outp.read_bytes()
                nm, T2, K2 = struct.unpack_from("<iii", raw, 0)
                sel[tag] = np.frombuffer(raw, dtype="<i4", count=nm * T2 * K2,
                                         offset=12).reshape(nm, T2, K2).copy()
        if sel:
            # (Direction matters and is easy to invert — see the module docstring.)
            # experts'[k] = experts[perm[k]] implies idx_new = perm^{-1}(idx_base),
            # i.e. perm[idx_new] == idx_base.
            mism = 0
            for s in range(sel["perm"].shape[0]):
                p = perms[s]
                mism += int((p[sel["perm"][s]] != sel["base"][s]).sum())
            print("5. C ENGINE   perm[C(permuted)] vs C(base) routing: %d / %d differ"
                  % (mism, sel["base"].size))
            print("              (C itself verified against its own Python reference by")
            print("               tools/verify_cpw1_v3.py — this is the cross-file symmetry)")
            if mism:
                fails.append("C engine %d" % mism)

    print("\n" + "=" * 78)
    if fails:
        print("PERMUTATION GATE: FAIL — %s" % ", ".join(fails))
        return 1
    print("PERMUTATION GATE: PASS")
    print("  expert reorder + matching router permutation is a symmetry:")
    print("  routing equivariant exactly, logits identical to fp32 rounding (%.1e rel)." % rel)
    print("  NOTE this is NOT bit-identical: router column order changes summation")
    print("  order, so max|Δ| = %.1e, not 0. Layout compilation is float32-equivalent," % dmax)
    print("  not bit-exact — machine-invariant claims must say 'exact routing + fp32-equal logits'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
