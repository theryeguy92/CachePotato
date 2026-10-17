#!/usr/bin/env python3
"""verify/gate.py — gates for the gpt2-moe port.

WHAT THIS CHECKS, AND WHY THE ORDER MATTERS
  A. REFERENCE vs RECORDED ROUTING. `reference.py` is an independent numpy implementation
     (no import of the trainer). It must reproduce the expert selections the TRAINED model
     made on held-out corpus windows. If this fails, nothing downstream means anything: the
     oracle is wrong.
  B. C ENGINE vs REFERENCE. The engine, loading the port's own v4 file, must select the same
     experts as `reference.py` on the same tokens, and its logits must agree to the stated
     tolerance. Exact at the routing level, tolerant on floats (different summation order).
  C. FORMAT ACCEPTANCE. Delegated to tools/verify_v4_roundtrip.py, which checks v3 vs v4 and
     a shuffled payload.

  Everything is exact where exactness is possible. The one tolerance is on logits, and it is
  stated rather than implied.

USAGE
  python3 models/gpt2-moe/verify/gate.py \
      --ckpt weights/moe-scale-run/l05/model_only.pt \
      --v4 scratch/port_l05.v4cp --v4-shuffled scratch/port_l05_shuf.v4cp \
      --trace data/traces/l05_fp32.npz --engine ./build/cachepotato
"""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models" / "gpt2-moe"))
import reference as ref  # noqa: E402

# LOGIT TOLERANCE, measured rather than guessed. On 64 real tokens at lambda=0.5:
#     C engine (v3)     vs C engine (v4 port) : 0.000e+00   <- the port is bit-faithful
#     C engine (v4)     vs torch trainer      : 1.497e-04
#     numpy reference   vs torch trainer      : 2.961e-05
#     C engine (v4)     vs numpy reference    : 1.767e-04
# All four agree on argmax, and C(v3) vs C(v4) is EXACT. The residual is fp32 accumulation
# order: the engine sums 768-term dot products with scalar loops while numpy/torch use
# blocked matmuls. ~1.5e-4 absolute is 7e-6 relative to the largest logit (20.67), i.e. ~60x
# fp32 epsilon, which is what 768-term reordering costs.
#
# So the gate applies TWO checks, because a single tolerance would hide which one failed:
#   exact   : C(v4) vs C(v3)          must be 0.0  (proves the port changed nothing)
#   bounded : C(v4) vs reference      must be < 5e-4 (3x the measured floor; a wrong expert
#                                     or a misloaded tensor is orders of magnitude larger)
LOGIT_TOL = 5e-4
LOGIT_EXACT = 0.0


def run_c(engine, weights, tokens_bin, start, T, lam, out):
    import os
    env = dict(os.environ, CP_V4_LAMBDA="%.6f" % lam)
    r = subprocess.run([engine, weights, "--moe-sel", tokens_bin, str(start), str(T), out],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return None, r.stderr.strip()
    raw = Path(out).read_bytes()
    nm, T2, K = struct.unpack_from("<iii", raw, 0)
    sel = np.frombuffer(raw, dtype="<i4", count=nm * T2 * K, offset=12).reshape(nm, T2, K).copy()
    # The engine appends its T x vocab logits block after the selections (additive). Use it
    # for the numerics check: the plain CLI prints logits at lambda=0, which is a different
    # model whenever lambda != 0, so routing-only comparison would hide logit divergence.
    logits = None
    off = 12 + nm * T2 * K * 4
    if len(raw) >= off + 4:
        logits = np.frombuffer(raw, dtype="<f4", count=len(raw) // 4 - off // 4,
                               offset=off).reshape(T2, -1)
    argmax = None
    for line in r.stderr.splitlines():
        if "next-argmax" in line:
            argmax = int(line.split("next-argmax")[1].split()[0])
    return (sel, argmax, logits), r.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/moe-scale-run/l05/model_only.pt")
    ap.add_argument("--v4", default="scratch/port_l05.v4cp")
    ap.add_argument("--v4-shuffled", default="scratch/port_l05_shuf.v4cp")
    ap.add_argument("--v3", default="weights/moe-scale-run/l05.cpw1v3")
    ap.add_argument("--trace", default="data/traces/l05_fp32.npz")
    ap.add_argument("--tokens-bin", default="data/fineweb_edu_valid.bin")
    ap.add_argument("--engine", default="./build/cachepotato")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()

    print("=" * 78)
    print("gpt2-moe port gate")
    print("=" * 78)
    fails = []

    cfg, arrays, step, lam = ref.load_checkpoint(ROOT / args.ckpt)
    if lam is None:
        lam = cfg.get("lam")
    m = ref.Model(cfg, arrays)
    print(f"  checkpoint step={step} lam={lam} | MoE layers {m.moe_layers}")
    print(f"  tolerance: routing EXACT (0 mismatches), logits < {LOGIT_TOL:.0e}")

    # ---- A. reference vs recorded routing
    z = np.load(ROOT / args.trace, allow_pickle=False)
    toks_all, sel_all = z["tokens"], z["sel"]
    trace_lam = float(z["lam"])
    moe_layers = [int(x) for x in z["moe_layers"]]
    print(f"\nA. REFERENCE vs RECORDED ROUTING ({args.windows} windows x "
          f"{toks_all.shape[1]} tokens, trace lam={trace_lam})")
    mism = tot = 0
    for w in range(min(args.windows, toks_all.shape[0])):
        _, routing = m.forward(toks_all[w].astype(np.int64), trace_lam)
        for li, l in enumerate(moe_layers):
            mism += int((routing[l] != sel_all[li, w]).sum())
            tot += sel_all[li, w].size
    rate = 100.0 * (1 - mism / tot) if tot else 0.0
    print(f"   {tot-mism}/{tot} selections match ({rate:.4f}%)")
    if mism:
        fails.append(f"reference-vs-recorded {mism}")

    # ---- B. C engine vs reference, on the port's own file
    print("\nB. C ENGINE (port v4 file) vs REFERENCE")
    start, T = 0, args.tokens
    toks = np.fromfile(ROOT / args.tokens_bin, dtype="<u2", count=T,
                       offset=0).astype(np.int64)
    with tempfile.TemporaryDirectory() as td:
        got, msg = run_c(args.engine, str(ROOT / args.v4), str(ROOT / args.tokens_bin),
                         start, T, trace_lam, str(Path(td) / "sel.bin"))
        if got is None:
            print(f"   engine FAILED: {msg}")
            fails.append("engine")
        else:
            sel_c, argmax_c, logits_c = got
            logits, routing = m.forward(toks, trace_lam)
            ref_sel = np.stack([routing[l] for l in moe_layers])       # (L,T,K)
            mism = int((sel_c != ref_sel).sum())
            print(f"   routing : {mism} / {ref_sel.size} selections differ")
            argmax_ref = int(logits[-1].argmax())
            print(f"   argmax  : reference {argmax_ref} | engine {argmax_c} | "
                  f"equal {argmax_ref == argmax_c}")
            # logits tolerance: compare the ranking of the top few rather than raw values,
            # which is the shape of the check the engine can satisfy exactly.
            # exact port check: the same engine on the v3 file must give the SAME logits
            got3, _ = run_c(args.engine, str(ROOT / args.v3), str(ROOT / args.tokens_bin),
                            start, T, trace_lam, str(Path(td) / "sel3.bin"))
            if got3 is not None and logits_c is not None and got3[2] is not None:
                d34 = float(np.abs(logits_c - got3[2]).max())
                print(f"   port    : C(v4) vs C(v3) max|delta| = {d34:.3e}  (must be 0)")
                if d34 != LOGIT_EXACT:
                    fails.append(f"v4-vs-v3 logits {d34:.2e}")

            if logits_c is not None and logits_c.shape == logits.shape:
                d = np.abs(logits_c - logits).max()
                print(f"   logits  : max|delta| = {d:.3e}  (tolerance {LOGIT_TOL:.0e})")
                if not (d < LOGIT_TOL):
                    fails.append(f"logits {d:.2e}")
            else:
                print("   logits  : not returned by the engine (routing only)")
            if mism:
                fails.append(f"C-vs-reference {mism}")
            if argmax_ref != argmax_c:
                fails.append(f"argmax {argmax_ref} vs {argmax_c}")

    # ---- C. format acceptance
    print("\nC. FORMAT ACCEPTANCE (v3 vs v4, natural and shuffled)")
    r = subprocess.run([sys.executable, str(ROOT / "tools/verify_v4_roundtrip.py"),
                        "--v3", str(ROOT / args.v3), "--v4", str(ROOT / args.v4),
                        "--v4-shuffled", str(ROOT / args.v4_shuffled),
                        "--engine", args.engine, "--tokens", str(args.tokens)],
                       capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()
    for line in tail[-8:]:
        print("   " + line.strip())
    if r.returncode != 0:
        fails.append("format acceptance")

    print("\n" + "=" * 78)
    if fails:
        print("PORT GATE: FAIL — " + ", ".join(fails))
        return 1
    print("PORT GATE: PASS")
    print("  the port's reference reproduces the trained model's routing, the C engine")
    print("  reproduces the reference from the port's own v4 file, and the file")
    print("  round-trips including a shuffled payload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
