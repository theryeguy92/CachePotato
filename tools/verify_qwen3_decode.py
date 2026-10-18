#!/usr/bin/env python3
"""verify_qwen3_decode.py — the step gate: incremental decode == full-sequence prefill.

WHY THIS EXISTS
  C2's numbers come from a decode loop (one token at a time against a KV cache), because a
  T-token prefill measures the wrong thing twice: token t's experts are touched ~200 MB
  after token t-1's, which inflates reuse, and its tok/s is a prefill rate. But a decode
  path is a second way of computing attention, and a second way is a place for the cache
  indexing, the RoPE position or the causal bound to be wrong while every number still
  looks plausible.

  So decode is gated the way gate 1 gates the GPT-2 MoE's incremental path: the logits
  produced one token at a time must be BIT-IDENTICAL to the logits the prefill path
  produces for the same position. Not "close": the two paths read the same K/V rows in
  the same order, so there is no tolerance to spend.

WHY BIT-IDENTITY IS THE RIGHT BAR HERE
  Decode uses qwen3_layer_kv(T=1, pos0=n) and prefill uses qwen3_layer_kv(T=n+1, pos0=0).
  The K/V rows, the causal bound (j <= pos0+i) and the summations are the same sequence of
  operations, so any difference is an indexing bug, not rounding. The only thing that
  legitimately differs is the order in which layers are visited — which is the same.

USAGE
  python3 tools/verify_qwen3_decode.py                    # 1 prompt, 8 generated tokens
  python3 tools/verify_qwen3_decode.py --gen 4 --tokens 785,6722,315,9625,374
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def run(engine, model, ids, prefix, gen=0):
    cmd = [engine, model, "48", ids, prefix]
    if gen:
        cmd += ["--decode", str(gen)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-1200:])
        raise SystemExit(f"engine failed: {' '.join(cmd)}")
    return r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/qwen3_30b_a3b_q8.v4cp")
    ap.add_argument("--engine", default="./build/qwen3_eval")
    ap.add_argument("--tokens", default="785,6722,315,9625,374")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--keep", default="scratch/c2gate")
    args = ap.parse_args()

    prompt = [int(x) for x in args.tokens.split(",")]
    T = len(prompt)
    print("=" * 78)
    print(f"C2 step gate: decode logits == prefill logits at the same position")
    print("=" * 78)
    print(f"  model {args.model}  prompt {T} tokens  generate {args.gen}")

    pre = Path(args.keep)
    pre.parent.mkdir(parents=True, exist_ok=True)
    run(args.engine, args.model, args.tokens, str(pre) + "_dec", gen=args.gen)

    gtok = np.fromfile(str(pre) + "_dec.tokens", dtype="<i4")
    glog = np.fromfile(str(pre) + "_dec.logits", dtype="<f4").reshape(args.gen, -1)
    if len(gtok) != args.gen:
        print(f"  decode emitted {len(gtok)} tokens, expected {args.gen}")
        return 1

    # The prefill that must agree: the prompt plus every generated token EXCEPT the last
    # one (the last is what the final logits row predicts, it is never fed).
    full = prompt + gtok[:-1].tolist()
    run(args.engine, args.model, ",".join(str(t) for t in full), str(pre) + "_full")
    flog = np.fromfile(str(pre) + "_full.logits", dtype="<f4").reshape(len(full), -1)

    ok = True
    worst = 0.0
    for s in range(args.gen):
        pos = T - 1 + s                      # prefill row for the same position
        a, b = glog[s], flog[pos]
        same = np.array_equal(a, b)
        if not same:
            ok = False
            worst = max(worst, float(np.abs(a - b).max()))
        print(f"  position {pos:>2d}: token {int(gtok[s]):>6d}  "
              f"{'BIT-IDENTICAL' if same else 'DIFFERS'}")

    # The greedy chain is part of the claim: prefill's argmax must reproduce the tokens.
    chain = [int(flog[T - 1 + s].argmax()) for s in range(args.gen)]
    if chain != gtok.tolist():
        ok = False
        print(f"  greedy chain differs: prefill {chain} vs decode {gtok.tolist()}")
    else:
        print(f"  greedy chain: {gtok.tolist()} (prefill argmax agrees at every position)")

    print()
    if ok:
        print("C2 STEP GATE: PASS — decode is the same computation as prefill")
    else:
        print(f"C2 STEP GATE: FAIL — max abs difference {worst:.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
