#!/usr/bin/env python3
"""check_qwen3_baseline.py — is the recorded baseline a real 48-layer model, or well-formed rubbish?

The shard-1 recording passed every structural check and had randomly-initialised experts; only a
157x disagreement exposed it. So this checks things that a plausible-looking file would fail:

  * every one of the 48 layers has a routing trace, of the right shape;
  * routing weights are a distribution (sum to 1) — random experts do not produce that;
  * the trace is CONSISTENT with the recorded router logits: top-8 recomputed from `mlp.gate` must
    select the same SET as recorded, and the weights must match softmax+renormalise;
  * logits are finite, the right width, and their argmax matches what was printed;
  * no array anywhere is NaN or inf.
"""
import json, sys
from pathlib import Path
import numpy as np

def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/traces/qwen3_30b_a3b_full.npz")
    z = np.load(path, allow_pickle=False)
    keys = set(z.files)
    meta = json.loads(str(z["meta"])) if "meta" in keys else {}
    print("=" * 78); print(f"QWEN3 BASELINE: {path} ({path.stat().st_size/1e6:.1f} MB)"); print("=" * 78)
    print(f"  dtype recorded: {meta.get('dtype')}   layers: {meta.get('layers')}   "
          f"arrays: {len(keys)}")
    bad = 0

    # 1. no NaN/inf anywhere
    poisoned = [k for k in keys if z[k].dtype.kind == "f" and not np.isfinite(z[k]).all()]
    print(f"  non-finite arrays: {poisoned if poisoned else 'none'}")
    bad += bool(poisoned)

    # 2. routing trace complete and shaped right
    L = int(meta.get("layers") or 48)
    missing = [i for i in range(L) if f"L{i}.top8" not in keys or f"L{i}.top8_w" not in keys]
    print(f"  routing traces: {L - len(missing)}/{L}" + (f"  MISSING {missing[:5]}" if missing else ""))
    bad += bool(missing)

    # 3. weights are a distribution, and the trace agrees with the router logits
    sum_err = 0.0; set_mismatch = []; w_err = 0.0; ties = 0
    for i in range(L):
        k8, kw, kg = f"L{i}.top8", f"L{i}.top8_w", f"L{i}.mlp.gate"
        if k8 not in keys:
            continue
        w = z[kw]
        sum_err = max(sum_err, float(np.abs(w.sum(-1) - 1.0).max()))
        if kg in keys:
            lg = z[kg]
            p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
            # TIES ARE EXPECTED, NOT AN ERROR. Router logits are recorded in bf16 (~8 mantissa
            # bits), so at 128 experts two probabilities can be EXACTLY equal — and then "the top
            # 8" has no single answer. A selected set is valid if every chosen expert is at least
            # as likely as the 8th largest, and no unchosen expert is strictly more likely than
            # that threshold. Demanding one exact set called 5 of 48 layers broken while the
            # recorded WEIGHTS matched softmax to 1e-7 — the giveaway that the indices were right.
            kth = np.sort(p, axis=-1)[:, -8]
            for ti in range(p.shape[0]):
                rec = set(z[k8][ti].tolist())
                thr = kth[ti]
                chosen_ok = all(p[ti, j] >= thr - 1e-12 for j in rec)
                above = {j for j in range(p.shape[1]) if p[ti, j] > thr + 1e-12}
                unchosen_ok = above <= rec
                n_above_sel = sum(1 for j in rec if p[ti, j] > thr + 1e-12)
                if not (chosen_ok and unchosen_ok) or n_above_sel != len(above):
                    set_mismatch.append(i); break
                if len(above) < len(rec):
                    ties += 1
            got = np.take_along_axis(p, z[k8], -1)
            want = got / got.sum(-1, keepdims=True)
            w_err = max(w_err, float(np.abs(want - w).max()))
    print(f"  routing weights sum to 1 (max |err|): {sum_err:.2e}")
    print(f"  trace vs recorded router logits: "
          f"{'AGREE (all layers)' if not set_mismatch else f'MISMATCH in {set_mismatch[:5]}'}"
          + (f"  [{ties} selections resolved by a tie]" if ties else ""))
    print(f"  weights vs softmax+renormalise (max |err|): {w_err:.2e}")
    bad += (sum_err > 1e-4) or bool(set_mismatch) or (w_err > 1e-4)

    # 4. logits and argmax
    for k in sorted(x for x in keys if x.startswith("logits_")):
        lg = z[k]
        am = int(lg[0, -1].argmax())
        print(f"  {k:12} shape {str(lg.shape):16} argmax {am:7d}  peak {float(lg.max()):8.3f}")
        bad += (lg.shape[-1] != 151936) or (not np.isfinite(lg).all())

    # 5. how much routing actually varies — a real MoE spreads across all 128 experts
    if "L0.top8" in keys:
        used = len(set(z["L0.top8"].ravel().tolist()))
        print(f"  distinct experts in layer 0's trace: {used}/128")
    print()
    print(f"QWEN3 BASELINE: {'PASS' if not bad else f'FAIL ({bad})'}")
    return 0 if not bad else 1

if __name__ == "__main__":
    raise SystemExit(main())
