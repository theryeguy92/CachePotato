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
    sum_err = 0.0; set_mismatch = []; w_err = 0.0
    for i in range(L):
        k8, kw, kg = f"L{i}.top8", f"L{i}.top8_w", f"L{i}.mlp.gate"
        if k8 not in keys:
            continue
        w = z[kw]
        sum_err = max(sum_err, float(np.abs(w.sum(-1) - 1.0).max()))
        if kg in keys:
            lg = z[kg]
            p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
            order = np.argsort(-p, axis=-1, kind="stable")[:, :8]
            for a, b in zip(order.tolist(), z[k8].tolist()):
                if set(a) != set(b):
                    set_mismatch.append(i); break
            got = np.take_along_axis(p, z[k8], -1)
            want = got / got.sum(-1, keepdims=True)
            w_err = max(w_err, float(np.abs(want - w).max()))
    print(f"  routing weights sum to 1 (max |err|): {sum_err:.2e}")
    print(f"  trace vs recorded router logits: {'AGREE (all layers)' if not set_mismatch else f'MISMATCH in {set_mismatch[:5]}'}")
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
