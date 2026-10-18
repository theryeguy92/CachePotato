#!/usr/bin/env python3
"""compare_qwen3_eval.py — B5: the C engine vs the recorded HuggingFace baseline.

Three different standards, because they answer three different questions:
  logits   TIER C quality bound — the baseline is bf16, our file is Q8, so exactness is not
           available and would not mean anything. Reports relative error and argmax agreement.
  routing  EXACT under the tie-aware rule. bf16 makes the 8th/9th expert tie at essentially every
           position, so "same set" is not well defined; "a valid top-8 given ties" is. Anything
           less than validity is a real bug.
  router   per-layer correlation of the router LOGITS, so a failure names a layer instead of being
           an end-to-end disagreement.
"""
import sys
from pathlib import Path
import numpy as np

def tie_aware_agree(logits, chosen):
    """True when `chosen` is a valid top-k for `logits` given ties."""
    p = np.exp(logits - logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    k = chosen.shape[-1]
    thr = np.sort(p, -1)[:, -k]
    for t in range(p.shape[0]):
        rec = set(chosen[t].tolist())
        if any(p[t, j] < thr[t] - 1e-12 for j in rec):
            return False, t
        above = {j for j in range(p.shape[1]) if p[t, j] > thr[t] + 1e-12}
        if not above <= rec:
            return False, t
    return True, -1

def main():
    npz = Path(sys.argv[1]); prefix = sys.argv[2]
    z = np.load(npz)
    nprompt = sys.argv[3] if len(sys.argv) > 3 else "p0"
    L = int(__import__("json").loads(str(z["meta"]))["layers"])
    T = len(z[f"input_ids_{nprompt}"])
    ref = z[f"logits_{nprompt}"][0][:T].astype(np.float64)
    got = np.fromfile(prefix + ".logits", dtype="<f4").reshape(T, -1).astype(np.float64)
    print("=" * 78); print(f"B5: C engine vs HF baseline  (prompt {nprompt}, {T} tokens, {L} layers)")
    print("=" * 78)
    rel = float(np.abs(got - ref).max() / max(float(np.abs(ref).max()), 1e-30))
    am_ref = ref.argmax(-1); am_got = got.argmax(-1)
    agree = int((am_ref == am_got).sum())
    print(f"  logits   shape {got.shape} rel {rel:.3e}   argmax {agree}/{T}")
    cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref)))
    print(f"  cosine   {cos:.8f}")

    route = np.fromfile(prefix + ".route", dtype="<i4").reshape(L, T, -1)
    bad_layers = []
    for l in range(L):
        if f"L{l}.mlp.gate" not in z:
            continue
        ok, at = tie_aware_agree(z[f"L{l}.mlp.gate"].astype(np.float64), route[l])
        if not ok:
            bad_layers.append((l, at))
    print(f"  routing  {L - len(bad_layers)}/{L} layers valid under the tie-aware rule"
          + (f"   FIRST BAD {bad_layers[:3]}" if bad_layers else ""))

    router = np.fromfile(prefix + ".router", dtype="<f4").reshape(L, T, -1)
    worst = None
    for l in range(L):
        k = f"L{l}.mlp.gate"
        if k not in z:
            continue
        a, b = router[l].ravel().astype(np.float64), z[k].astype(np.float64).ravel()
        c = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        if worst is None or c < worst[1]:
            worst = (l, c)
    if worst:
        print(f"  router   worst layer correlation {worst[1]:.5f} (layer {worst[0]})")
    print()
    ok = (agree == T) and not bad_layers
    print("B5: " + ("PASS — engine reproduces the model (quality bound met, routing valid)"
                    if ok else "FAIL — see above"))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
