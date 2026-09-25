#!/usr/bin/env python3
"""Regression gate: enabling the TRACE hook must not perturb the model.

The Phase 6 scale run executed train_moe.py with TRACE = None (the module
default), so the training path is the un-instrumented branch of every
`if TRACE is not None` guard. This test pins that invariant: with identical
weights and an identical batch, logits and loss must be bit-identical whether
the hook is disabled or enabled.

Run from repo root:  python tools/verify_trace_hook.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_moe as tm  # noqa: E402


def main():
    cfg = tm.Config.spec(0.5)
    torch.manual_seed(1234)
    model = tm.MoEModel(cfg).eval()

    torch.manual_seed(7)
    ids = torch.randint(0, 50257, (2, 128))
    ids[0, 60] = tm.EOS          # exercise the EOS locality reset

    with torch.no_grad():
        l_off, loss_off, _ = model(ids, ids)

    tm.TRACE = {}
    with torch.no_grad():
        l_on, loss_on, _ = model(ids, ids)
    trace = tm.TRACE
    tm.TRACE = None

    dl = (l_off - l_on).abs().max().item()
    dloss = abs(float(loss_off) - float(loss_on))
    print("hook OFF vs ON:  max|dlogits| = %.3e   |dloss| = %.3e" % (dl, dloss))

    layers = sorted((trace or {}).keys())
    print("captured MoE layers: %s" % layers)
    ok_shapes = True
    for l in layers:
        t = torch.stack(trace[l])          # (T, B, K)
        if tuple(t.shape) != (ids.shape[1], ids.shape[0], cfg.topk):
            ok_shapes = False
            print("  BAD shape layer %d: %s" % (l, tuple(t.shape)))
    print("captured shapes OK: %s" % ok_shapes)

    ok = (dl == 0.0 and dloss == 0.0 and ok_shapes and len(layers) > 0)
    print("\nTRACE-HOOK GATE:", "PASS (bit-identical)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
