#!/usr/bin/env python3
"""check_spec_conformance.py — compare a model against the architecture spec.

WHY THIS EXISTS
  G1 (bridge) compares PyTorch against the C trainer. G2 (smoke) compares curves.
  Neither compares EITHER implementation to Model-Architecture-Spec. That is how
  the Phase 6 checkpoints ended up with MoE on layers 1-10 while the spec says
  2-10, with every gate passing: a two-implementation agreement test is blind to
  a misreading both implementations share.

  SPEC_LAYOUT below is transcribed from the vault's Model-Architecture-Spec. If
  the spec changes, update it here in the same commit.

USAGE
  python tools/check_spec_conformance.py --ckpt <model_only.pt>
  python tools/check_spec_conformance.py --expect-next-run

  Exit 0 = conformant, 1 = divergent (loud, by design).
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_moe as tm  # noqa: E402

# --- transcribed from 05-Architecture-Specs/Model-Architecture-Spec.md -------
SPEC_LAYOUT = {
    "vocab": 50257, "d_model": 768, "n_heads": 12, "d_ff": 3072,
    "n_layer": 12, "n_ctx": 1024, "n_exp": 32, "exp_hid": 192, "topk": 2,
    "dense": [0, 1, 11],
    "moe": [2, 3, 4, 5, 6, 7, 8, 9, 10],
}
# -----------------------------------------------------------------------------


def budget(cfg, model):
    """Live / paged / active parameter counts for the model as configured."""
    moe, dense = cfg.moe_layers(), cfg.dense_layers()
    ATTN = ["Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"]
    LNS = ["ln1_g", "ln1_b", "ln2_g", "ln2_b"]
    FFN = ["W1", "b1", "W2", "b2"]
    SH = ["W1s", "b1s", "W2s", "b2s"]
    RO = ["W1e", "b1e", "W2e", "b2e"]
    RT = ["Wr", "br"]
    lp = lambda l, gs: sum(getattr(model, g)[l].numel() for g in gs)  # noqa: E731
    live = (model.token_emb.numel() + model.pos_emb.numel()
            + model.lnf_g.numel() + model.lnf_b.numel())
    paged = active = 0
    for l in range(cfg.n_layer):
        live += lp(l, ATTN) + lp(l, LNS)
        active += lp(l, ATTN)
        if l in moe:
            live += lp(l, SH) + lp(l, RO) + lp(l, RT)
            paged += lp(l, RO)
            active += lp(l, SH) + lp(l, RT)
            active += 2 * (model.W1e[l][0].numel() + model.b1e[l][0].numel()
                           + model.W2e[l][0].numel() + model.b2e[l][0].numel())
        else:
            live += lp(l, FFN)
            active += lp(l, FFN)
    return live, paged, active


def report(cfg, label):
    moe, dense = cfg.moe_layers(), cfg.dense_layers()
    dims_ok = all(getattr(cfg, k) == v for k, v in SPEC_LAYOUT.items()
                  if k not in ("dense", "moe"))
    layout_ok = (sorted(moe) == SPEC_LAYOUT["moe"]
                 and sorted(dense) == SPEC_LAYOUT["dense"])
    model = tm.MoEModel(cfg)
    live, paged, active = budget(cfg, model)

    print(f"--- {label} ---")
    print(f"  moe_from          : {cfg.moe_from}")
    print(f"  actual MoE layers : {moe}")
    print(f"  actual dense      : {dense}")
    print(f"  spec  MoE layers  : {SPEC_LAYOUT['moe']}")
    print(f"  spec  dense       : {SPEC_LAYOUT['dense']}")
    print(f"  dims match spec   : {dims_ok}")
    print(f"  layout match spec : {layout_ok}")
    print(f"  live params       : {live:,}  (~{live/1e6:.1f}M)")
    print(f"  paged (routed)    : {paged:,}  (~{paged/1e6:.1f}M, {paged*4/1e6:.0f} MB)")
    print(f"  hot               : {live-paged:,}  (~{(live-paged)/1e6:.1f}M)")
    print(f"  active / token    : {active:,}  (~{active/1e6:.1f}M)")
    print(f"  expert blocks     : {len(moe)*cfg.n_exp}")
    print(f"  worst-case reads  : {len(moe)*2*295872*4/1e6:.1f} MB/token")
    return layout_ok and dims_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--expect-next-run", action="store_true",
                    help="check that Config.spec(moe_from=2) IS conformant")
    args = ap.parse_args()

    verdicts = []
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = tm.Config(**ck["config"])
        print(f"checkpoint {args.ckpt}  step={ck.get('step')}  "
              f"recorded config keys={sorted(ck['config'])}")
        verdicts.append(report(cfg, "AS-BUILT ARTIFACT"))
    if args.expect_next_run:
        verdicts.append(report(tm.Config.spec(0.5, moe_from=2), "NEXT-RUN TARGET"))

    if not verdicts:
        ap.error("give --ckpt and/or --expect-next-run")

    ok = all(verdicts)
    print("\nSPEC CONFORMANCE:", "PASS" if ok else
          "DIVERGENT (see Decision-Log 2026-09-25 — accepted deviation for the "
          "Phase 6 checkpoints; the next run must be conformant)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
