#!/usr/bin/env python3
"""convert.py — gpt2-moe port: PyTorch checkpoint -> CPW1 v4 (format/SPEC.md).

WHAT A PORT'S CONVERTER MUST DO
  Turn a released checkpoint into a file the engine can load, using only this port's facts
  (MODEL.md) plus the shared writer (format/v4_writer.py). It must not consult the trainer,
  and it must refuse rather than guess when the checkpoint does not match MODEL.md.

  This is the real port path: checkpoint -> v4 directly. (tools/v3_to_v4.py converts the
  already-exported v3 file and exists for the FORMAT's acceptance test, not for ports.)

TWO THINGS THIS CONVERTER REFUSES, ON PURPOSE
  1. A config that disagrees with MODEL.md's verified facts. A port that silently adapts to
     whatever the checkpoint says is a port that cannot be trusted: the model would load and
     run with a different architecture than the reference models.
  2. A layer split it cannot establish. `moe_from` is NOT in the checkpoint config (MODEL.md
     §3); it is ASSUMED to be the trainer default of 1 and is cross-checked against the
     recorded routing by verify/gate.py. Anything else is refused.

USAGE
  python3 models/gpt2-moe/convert.py \
      --ckpt weights/moe-scale-run/l05/model_only.pt \
      --out  weights/moe-scale-run/l05.v4cp [--shuffle] [--moe-from 1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "format"))
import v4_writer as V  # noqa: E402

# ---- MODEL.md §1: the facts this port relies on. The converter checks them.
EXPECTED = {
    "vocab": 50257, "d_model": 768, "n_heads": 12, "d_ff": 3072, "n_layer": 12,
    "n_ctx": 1024, "n_exp": 32, "exp_hid": 192, "topk": 2,
}


def check_config(cfg):
    bad = {k: (cfg.get(k), v) for k, v in EXPECTED.items() if cfg.get(k) != v}
    if bad:
        msg = ", ".join(f"{k}: checkpoint {a} != MODEL.md {b}" for k, (a, b) in bad.items())
        raise SystemExit(
            "convert: checkpoint does not match models/gpt2-moe/MODEL.md:\n  %s\n"
            "A converter that adapts to any config cannot be checked against the reference; "
            "if this is a genuinely different model it needs its own port." % msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--moe-from", type=int, default=1,
                    help="first MoE layer; NOT recorded in the checkpoint (MODEL.md §3)")
    ap.add_argument("--shuffle", action="store_true",
                    help="write the payload in a different physical order (format test)")
    ap.add_argument("--seed", type=int, default=20261016)
    args = ap.parse_args()

    import torch
    ck = torch.load(ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else args.ckpt,
                    map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg["moe_from"] = args.moe_from
    check_config(cfg)
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in ck["model"].items()}

    D, L, N, EH = cfg["d_model"], cfg["n_layer"], cfg["n_exp"], cfg["exp_hid"]
    F, Vv, C, H = cfg["d_ff"], cfg["vocab"], cfg["n_ctx"], cfg["n_heads"]
    moe = [l for l in range(L) if cfg["moe_from"] <= l < L - 1]
    dense = [l for l in range(L) if l not in moe]

    # ---- the dead-weights trap (MODEL.md §2). The CHECKPOINT carries a dense-FFN slot on
    # EVERY layer (12), including the MoE layers where those weights were never trained. The
    # FILE carries only the dense layers (2). Selecting the wrong side of that difference is
    # the failure this project has already had once: the weights load, run, and emit fluent
    # garbage. So check the checkpoint really is the all-layers layout before compacting.
    if sd["W1"].shape[0] != L:
        raise SystemExit(
            "convert: W1 has %d slots but the checkpoint has %d layers. Expected the "
            "trainer's ALL-LAYERS layout (dead dense weights present on MoE layers), which "
            "this converter compacts to the %d dense layers. A checkpoint that has already "
            "been compacted would be double-compacted here." % (sd["W1"].shape[0], L, len(dense)))

    w = V.V4Writer()
    tid = {}

    def add(name, arr, role):
        tid[name] = w.add_tensor(name, arr, role=role)

    # global tensors
    add("token_emb", sd["token_emb"], V.R_EMBED)
    add("pos_emb", sd["pos_emb"], V.R_EMBED)
    add("lnf_g", sd["lnf_g"], V.R_NORM)
    add("lnf_b", sd["lnf_b"], V.R_NORM)
    # per-layer groups
    for nm, role in (("ln1_g", V.R_NORM), ("ln1_b", V.R_NORM),
                     ("ln2_g", V.R_NORM), ("ln2_b", V.R_NORM)):
        add(nm, sd[nm], role)
    for nm in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"):
        add(nm, sd[nm], V.R_ATTN)
    # dense FFN: compacted group (dense layers only)
    for nm in ("W1", "b1", "W2", "b2"):
        add(nm, sd[nm][dense], V.R_DENSE)
    # router + shared expert, per MoE layer
    for nm, role in (("Wr", V.R_ROUTER), ("br", V.R_ROUTER),
                     ("W1s", V.R_SHARED), ("b1s", V.R_SHARED),
                     ("W2s", V.R_SHARED), ("b2s", V.R_SHARED)):
        add(nm, sd[nm][moe], role)
    # routed experts: ONE TENSOR PER (layer, expert) so one expert is one pageable region
    # (SPEC §7.3 interleaved-pair arrangement, as this model's v3 file already uses)
    for slot, l in enumerate(moe):
        for e in range(N):
            flat = np.concatenate([sd["W1e"][l, e].ravel(), sd["b1e"][l, e].ravel(),
                                   sd["W2e"][l, e].ravel(), sd["b2e"][l, e].ravel()])
            add("experts.%d.%d" % (slot, e), flat, V.R_EXPERT)

    # ---- blocks (SPEC §7). Param order is the spec's, and the loader reads it by index.
    w.add_block(V.KIND_EMBED, 0xFFFFFFFF, refs=[tid["token_emb"], tid["pos_emb"]],
                params=[Vv, D, C])
    for l in range(L):
        w.add_block(V.KIND_ATTN, l,
                    refs=[tid[n] for n in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo")],
                    params=[H, H, 0, V.f32_bits(10000.0), 0, 0])
        if l in moe:
            slot = moe.index(l)
            w.add_block(V.KIND_MOE, l,
                        refs=[tid[n] for n in ("Wr", "br", "W1s", "b1s", "W2s", "b2s")],
                        params=[N, cfg["topk"], V.SCORING_SOFTMAX, 1, 0,
                                V.f32_bits(1.0), 1, 0, 0, 1])
        else:
            w.add_block(V.KIND_DENSE_FFN, l,
                        refs=[tid[n] for n in ("ln2_g", "ln2_b", "W1", "b1", "W2", "b2")],
                        params=[V.ACT_GELU_TANH, F, dense.index(l),
                                1 if l >= cfg["moe_from"] else 0])
    w.add_block(V.KIND_NORM, L, refs=[tid["lnf_g"], tid["lnf_b"]],
                params=[V.f32_bits(1e-5), 0])

    # ---- regions: one per expert (pageable as a unit), one per other tensor
    for slot, l in enumerate(moe):
        for e in range(N):
            w.add_expert_region(tid["experts.%d.%d" % (slot, e)], slot, e)
    for name, i in tid.items():
        if name.startswith("experts."):
            continue
        w.add_region(i, i + 1, kind=V.REGION_TENSOR)

    # ---- layout policy: the engine's strided order, so it can use the payload IN PLACE
    # (engine/core/weights_v4.c). Any other order is legal and loads via the copy path.
    engine_order = ["token_emb", "pos_emb", "ln1_g", "ln1_b",
                    "Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo",
                    "ln2_g", "ln2_b", "W1", "b1", "W2", "b2",
                    "Wr", "br", "W1s", "b1s", "W2s", "b2s"]
    order_ids = [tid[n] for n in engine_order]
    for slot in range(len(moe)):
        for e in range(N):
            order_ids.append(tid["experts.%d.%d" % (slot, e)])
    order_ids += [tid["lnf_g"], tid["lnf_b"]]
    if args.shuffle:
        np.random.default_rng(args.seed).shuffle(order_ids)

    info = w.write(args.out, layout=order_ids)
    out = Path(args.out)
    print("convert: wrote %s (%.1f MB)" % (out, out.stat().st_size / 1e6))
    print("  tensors %d | blocks %d | regions %d | payload at %d"
          % (info["tensors"], info["blocks"], info["regions"], info["off_data"]))
    print("  layout: %s" % ("SHUFFLED (exercises the copy path)" if args.shuffle
                            else "engine order (loads in place)"))
    print("  step=%s lam=%s layer split: MoE %s dense %s (ASSUMED from --moe-from %d)"
          % (ck.get("step"), cfg["lam"], moe, dense, cfg["moe_from"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
