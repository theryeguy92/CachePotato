#!/usr/bin/env python3
"""convert.py — Qwen3-30B-A3B: safetensors -> CPW1 v4.

The first sparse port, so this is also the first file whose block table says `moe` with 128 experts
and `n_shared = 0`, and whose attention block sets `qk_norm = 1` (the flag the config does not have,
found in the weights).

**One region per expert.** The checkpoint already ships one tensor per expert, which is exactly the
layout paging wants; HuggingFace fuses the three matrices per expert at load time for kernel
efficiency, which is the opposite of what a region-per-expert pager needs. Reading the checkpoint
form is therefore both simpler and better for this engine.

`--shard1` converts only the layers present in shard 1 (0-2), for gating the wiring against the
reference before the full 61 GB checkpoint exists.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "format"))
import v4_writer as V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/qwen3-30b-a3b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard1", action="store_true", help="only layers present in shard 1")
    ap.add_argument("--layers", type=int, default=None, help="override the layer count")
    ap.add_argument("--quant", choices=["none", "q8_0"], default="none")
    args = ap.parse_args()

    md = ROOT / args.model
    cfg = json.load(open(md / "config.json"))
    D = cfg["hidden_size"]; H = cfg["num_attention_heads"]; KV = cfg["num_key_value_heads"]
    HD = cfg["head_dim"]; E = cfg["num_experts"]; K = cfg["num_experts_per_tok"]
    F = cfg["moe_intermediate_size"]; Voc = cfg["vocab_size"]
    THETA = float(cfg["rope_theta"]); EPS = float(cfg["rms_norm_eps"])
    total_layers = cfg["num_hidden_layers"]

    from safetensors import safe_open
    shard1 = safe_open(str(md / "model-00001-of-00016.safetensors"), framework="pt")
    # A shard boundary can cut MID-LAYER: shard 1 contains layer 2's experts but not its norms,
    # so counting distinct layer indices overstates what is usable. Only COMPLETE layers count.
    keys = set(shard1.keys())
    need = ("input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "mlp.gate.weight")
    idx = sorted({int(k.split(".")[2]) for k in keys if k.startswith("model.layers.")})
    complete = [l for l in idx
                if all(f"model.layers.{l}.{x}" in keys for x in need)
                and all(f"model.layers.{l}.mlp.experts.{e}.{k}.weight" in keys
                        for e in range(E) for k in ("gate_proj", "up_proj", "down_proj"))]
    L = args.layers if args.layers else (len(complete) if args.shard1 else total_layers)
    print(f"layers: {L} (shard1 indices {idx[0]}..{idx[-1]}, complete {complete}; "
          f"model has {total_layers})")
    if args.shard1 and L == 0:
        raise SystemExit("convert: no complete layer in shard 1")

    def t(name):
        return shard1.get_tensor(name).float().numpy()

    w = V.V4Writer()
    tid = {}
    def add(name, arr, role, enc="f32"):
        use = "q8_0" if (enc == "q8_0" and args.quant == "q8_0") else "f32"
        tid[name] = w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32), role=role, enc=use)

    # embedding table (also the shard-1 input); the model's head is UNTIED
    add("embed", t("model.embed_tokens.weight"), V.R_EMBED)
    for l in range(L):
        p = f"model.layers.{l}."
        add(f"ln1.{l}", t(p + "input_layernorm.weight"), V.R_NORM)
        add(f"ln2.{l}", t(p + "post_attention_layernorm.weight"), V.R_NORM)
        add(f"q_norm.{l}", t(p + "self_attn.q_norm.weight"), V.R_NORM)
        add(f"k_norm.{l}", t(p + "self_attn.k_norm.weight"), V.R_NORM)
        for nm, key in (("Wq", "q_proj"), ("Wk", "k_proj"), ("Wv", "v_proj"), ("Wo", "o_proj")):
            add(f"{nm}.{l}", t(p + f"self_attn.{key}.weight"), V.R_ATTN, enc="q8_0")
        add(f"Wgate.{l}", t(p + "mlp.gate.weight"), V.R_ROUTER)
        for e in range(E):
            for nm, key in (("Wg", "gate_proj"), ("Wu", "up_proj"), ("Wd", "down_proj")):
                add(f"expert.{l}.{e}.{nm}", t(p + f"mlp.experts.{e}.{key}.weight"),
                    V.R_EXPERT, enc="q8_0")

    w.add_block(V.KIND_EMBED, 0xFFFFFFFF, refs=[tid["embed"]], params=[Voc, D, 0])
    for l in range(L):
        # attention: params are (n_heads, n_kv_heads, head_dim, rope_theta, qk_norm, sliding_window).
        # qk_norm = 1 means the block reads q_norm and k_norm AFTER Wo -> see SPEC §7.
        w.add_block(V.KIND_ATTN, l,
                    refs=[tid[f"{n}.{l}"] for n in ("Wq", "Wk", "Wv", "Wo", "q_norm", "k_norm")],
                    params=[H, KV, HD, V.f32_bits(THETA), 1, 0])
        # MoE: (n_experts, top_k, scoring, norm_topk_prob, latent_size, routed_scaling,
        #       n_shared, shared_overlap, group_limited, n_group). n_shared = 0: this model has
        #       NO shared expert, so every FFN read is an expert read.
        w.add_block(V.KIND_MOE, l, refs=[tid[f"Wgate.{l}"]],
                    params=[E, K, V.SCORING_SOFTMAX, 1, 0, V.f32_bits(1.0), 0, 0, 0, 1])

    for e in range(L * E):
        l, ex = divmod(e, E)
        lo = min(tid[f"expert.{l}.{ex}.{n}"] for n in ("Wg", "Wu", "Wd"))
        hi = max(tid[f"expert.{l}.{ex}.{n}"] for n in ("Wg", "Wu", "Wd"))
        w.add_region(lo, hi + 1, kind=V.REGION_EXPERT, slot=l, mlo=ex, mhi=ex + 1)
    for name, i in tid.items():
        if not name.startswith("expert."):
            w.add_region(i, i + 1, kind=V.REGION_TENSOR)

    order = list(range(len(w.tensors)))
    order = [i for i in order if w.tensors[i]["enc"] == "f32"] + \
            [i for i in order if w.tensors[i]["enc"] != "f32"]
    info = w.write(args.out, layout=order)
    out = Path(args.out)
    print(f"convert: {out} ({out.stat().st_size/1e6:.0f} MB) | tensors {info['tensors']} "
          f"blocks {info['blocks']} regions {info['regions']}")
    print(f"  D={D} H={H}/{KV} HD={HD} experts={E} top{K} d_exp={F} vocab={Voc} layers={L}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
