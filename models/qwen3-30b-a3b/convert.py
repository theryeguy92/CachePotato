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
    ap.add_argument("--manifest", default=None,
                    help="write name/offset/size so the payloads can be checked against the source")
    # TIER B: placement-only expert reordering. The order file names a permutation per layer;
    # `--group N` puts N consecutive experts of that order into ONE region. Expert ids, tensor
    # names and the router are untouched, so this changes where the bytes live and nothing about
    # what the model computes — which makes the gate bit-identity rather than a tolerance.
    ap.add_argument("--expert-order", default=None,
                    help="JSON from tools/fit_expert_pairs.py: per-layer expert permutations")
    ap.add_argument("--group", type=int, default=2,
                    help="experts per region when --expert-order is given (default 2)")
    # REGION GRANULARITY vs ACCESS GRANULARITY. `embed` is a lookup table: a token reads ONE
    # row (2 KB at Q8) and never the other 151,935. Declared as one 330 MB region it costs a
    # pager that owns its bytes a whole region per token for one row — measured as the reason
    # the arena lost. This splits it into row-chunks so a region is at most a few MB.
    ap.add_argument("--embed-rows", type=int, default=1024,
                    help="rows per embed region (0 = one region for the whole table); "
                         "the block ref count is u8, so Voc/embed-rows must be <= 254")
    args = ap.parse_args()

    md = ROOT / args.model
    cfg = json.load(open(md / "config.json"))
    D = cfg["hidden_size"]; H = cfg["num_attention_heads"]; KV = cfg["num_key_value_heads"]
    HD = cfg["head_dim"]; E = cfg["num_experts"]; K = cfg["num_experts_per_tok"]
    F = cfg["moe_intermediate_size"]; Voc = cfg["vocab_size"]
    THETA = float(cfg["rope_theta"]); EPS = float(cfg["rms_norm_eps"])
    total_layers = cfg["num_hidden_layers"]

    from safetensors import safe_open
    # A1: 48 layers span ALL 16 shards. Opening shard 1 only was a shard-shaped converter:
    # it worked for the gate and would have failed on the first tensor outside that file.
    index = json.load(open(md / "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    _handles = {}
    def _h(name):
        sh = weight_map[name]
        if sh not in _handles:
            _handles[sh] = safe_open(str(md / sh), framework="pt")
        return _handles[sh]
    shard1 = None
    # A shard boundary can cut MID-LAYER: shard 1 contains layer 2's experts but not its norms,
    # so counting distinct layer indices overstates what is usable. Only COMPLETE layers count.
    keys = set(weight_map.keys())
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

    order = None
    group = 1
    if args.expert_order:
        spec = json.load(open(args.expert_order))
        order = spec["layers_order"]
        group = int(spec.get("group", args.group))
        if len(order) < L:
            raise SystemExit(f"expert order has {len(order)} layers, converting {L}")
        for l in range(L):
            if sorted(order[l]) != list(range(E)):
                raise SystemExit(f"expert order for layer {l} is not a permutation of 0..{E-1}")
        print(f"expert order: {args.expert_order} ({group} experts per region)")

    def t(name):
        # the tensor, from whichever shard holds it
        return _h(name).get_tensor(name).float().numpy()

    import importlib.util, sys as _s
    _spec = importlib.util.spec_from_file_location("format_q8", str(ROOT / "format/q8.py"))
    _q8 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_q8)

    w = V.V4Writer()
    tid = {}

    _src = {}

    def nbytes_for(shape, enc):
        n = 1
        for x in shape: n *= int(x)
        if enc == "q8_0":
            assert n % 32 == 0, (shape, n)
            return (n // 32) * 34
        return n * 4

    def add(name, shape, src_name, role, enc="f32", rows=None):
        """Register LAZILY: the payload is read and quantized at write time, one tensor at a time.

        Every tensor goes through this path, not just the experts. Materialising them would need
        the sum of all payloads (61 GB fp32, 16.2 GB at Q8) against 16 GB of RAM.
        """
        use = "q8_0" if (enc == "q8_0" and args.quant == "q8_0") else "f32"
        def source(_n=src_name, _u=use, _r=rows):
            if _r is None:
                a = np.ascontiguousarray(t(_n), dtype=np.float32)
            else:
                # read only the rows this chunk needs: the full table is 330 MB at Q8 and
                # materialising it once per chunk would dominate the conversion
                r0, r1 = _r
                a = np.ascontiguousarray(_h(_n).get_slice(_n)[r0:r1].float().numpy(),
                                         dtype=np.float32)
            return _q8.encode_q8_0(a)[0] if _u == "q8_0" else a.tobytes()
        tid[name] = w.add_tensor_lazy(name, shape, nbytes_for(shape, use),
                                     role=role, enc=use, source=source)
        _src[name] = (src_name, tuple(int(x) for x in shape), use)

    # embedding table (also the shard-1 input); the model's head is UNTIED
    embed_tids = []
    EMB = args.embed_rows
    if EMB and 0 < EMB < Voc:
        if (Voc + EMB - 1) // EMB > 254:
            raise SystemExit(f"--embed-rows {EMB} needs {(Voc+EMB-1)//EMB} regions but a block "
                             f"holds at most 255 refs; use >= {(Voc + 253)//254} rows")
        for c, r0 in enumerate(range(0, Voc, EMB)):
            r1 = min(r0 + EMB, Voc)
            add(f"embed.{c}", (r1 - r0, D), "model.embed_tokens.weight", V.R_EMBED,
                enc="q8_0", rows=(r0, r1))
            embed_tids.append(tid[f"embed.{c}"])
        print(f"  embed: {len(embed_tids)} regions of {EMB} rows")
    else:
        add("embed", (Voc, D), "model.embed_tokens.weight", V.R_EMBED, enc="q8_0")
        embed_tids = [tid["embed"]]
    for l in range(L):
        p = f"model.layers.{l}."
        add(f"ln1.{l}", (D,), p + "input_layernorm.weight", V.R_NORM)
        add(f"ln2.{l}", (D,), p + "post_attention_layernorm.weight", V.R_NORM)
        add(f"q_norm.{l}", (HD,), p + "self_attn.q_norm.weight", V.R_NORM)
        add(f"k_norm.{l}", (HD,), p + "self_attn.k_norm.weight", V.R_NORM)
        for nm, key in (("Wq", "q_proj"), ("Wk", "k_proj"), ("Wv", "v_proj"), ("Wo", "o_proj")):
            shp = (H * HD, D) if nm == "Wq" else (KV * HD, D) if nm in ("Wk", "Wv") else (D, H * HD)
            add(f"{nm}.{l}", shp, p + f"self_attn.{key}.weight", V.R_ATTN, enc="q8_0")
        add(f"Wgate.{l}", (E, D), p + "mlp.gate.weight", V.R_ROUTER)
        for e in (order[l] if order else range(E)):
            for nm, key in (("Wg", "gate_proj"), ("Wu", "up_proj"), ("Wd", "down_proj")):
                shp = (D, F) if nm == "Wd" else (F, D)
                add(f"expert.{l}.{e}.{nm}", shp, p + f"mlp.experts.{e}.{key}.weight",
                    V.R_EXPERT, enc="q8_0")

    # A2: `model.norm` (final RMSNorm) and `lm_head` live in the LAST shard and were never
    # emitted, because the converter only ever opened shard 1. Without them the artifact loads,
    # has 48 correct layers, and cannot produce a single logit — and this model's head is UNTIED,
    # so there is no embedding to fall back on. A silent hole, not a crash.
    add("final_norm", (D,), "model.norm.weight", V.R_NORM)
    add("lm_head", (Voc, D), "lm_head.weight", V.R_HEAD, enc="q8_0")
    w.add_block(V.KIND_EMBED, 0xFFFFFFFF, refs=embed_tids, params=[Voc, D, 0])
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

    if order is None:
        for e in range(L * E):
            l, ex = divmod(e, E)
            lo = min(tid[f"expert.{l}.{ex}.{n}"] for n in ("Wg", "Wu", "Wd"))
            hi = max(tid[f"expert.{l}.{ex}.{n}"] for n in ("Wg", "Wu", "Wd"))
            w.add_region(lo, hi + 1, kind=V.REGION_EXPERT, slot=l, mlo=ex, mhi=ex + 1)
    else:
        # One region per GROUP of consecutive experts in the emitted order. The members' tensors
        # are contiguous because the experts were emitted in that order, which is what lets a
        # region be one sequential read.
        for l in range(L):
            seq = order[l]
            for g0 in range(0, E, group):
                members = seq[g0:g0 + group]
                lo = min(tid[f"expert.{l}.{e}.Wg"] for e in members)
                hi = max(tid[f"expert.{l}.{e}.Wd"] for e in members)
                w.add_region(lo, hi + 1, kind=V.REGION_EXPERT, slot=l,
                             mlo=g0, mhi=g0 + len(members))
    for name, i in tid.items():
        if not name.startswith("expert."):
            w.add_region(i, i + 1, kind=V.REGION_TENSOR)

    order = list(range(len(w.tensors)))
    order = [i for i in order if w.tensors[i]["enc"] == "f32"] + \
            [i for i in order if w.tensors[i]["enc"] != "f32"]
    info = w.write(args.out, layout=order)
    if args.manifest:
        man = []
        for i, tt in enumerate(w.tensors):
            nm = tt["name"]
            if nm in _src:
                src, shp, use = _src[nm]
                man.append(dict(name=nm, src=src, shape=list(shp), enc=use,
                                offset=int(tt["offset"]), nbytes=int(w._nbytes(tt))))
        json.dump(dict(off_data=info["off_data"], tensors=man), open(args.manifest, "w"), indent=0)
        print(f"  manifest: {args.manifest} ({len(man)} tensors)")
    out = Path(args.out)
    print(f"convert: {out} ({out.stat().st_size/1e6:.0f} MB) | tensors {info['tensors']} "
          f"blocks {info['blocks']} regions {info['regions']}")
    print(f"  D={D} H={H}/{KV} HD={HD} experts={E} top{K} d_exp={F} vocab={Voc} layers={L}"
          + ("" if order is None else f" | {group} experts/region"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
