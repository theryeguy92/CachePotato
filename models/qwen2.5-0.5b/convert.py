#!/usr/bin/env python3
"""convert.py — Qwen2.5-0.5B-Instruct: safetensors -> CPW1 v4.

Uses the shared writer (format/v4_writer.py) and the block params the spec already defines for
this architecture: `norm` kind=1 (RMSNorm), `attention` n_kv_heads + rope_theta, and
`dense_ffn` gated=1 (SwiGLU).

Tensors are stacked per-layer (L, ...) exactly as the GPT-2 port does, so the engine's per-layer
pointer arithmetic is the same shape of problem for both ports.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "format"))
import v4_writer as V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/qwen2.5-0.5b-instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", choices=["none", "q8_0"], default="none")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file
    md = ROOT / args.model
    cfg = json.load(open(md / "config.json"))
    sd = {k: v.to(torch.float32).numpy() for k, v in load_file(str(md / "model.safetensors")).items()}
    torch = None  # container only

    L, D = cfg["num_hidden_layers"], cfg["hidden_size"]
    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    HD = D // H
    F, Voc = cfg["intermediate_size"], cfg["vocab_size"]
    THETA = float(cfg.get("rope_theta", 10000.0))
    EPS = float(cfg["rms_norm_eps"])
    Z = np.zeros(D, dtype=np.float32)          # o_proj has no bias; spec's refs need one

    w = V.V4Writer()
    tid = {}
    def add(name, arr, role, enc="f32"):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        use = "q8_0" if (enc == "q8_0" and args.quant == "q8_0") else "f32"
        tid[name] = w.add_tensor(name, a, role=role, enc=use)

    def stack(nm):
        return np.stack([sd[f"model.layers.{l}.{nm}"].astype(np.float32) for l in range(L)])

    # The tied head is 151936 x 896 and is 27% of the model: quantizing it is the largest
    # single byte saving left. It is also the embedding table, so the LOOKUP path must
    # dequantize one row per token.
    add("embed", sd["model.embed_tokens.weight"], V.R_EMBED, enc="q8_0")
    add("norm", sd["model.norm.weight"], V.R_NORM)
    add("ln1", stack("input_layernorm.weight"), V.R_NORM)
    add("ln2", stack("post_attention_layernorm.weight"), V.R_NORM)
    for nm, role in (("Wq", V.R_ATTN), ("Wk", V.R_ATTN), ("Wv", V.R_ATTN), ("Wo", V.R_ATTN)):
        add(nm, stack(f"self_attn.{nm[1].lower()}_proj.weight"), role, enc="q8_0")
    for nm in ("bq", "bk", "bv"):
        add(nm, stack(f"self_attn.{nm[1].lower()}_proj.bias"), V.R_ATTN)
    add("bo", np.zeros((L, D), dtype=np.float32), V.R_ATTN)   # absent in the checkpoint
    add("Wg", stack("mlp.gate_proj.weight"), V.R_DENSE, enc="q8_0")
    add("Wu", stack("mlp.up_proj.weight"), V.R_DENSE, enc="q8_0")
    add("Wd", stack("mlp.down_proj.weight"), V.R_DENSE, enc="q8_0")

    w.add_block(V.KIND_EMBED, 0xFFFFFFFF, refs=[tid["embed"]], params=[Voc, D, 0])
    for l in range(L):
        w.add_block(V.KIND_ATTN, l,
                    refs=[tid[n] for n in ("Wq","bq","Wk","bk","Wv","bv","Wo","bo")],
                    params=[H, KV, HD, V.f32_bits(THETA), 0, 0])
        w.add_block(V.KIND_DENSE_FFN, l, refs=[tid[n] for n in ("Wg","Wu","Wd")],
                    params=[V.ACT_SILU, F, l, 0, 1])          # gated = 1
    # RMSNorm blocks: one per site, plus the final norm. The spec's norm block carries
    # (eps_bits, kind); kind 1 = rmsnorm.
    for l in range(L):
        w.add_block(V.KIND_NORM, l, refs=[tid["ln1"], tid["ln2"]],
                    params=[V.f32_bits(EPS), 1])
    w.add_block(V.KIND_NORM, L, refs=[tid["norm"]], params=[V.f32_bits(EPS), 1])

    for name, i in tid.items():
        w.add_region(i, i + 1, kind=V.REGION_TENSOR)

    order = list(range(len(w.tensors)))          # f32 first, then q8, is enforced below
    f32 = [i for i in order if w.tensors[i]["enc"] == "f32"]
    q8 = [i for i in order if w.tensors[i]["enc"] != "f32"]
    order = f32 + q8
    if args.shuffle:
        np.random.default_rng(7).shuffle(order)

    info = w.write(args.out, layout=order)
    out = Path(args.out)
    print(f"convert: {out} ({out.stat().st_size/1e6:.1f} MB) | tensors {info['tensors']} "
          f"blocks {info['blocks']} regions {info['regions']} | encodings {w.ENCODINGS}")
    print(f"  L={L} d={D} heads={H}/{KV} hd={HD} d_ff={F} vocab={Voc} theta={THETA} eps={EPS:g}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
