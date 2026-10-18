#!/usr/bin/env python3
"""verify/c_forward.py — does the C engine reproduce the recorded HuggingFace logits?

Packs the published weights into the flat blob qwen2.c reads, runs the C forward on the exact
token ids from the baseline, and compares. This is the gate that says the ENGINE can run this
architecture; the format/loader work comes after, so a mismatch here is unambiguously arithmetic.

Tolerance: 5e-4 relative, the same fp32-accumulation bound used elsewhere. The C sums in scalar
order and numpy uses blocked matmuls, so the two cannot be bit-identical — but a wrong RMSNorm,
RoPE pairing or GQA head mapping is orders of magnitude larger than this bound.
"""
import json, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REL_TOL = 5e-4

def main():
    md = ROOT / "weights/qwen2.5-0.5b-instruct"
    cfg = json.load(open(md / "config.json"))
    from safetensors.torch import load_file
    import torch
    sd = {k: v.to(torch.float32).numpy() for k, v in load_file(str(md / "model.safetensors")).items()}

    L = cfg["num_hidden_layers"]; D = cfg["hidden_size"]
    H = cfg["num_attention_heads"]; KV = cfg["num_key_value_heads"]
    HD = D // H; F = cfg["intermediate_size"]; V = cfg["vocab_size"]
    KVH = KV * HD

    order = [("model.embed_tokens.weight", (V, D)), ("model.norm.weight", (D,))]
    for name, shape in (("input_layernorm.weight", (D,)), ("post_attention_layernorm.weight", (D,))):
        order.append(("STACK:" + name, (L, *shape)))
    for name, shape in (("self_attn.q_proj.weight", (D, D)), ("self_attn.q_proj.bias", (D,)),
                        ("self_attn.k_proj.weight", (KVH, D)), ("self_attn.k_proj.bias", (KVH,)),
                        ("self_attn.v_proj.weight", (KVH, D)), ("self_attn.v_proj.bias", (KVH,)),
                        ("self_attn.o_proj.weight", (D, D)),
                        ("mlp.gate_proj.weight", (F, D)), ("mlp.up_proj.weight", (F, D)),
                        ("mlp.down_proj.weight", (D, F))):
        order.append(("STACK:" + name, (L, *shape)))

    blob = bytearray()
    blob += struct.pack("<7i", L, D, H, KV, HD, F, V)
    blob += struct.pack("<2f", float(cfg.get("rope_theta", 10000.0)), float(cfg["rms_norm_eps"]))
    for key, shape in order:
        if key.startswith("STACK:"):
            nm = key[6:]
            arr = np.stack([sd[f"model.layers.{l}.{nm}"].astype(np.float32) for l in range(L)])
        else:
            arr = sd[key].astype(np.float32)
        assert arr.shape == shape, (key, arr.shape, shape)
        blob += np.ascontiguousarray(arr, dtype="<f4").tobytes()

    base = np.load(ROOT / "data/traces/qwen25_05b_hf.npz")
    ids = base["raw_ids"].astype(np.int32)
    want = base["raw_logits"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "w.bin").write_bytes(bytes(blob))
        (td / "ids.bin").write_bytes(ids.tobytes())
        # Build the runner HERE so the gate can never test a stale binary — the suite has
        # already been fooled once by a stale build.
        exe = td / "qwen_run"
        src = td / "run.c"
        src.write_text("""#include "blocks/qwen2.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    (void)argc; QW2 m; float *owned = NULL;
    if (qwen2_load_blob(&m, argv[1], &owned)) return 1;
    int T = atoi(argv[2]);
    int *tok = malloc(sizeof(int) * T);
    FILE *f = fopen(argv[3], "rb");
    if (fread(tok, sizeof(int), T, f) != (size_t)T) return 1;
    fclose(f);
    float *logits = malloc(sizeof(float) * (size_t)T * m.vocab);
    if (qwen2_forward(&m, tok, T, logits)) return 1;
    f = fopen(argv[4], "wb"); fwrite(logits, sizeof(float), (size_t)T * m.vocab, f); fclose(f);
    fprintf(stderr, "qwen2: T=%d vocab=%d layers=%d heads=%d/%d hd=%d d_ff=%d\\n",
            T, m.vocab, m.n_layers, m.n_heads, m.n_kv, m.head_dim, m.d_ff);
    free(logits); free(tok); free(owned);
    return 0;
}""")
        b = subprocess.run(["cc", "-O3", "-march=native", "-Iengine", "-o", str(exe), str(src),
                            "engine/blocks/qwen2.c", "engine/kernels/kernels.c",
                            "engine/kernels/parallel.c", "-lm", "-lpthread"],
                           cwd=ROOT, capture_output=True, text=True)
        if b.returncode:
            print("build failed:", b.stderr[-300:]); return 1
        r = subprocess.run([str(exe), str(td / "w.bin"), str(len(ids)),
                            str(td / "ids.bin"), str(td / "out.bin")], capture_output=True, text=True)
        if r.returncode:
            print("C forward FAILED:", r.stderr[-400:]); return 1
        print(r.stderr.strip())
        got = np.frombuffer((td / "out.bin").read_bytes(), dtype="<f4").reshape(want.shape)

    d = np.abs(got - want)
    rel = float(d.max() / np.abs(want).max())
    print(f"  C vs HF logits: max|delta| {d.max():.3e}  rel {rel:.2e}  (tol {REL_TOL:.0e})")
    print(f"  argmax last position: C {int(got[-1].argmax())} | HF {int(want[-1].argmax())}")
    print(f"  argmax agreement     : {int((got.argmax(-1)==want.argmax(-1)).sum())}/{want.shape[0]}")
    ok = rel < REL_TOL and int(got[-1].argmax()) == int(want[-1].argmax())
    print()
    print("C FORWARD: PASS" if ok else "C FORWARD: FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
