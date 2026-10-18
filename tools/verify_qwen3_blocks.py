#!/usr/bin/env python3
"""verify_qwen3_blocks.py — the C port of the three new blocks vs the shard-1 reference.

QK-norm, top-8 routing and the gated expert SwiGLU, in C, against targets taken from real weights.
Tolerance 2e-5 relative: the C sums in scalar order and numpy uses blocked matmuls, so they cannot
be bit-identical — but a misplaced QK-norm or a wrong expert index is orders of magnitude larger.
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REL = 2e-5
D, H, KV, HD, E, K = 2048, 32, 4, 128, 128, 8

RUNNER = r'''
#include "blocks/qwen3_moe.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    (void)argc;
    QW3Layer m; float *owned = NULL;
    if (qwen3_load_blob(&m, argv[1], &owned)) return 1;
    int T = atoi(argv[2]);
    FILE *f = fopen(argv[3], "rb");
    float *x = malloc(sizeof(float) * (size_t)T * m.d_model);
    if (fread(x, sizeof(float), (size_t)T * m.d_model, f) != (size_t)T * m.d_model) return 1;
    fclose(f);
    QW3Out o;
    if (qwen3_layer_alloc(&o, &m, T)) return 1;
    if (qwen3_layer(&m, x, T, &o)) return 1;
    f = fopen(argv[4], "wb");
    fwrite(o.attn, sizeof(float), (size_t)T * m.d_model, f);
    fwrite(o.mlp, sizeof(float), (size_t)T * m.d_model, f);
    fwrite(o.router, sizeof(float), (size_t)T * m.n_experts, f);
    fwrite(o.topk_idx, sizeof(int), (size_t)T * m.topk, f);
    fclose(f);
    return 0;
}'''

def main():
    from safetensors import safe_open
    z = np.load(ROOT / "data/traces/qwen3_30b_a3b_shard1.npz")
    ids = z["input_ids"]; T = len(ids)
    shard = str(ROOT / "weights/qwen3-30b-a3b/model-00001-of-00016.safetensors")
    f = safe_open(shard, framework="pt")
    W = lambda n: f.get_tensor("model.layers.0." + n).float().numpy()
    embed = f.get_tensor("model.embed_tokens.weight").float().numpy()
    x = embed[ids].astype(np.float32)
    print("=" * 78); print("QWEN3 C BLOCKS vs the numpy reference targets"); print("=" * 78)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        blob = bytearray()
        blob += np.array([D, H, KV, HD, D and 768, E, K, 151936], dtype=np.int32).tobytes()
        blob += np.array([1e6, 1e-6], dtype=np.float32).tobytes()
        for a in (W("input_layernorm.weight"), W("post_attention_layernorm.weight"),
                  W("self_attn.q_norm.weight"), W("self_attn.k_norm.weight"),
                  W("self_attn.q_proj.weight"), W("self_attn.k_proj.weight"),
                  W("self_attn.v_proj.weight"), W("self_attn.o_proj.weight"),
                  W("mlp.gate.weight")):
            blob += np.ascontiguousarray(a, dtype="<f4").tobytes()
        for key in ("gate_proj", "up_proj", "down_proj"):
            blob += np.ascontiguousarray(np.stack([
                f.get_tensor(f"model.layers.0.mlp.experts.{e}.{key}.weight").float().numpy()
                for e in range(E)]), dtype="<f4").tobytes()
        (td / "w.bin").write_bytes(bytes(blob))
        (td / "x.bin").write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())
        src = td / "r.c"; src.write_text(RUNNER)
        exe = td / "q3"
        b = subprocess.run(["cc", "-O3", "-march=native", "-Iengine", "-o", str(exe), str(src),
                            "engine/blocks/qwen3_moe.c", "engine/kernels/kernels.c",
                            "engine/kernels/parallel.c", "-lm", "-lpthread"],
                           cwd=ROOT, capture_output=True, text=True)
        if b.returncode:
            print(b.stderr[-600:]); return 1
        r = subprocess.run([str(exe), str(td / "w.bin"), str(T), str(td / "x.bin"),
                            str(td / "out.bin")], capture_output=True, text=True)
        if r.returncode:
            print("C layer FAILED:", r.stderr[-400:]); return 1
        raw = (td / "out.bin").read_bytes()
        n = T * D
        attn = np.frombuffer(raw, dtype="<f4", count=n, offset=0).reshape(T, D)
        mlp = np.frombuffer(raw, dtype="<f4", count=n, offset=n * 4).reshape(T, D)
        router = np.frombuffer(raw, dtype="<f4", count=T * E, offset=2 * n * 4).reshape(T, E)
        top = np.frombuffer(raw, dtype="<i4", count=T * K, offset=(2 * n + T * E) * 4).reshape(T, K)

    bad = 0
    def rep(name, got, want):
        nonlocal bad
        w = np.asarray(want, np.float32).reshape(np.shape(got))
        d = float(np.abs(got - w).max()); peak = max(float(np.abs(w).max()), 1e-30)
        ok = d / peak < REL
        print(f"  {name:16} max|d| {d:.3e}  rel {d/peak:.2e}  {'ok' if ok else 'FAIL'}")
        bad += not ok
    rep("attention", attn, z["L0.self_attn"][0])
    rep("router logits", router, z["L0.mlp.gate"])
    rep("moe output", mlp, z["L0.mlp"][0])
    same = all(set(a) == set(b) for a, b in zip(top.tolist(), z["L0.top8"].tolist()))
    print(f"  {'top-8 selection':16} {'SAME SET' if same else 'DIFFERS'}")
    bad += (not same)
    print()
    print("QWEN3 C BLOCKS: PASS" if not bad else f"QWEN3 C BLOCKS: FAIL ({bad})")
    return 0 if not bad else 1

if __name__ == "__main__":
    raise SystemExit(main())
