#!/usr/bin/env python3
"""verify_qwen3_v4.py — A5: the FORMAT path must equal the BLOB path, bit for bit.

Both loaders are handed the same weights, so any difference is plumbing: a wrong table offset, a
mis-read encoding, an expert index off by one. That is why this compares against an fp32 v4 file
and not the Q8 one — with Q8 in the middle the difference would be quantization, and the test would
no longer be able to distinguish "the parser is wrong" from "Q8 is lossy".
"""
import subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = r'''
#include "blocks/qwen3_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) { (void)argc;
    QW3Layer m; memset(&m, 0, sizeof m);
    float *owned = NULL;
    if (!strcmp(argv[1], "blob")) {
        if (qwen3_load_blob(&m, argv[2], &owned)) return 1;
    } else {
        if (qwen3_load_v4(&m, argv[2], atoi(argv[3]))) return 1;
    }
    int T = atoi(argv[4]);
    FILE *f = fopen(argv[5], "rb");
    float *x = malloc(sizeof(float) * (size_t)T * m.d_model);
    if (fread(x, sizeof(float), (size_t)T * m.d_model, f) != (size_t)T * m.d_model) return 1;
    fclose(f);
    QW3Out o;
    if (qwen3_layer_alloc(&o, &m, T)) return 1;
    if (qwen3_layer(&m, x, T, &o)) return 1;
    f = fopen(argv[6], "wb");
    fwrite(o.attn, 4, (size_t)T * m.d_model, f);
    fwrite(o.mlp,  4, (size_t)T * m.d_model, f);
    fwrite(o.router, 4, (size_t)T * m.n_experts, f);
    fwrite(o.topk_idx, sizeof(int), (size_t)T * m.topk, f);
    fclose(f);
    return 0;
}'''

def main():
    import numpy as np
    from safetensors import safe_open
    v4path = sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen3_l1f.v4cp"
    z = np.load(ROOT / "data/traces/qwen3_30b_a3b_shard1.npz")
    ids = z["input_ids"]; T = len(ids)
    f = safe_open(str(ROOT / "weights/qwen3-30b-a3b/model-00001-of-00016.safetensors"), framework="pt")
    W = lambda n: f.get_tensor("model.layers.0." + n).float().numpy()
    x = f.get_tensor("model.embed_tokens.weight").float().numpy()[ids].astype(np.float32)
    D, H, KV, HD, E, K, F = 2048, 32, 4, 128, 128, 8, 768
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        blob = bytearray()
        blob += np.array([D, H, KV, HD, F, E, K, 151936], dtype=np.int32).tobytes()
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
        (td / "r.c").write_text(RUNNER)
        exe = td / "q3v"
        b = subprocess.run(["cc", "-O3", "-march=native", "-Iengine", "-o", str(exe), str(td / "r.c"),
                            "engine/blocks/qwen3_moe.c", "engine/kernels/kernels.c",
                            "engine/kernels/parallel.c", "-lm", "-lpthread"],
                           cwd=ROOT, capture_output=True, text=True)
        if b.returncode:
            print(b.stderr[-800:]); return 1
        outs = {}
        for mode in ("blob", "v4"):
            args = [str(exe), mode, str(td / "w.bin") if mode == "blob" else str(ROOT / v4path),
                    "0", str(T), str(td / "x.bin"), str(td / f"{mode}.bin")]
            r = subprocess.run(args, capture_output=True, text=True)
            if r.returncode:
                print(f"{mode} path FAILED:", r.stderr[-400:]); return 1
            outs[mode] = (td / f"{mode}.bin").read_bytes()
    print("=" * 78); print("A5: CPW1 v4 path vs flat blob path (same f32 weights)"); print("=" * 78)
    same = outs["blob"] == outs["v4"]
    print(f"  outputs: {len(outs['blob'])} bytes each, bit-identical = {same}")
    if not same:
        a, c = outs["blob"], outs["v4"]
        i = next((k for k in range(min(len(a), len(c))) if a[k] != c[k]), None)
        print(f"  first differing byte: {i}")
    print()
    print("A5: " + ("PASS — the format path computes what the blob path computes"
                    if same else "FAIL — plumbing, not arithmetic"))
    return 0 if same else 1

if __name__ == "__main__":
    raise SystemExit(main())
