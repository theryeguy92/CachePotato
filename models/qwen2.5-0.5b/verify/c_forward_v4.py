#!/usr/bin/env python3
"""verify/c_forward_v4.py — run Qwen2.5 from the CPW1 v4 FILE (not the blob) and check logits.

This is the gate that closes the loop: values proven to be in the file (v4_values) and arithmetic
proven correct (c_forward) are only useful together if the ENGINE can get from the file to the
same logits. Geometry comes from the file's block table, so a wrong param ordering shows up here
rather than as a silently wrong model.
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REL_TOL = 5e-4

RUNNER = r'''
#include "blocks/qwen2.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    (void)argc;
    QW2 m; void *map = NULL; size_t maplen = 0;
    if (qwen2_load_v4(&m, argv[1], &map, &maplen)) return 1;
    FILE *f = fopen(argv[2], "rb");
    int T = atoi(argv[3]);
    int *tok = malloc(sizeof(int) * T);
    if (fread(tok, sizeof(int), T, f) != (size_t)T) return 1;
    fclose(f);
    float *logits = malloc(sizeof(float) * (size_t)T * m.vocab);
    if (qwen2_forward(&m, tok, T, logits)) return 1;
    f = fopen(argv[4], "wb");
    fwrite(logits, sizeof(float), (size_t)T * m.vocab, f);
    fclose(f);
    fprintf(stderr, "v4 loader: L=%d d=%d heads=%d/%d hd=%d d_ff=%d vocab=%d "
                    "theta=%.0f eps=%g\n", m.n_layers, m.d_model, m.n_heads, m.n_kv,
            m.head_dim, m.d_ff, m.vocab, m.rope_theta, m.rms_eps);
    free(logits); free(tok);
    return 0;
}
'''

def main():
    ap_file = sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen.v4cp"
    base = np.load(ROOT / "data/traces/qwen25_05b_hf.npz")
    ids = base["raw_ids"].astype(np.int32)
    want = base["raw_logits"]
    print("=" * 78); print("C FORWARD FROM THE v4 FILE vs recorded HF logits"); print("=" * 78)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "r.c"; src.write_text(RUNNER)
        exe = td / "r"
        b = subprocess.run(["cc", "-O3", "-march=native", "-Iengine", "-o", str(exe), str(src),
                            "engine/blocks/qwen2.c", "engine/kernels/kernels.c",
                            "engine/kernels/parallel.c", "-lm", "-lpthread"],
                           cwd=ROOT, capture_output=True, text=True)
        if b.returncode:
            print("build failed:", b.stderr[-400:]); return 1
        (td / "ids.bin").write_bytes(ids.tobytes())
        r = subprocess.run([str(exe), str(ROOT / ap_file), str(td / "ids.bin"),
                            str(len(ids)), str(td / "out.bin")], capture_output=True, text=True)
        if r.returncode:
            print("LOAD/RUN FAILED:", r.stderr[-400:]); return 1
        print("  " + r.stderr.strip())
        got = np.frombuffer((td / "out.bin").read_bytes(), dtype="<f4").reshape(want.shape)
    d = np.abs(got - want)
    rel = float(d.max() / np.abs(want).max())
    same = int((got.argmax(-1) == want.argmax(-1)).sum())
    print(f"  logits  : max|delta| {d.max():.3e}  rel {rel:.2e}  (tol {REL_TOL:.0e})")
    print(f"  argmax  : {same}/{want.shape[0]} positions agree")
    ok = rel < REL_TOL and same == want.shape[0]
    print()
    print("C FORWARD (v4): PASS" if ok else "C FORWARD (v4): FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
