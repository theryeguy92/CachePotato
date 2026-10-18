#!/usr/bin/env python3
"""verify_qwen_kernels.py — the C primitives must match the independent numpy reference.

Each of RMSNorm, RoPE and SiLU is a place where a plausible-looking implementation is wrong:
RMSNorm written as LayerNorm, RoPE rotating the wrong pair or using the duplicated table twice,
SiLU confused with GELU. All three still produce fluent text. So each is checked against
models/qwen2.5-0.5b/reference.py on random data, with a stated tolerance.
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/qwen2.5-0.5b"))
import reference as R

HARNESS = r'''
#include "kernels/kernels.h"
#include <stdio.h>
#include <stdlib.h>
/* argv: mode in out [dims...]; binary float I/O so the comparison is exact about layout */
int main(int argc, char **argv) {
    (void)argc;
    int mode = atoi(argv[1]);
    FILE *fi = fopen(argv[2], "rb"), *fo = fopen(argv[3], "wb");
    if (mode == 0) {                 /* rmsnorm: in = n + n + n (x, g, out) sized by argv */
        int n = atoi(argv[4]);
        float *x = malloc(4 * n), *g = malloc(4 * n), *o = malloc(4 * n);
        if (fread(x, 4, n, fi) != (size_t)n || fread(g, 4, n, fi) != (size_t)n) return 2;
        rmsnorm(o, x, g, n, 1e-6f);
        fwrite(o, 4, n, fo);
    } else if (mode == 1) {          /* rope: T, nq, nkv, hd, theta */
        int T = atoi(argv[4]), nq = atoi(argv[5]), nkv = atoi(argv[6]), hd = atoi(argv[7]);
        float theta = atof(argv[8]);
        size_t nq_n = (size_t)T * nq * hd, nk_n = (size_t)T * nkv * hd;
        float *q = malloc(4 * nq_n), *k = malloc(4 * nk_n);
        if (fread(q, 4, nq_n, fi) != nq_n || fread(k, 4, nk_n, fi) != nk_n) return 2;
        rope_apply(q, k, T, nq, nkv, hd, theta);
        fwrite(q, 4, nq_n, fo); fwrite(k, 4, nk_n, fo);
    } else {                         /* silu */
        int n = atoi(argv[4]);
        float *x = malloc(4 * n), *o = malloc(4 * n);
        if (fread(x, 4, n, fi) != (size_t)n) return 2;
        for (int i = 0; i < n; i++) o[i] = silu(x[i]);
        fwrite(o, 4, n, fo);
    }
    fclose(fi); fclose(fo);
    return 0;
}
'''

def build(td):
    exe = Path(td) / "qk"
    (Path(td) / "h.c").write_text(HARNESS)
    r = subprocess.run(["cc", "-O2", "-Iengine", "-o", str(exe), str(Path(td) / "h.c"),
                        "engine/kernels/kernels.c", "engine/kernels/parallel.c",
                        "-lm", "-lpthread"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr); raise SystemExit(1)
    return exe

def main():
    print("=" * 78); print("QWEN KERNELS: C vs the independent numpy reference"); print("=" * 78)
    rng = np.random.default_rng(20261017)
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        exe = build(td)
        d = Path(td)
        def run(mode, arrays, dims):
            fi, fo = d / "in.bin", d / "out.bin"
            fi.write_bytes(b"".join(a.astype("<f4").tobytes() for a in arrays))
            r = subprocess.run([str(exe), str(mode), str(fi), str(fo), *map(str, dims)],
                               capture_output=True)
            assert r.returncode == 0, r.stderr
            return np.frombuffer(fo.read_bytes(), dtype="<f4")

        # RMSNorm
        n = 896
        x = rng.standard_normal(n).astype(np.float32)
        g = (1 + 0.1 * rng.standard_normal(n)).astype(np.float32)
        got = run(0, [x, g], [n])
        want = R.rmsnorm(x[None, :], g)[0]
        rel = float(np.abs(got - want).max() / np.abs(want).max())
        ok = rel < 1e-6
        print(f"  {'RMSNorm':10} rel {rel:.2e} (tol 1e-06)  {'ok' if ok else 'FAIL'}")
        bad += not ok

        # RoPE
        T, nq, nkv, hd = 5, 14, 2, 64
        q = rng.standard_normal(T * nq * hd).astype(np.float32)
        k = rng.standard_normal(T * nkv * hd).astype(np.float32)
        got = run(1, [q, k], [T, nq, nkv, hd, 1e6])
        cos, sin = R.rope_tables(T, hd, 1e6)
        wq, wk = R.apply_rope(q.reshape(T, nq, hd).astype(np.float64),
                              k.reshape(T, nkv, hd).astype(np.float64),
                              cos.astype(np.float64), sin.astype(np.float64))
        idx = T * nq * hd
        relq = float(np.abs(got[:idx] - wq.reshape(-1)).max() / np.abs(wq).max())
        relk = float(np.abs(got[idx:] - wk.reshape(-1)).max() / np.abs(wk).max())
        ok = relq < 1e-5 and relk < 1e-5
        print(f"  {'RoPE':10} q rel {relq:.2e}  k rel {relk:.2e} (tol 1e-05)  "
              f"{'ok' if ok else 'FAIL'}")
        bad += not ok

        # SiLU
        x = (rng.standard_normal(4096) * 3).astype(np.float32)
        got = run(2, [x], [x.size])
        want = (x / (1.0 + np.exp(-x))).astype(np.float32)
        rel = float(np.abs(got - want).max() / np.abs(want).max())
        ok = rel < 1e-6
        print(f"  {'SiLU':10} rel {rel:.2e} (tol 1e-06)  {'ok' if ok else 'FAIL'}")
        bad += not ok

    print()
    print("QWEN KERNELS: PASS" if not bad else f"QWEN KERNELS: FAIL ({bad})")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
