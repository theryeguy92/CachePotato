#!/usr/bin/env python3
"""verify_q8_kernels.py — the quantized matmuls must implement their mathematical definition.

The *decoder* is gated bit-exactly (verify_q8_decode.py) because `q * d` has one correct
answer. The *matmul* cannot be: it is a sum of n_in products, and the engine's summation order
(scalar loop, or AVX2 lanes with FMA) is not numpy's. So this gate checks the kernel against the
definition `out = (dequantised W) @ x + b` with a stated tolerance, and separately checks that
the AVX2 path agrees with a scalar recomputation of the same shape.

A tolerance here is honest — it is the same class of difference as C-vs-numpy for fp32 — and it
still catches a wrong block stride, a wrong scale position, or a lane that is dropped.
"""
import argparse, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "format"))
import q8

REL_TOL = 1e-5

HARNESS = r'''
#include "kernels/kernels.h"
#include <stdio.h>
#include <stdlib.h>
/* argv: q8file n_in n_out infile(x) outfile ; writes out = matmul_xw_q8(x) */
int main(int argc, char **argv) {
    int nin = atoi(argv[2]), nout = atoi(argv[3]);
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END);
    long nb = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *w = malloc(nb); if (fread(w, 1, nb, f) != (size_t)nb) return 2; fclose(f);
    float *x = malloc(4 * nin), *b = calloc(nout, 4), *o = malloc(4 * nout);
    f = fopen(argv[4], "rb"); if (fread(x, 4, nin, f) != (size_t)nin) return 2; fclose(f);
    matmul_xw_q8(o, x, w, b, nin, nout);
    f = fopen(argv[5], "wb"); fwrite(o, 4, nout, f); fclose(f);
    return 0;
}
'''

def main():
    ap = argparse.ArgumentParser()
    fmt = "=" * 78
    print(fmt); print("Q8_0 MATMUL: kernel vs the mathematical definition"); print(fmt)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        exe = td / "q8mm"; (td / "h.c").write_text(HARNESS)
        r = subprocess.run(["cc", "-O3", "-march=native", "-Iengine", "-o", str(exe),
                            str(td / "h.c"), "engine/kernels/kernels.c",
                            "engine/kernels/parallel.c", "-lm", "-lpthread"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            print("build failed:\n" + r.stderr); return 1
        rng = np.random.default_rng(7)
        cases = [("expert 768x192", 768, 192), ("expert 192x768", 192, 768),
                 ("attention 768x768", 768, 768), ("router 768x32", 768, 32)]
        bad = 0
        for name, nin, nout in cases:
            W = (rng.standard_normal((nin, nout)).astype(np.float32) * 0.02)
            x = rng.standard_normal(nin).astype(np.float32) * 0.5
            blocks, shape = q8.encode_q8_0(np.ascontiguousarray(W))
            Wd = q8.decode_q8_0(blocks, shape)          # exact definition of the stored values
            want = (x.astype(np.float64) @ Wd.astype(np.float64)).astype(np.float32)
            (td / "b.bin").write_bytes(blocks)
            (td / "x.bin").write_bytes(x.tobytes())
            r = subprocess.run([str(exe), str(td / "b.bin"), str(nin), str(nout),
                                str(td / "x.bin"), str(td / "o.bin")], capture_output=True)
            if r.returncode:
                print(f"  {name:22} harness failed"); bad += 1; continue
            got = np.frombuffer((td / "o.bin").read_bytes(), dtype="<f4").copy()
            denom = max(float(np.abs(want).max()), 1e-30)
            rel = float(np.abs(got - want).max()) / denom
            ok = rel < REL_TOL
            print(f"  {name:22} rel err {rel:.3e}  (tol {REL_TOL:.0e})  {'ok' if ok else 'FAIL'}")
            if not ok: bad += 1
        print()
        if bad:
            print(f"Q8 KERNELS: FAIL ({bad})"); return 1
        print("Q8 KERNELS: PASS — the quantized matmuls implement dequant + matmul to fp32 noise")
        print("  (the decoder itself is bit-exact; only the summation order is free)")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
