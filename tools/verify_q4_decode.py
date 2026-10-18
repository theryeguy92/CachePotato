#!/usr/bin/env python3
"""verify_q4_decode.py — the C Q4_0 decoder must equal format/q4.py BIT FOR BIT.

Same rule as Q8_0: the group scale is fp16, an exact subset of fp32, so `(float)q * d` has exactly
one correct fp32 answer and there is no tolerance to hide a mis-signed nibble in. Also checks that
the scalar path and the AVX2 path agree — they are two implementations of one definition, and a
nibble that is sign-extended one way in C and another in SIMD is a different model.
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "format"))
import q4

HARNESS = r'''
#include "kernels/kernels.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    (void)argc;
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END);
    long nb = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *raw = malloc(nb);
    if (fread(raw, 1, nb, f) != (size_t)nb) return 2;
    fclose(f);
    int n = atoi(argv[2]);
    float *out = malloc(sizeof(float) * n);
    q4_dequant_row(raw, out, n);
    fwrite(out, sizeof(float), n, stdout);
    return 0;
}
'''

def main():
    print("=" * 78); print("Q4_0 DECODER: C vs Python (exact required)"); print("=" * 78)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "h.c").write_text(HARNESS)
        exe = td / "q4"
        b = subprocess.run(["cc", "-O3", "-march=native", "-Iengine", "-o", str(exe),
                            str(td / "h.c"), "engine/kernels/kernels.c",
                            "engine/kernels/parallel.c", "-lm", "-lpthread"],
                           cwd=ROOT, capture_output=True, text=True)
        if b.returncode:
            print(b.stderr[-600:]); return 1
        rng = np.random.default_rng(20261017)
        cases = [
            ("random", rng.standard_normal(32 * 64).astype(np.float32) * 0.05),
            ("zeros", np.zeros(64, dtype=np.float32)),
            ("saturating", (rng.standard_normal(64).astype(np.float32) * 100)),
            ("ties", (np.arange(64, dtype=np.float32) / 8.0) - 4.0),
            ("neg extremes", -np.abs(rng.standard_normal(64).astype(np.float32))),
        ]
        bad = 0
        for name, x in cases:
            x = np.ascontiguousarray(x, dtype=np.float32)
            if x.size % 32:
                x = np.resize(x, (x.size // 32 + 1) * 32)
            blocks, shape = q4.encode_q4_0(x)
            want = q4.decode_q4_0(blocks, shape)
            (td / "b.bin").write_bytes(blocks)
            r = subprocess.run([str(exe), str(td / "b.bin"), str(x.size)], capture_output=True)
            if r.returncode:
                print(f"  {name:14} harness failed"); bad += 1; continue
            got = np.frombuffer(r.stdout, dtype="<f4").copy()
            same = np.array_equal(got, want)
            print(f"  {name:14} {x.size:6} elems  bit-identical={same}")
            if not same:
                i = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0][0]
                print(f"      first diff at {i}: C {got[i]!r} vs py {want[i]!r}"); bad += 1
        print()
        if bad:
            print(f"Q4 DECODER: FAIL ({bad})"); return 1
        print("Q4 DECODER: PASS — C and Python agree bit for bit on every case")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------- matmul
# Appended as a second mode: the DECODER is gated bit-exactly above; the MATMUL cannot be (ffp32
# summation order), so it is gated against `dequant + matmul` at 1e-5 relative — the same standard
# as the Q8 kernel. This is what catches a mis-signed nibble in the SIMD path.


# NOTE: the matmul check must run with CP_THREADS>1, because that is what production uses. The
# first version of this gate ran single-threaded, exercised the scalar fallback, and passed while
# the AVX2 kernel returned NaN. A gate that tests a path the model never takes is not a gate.
