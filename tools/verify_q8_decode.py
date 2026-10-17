#!/usr/bin/env python3
"""verify_q8_decode.py — the C Q8_0 decoder must equal format/q8.py BIT FOR BIT.

Why exact and not a tolerance: the group scale is fp16, an exact subset of fp32, so
`(float)q[i] * d` has exactly one correct fp32 result. A tolerance here would hide a
decoder that is one LSB off — which for a quantized weight matrix is not a rounding detail,
it is a different model that still runs.

Builds a tiny harness against engine/kernels/kernels.c so the comparison is against the
real decoder, not a copy of it.
"""
import argparse, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "format"))
import q8

HARNESS = r'''
#include "kernels/kernels.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *raw = malloc(n);
    if (fread(raw, 1, n, f) != (size_t)n) return 2;
    fclose(f);
    int count = atoi(argv[2]);
    float *out = malloc(sizeof(float) * count);
    q8_dequant_row(raw, out, count);
    fwrite(out, sizeof(float), count, stdout);
    return 0;
}
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./build/verify_q8_decode")
    args = ap.parse_args()
    print("=" * 78); print("Q8_0 DECODER: C vs Python (exact required)"); print("=" * 78)
    with tempfile.TemporaryDirectory() as td:
        exe = Path(td) / "q8dec"
        src = Path(td) / "h.c"; src.write_text(HARNESS)
        cc = subprocess.run(["cc", "-O2", "-Iengine", "-o", str(exe), str(src),
                             "engine/kernels/kernels.c", "engine/kernels/parallel.c", "-lm", "-lpthread"],
                            cwd=ROOT, capture_output=True, text=True)
        if cc.returncode:
            print("build failed:\n" + cc.stderr); return 1
        rng = np.random.default_rng(20261017)
        cases = [
            ("random small", (rng.standard_normal(32 * 40).astype(np.float32) * 0.05)),
            ("random large", (rng.standard_normal(768 * 192).astype(np.float32) * 0.02)),
            ("all zeros", np.zeros(64, dtype=np.float32)),
            ("exact powers", (np.arange(64, dtype=np.float32) / 64.0)),
            ("large dynamic range", np.concatenate([rng.standard_normal(32).astype(np.float32),
                                                    rng.standard_normal(32).astype(np.float32) * 1000]),
             ),
        ]
        bad = 0
        for name, x in cases:
            x = np.ascontiguousarray(x, dtype=np.float32)
            if x.size % 32:
                x = np.resize(x, (x.size // 32 + 1) * 32)
            blocks, shape = q8.encode_q8_0(x)
            want = q8.decode_q8_0(blocks, shape)
            (Path(td) / "b.bin").write_bytes(blocks)
            r = subprocess.run([str(exe), str(Path(td) / "b.bin"), str(x.size)],
                               capture_output=True)
            if r.returncode:
                print(f"  {name:22} harness failed"); bad += 1; continue
            got = np.frombuffer(r.stdout, dtype="<f4").copy()
            same = np.array_equal(got, want)
            nbits = 0 if same else int((got.view(np.uint32) != want.view(np.uint32)).sum())
            print(f"  {name:22} {x.size:7} elems  bit-identical={same}"
                  + ("" if same else f"  DIFFERING {nbits}"))
            if not same:
                i = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0][0]
                print(f"      first diff at {i}: C {got[i]!r} vs py {want[i]!r}")
                bad += 1
        print()
        if bad:
            print(f"Q8 DECODER: FAIL ({bad} case(s))"); return 1
        print("Q8 DECODER: PASS — C and Python agree bit for bit on every case")
        print("  (fp16 scale is an exact subset of fp32, so 'q * d' has one correct answer;")
        print("   this gate has no tolerance to hide a one-LSB mistake in.)")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
