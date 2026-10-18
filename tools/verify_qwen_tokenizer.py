#!/usr/bin/env python3
"""verify_qwen_tokenizer.py — the C tokenizer must reproduce HF's ids EXACTLY on the corpus."""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include "tokenizers/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
int tokenizer_qwen_load_unicode(const char *path);
int tokenizer_qwen_encode(const Tokenizer *t, const char *text, int *ids, int max_ids);
int main(int argc, char **argv) {
    (void)argc;
    if (tokenizer_qwen_load_unicode("weights/qwen2.5-unicode.bin")) return 1;
    Tokenizer t;
    if (tokenizer_load(&t, "weights/qwen2.5-tokenizer.bin")) return 1;
    FILE *f = fopen(argv[1], "rb");
    static char buf[65536];
    size_t n = fread(buf, 1, sizeof buf - 1, f); buf[n] = 0; fclose(f);
    static int ids[16384];
    int k = tokenizer_qwen_encode(&t, buf, ids, 16384);
    FILE *o = fopen(argv[2], "wb");
    fwrite(ids, sizeof(int), (size_t)k, o); fclose(o);
    fprintf(stderr, "%d\n", k);
    return 0;
}
'''

def main():
    z = np.load(ROOT / "data/traces/qwen25_05b_tokenizer.npz")
    n_texts = int(z["n_texts"][0])
    print("=" * 78); print("QWEN TOKENIZER: C vs HF ids (exact required)"); print("=" * 78)
    bad = total = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "h.c").write_text(HARNESS)
        exe = td / "tk"
        b = subprocess.run(["cc", "-O2", "-Iengine", "-o", str(exe), str(td / "h.c"),
                            "engine/tokenizers/tokenizer.c", "engine/tokenizers/tokenizer_qwen.c"],
                           cwd=ROOT, capture_output=True, text=True)
        if b.returncode:
            print(b.stderr[-800:]); return 1
        for i in range(n_texts):
            text = np.asarray(z[f"text{i}"]).tobytes().decode("utf-8")
            want = z[f"ids{i}"].tolist()
            (td / "in.txt").write_bytes(text.encode("utf-8"))
            r = subprocess.run([str(exe), str(td / "in.txt"), str(td / "out.bin")],
                               cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                print(f"  text{i}: encoder failed"); bad += 1; continue
            got = np.frombuffer((td / "out.bin").read_bytes(), dtype="<i4").tolist()
            total += 1
            if got != want:
                bad += 1
                n = min(len(got), len(want))
                first = next((j for j in range(n) if got[j] != want[j]), n)
                print(f"  text{i}: MISMATCH len {len(got)} vs {len(want)}, first diff at {first}")
                print(f"     text  : {text[:50]!r}")
                print(f"     C     : {got[max(0,first-2):first+4]}")
                print(f"     HF    : {want[max(0,first-2):first+4]}")
            else:
                print(f"  text{i}: ok ({len(want)} ids)")
    print()
    print(f"QWEN TOKENIZER: {'PASS' if not bad else 'FAIL'} ({total-bad}/{total} exact)")
    return 0 if not bad else 1

if __name__ == "__main__":
    raise SystemExit(main())
