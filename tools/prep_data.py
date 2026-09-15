"""prep_data.py — tokenize TinyStories into flat uint16 token-id binaries
for the C trainer (Phase 5).

Format: raw little-endian uint16 stream, no header (llama2.c convention).
GPT-2 vocab 50257 < 65536, so uint16 suffices. Tokenizer: GPT-2 BPE via
transformers (Rust-fast); the C engine's tokenizer is for serving, not for
bulk dataset prep.

Output: data/tinystories_train.bin, data/tinystories_valid.bin
Run from repo root:  python3 tools/prep_data.py [max_mb_per_split]
"""

import os
import sys

import numpy as np
from transformers import GPT2TokenizerFast

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
CHUNK = 1 << 20  # 1MB of text per tokenizer call

tok = GPT2TokenizerFast.from_pretrained("gpt2")


def tokenize_file(src, dst, max_bytes=None):
    total_ids = 0
    with open(src, encoding="utf-8", errors="ignore") as f, open(dst, "wb") as out:
        while True:
            text = f.read(CHUNK)
            if not text:
                break
            ids = tok.encode(text)
            np.array(ids, dtype="<u2").tofile(out)
            total_ids += len(ids)
            if max_bytes and f.tell() >= max_bytes:
                break
    print(f"  {os.path.basename(dst)}: {total_ids:,} tokens")
    return total_ids


def main():
    max_mb = int(sys.argv[1]) if len(sys.argv) > 1 else None
    max_bytes = max_mb * (1 << 20) if max_mb else None
    for split, src_name in [("valid", "TinyStories-valid.txt"),
                            ("train", "TinyStories-train.txt")]:
        src = os.path.join(DATA, src_name)
        if not os.path.exists(src):
            print(f"  skip {src_name} (not downloaded yet)")
            continue
        print(f"tokenizing {src_name}" + (f" (first {max_mb}MB)" if max_mb else ""))
        tokenize_file(src, os.path.join(DATA, f"tinystories_{split}.bin"), max_bytes)


if __name__ == "__main__":
    main()
