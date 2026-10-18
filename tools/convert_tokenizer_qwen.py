#!/usr/bin/env python3
"""convert_tokenizer_qwen.py — Qwen2 tokenizer.json -> CPT1 binary + Unicode class table.

CPT1 (see tools/convert_tokenizer.py) already stores the vocab as RAW BYTES and the merges as
(id1, id2, out), which is exactly what a byte-level BPE needs. Qwen uses the SAME bytes-to-unicode
mapping as GPT-2, so the inversion is identical and the C side needs no change to its BPE loop.

What Qwen DOES need and GPT-2 does not:
  1. a Unicode-aware pre-tokenizer (`\\p{L}`, `\\p{N}`), so the split rules need real categories;
  2. single-character `\\p{N}` (a run of digits splits per digit), not runs;
  3. NFC normalization;
  4. special tokens matched BEFORE BPE.

The category table is emitted as sorted codepoint RANGES from Python's own unicodedata, so the C
side does a binary search instead of carrying hand-copied tables. Two classes are needed:
L (letter) and N (number); everything else is "other". The Unicode version used here is Python's,
which need not match HF's Rust tables exactly — so the tokenizer gate (zero mismatches on the
recorded corpus) is what decides whether they agree in practice, not this comment.

Writes: weights/qwen2.5-tokenizer.bin  (CPT1)
        weights/qwen2.5-unicode.bin    (ranges)
"""
import json
import struct
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "weights/qwen2.5-0.5b-instruct/tokenizer.json"
OUT_TOK = ROOT / "weights/qwen2.5-tokenizer.bin"
OUT_UNI = ROOT / "weights/qwen2.5-unicode.bin"

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def cls_of(cp):
    """0 = other, 1 = letter (\\p{L}), 2 = number (\\p{N})."""
    c = unicodedata.category(chr(cp))
    if c.startswith("L"): return 1
    if c.startswith("N"): return 2
    return 0

def ranges():
    out = []
    cur = None
    for cp in range(0x110000):
        k = cls_of(cp)
        if cp >= 0xD800 and cp <= 0xDFFF: k = 0     # surrogates: never valid in UTF-8
        if cur and cur[2] == k and cur[1] + 1 == cp:
            cur = (cur[0], cp, k)
        else:
            if cur: out.append(cur)
            cur = (cp, cp, k)
    out.append(cur)
    return out

def main():
    t = json.loads(SRC.read_text())
    b2u = bytes_to_unicode()
    vocab = t["model"]["vocab"]           # token_str -> id
    n_vocab = max(vocab.values()) + 1
    # added/special tokens may have ids beyond the BPE vocab
    specials = {}
    for a in t.get("added_tokens", []):
        specials[a["id"]] = a["content"]
        n_vocab = max(n_vocab, a["id"] + 1)
    raw = [b""] * n_vocab
    for tok, tid in vocab.items():
        raw[tid] = "".join(ch for ch in tok).encode("utf-8")   # already byte-space chars
    # invert the mapping: each char in the token string maps back to one byte
    u2b = {v: k for k, v in b2u.items()}
    for tid, s in enumerate(raw):
        if s:
            try:
                raw[tid] = bytes(u2b[ch] for ch in s.decode("utf-8"))
            except (KeyError, UnicodeDecodeError):
                pass
    for tid, content in specials.items():
        raw[tid] = content.encode("utf-8")

    merges = t["model"]["merges"]
    if isinstance(merges[0], str):
        merges = [m.split(" ") for m in merges]

    with open(OUT_TOK, "wb") as f:
        f.write(struct.pack("<4I", 0x31545043, 1, n_vocab, len(merges)))
        for s in raw:
            f.write(struct.pack("<H", len(s)))
            f.write(s)
        for a, b in merges:
            f.write(struct.pack("<3I", vocab[a], vocab[b], vocab[a + b]))

    rg = ranges()
    with open(OUT_UNI, "wb") as f:
        f.write(struct.pack("<2I", 0x31494E55, len(rg)))   # "UNI1"
        for lo, hi, k in rg:
            f.write(struct.pack("<3I", lo, hi, k))

    print(f"wrote {OUT_TOK.name}: {n_vocab} tokens, {len(merges)} merges, "
          f"{OUT_TOK.stat().st_size/1e6:.1f} MB")
    print(f"wrote {OUT_UNI.name}: {len(rg)} codepoint ranges, {OUT_UNI.stat().st_size/1e3:.0f} KB")
    print(f"  specials: {len(specials)} (first: {list(specials.items())[:2]})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
