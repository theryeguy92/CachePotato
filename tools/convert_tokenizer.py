"""convert_tokenizer.py — GPT-2 BPE (vocab.json + merges.txt) -> CPT1 binary.

CPT1 format, little-endian:
    [magic u32 "CPT1"][version u32=1][n_vocab u32][n_merges u32]
    vocab:   n_vocab x { len u16, bytes }        (id = index; raw BYTES,
                                                  unicode mapping inverted)
    merges:  n_merges x { id1 u32, id2 u32, id_merged u32 }
             (file order = merge rank, ascending)

The C tokenizer works purely on raw bytes — GPT-2's bytes-to-unicode
mapping is inverted HERE, once, in Python where it's trivial.

Run from repo root:  python3 tools/convert_tokenizer.py
"""

import json
import os
import struct

from huggingface_hub import hf_hub_download

out = os.path.join(os.path.dirname(__file__), "..", "weights", "tokenizer.bin")

vocab_path = hf_hub_download("gpt2", "vocab.json")
merges_path = hf_hub_download("gpt2", "merges.txt")
vocab = json.load(open(vocab_path))  # token_str -> id
merge_lines = open(merges_path, encoding="utf-8").read().splitlines()

# invert GPT-2's bytes_to_unicode mapping: vocab strings live in unicode
# space; we want the raw bytes each token represents
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

def token_to_bytes(s):
    return bytes(byte_decoder[ch] for ch in s)

n_vocab = max(vocab.values()) + 1
vocab_bytes = [b""] * n_vocab
for s, i in vocab.items():
    vocab_bytes[i] = token_to_bytes(s)

# merges.txt: first line is a version header, then one "A B" pair per
# line, in rank order
rows = []
for line in merge_lines:
    if line.startswith("#") or not line.strip():
        continue
    a, b = line.split(" ")
    merged = a + b
    if a not in vocab or b not in vocab or merged not in vocab:
        continue
    rows.append((vocab[a], vocab[b], vocab[merged]))

MAGIC = int.from_bytes(b"CPT1", "little")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as f:
    f.write(struct.pack("<IIII", MAGIC, 1, n_vocab, len(rows)))
    for bs in vocab_bytes:
        f.write(struct.pack("<H", len(bs)))
        f.write(bs)
    for id1, id2, idm in rows:
        f.write(struct.pack("<III", id1, id2, idm))

print(f"wrote {os.path.normpath(out)}: {n_vocab} vocab entries, {len(rows)} merges")
print(f"  sanity: id 464 = {vocab_bytes[464]!r} ('The'),"
      f" id 318 = {vocab_bytes[318]!r} (' is'),"
      f" id 262 = {vocab_bytes[262]!r} (' the')")
