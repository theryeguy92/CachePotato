"""export_toy_weights.py — dump the forward_by_hand toy weights to our binary
format (weights/toy.bin) so the C engine can mmap them.

Format ("CPW1"), all little-endian:
    [magic u32][reserved u32][vocab i32][d_model i32][n_heads i32][d_ff i32][n_layer i32]
    then float32 tensors in this exact order:
    token_emb, Wq, Wk, Wv, Wo, W1, W2, ln1_g, ln2_g, lnf_g, unemb

The C loader validates magic + total file size, so a wrong or stale file
fails LOUDLY instead of slicing silently (the llama2.c lesson).

Run from repo root:  python3 tools/export_toy_weights.py
"""

import os
import struct

import numpy as np

rng = np.random.default_rng(42)
VOCAB, D, H, D_FF, N_LAYER = 8, 4, 2, 8, 1


def rand_weight(*shape):
    return rng.standard_normal(shape, dtype=np.float32) * 0.2


# SAME draw order as forward_by_hand.py — do not reorder.
W = [
    ("token_emb", rand_weight(VOCAB, D)),
    ("Wq", rand_weight(D, D)),
    ("Wk", rand_weight(D, D)),
    ("Wv", rand_weight(D, D)),
    ("Wo", rand_weight(D, D)),
    ("W1", rand_weight(D, D_FF)),
    ("W2", rand_weight(D_FF, D)),
    ("ln1_g", np.ones(D, dtype=np.float32)),
    ("ln2_g", np.ones(D, dtype=np.float32)),
    ("lnf_g", np.ones(D, dtype=np.float32)),
    ("unemb", rand_weight(VOCAB, D)),
]

MAGIC = int.from_bytes(b"CPW1", "little")

out = os.path.join(os.path.dirname(__file__), "..", "weights", "toy.bin")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as f:
    f.write(struct.pack("<II5i", MAGIC, 0, VOCAB, D, H, D_FF, N_LAYER))
    for _, a in W:
        f.write(a.astype("<f4").tobytes())

n_floats = sum(a.size for _, a in W)
print(f"wrote {os.path.normpath(out)}: {28 + 4 * n_floats} bytes ({n_floats} floats)")
d = dict(W)
print("ground truth for the C loader checkpoint:")
print("  token_emb[5] =", d["token_emb"][5])
print("  unemb[0]     =", d["unemb"][0])
