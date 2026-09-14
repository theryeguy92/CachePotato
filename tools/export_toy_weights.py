"""export_toy_weights.py — dump toy weights (same seed as forward_by_hand.py)
to our binary format v2 (weights/toy.bin) so the C engine can mmap them.

Format ("CPW1" v2), all little-endian:
    header: [magic u32][version u32=2][vocab i32][d_model i32][n_heads i32]
            [d_ff i32][n_layer i32][n_ctx i32]  = 32 bytes
    float32 tensors in this exact order (per-tensor-type across layers):
    token_emb, pos_emb,
    ln1_g, ln1_b, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
    ln2_g, ln2_b, W1, b1, W2, b2,
    lnf_g, lnf_b
    (unembedding is TIED to token_emb — not stored)

Toy values: drawn weights match forward_by_hand.py's seed/order; all biases
and pos_emb are ZEROS (no-ops), so the toy ground truth needs no bias terms.

Run from repo root:  python3 tools/export_toy_weights.py
"""

import os
import struct

import numpy as np

rng = np.random.default_rng(42)
VOCAB, D, H, D_FF, N_LAYER, N_CTX = 8, 4, 2, 8, 1, 8


def rand_weight(*shape):
    return rng.standard_normal(shape, dtype=np.float32) * 0.2


# SAME draw order as forward_by_hand.py (minus unemb, now tied) — do not reorder.
token_emb = rand_weight(VOCAB, D)
Wq, Wk, Wv, Wo = rand_weight(D, D), rand_weight(D, D), rand_weight(D, D), rand_weight(D, D)
W1, W2 = rand_weight(D, D_FF), rand_weight(D_FF, D)

TENSORS = [
    ("token_emb", token_emb),
    ("pos_emb", np.zeros((N_CTX, D), dtype=np.float32)),
    ("ln1_g", np.ones(D, dtype=np.float32)),
    ("ln1_b", np.zeros(D, dtype=np.float32)),
    ("Wq", Wq), ("bq", np.zeros(D, dtype=np.float32)),
    ("Wk", Wk), ("bk", np.zeros(D, dtype=np.float32)),
    ("Wv", Wv), ("bv", np.zeros(D, dtype=np.float32)),
    ("Wo", Wo), ("bo", np.zeros(D, dtype=np.float32)),
    ("ln2_g", np.ones(D, dtype=np.float32)),
    ("ln2_b", np.zeros(D, dtype=np.float32)),
    ("W1", W1), ("b1", np.zeros(D_FF, dtype=np.float32)),
    ("W2", W2), ("b2", np.zeros(D, dtype=np.float32)),
    ("lnf_g", np.ones(D, dtype=np.float32)),
    ("lnf_b", np.zeros(D, dtype=np.float32)),
]

MAGIC = int.from_bytes(b"CPW1", "little")

out = os.path.join(os.path.dirname(__file__), "..", "weights", "toy.bin")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as f:
    f.write(struct.pack("<II6i", MAGIC, 2, VOCAB, D, H, D_FF, N_LAYER, N_CTX))
    for _, a in TENSORS:
        f.write(a.astype("<f4").tobytes())

n_floats = sum(a.size for _, a in TENSORS)
print(f"wrote {os.path.normpath(out)}: {32 + 4 * n_floats} bytes ({n_floats} floats)")
print("ground truth for the C loader checkpoint:")
print("  token_emb[5] =", token_emb[5])
