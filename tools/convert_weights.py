"""convert_weights.py — HuggingFace GPT-2 -> CPW1 v2 (weights/gpt2.bin).

Layout notes:
- HF Conv1D stores weights (in, out) — exactly our engine's xw layout,
  so NO transposes anywhere; c_attn just gets split into q/k/v slices.
- Unembedding is tied in GPT-2 (lm_head = wte), so it is NOT written —
  the C loader ties it too.
- First run downloads ~500MB to the HF cache.

Run from repo root:  python3 tools/convert_weights.py [hf_name] [out.bin]
"""

import os
import struct
import sys

import numpy as np
from transformers import AutoModelForCausalLM

hf_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(__file__), "..", "weights", "gpt2.bin")

model = AutoModelForCausalLM.from_pretrained(hf_name)
sd = model.state_dict()
cfg = model.config
V, D, H, F = cfg.vocab_size, cfg.n_embd, cfg.n_head, cfg.n_embd * 4
L, C = cfg.n_layer, cfg.n_positions
print(f"{hf_name}: vocab={V} d_model={D} n_heads={H} d_ff={F} n_layer={L} n_ctx={C}")


def t(name):
    return sd[name].numpy().astype("<f4")


def stack(name):
    return np.stack([t(name.format(i=i)) for i in range(L)])


# c_attn: (D, 3D) in,out layout -> split into q/k/v of (D, D) each, then stack layers
attn_w = np.stack([t(f"transformer.h.{i}.attn.c_attn.weight") for i in range(L)])  # (L, D, 3D)
attn_b = np.stack([t(f"transformer.h.{i}.attn.c_attn.bias") for i in range(L)])    # (L, 3D)

TENSORS = [
    ("token_emb", t("transformer.wte.weight")),
    ("pos_emb", t("transformer.wpe.weight")),
    ("ln1_g", stack("transformer.h.{i}.ln_1.weight")),
    ("ln1_b", stack("transformer.h.{i}.ln_1.bias")),
    ("Wq", attn_w[:, :, 0 * D:1 * D]), ("bq", attn_b[:, 0 * D:1 * D]),
    ("Wk", attn_w[:, :, 1 * D:2 * D]), ("bk", attn_b[:, 1 * D:2 * D]),
    ("Wv", attn_w[:, :, 2 * D:3 * D]), ("bv", attn_b[:, 2 * D:3 * D]),
    ("Wo", stack("transformer.h.{i}.attn.c_proj.weight")),
    ("bo", stack("transformer.h.{i}.attn.c_proj.bias")),
    ("ln2_g", stack("transformer.h.{i}.ln_2.weight")),
    ("ln2_b", stack("transformer.h.{i}.ln_2.bias")),
    ("W1", stack("transformer.h.{i}.mlp.c_fc.weight")),
    ("b1", stack("transformer.h.{i}.mlp.c_fc.bias")),
    ("W2", stack("transformer.h.{i}.mlp.c_proj.weight")),
    ("b2", stack("transformer.h.{i}.mlp.c_proj.bias")),
    ("lnf_g", t("transformer.ln_f.weight")),
    ("lnf_b", t("transformer.ln_f.bias")),
]

MAGIC = int.from_bytes(b"CPW1", "little")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as f:
    f.write(struct.pack("<II6i", MAGIC, 2, V, D, H, F, L, C))
    for _, a in TENSORS:
        f.write(np.ascontiguousarray(a).astype("<f4").tobytes())

n_floats = sum(a.size for _, a in TENSORS)
print(f"wrote {os.path.normpath(out)}: {32 + 4 * n_floats:,} bytes ({n_floats:,} floats)")
