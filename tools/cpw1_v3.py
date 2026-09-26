#!/usr/bin/env python3
"""cpw1_v3.py — the CPW1 v3 weight-file format (engine contract).

Extends v2 (src/weights.h). Dense tensors keep v2's per-tensor-across-layers
order; the MoE tensors are added, and crucially the payload holds ONLY the live
parameters. The PyTorch trainer allocates dense-FFN weights on all 12 layers
while only the dense ones use them, so every checkpoint carries ~68.6M dead
params; those are not written.

Header (60 bytes):
    u32 magic 'CPW1' | u32 version=3
    i32 vocab, d_model, n_heads, d_ff, n_layer, n_ctx,
        n_exp, exp_hid, topk, moe_from, n_moe_layers, n_dense_layers
    f32 lambda                       <- the locality bias is part of the model:
                                        serving it at a different lambda routes
                                        differently from what was trained

Payload (float32, this exact order):
    token_emb (v,d) | pos_emb (c,d)
    ln1_g, ln1_b (L,d)
    Wq,bq,Wk,bk,Wv,bv,Wo,bo (L,d,d)/(L,d)
    ln2_g, ln2_b (L,d)
    --- DENSE FFN, dense layers only (n_dense) ---
    W1 (n_dense,d,f) | b1 (n_dense,f) | W2 (n_dense,f,d) | b2 (n_dense,d)
    --- ROUTER, MoE layers only (n_moe) ---
    Wr (n_moe,d,N) | br (n_moe,N)
    --- SHARED EXPERT, MoE layers only (n_moe) ---
    W1s (n_moe,d,d) | b1s (n_moe,d) | W2s (n_moe,d,d) | b2s (n_moe,d)
    --- ROUTED EXPERTS: ONE CONTIGUOUS BLOCK PER (layer, expert) ---
    experts (n_moe, N, EXPERT_FLOATS), each block = W1e(d,EH) b1e(EH) W2e(EH,d) b2e(d)
    lnf_g, lnf_b (d)

The per-expert contiguity is the whole point of the layout: the pager regions are
(layer, expert) so one fault pulls one complete expert in a single sequential
read (spec: "each expert one contiguous 1.18MB block = one sequential page-in").
"""
import struct

import numpy as np

MAGIC = 0x31575043          # 'CPW1'
VERSION = 3
HDR_INT_FIELDS = ("vocab", "d_model", "n_heads", "d_ff", "n_layer", "n_ctx",
                  "n_exp", "exp_hid", "topk", "moe_from",
                  "n_moe_layers", "n_dense_layers")
HDR_INT_BYTES = 4 * len(HDR_INT_FIELDS)
HDR_SIZE = 8 + HDR_INT_BYTES + 4      # + f32 lambda

_MOE_GROUP = ("Wr", "br", "W1s", "b1s", "W2s", "b2s")
_DENSE_GROUP = ("W1", "b1", "W2", "b2")


def expert_floats(cfg):
    """Floats in one routed-expert block: W1e, b1e, W2e, b2e."""
    return cfg.d_model * cfg.exp_hid + cfg.exp_hid \
        + cfg.exp_hid * cfg.d_model + cfg.d_model


def expert_slices(cfg):
    """(name, offset, size) inside one expert block."""
    d, eh = cfg.d_model, cfg.exp_hid
    o = 0
    out = []
    for name, size in (("W1e", d * eh), ("b1e", eh), ("W2e", eh * d), ("b2e", d)):
        out.append((name, o, size))
        o += size
    return out


def payload_layout(cfg):
    """Ordered [(name, shape)] of the v3 payload."""
    d, f, L, c, v = cfg.d_model, cfg.d_ff, cfg.n_layer, cfg.n_ctx, cfg.vocab
    N, EH = cfg.n_exp, cfg.exp_hid
    nm, nd = len(cfg.moe_layers()), len(cfg.dense_layers())
    return [
        ("token_emb", (v, d)), ("pos_emb", (c, d)),
        ("ln1_g", (L, d)), ("ln1_b", (L, d)),
        ("Wq", (L, d, d)), ("bq", (L, d)),
        ("Wk", (L, d, d)), ("bk", (L, d)),
        ("Wv", (L, d, d)), ("bv", (L, d)),
        ("Wo", (L, d, d)), ("bo", (L, d)),
        ("ln2_g", (L, d)), ("ln2_b", (L, d)),
        ("W1", (nd, d, f)), ("b1", (nd, f)),
        ("W2", (nd, f, d)), ("b2", (nd, d)),
        ("Wr", (nm, d, N)), ("br", (nm, N)),
        ("W1s", (nm, d, d)), ("b1s", (nm, d)),
        ("W2s", (nm, d, d)), ("b2s", (nm, d)),
        ("experts", (nm, N, expert_floats(cfg))),
        ("lnf_g", (d,)), ("lnf_b", (d,)),
    ]


def payload_floats(cfg):
    return sum(int(np.prod(s)) for _, s in payload_layout(cfg))


def header_ints(cfg):
    return (cfg.vocab, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_layer, cfg.n_ctx,
            cfg.n_exp, cfg.exp_hid, cfg.topk, cfg.moe_from,
            len(cfg.moe_layers()), len(cfg.dense_layers()))


# ------------------------------------------------------------------ writer
def state_to_arrays(sd, cfg):
    """MoEModel state_dict (allocated layout) -> v3 payload arrays."""
    moe, dense = cfg.moe_layers(), cfg.dense_layers()
    get = lambda n: sd[n].detach().cpu().numpy()          # noqa: E731
    out = {}
    for name, shape in payload_layout(cfg):
        if name == "experts":
            blocks = []
            W1e, b1e, W2e, b2e = get("W1e"), get("b1e"), get("W2e"), get("b2e")
            for li in moe:
                for e in range(cfg.n_exp):
                    blocks.append(np.concatenate([
                        W1e[li, e].ravel(), b1e[li, e].ravel(),
                        W2e[li, e].ravel(), b2e[li, e].ravel()]))
            out[name] = np.stack(blocks).reshape(shape)
        elif name in _DENSE_GROUP:
            out[name] = get(name)[dense]
        elif name in _MOE_GROUP:
            out[name] = get(name)[moe]
        else:
            out[name] = get(name)
    return out


def write(path, sd, cfg):
    arrays = state_to_arrays(sd, cfg)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        f.write(struct.pack("<%di" % len(HDR_INT_FIELDS), *header_ints(cfg)))
        f.write(struct.pack("<f", float(cfg.lam)))
        for name, _shape in payload_layout(cfg):
            f.write(np.ascontiguousarray(arrays[name], dtype="<f4").tobytes())
    return arrays


# ------------------------------------------------------------------ reader
def read_header(path):
    with open(path, "rb") as f:
        raw = f.read(HDR_SIZE)
    if len(raw) < HDR_SIZE:
        raise ValueError("file shorter than a CPW1 header")
    magic, version = struct.unpack_from("<II", raw, 0)
    if magic != MAGIC:
        raise ValueError("bad magic 0x%08x: not a CPW1 file" % magic)
    if version != VERSION:
        raise ValueError("CPW1 v%d, expected v%d" % (version, VERSION))
    vals = struct.unpack_from("<%di" % len(HDR_INT_FIELDS), raw, 8)
    lam = struct.unpack_from("<f", raw, 8 + HDR_INT_BYTES)[0]
    hdr = dict(zip(HDR_INT_FIELDS, vals))
    hdr["lam"] = lam
    return hdr


def read_arrays(path, cfg):
    """Read the payload back as {name: ndarray}. Validates the exact file size,
    which is the loud-failure guarantee the format exists for."""
    hdr = read_header(path)
    exp_size = HDR_SIZE + payload_floats(cfg) * 4
    import os
    got = os.path.getsize(path)
    if got != exp_size:
        raise ValueError("size mismatch: file %d bytes, config implies %d "
                         "(magic ok, so this is a layout/version skew)" % (got, exp_size))
    out = {}
    with open(path, "rb") as f:
        f.seek(HDR_SIZE)
        for name, shape in payload_layout(cfg):
            n = int(np.prod(shape))
            out[name] = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(shape)
    return hdr, out


def arrays_to_state(arrays, cfg, model):
    """v3 payload -> a full allocated state_dict (unused slots left at zero)."""
    import torch
    sd = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    moe, dense = cfg.moe_layers(), cfg.dense_layers()
    for name, _shape in payload_layout(cfg):
        a = torch.from_numpy(np.array(arrays[name], dtype=np.float32))
        if name == "experts":
            blocks = arrays[name].reshape(len(moe), cfg.n_exp, expert_floats(cfg))
            for li, l in enumerate(moe):
                for e in range(cfg.n_exp):
                    flat = torch.from_numpy(np.array(blocks[li, e], dtype=np.float32))
                    for sub, off, size in expert_slices(cfg):
                        tgt = sd[sub].reshape(sd[sub].shape[0], sd[sub].shape[1], -1)[l, e]
                        tgt.copy_(flat[off:off + size])
        elif name in _DENSE_GROUP:
            for i, l in enumerate(dense):
                sd[name][l] = a[i]
        elif name in _MOE_GROUP:
            for i, l in enumerate(moe):
                sd[name][l] = a[i]
        elif name not in sd:
            raise KeyError(name)
        else:
            sd[name].copy_(a)
    return sd
