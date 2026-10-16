#!/usr/bin/env python3
"""v3_to_v4.py — express an existing CPW1 v3 file as CPW1 v4 (see format/SPEC.md).

WHY THIS EXISTS
  format/SPEC.md §9.2 makes "the existing model round-trips into v4" the acceptance test
  for the format itself: a format that cannot express the one model we have is not ready.
  This writer is that test's input side; the C loader is the other.

WHAT IT DOES NOT DO
  No numerics. Every tensor is raw f32 with byte-identical payload to the v3 source, so the
  loader must reproduce the source's logits and routing exactly. Values are never
  re-encoded — quantization is a later, separate concern.

THE STRONGER TEST (`--shuffle`)
  Emitting the same values in the same order would prove only that the loader can read a
  table; it would not exercise the format's central claim, that tensors are addressed by
  OFFSET rather than by position. `--shuffle` writes the same tensors in a different
  physical order. The shuffled file must load and produce byte-identical output to the
  unshuffled one and to the v3 source. It fails if any loader code assumes a layout.

USAGE
  python3 tools/v3_to_v4.py --src weights/moe-scale-run/l05.cpw1v3 \
      --dst scratch/l05.v4cp [--shuffle] [--ckpt weights/moe-scale-run/l05/model_only.pt]
"""
import argparse
import zlib
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import cpw1_v3 as v3  # noqa: E402

MAGIC = 0x34575043          # "CPW4"
VERSION = 4
HDR_BYTES = 88   # 4*4 + 4*4 + 5*8 + 4*4 (SPEC section 3; earlier drafts said 64
                 # and 68, both simple miscounts -- now asserted, not computed by hand)
ALIGN = 64

ROLE = {"other": 0, "embedding": 1, "norm": 2, "attn": 3, "router": 4,
        "expert": 5, "shared_expert": 6, "dense_ffn": 7, "head": 8}
KIND_EMBED, KIND_NORM, KIND_ATTN, KIND_DENSE_FFN, KIND_MOE = 1, 2, 0x10, 0x11, 0x20
ACT_GELU_TANH, SCORING_SOFTMAX = 0, 0
NO_BLOCK = 0xFFFFFFFF


class W:
    """Little-endian writer following SPEC.md's string/align conventions."""

    def __init__(self):
        self.b = bytearray()

    def u8(self, v):  self.b += struct.pack("<B", v & 0xFF); return self
    def u16(self, v): self.b += struct.pack("<H", v & 0xFFFF); return self
    def u32(self, v): self.b += struct.pack("<I", v & 0xFFFFFFFF); return self
    def u64(self, v): self.b += struct.pack("<Q", v & 0xFFFFFFFFFFFFFFFF); return self
    def string(self, s):
        e = s.encode(); self.u16(len(e)); self.b += e
        return self          # NOTE: must return self, not the bytes. Returning self.b + e
                             # (a list) silently REPLACED the builder at the call site, so
                             # the encoding table came out as 5 stray bytes and every
                             # subsequent table was misaligned.

    def pad_to(self, off):
        assert len(self.b) <= off, "overshoot: %d > %d" % (len(self.b), off)
        self.b += b"\0" * (off - len(self.b))
        return self


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def role_of(name):
    if name in ("token_emb", "pos_emb"):  return ROLE["embedding"]
    if name.startswith("ln"):             return ROLE["norm"]
    if name in ("Wr", "br"):              return ROLE["router"]
    if name.startswith("experts"):        return ROLE["expert"]
    if name in ("W1s", "b1s", "W2s", "b2s"): return ROLE["shared_expert"]
    if name in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"): return ROLE["attn"]
    if name in ("W1", "b1", "W2", "b2"):  return ROLE["dense_ffn"]
    return ROLE["other"]


def region_digest(data):
    """CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320, init 0xFFFFFFFF, final xor
    0xFFFFFFFF) over the region's bytes, as a u64.

    Replaced FNV-1a here. FNV was chosen first and cost ~5.5 s per 735 MB pass in Python
    (a byte-at-a-time loop), which made writing or checking a file slower than loading it.
    CRC-32 is C-implemented in zlib (~1.5 GB/s) and a ten-line table-driven loop in C, so
    the engine keeps its no-dependency rule while the check becomes free.

    Verified against the standard vector: region_digest(b"123456789") == 0xCBF43926.
    engine/core/weights_v4.c implements the same table-driven CRC; all three
    implementations are cross-checked by tools/verify_v4_file.py --digest-check.
    """
    return zlib.crc32(data) & 0xFFFFFFFFFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ckpt", default="weights/moe-scale-run/l05/model_only.pt")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=20261016)
    args = ap.parse_args()

    import torch
    import train_moe as tm
    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    cfg = tm.Config(**ck["config"])

    hdr = v3.read_header(args.src)
    assert tuple(hdr[k] for k in v3.HDR_INT_FIELDS) == v3.header_ints(cfg), \
        "checkpoint config does not match the file header"

    arrays = {}
    with open(args.src, "rb") as f:
        f.seek(v3.HDR_SIZE)
        for name, shape in v3.payload_layout(cfg):
            n = int(np.prod(shape))
            arrays[name] = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(shape).copy()

    D, L, N, EH = cfg.d_model, cfg.n_layer, cfg.n_exp, cfg.exp_hid
    moe_layers, dense_layers = cfg.moe_layers(), cfg.dense_layers()
    n_moe, n_dense = len(moe_layers), len(dense_layers)

    # ---------------- logical order: v3 payload, with the expert pool expanded
    logical = []          # (tensor_name, name_suffix, ndarray, is_expert, slot, e)
    for name, shape in v3.payload_layout(cfg):
        if name == "experts":
            for s in range(shape[0]):
                for e in range(shape[1]):
                    # name carries BOTH indices: "experts.<e>" collided across the 10
                    # slots, so a reader keyed by name silently kept whichever slot came
                    # last and 320 of 346 tensors "differed". The region table already
                    # disambiguates by slot, but the tensor NAME must too.
                    logical.append((name, "%d.%d" % (s, e), arrays[name][s, e], True, s, e, s, e))
        else:
            logical.append((name, "", arrays[name], False, -1, -1, -1, -1))

    order = list(range(len(logical)))
    if args.shuffle:
        np.random.default_rng(args.seed).shuffle(order)

    # ---------------- place payload in the chosen physical order
    placed = [None] * len(logical)
    off = 0
    for i in order:
        name, suffix, arr, is_exp, e_slot, e_id, slot, e = logical[i]
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        pad = (-len(raw)) % ALIGN
        full = "%s.%s" % (name, suffix) if suffix else name
        placed[i] = dict(name=full, role=(ROLE["expert"] if is_exp else role_of(name)),
                         shape=tuple(int(x) for x in arr.shape), offset=off,
                         length=len(raw), raw=raw, digest=region_digest(raw),
                         is_expert=is_exp, slot=slot, e=e)
        off += len(raw) + pad
    data_len = off

    # index tensors by name for block references
    tid = {p["name"]: i for i, p in enumerate(placed)}

    # ---------------- encoding table: one entry, raw f32
    enc = W().u32(0).u8(0).u8(0).u16(0).u16(4).u8(1).u8(0).u8(4).string("f32")

    # ---------------- blocks
    blocks = []
    blocks.append(dict(kind=KIND_EMBED, layer=NO_BLOCK,
                       refs=[tid["token_emb"], tid["pos_emb"]],
                       params=[cfg.vocab, D, cfg.n_ctx]))
    for l in range(L):
        if l in moe_layers:
            s = moe_layers.index(l)
            refs = [tid["Wr"], tid["br"], tid["W1s"], tid["b1s"], tid["W2s"], tid["b2s"]]
            # routed experts are addressed as regions, not as block inputs (SPEC §7.3)
            blocks.append(dict(kind=KIND_MOE, layer=l, refs=refs,
                               params=[N, cfg.topk, SCORING_SOFTMAX, 1,
                                       f32_bits(cfg.lam), s, 1, 0, 1, 1]))
        else:
            d = dense_layers.index(l)
            refs = [tid["ln2_g"], tid["ln2_b"], tid["W1"], tid["b1"], tid["W2"], tid["b2"]]
            blocks.append(dict(kind=KIND_DENSE_FFN, layer=l, refs=refs,
                               params=[ACT_GELU_TANH, cfg.d_ff, d]))
    blocks.append(dict(kind=KIND_NORM, layer=L,
                       refs=[tid["lnf_g"], tid["lnf_b"]],
                       params=[f32_bits(1e-5), 0]))

    # ---------------- tensor + region tables (block_id assigned by tensor role)
    tens = W().u32(len(placed))
    regions = []
    for i, p in enumerate(placed):
        block_id = NO_BLOCK
        if p["is_expert"]:
            block_id = p["slot"]
        tens.string(p["name"]).u32(0).u8(len(p["shape"]))
        for d in p["shape"]:
            tens.u32(d)
        tens.u64(p["offset"]).u64(p["length"]).u32(block_id).u16(p["role"])
        if p["is_expert"]:
            regions.append((p["offset"], p["length"], i, i + 1, 1, p["slot"], p["e"], p["e"] + 1, p["digest"]))
        else:
            regions.append((p["offset"], p["length"], i, i + 1, 0, 0xFFFF, 0, 1, p["digest"]))

    blk = W().u32(len(blocks))
    for b in blocks:
        blk.u8(b["kind"]).u32(b["layer"]).u8(len(b["refs"]))
        for r in b["refs"]:
            blk.u32(r)
        blk.u8(len(b["params"]))
        for q in b["params"]:
            blk.u32(q)

    reg = W().u32(len(regions))
    for roff, rlen, lo, hi, k, rslot, mlo, mhi, dig in regions:
        reg.u64(roff).u64(rlen).u32(lo).u32(hi).u8(k).u16(rslot)
        reg.u32(mlo).u32(mhi).u64(dig)

    # ---------------- header + file assembly
    off_enc = HDR_BYTES
    off_tens = off_enc + len(enc.b)
    off_blk = off_tens + len(tens.b)
    off_reg = off_blk + len(blk.b)
    off_data = (off_reg + len(reg.b) + ALIGN - 1) // ALIGN * ALIGN

    # One explicit struct, so the header size cannot drift from HDR_BYTES:
    #   4 u32 (magic/version/header_bytes/flags) + 4 u32 counts
    #   + 5 u64 offsets + 4 u32 reserved = 16 + 16 + 40 + 16 = 88? No -- see below.
    # Counted properly: MAGIC,VERSION,HDR,FLAGS (16) + n_tensors,n_blocks,n_regions,
    # n_enc (16) = 32; five u64 = 40 -> 72; four reserved u32 = 16 -> 88.
    # The definition of HDR_BYTES is therefore 88, not 64 or 68. Rather than patch a
    # constant again, build the bytes and assert against the named size.
    hb = struct.pack("<IIII", MAGIC, VERSION, HDR_BYTES, 0)
    hb += struct.pack("<IIII", len(placed), len(blocks), len(regions), 1)
    hb += struct.pack("<QQQQQ", off_enc, off_tens, off_blk, off_reg, off_data)
    hb += struct.pack("<IIII", 0, 0, 0, 0)
    assert len(hb) == HDR_BYTES, "header %d != HDR_BYTES %d" % (len(hb), HDR_BYTES)
    head = W(); head.b += hb

    out = bytearray(head.b)
    out += enc.b; out += tens.b; out += blk.b; out += reg.b
    out += b"\0" * (off_data - len(out))
    # Payload must be emitted in PHYSICAL order (by offset), not table order: with
    # --shuffle the two differ, and emitting by table index put token_emb (offset
    # 251918336) at position 0. The tensor table is what maps offsets to names, so the
    # table may be in any order -- the bytes may not.
    for p in sorted(placed, key=lambda q: q["offset"]):
        assert len(out) == off_data + p["offset"], (len(out), p["offset"], p["name"])
        out += p["raw"] + b"\0" * ((-len(p["raw"])) % ALIGN)
    assert len(out) == off_data + data_len, (len(out), off_data + data_len)

    Path(args.dst).write_bytes(bytes(out))
    print("wrote %s (%d bytes)" % (args.dst, len(out)))
    print("  tensors %d (expert pool expanded per expert) | blocks %d | regions %d"
          % (len(placed), len(blocks), len(regions)))
    print("  tables end %d | payload at %d | payload %d bytes%s"
          % (off_reg + len(reg.b), off_data, data_len,
             "   SHUFFLED physical order (seed %d)" % args.seed if args.shuffle else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
