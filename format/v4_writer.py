#!/usr/bin/env python3
"""v4_writer.py — reference CPW1 v4 writer (format/SPEC.md).

Every port's `convert.py` builds its file through this class, so the byte-level format lives
in exactly one place. A port supplies tensors, blocks and regions; it does not decide how a
string is length-prefixed or where a table starts.

Design notes that matter:

- **Offsets are assigned by this class, not by the caller.** A port says "here is a tensor",
  and gets back its id; the physical placement is a writer concern. That is what lets
  `layout` be a *policy* (v3 order, co-activation order, per-machine order) rather than
  something every port re-derives.
- **Region digests are CRC-32** (IEEE), computed here over exactly the bytes written, and
  verified by the loader at open. Two implementations of a checksum that disagree are worse
  than no checksum — that mistake has been made once in this repo already — so this is the
  single definition; `engine/core/weights_v4.c` must match it and
  `tools/verify_v4_file.py --digest-check` compares the two.
- **Payload is emitted in physical offset order**, not table order. With a shuffled layout
  those differ, and writing by table index put a tensor at the wrong position while every
  table still looked consistent.

Typical use:

    w = V4Writer()
    t = {name: w.add_tensor(name, arr, role) for ...}
    w.add_block(KIND_ATTN, layer, refs=[...], params=[...])
    w.add_expert_region(name, arr, slot, expert)   # one pageable region per expert
    w.write("out.v4cp", header_params)
"""
import struct
import zlib
from pathlib import Path

import numpy as np

import q8 as _q8

MAGIC = 0x34575043          # "CPW4"
VERSION = 4
HDR_BYTES = 88
ALIGN = 64

# block kinds (SPEC §7)
KIND_EMBED, KIND_NORM, KIND_ATTN, KIND_DENSE_FFN = 1, 2, 0x10, 0x11
KIND_MOE, KIND_MOE_LATENT, KIND_SSM = 0x20, 0x21, 0x30

# tensor roles (SPEC §5)
R_OTHER, R_EMBED, R_NORM, R_ATTN, R_ROUTER = 0, 1, 2, 3, 4
R_EXPERT, R_SHARED, R_DENSE, R_HEAD = 5, 6, 7, 8

# activations / routing (SPEC §7)
ACT_GELU_TANH, ACT_SILU, ACT_RELU2 = 0, 1, 2
SCORING_SOFTMAX, SCORING_SIGMOID = 0, 1

NO_BLOCK = 0xFFFFFFFF
REGION_TENSOR, REGION_EXPERT, REGION_GROUP, REGION_PINNED = 0, 1, 2, 3


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class _W:
    def __init__(self):
        self.b = bytearray()

    def u8(self, v):  self.b += struct.pack("<B", v & 0xFF); return self
    def u16(self, v): self.b += struct.pack("<H", v & 0xFFFF); return self
    def u32(self, v): self.b += struct.pack("<I", v & 0xFFFFFFFF); return self
    def u64(self, v): self.b += struct.pack("<Q", v & 0xFFFFFFFFFFFFFFFF); return self

    def string(self, s):
        e = s.encode()
        if len(e) >= 65536:
            raise ValueError("name too long: %r" % s)
        self.u16(len(e)); self.b += e
        return self


class V4Writer:
    def __init__(self, align=ALIGN):
        self.align = align
        self.tensors = []      # dicts, index == tensor id
        self.blocks = []
        self.regions = []
        self.ENCODINGS = []

    # ---------------- tensors
    def add_tensor(self, name, array, role=R_OTHER, enc="f32"):
        """Register a tensor; offset is assigned at write() time. Returns its id.

        `enc` is "f32" or "q8_0". The SHAPE stays logical (elements) and the stored length
        differs for a block encoding — the loader checks `length` against the encoding's
        group/block size, so the two are not interchangeable (SPEC §5).
        """
        a = np.ascontiguousarray(array, dtype="<f4")
        self._use_enc(enc)
        if enc == "q8_0":
            raw, _ = _q8.encode_q8_0(a)
        elif enc == "q4_0":
            import q4 as _q4
            raw, _ = _q4.encode_q4_0(a)
        elif enc == "f32":
            raw = a.tobytes()
        else:
            raise ValueError(f"unknown encoding {enc!r}")
        self.tensors.append(dict(name=name, shape=tuple(int(x) for x in a.shape),
                                 role=role, enc=enc, raw=raw,
                                 offset=None, pad=0))
        return len(self.tensors) - 1

    def _use_enc(self, enc):
        if enc not in self.ENCODINGS:
            self.ENCODINGS.append(enc)

    def tensor_id(self, name):
        for i, t in enumerate(self.tensors):
            if t["name"] == name:
                return i
        raise KeyError(name)

    # ---------------- blocks
    def add_block(self, kind, layer, refs=(), params=()):
        if len(refs) > 255 or len(params) > 255:
            raise ValueError("block refs/params exceed u8 count")
        for r in refs:
            if not (0 <= r < len(self.tensors)):
                raise ValueError("block ref %r is not a registered tensor id" % (r,))
        self.blocks.append(dict(kind=kind, layer=layer, refs=list(refs),
                                params=[int(p) & 0xFFFFFFFF for p in params]))

    # ---------------- regions
    def add_region(self, lo, hi, offset_bytes=None, kind=REGION_TENSOR,
                   slot=0xFFFF, mlo=0, mhi=1):
        """Declare a pageable region covering tensors [lo, hi).

        For REGION_EXPERT it spans exactly one expert's tensor in this port (the file's
        "one expert = one sequential read" property); `slot`/`mlo..mhi` name the (layer,
        expert) it is so a pager can reason about grouping without parsing names.
        """
        self.regions.append(dict(lo=lo, hi=hi, kind=kind, slot=slot, mlo=mlo, mhi=mhi,
                                 offset=offset_bytes))

    def add_expert_region(self, tensor_id_, slot, expert):
        self.add_region(tensor_id_, tensor_id_ + 1, kind=REGION_EXPERT,
                        slot=slot, mlo=expert, mhi=expert + 1)

    # ---------------- write
    def _place(self, order=None):
        """Assign physical offsets. `order` is a list of tensor ids; default is table order."""
        ids = list(range(len(self.tensors))) if order is None else list(order)
        if sorted(ids) != list(range(len(self.tensors))):
            raise ValueError("layout must be a permutation of the tensor ids")
        off = 0
        for i in ids:
            t = self.tensors[i]
            t["offset"] = off
            pad = (-len(t["raw"])) % self.align
            t["pad"] = pad
            off += len(t["raw"]) + pad
        return off

    def write(self, path, n_encodings=1, encoding=(0, 0, 0, 0, 4, 1, 0, 4, "f32"),
              layout=None):
        """Emit the file. `encoding` is (id, kind, scalar, group, block_bytes, sub, flags,
        align, name) per SPEC §4; the default is one raw-f32 entry."""
        if self._place(layout) == 0 and self.tensors:
            raise ValueError("no payload")
        if not self.ENCODINGS:
            self.ENCODINGS = ["f32"]
        n_encodings = len(self.ENCODINGS)

        # One record per encoding actually used. f32 and Q8_0 have different block geometry,
        # and the loader checks each tensor's stored length against its own record.
        DESC = {"f32":  (0, 0, 0, 4, 1, 0, 4, "f32"),
                "q8_0": (1, 3, 32, 34, 1, 3, 64, "Q8_0"),
                "q4_0": (1, 4, 32, 18, 1, 3, 64, "Q4_0")}
        enc = _W()
        for i, name in enumerate(self.ENCODINGS):
            ek, esc, egrp, eblk, esub, eflag, ealign, ename = DESC[name]
            enc.u32(i).u8(ek).u8(esc).u16(egrp).u16(eblk)
            enc.u8(esub).u8(eflag).u8(ealign).string(ename)

        tens = _W().u32(len(self.tensors))
        for t in self.tensors:
            tens.string(t["name"]).u32(self.ENCODINGS.index(t["enc"])).u8(len(t["shape"]))
            for d in t["shape"]:
                tens.u32(d)
            tens.u64(t["offset"]).u64(len(t["raw"]))
            tens.u32(NO_BLOCK).u16(t["role"])

        blk = _W().u32(len(self.blocks))
        for b in self.blocks:
            blk.u8(b["kind"]).u32(b["layer"]).u8(len(b["refs"]))
            for r in b["refs"]:
                blk.u32(r)
            blk.u8(len(b["params"]))
            for q in b["params"]:
                blk.u32(q)

        reg = _W().u32(len(self.regions))
        for r in self.regions:
            t = self.tensors[r["lo"]]
            if r["hi"] <= r["lo"]:
                raise ValueError("region tensor range must be non-empty")
            first, last = self.tensors[r["lo"]], self.tensors[r["hi"] - 1]
            off = first["offset"]
            length = (last["offset"] + len(last["raw"])) - off
            # digest over exactly the bytes this region covers
            blob = b"".join(self.tensors[i]["raw"] + b"\0" * self.tensors[i]["pad"]
                            for i in range(r["lo"], r["hi"]))
            digest = zlib.crc32(blob) & 0xFFFFFFFFFFFFFFFF
            reg.u64(off).u64(length).u32(r["lo"]).u32(r["hi"])
            reg.u8(r["kind"]).u16(r["slot"]).u32(r["mlo"]).u32(r["mhi"]).u64(digest)

        off_enc = HDR_BYTES
        off_tens = off_enc + len(enc.b)
        off_blk = off_tens + len(tens.b)
        off_reg = off_blk + len(blk.b)
        off_data = (off_reg + len(reg.b) + self.align - 1) // self.align * self.align

        hb = struct.pack("<IIII", MAGIC, VERSION, HDR_BYTES, 0)
        hb += struct.pack("<IIII", len(self.tensors), len(self.blocks),
                          len(self.regions), n_encodings)
        hb += struct.pack("<QQQQQ", off_enc, off_tens, off_blk, off_reg, off_data)
        hb += struct.pack("<IIII", 0, 0, 0, 0)
        assert len(hb) == HDR_BYTES

        out = bytearray(hb) + enc.b + tens.b + blk.b + reg.b
        out += b"\0" * (off_data - len(out))
        for i in sorted(range(len(self.tensors)), key=lambda k: self.tensors[k]["offset"]):
            t = self.tensors[i]
            assert len(out) == off_data + t["offset"], (len(out), t["offset"], t["name"])
            out += t["raw"] + b"\0" * t["pad"]

        Path(path).write_bytes(bytes(out))
        return dict(bytes=len(out), off_data=off_data, tensors=len(self.tensors),
                    blocks=len(self.blocks), regions=len(self.regions))
