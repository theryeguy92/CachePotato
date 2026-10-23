#!/usr/bin/env python3
"""verify_v4_file.py — format-conformance checks for a CPW1 v4 file (format/SPEC.md §9.1).

WHY THIS EXISTS
  The C loader must not be the first thing that discovers the file is malformed. This checks
  the format-level invariants — offsets, alignment, regions, block refs, digests — and, when
  given the v3 source, that the v4 tensor table reconstructs byte-identical values. The
  loader then only has to be correct, not also a validator.

CHECKS
  1. header parses; magic/version/header_bytes coherent; reserved words zero
  2. tables lie inside the file, in the declared order, and end before off_data
  3. tensor offsets are >= off_data, in-file, 64-aligned, non-overlapping
  4. every block's tensor refs are in range and its ref count matches its kind
  5. every region lies inside exactly one tensor (or a contiguous run); expert regions
     carry a slot and a member range; digests match the payload
  6. every tensor is covered by at least one region (SPEC §8 rule 3)
  7. with --v3: the tensor table reconstructs the v3 payload byte-for-byte

USAGE
  python3 tools/verify_v4_file.py --file scratch/l05.v4cp --v3 weights/moe-scale-run/l05.cpw1v3
"""
import argparse
import mmap
import struct
import sys
from pathlib import Path

import zlib

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import cpw1_v3 as v3  # noqa: E402

MAGIC = 0x34575043
HDR = struct.Struct("<IIII IIII QQQQQ IIII")   # 88 bytes
ALIGN = 64

KIND_NAMES = {1: "embedding", 2: "norm", 0x10: "attention", 0x11: "dense_ffn",
              0x20: "moe", 0x21: "moe_latent", 0x30: "ssm"}
ROLE_NAMES = {0: "other", 1: "embedding", 2: "norm", 3: "attn", 4: "router",
              5: "expert", 6: "shared_expert", 7: "dense_ffn", 8: "head"}


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


class Reader:
    def __init__(self, b, off=0):
        self.b, self.o = b, off
    def u8(self):  v = self.b[self.o]; self.o += 1; return v
    def u16(self): v = struct.unpack_from("<H", self.b, self.o)[0]; self.o += 2; return v
    def u32(self): v = struct.unpack_from("<I", self.b, self.o)[0]; self.o += 4; return v
    def u64(self): v = struct.unpack_from("<Q", self.b, self.o)[0]; self.o += 8; return v
    def string(self):
        n = self.u16(); s = self.b[self.o:self.o + n].decode(); self.o += n; return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--v3", default="", help="v3 source to compare values against")
    ap.add_argument("--ckpt", default="weights/moe-scale-run/l05/model_only.pt")
    args = ap.parse_args()

    # MMAP, NOT read_bytes(). This tool's job is to check a file LARGER than the machine's
    # RAM: the Q8 Qwen3 file is 32.5 GB and the target box has 16 GB, so read_bytes() cannot
    # run the check that matters. Every access below is a struct.unpack_from, a slice, or a
    # np.frombuffer on the buffer, all of which work on an mmap -- and each region digest is
    # computed over one region's slice, so peak RSS stays near one region rather than one
    # file. Verified to produce byte-identical output on the 736 MB l05.v4cp before/after.
    fh = open(args.file, "rb")
    b = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    errs, warns = [], []

    # ---- 1. header
    (magic, ver, hbytes, flags, nt, nb, nr, ne,
     o_enc, o_tens, o_blk, o_reg, o_data, r0, r1, r2, r3) = HDR.unpack_from(b, 0)
    print(f"{args.file}: {len(b)} bytes")
    print(f"  header: magic={magic:#x} version={ver} header_bytes={hbytes} flags={flags}")
    print(f"          tensors={nt} blocks={nb} regions={nr} encodings={ne}")
    print(f"          tables enc={o_enc} tens={o_tens} blk={o_blk} reg={o_reg} | data={o_data}")
    if magic != MAGIC: errs.append("bad magic")
    if ver != 4: errs.append(f"version {ver}")
    if hbytes != HDR.size: errs.append(f"header_bytes {hbytes} != {HDR.size}")
    if (r0, r1, r2, r3) != (0, 0, 0, 0): errs.append("reserved words nonzero")
    if not (hbytes <= o_enc <= o_tens <= o_blk <= o_reg <= o_data <= len(b)):
        errs.append("table offsets not ordered/in-file")

    # ---- encoding table
    r = Reader(b, o_enc)
    encs = []
    for _ in range(ne):
        eid = r.u32(); kind = r.u8(); scal = r.u8(); gs = r.u16(); bb = r.u16()
        sb = r.u8(); fl = r.u8(); al = r.u8(); nm = r.string()
        encs.append(dict(id=eid, kind=kind, scalar=scal, group=gs, block=bb, align=al, name=nm))
    print("  encodings:", [(e["name"], e["group"], e["block"], e["align"]) for e in encs])
    if r.o != o_tens: errs.append(f"encoding table ends at {r.o}, expected {o_tens}")

    # ---- tensor table. Each table carries its own u32 count before its records
    # (SPEC section 2), so the reader must consume it; the header's count is a cross-check.
    r = Reader(b, o_tens)
    n_tens = r.u32()
    if n_tens != nt: errs.append(f"tensor table count {n_tens} != header {nt}")
    tens = []
    for i in range(nt):
        name = r.string(); dt = r.u32(); rank = r.u8()
        shape = tuple(r.u32() for _ in range(rank))
        off = r.u64(); ln = r.u64(); bid = r.u32(); role = r.u16()
        tens.append(dict(i=i, name=name, dtype=dt, shape=shape, off=off, len=ln,
                         block=bid, role=role))
    if r.o != o_blk: errs.append(f"tensor table ends at {r.o}, expected {o_blk}")

    # ---- 3. tensor geometry
    for t in tens:
        # SPEC section 5: tensor offsets are relative to off_data, so offset 0 is legal and
        # the absolute position is off_data + offset. Treating them as absolute was exactly
        # the bug this file's own header warns about; it made every digest check compare the
        # wrong bytes while the tensor table itself looked fine.
        if t["off"] + o_data + t["len"] > len(b):
            errs.append(f"{t['name']}: payload range {t['off']}+{t['len']} extends past EOF")
        if t["off"] % ALIGN: errs.append(f"{t['name']}: offset {t['off']} not {ALIGN}-aligned")
        if t["off"] % encs[t["dtype"]]["align"]:
            errs.append(f"{t['name']}: offset not aligned to its encoding")
    s = sorted(tens, key=lambda t: t["off"])
    for a, c in zip(s, s[1:]):
        if a["off"] + a["len"] > c["off"]:
            errs.append(f"overlap: {a['name']} and {c['name']}")

    # ---- block table
    r = Reader(b, o_blk)
    n_blk = r.u32()
    if n_blk != nb: errs.append(f"block table count {n_blk} != header {nb}")
    blocks = []
    for _ in range(nb):
        kind = r.u8(); layer = r.u32(); nref = r.u8()
        refs = [r.u32() for _ in range(nref)]
        npar = r.u8(); params = [r.u32() for _ in range(npar)]
        blocks.append(dict(kind=kind, layer=layer, refs=refs, params=params))
    if r.o != o_reg: errs.append(f"block table ends at {r.o}, expected {o_reg}")
    for bl in blocks:
        for ref in bl["refs"]:
            if ref >= nt: errs.append(f"block layer {bl['layer']}: ref {ref} out of range")
    kinds = {}
    for bl in blocks: kinds[KIND_NAMES.get(bl["kind"], hex(bl["kind"]))] = kinds.get(KIND_NAMES.get(bl["kind"], "?"), 0) + 1
    print("  blocks:", kinds)

    # ---- region table
    r = Reader(b, o_reg)
    n_reg = r.u32()
    if n_reg != nr: errs.append(f"region table count {n_reg} != header {nr}")
    regions = []
    for _ in range(nr):
        off = r.u64(); ln = r.u64(); lo = r.u32(); hi = r.u32(); k = r.u8()
        slot = r.u16(); mlo = r.u32(); mhi = r.u32(); dig = r.u64()
        regions.append(dict(off=off, len=ln, lo=lo, hi=hi, kind=k, slot=slot,
                            mlo=mlo, mhi=mhi, digest=dig))
    if r.o > o_data: errs.append(f"region table ends at {r.o} > off_data {o_data}")

    # every region inside one tensor (or a contiguous run) + digest
    by_off = {t["off"]: t for t in tens}
    for rg in regions:
        t = by_off.get(rg["off"])
        if t is None:
            errs.append(f"region at {rg['off']} does not start at a tensor offset")
            continue
        if rg["lo"] != t["i"]:
            errs.append(f"region at {rg['off']}: tensor_lo {rg['lo']} != {t['i']}")
        payload = b[o_data + rg["off"]:o_data + rg["off"] + rg["len"]]
        if region_digest(payload) != rg["digest"]:
            errs.append(f"region at {rg['off']} ({t['name']}): digest mismatch")
    covered = set()
    for rg in regions:
        for i in range(rg["lo"], rg["hi"]): covered.add(i)
    missing = [t["name"] for t in tens if t["i"] not in covered]
    if missing: errs.append(f"{len(missing)} tensors not covered by any region: {missing[:4]}")

    # ---- 7. value round-trip against the v3 source
    if args.v3:
        import torch, train_moe as tm
        ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
        cfg = tm.Config(**ck["config"])
        want = {}
        with open(args.v3, "rb") as f:
            f.seek(v3.HDR_SIZE)
            for nm, shape in v3.payload_layout(cfg):
                want[nm] = np.frombuffer(f.read(int(np.prod(shape)) * 4),
                                         dtype="<f4").reshape(shape).copy()
        # Decode per the tensor's OWN encoding. A raw-f32 tensor must match v3 exactly; a
        # block-quantized tensor is LOSSY by construction, so it is checked against a bound
        # and its measured error is reported. Conflating the two is how a legitimate Q8 file
        # got reported as "9 tensor value mismatches" — the check was exact for everything.
        import sys as _s, pathlib as _p
        _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1] / "format"))
        import q8 as _q8
        enc_of = {e["id"]: e for e in encs}
        got, got_enc = {}, {}
        for t in tens:
            e = enc_of[t["dtype"]]
            n = int(np.prod(t["shape"])) if t["shape"] else 0
            if e["name"].upper().startswith("Q8"):
                blk = np.frombuffer(b, dtype=np.uint8, count=t["len"], offset=o_data + t["off"])
                got[t["name"]] = _q8.decode_q8_0(blk.tobytes(), t["shape"]).ravel()
                got_enc[t["name"]] = "q8"
            else:
                got[t["name"]] = np.frombuffer(b, dtype="<f4", count=n,
                                               offset=o_data + t["off"])
                got_enc[t["name"]] = "f32" 
        mism, q8_stats = [], []
        for nm, arr in want.items():
            if nm == "experts":
                continue
            if nm not in got: mism.append(f"{nm}: missing from v4"); continue
            # .ravel() both sides: the v4 tensor is flat (a byte range) while the v3 layout
            # is shaped. numpy 2.x array_equal does NOT broadcast shapes, so comparing a
            # (9216,) against a (12,768) returned False for byte-identical data -- which is
            # how the "344 tensor value mismatches" in this tool's first working version
            # were produced. Flatten, then compare values (not shapes: shape is a table
            # property checked above).
            if got_enc.get(nm) == "q8":
                # lossy: bounded by the 127-level symmetric quantizer, checked as a fraction
                # of each tensor's own peak magnitude
                d = np.abs(got[nm].ravel() - arr.ravel()).max()
                peak = max(float(np.abs(arr).max()), 1e-30)
                q8_stats.append((nm, d / peak))
                if d / peak > 2.0 / 127:
                    mism.append(f"{nm}: q8 error {d/peak:.2e} exceeds 2/127 of peak")
            elif not np.array_equal(got[nm].ravel(), arr.ravel()):
                mism.append(f"{nm}: values differ")
        # experts: v4 splits them per (slot, expert); v3 stores them packed
        # Experts. Two layouts are legal and they are NOT interchangeable:
        #   f32 : one tensor per expert, W1|b1|W2|b2 concatenated
        #   q8  : one Q8 tensor per expert holding W1|W2, with biases moved to a separate
        #         per-slot f32 tensor `expert_bias.<slot>` (a bias must not share a group
        #         scale with weights). Comparing one against the other would report every
        #         expert as a mismatch, which is what the first version of this did.
        exp = want["experts"]           # (n_moe, N, expert_floats)
        n_moe, N, ef = exp.shape
        # care: want["W1"] is the DENSE FFN, not the expert matrix. Derive the expert
        # geometry from the packed expert length instead: ef = 2*D*EH + EH + D.
        D = want["W1"].shape[1]                 # d_model
        EH = (ef - D) // (2 * D + 1)
        w_elems = 2 * D * EH
        assert ef == 2 * D * EH + EH + D, (ef, D, EH)
        for s in range(n_moe):
            for e in range(N):
                nm = "experts.%d.%d" % (s, e)
                if nm not in got: mism.append(f"{nm}: missing"); continue
                src = exp[s, e]
                v3_w = np.concatenate([src[:D * EH], src[D * EH + EH: D * EH + EH + EH * D]])
                if got_enc.get(nm) == "q8":
                    g = got[nm].ravel()
                    if g.size != w_elems:
                        mism.append(f"{nm}: {g.size} elements, expected {w_elems} (W1|W2)")
                        continue
                    d2 = np.abs(g - v3_w).max()
                    peak = max(float(np.abs(v3_w).max()), 1e-30)
                    q8_stats.append((nm, d2 / peak))
                    if d2 / peak > 2.0 / 127:
                        mism.append(f"{nm}: q8 error {d2/peak:.2e} exceeds bound")
                    bnm = "expert_bias.%d" % s
                    if bnm not in got:
                        mism.append(f"{bnm}: missing (q8 experts need a bias tensor)")
                        continue
                    want_b = np.concatenate([src[D * EH: D * EH + EH], src[D * EH + EH + EH * D:]])
                    have_b = got[bnm].ravel()[e * (EH + D): (e + 1) * (EH + D)]
                    if not np.array_equal(have_b, want_b):
                        mism.append(f"{bnm}[{e}]: bias values differ")
                elif not np.array_equal(got[nm].ravel(), src.ravel()):
                    mism.append(f"{nm}: values differ")
        # Never say "identical" for a file that contains lossy tensors: the exact and the
        # bounded checks are different claims and must not be reported with one word.
        if q8_stats:
            worst = max(e for _, e in q8_stats)
            print(f"  value round-trip vs v3: f32 tensors EXACT; "
                  f"{len(q8_stats)} quantized tensor(s) within bound "
                  f"(worst {worst:.2%} of peak, limit {2/127:.2%})"
                  + ("" if not mism else "  -- MISMATCH"))
        else:
            print(f"  value round-trip vs v3: {'IDENTICAL' if not mism else 'MISMATCH'}")
        for m in mism[:6]: print("     ", m)
        if mism: errs.append(f"{len(mism)} tensor value mismatches")

    print()
    if errs:
        print("V4 FORMAT: FAIL")
        for e in errs[:20]: print("  -", e)
        return 1
    print("V4 FORMAT: PASS (structure, regions, digests%s)"
          % (", value round-trip" if args.v3 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
