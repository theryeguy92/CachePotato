#!/usr/bin/env python3
"""permute_experts_cpw1_v3.py — apply an expert permutation to a CPW1 v3 file.

WHY THIS EXISTS
  The Tier-B ("layout compilation") direction claims that a compiled weight layout
  is a *permutation* of the base model: same arithmetic, different byte order, so
  the engine's output must be bit-identical to the unpermuted file. That claim is
  the foundation of the whole direction and is asserted nowhere — it is tested
  here, on an artifact we already have, before any format or engine work depends
  on it.

  Concretely: permuting expert blocks in the file is only a symmetry if the
  router's expert dimension (Wr's last axis, and br) is permuted to match. This
  writes the permuted file; verify_cpw1_v3.py then runs the existing gates over
  it (round-trip + recorded routing + C engine) expecting max|Δ| = 0.000e+00 and
  0 routing mismatches.

WHAT IT DOES NOT DO
  No numerics change. No canonical summation order is introduced anywhere; if the
  gate fails, the failure is reported, not worked around (a canonical sum would be
  a numerics change requiring explicit approval).

USAGE
  python3 tools/permute_experts_cpw1_v3.py --src weights/moe-scale-run/l05.cpw1v3 \
      --dst scratch/l05_permuted.cpw1v3 --seed 20261016 [--report scratch/perm.json]
"""
import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from cpw1_v3 import HDR_INT_FIELDS, HDR_SIZE, payload_layout  # noqa: E402


def read_header(path):
    raw = open(path, "rb").read(HDR_SIZE)
    if len(raw) < HDR_SIZE:
        raise ValueError("file shorter than a CPW1 header")
    magic, version = struct.unpack_from("<II", raw, 0)
    if magic != 0x31575043:
        raise ValueError("bad magic 0x%08x" % magic)
    if version != 3:
        raise ValueError("expected CPW1 v3, got v%d" % version)
    vals = struct.unpack_from("<%di" % len(HDR_INT_FIELDS), raw, 8)
    hdr = dict(zip(HDR_INT_FIELDS, vals))
    hdr["magic"], hdr["version"] = magic, version
    return hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    hdr = read_header(args.src)
    L, d, N = hdr["n_layer"], hdr["d_model"], hdr["n_exp"]
    n_moe, nd = hdr["n_moe_layers"], hdr["n_dense_layers"]
    print("source   : %s" % args.src)
    print("header   : d=%d N=%d layers=%d (moe %d, dense %d)"
          % (d, N, L, n_moe, nd))

    src_size = Path(args.src).stat().st_size
    # copy the payload verbatim, then patch the two regions that must move together
    shutil.copyfile(args.src, args.dst)
    with open(args.dst, "r+b") as f:
        f.seek(HDR_SIZE)
        arrays, offset = {}, HDR_SIZE
        for name, shape in payload_layout(_Cfg(hdr)):
            n = int(np.prod(shape))
            f.seek(offset)
            arrays[name] = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(shape).copy()
            offset += n * 4
        assert offset == src_size, "payload walk did not consume the file"

        perms = {}
        for slot in range(n_moe):
            rng = np.random.default_rng(args.seed + slot)
            perm = rng.permutation(N)
            # experts[slot, new_pos, :] = original_experts[slot, perm[new_pos], :]
            arrays["experts"][slot] = arrays["experts"][slot][perm]
            # router must move with them: Wr[slot, :, new_pos] = Wr[slot, :, perm[new_pos]]
            arrays["Wr"][slot] = arrays["Wr"][slot][:, perm]
            arrays["br"][slot] = arrays["br"][slot][perm]
            perms[str(slot)] = perm.tolist()

        offset = HDR_SIZE
        for name, _shape in payload_layout(_Cfg(hdr)):
            f.seek(offset)
            f.write(np.ascontiguousarray(arrays[name], dtype="<f4").tobytes())
            offset += arrays[name].size * 4

    dst_size = Path(args.dst).stat().st_size
    assert dst_size == src_size, "size changed: %d -> %d" % (src_size, dst_size)

    digest = hashlib.sha256(Path(args.dst).read_bytes()).hexdigest()
    print("written  : %s (%d bytes, identical size, sha256 %s)"
          % (args.dst, dst_size, digest[:16]))
    print("permuted : %d experts/layer across %d MoE slots (seed %d, per-slot rng)"
          % (N, n_moe, args.seed))
    if args.report:
        Path(args.report).write_text(json.dumps(
            {"src": args.src, "dst": args.dst, "seed": args.seed,
             "N": N, "n_moe": n_moe, "sha256": digest, "perms": perms}, indent=1))
        print("report   : %s" % args.report)
    return 0


class _Cfg:
    """Minimal config shim: payload_layout() only reads these attributes."""

    def __init__(self, hdr):
        self.vocab, self.d_model, self.n_heads = hdr["vocab"], hdr["d_model"], hdr["n_heads"]
        self.d_ff, self.n_layer, self.n_ctx = hdr["d_ff"], hdr["n_layer"], hdr["n_ctx"]
        self.n_exp, self.exp_hid, self.topk = hdr["n_exp"], hdr["exp_hid"], hdr["topk"]
        self.moe_from = hdr["moe_from"]
        self._nm, self._nd = hdr["n_moe_layers"], hdr["n_dense_layers"]

    def moe_layers(self):
        return list(range(self._nm))

    def dense_layers(self):
        return list(range(self._nd))


if __name__ == "__main__":
    raise SystemExit(main())
