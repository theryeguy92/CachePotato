#!/usr/bin/env python3
"""npz_to_rtp1.py — convert a recorded routing trace (.npz) to the RTP1 format the
C replay simulator reads (src/replay.h).

RTP1 is a flat little-endian dump: a 32-byte header, then token ids, then the
selections. It exists so `cachepotato --pager-sim` can sweep layouts at C speed
without linking a NumPy/zip reader into a dependency-free C11 engine.

Layout (little-endian, packed):
    u32 magic "RTP1" (0x31505452)
    u32 version = 1
    i32 n_slots, n_windows, window_len, topk, n_exp, eos
    u32 lam_bits            (float32 bit pattern, provenance only)
    i32 tokens[n_windows * window_len]
    i16 sel[n_slots * n_windows * window_len * topk]

Usage:
    python3 tools/npz_to_rtp1.py --src data/traces/l05_fp32.npz \
        --dst data/traces/l05_fp32.rtp1
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x31505452  # 'RTP1'
VERSION = 1
HDR = struct.Struct("<IIiiiiiiI")  # 32 bytes


def convert(src, dst):
    z = np.load(src, allow_pickle=False)
    sel = z["sel"]
    if sel.ndim != 4:
        raise ValueError("sel must be (n_slots, n_windows, T, topk), got %r" % (sel.shape,))
    n_slots, n_windows, T, topk = (int(x) for x in sel.shape)
    tokens = z["tokens"].astype("<i4")
    if tokens.shape != (n_windows, T):
        raise ValueError("tokens %r does not match sel windows/T %r"
                         % (tokens.shape, (n_windows, T)))
    n_exp = int(sel.max()) + 1
    eos = int(z["eos"]) if "eos" in z.files else 50256
    lam = float(z["lam"]) if "lam" in z.files else float("nan")
    lam_bits = struct.unpack("<I", struct.pack("<f", lam))[0]

    with open(dst, "wb") as f:
        f.write(HDR.pack(MAGIC, VERSION, n_slots, n_windows, T, topk, n_exp, eos, lam_bits))
        f.write(tokens.astype("<i4").tobytes(order="C"))
        f.write(sel.astype("<i2").tobytes(order="C"))

    ntok = n_windows * T
    nsel = n_slots * ntok * topk
    exp_size = HDR.size + 4 * ntok + 2 * nsel
    got = Path(dst).stat().st_size
    assert got == exp_size, "size mismatch: wrote %d, expected %d" % (got, exp_size)
    print("wrote %s  (%d bytes)" % (dst, got))
    print("  slots %d | windows %d | T %d | topk %d | experts %d | lambda %.2f (bits 0x%08x)"
          % (n_slots, n_windows, T, topk, n_exp, lam, lam_bits))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    return convert(args.src, args.dst)


if __name__ == "__main__":
    raise SystemExit(main())
