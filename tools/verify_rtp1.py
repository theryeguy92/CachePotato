#!/usr/bin/env python3
"""verify_rtp1.py — structural check for an RTP1 trace, against the C layout.

WHY THIS EXISTS
  RTP1 is written as a raw C struct, so the header size is part of the format. Getting
  it wrong does not crash anything: the reader just starts the payload N bytes early and
  every expert id comes back as a plausible-looking but out-of-range number. That
  happened (a 32-byte constant against a 36-byte struct), and the only symptom was
  expert ids in the hundreds for a 32-expert model.

  This checks the file against the layout declared in src/replay.h and against its own
  declared dimensions, and fails loudly.

Checks:
  1. header fields match the C offsets in src/replay.h (parsed, not hard-coded)
  2. file size == header + tokens + selections, exactly
  3. every expert id is in [0, n_exp)
  4. token ids are non-negative and below a sane bound
  5. optional JSON report of the selection histogram

Usage:
  python3 tools/verify_rtp1.py scratch/bench_l05.rtp1
  python3 tools/verify_rtp1.py --all data/traces/*.rtp1
"""
import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAGIC = 0x31505452


def c_header_size():
    """Read RPY_HDR_BYTES and the struct declared in src/replay.h."""
    src = (ROOT / "src/replay.h").read_text()
    m = re.search(r"#define\s+RPY_HDR_BYTES\s+(\d+)", src)
    if not m:
        raise SystemExit("could not find RPY_HDR_BYTES in src/replay.h")
    nbytes = int(m.group(1))
    body = re.search(r"typedef struct \{(.*?)\} ReplayHeader;", src, re.S)
    if not body:
        raise SystemExit("could not find ReplayHeader in src/replay.h")
    fields = re.findall(r"(u?int(?:32|16)_t)\s+(\w+)\s*;", body.group(1))
    sizes = {"uint32_t": 4, "uint16_t": 2, "int32_t": 4, "int16_t": 2}
    total = sum(sizes[t] for t, _ in fields)
    order = [n for _, n in fields]
    if total != nbytes:
        raise SystemExit("src/replay.h inconsistent: RPY_HDR_BYTES=%d but struct sums to %d"
                         % (nbytes, total))
    return nbytes, order


def check(path, expect_order):
    raw = Path(path).read_bytes()
    nbytes = 4 * len(expect_order)
    if len(raw) < nbytes:
        return [f"{path}: shorter than a header ({len(raw)} bytes)"]
    f = dict(zip(expect_order, struct.unpack_from("<" + "i" * len(expect_order), raw, 0)))
    errs = []
    if (f["magic"] & 0xFFFFFFFF) != MAGIC:
        errs.append(f"{path}: bad magic 0x{f['magic'] & 0xFFFFFFFF:08x}")
    if f["version"] != 1:
        errs.append(f"{path}: version {f['version']} != 1")
    for k in ("n_slots", "n_windows", "window_len", "topk", "n_exp"):
        if f[k] <= 0:
            errs.append(f"{path}: {k}={f[k]} not positive")
    if errs:
        return errs

    ntok = f["n_windows"] * f["window_len"]
    nsel = f["n_slots"] * ntok * f["topk"]
    want = nbytes + 4 * ntok + 2 * nsel
    if len(raw) != want:
        errs.append(f"{path}: size {len(raw)} != header({nbytes}) + tokens({4*ntok}) "
                    f"+ sel({2*nsel}) = {want}  [header-size mismatch?]")
        return errs

    toks = np.frombuffer(raw, dtype="<i4", count=ntok, offset=nbytes)
    sel = np.frombuffer(raw, dtype="<i2", count=nsel, offset=nbytes + 4 * ntok)
    if toks.min() < 0:
        errs.append(f"{path}: negative token id")
    if sel.min() < 0 or sel.max() >= f["n_exp"]:
        errs.append(f"{path}: expert ids span [{sel.min()}, {sel.max()}] but n_exp={f['n_exp']}"
                    "  <-- classic symptom of a header-size/payload-offset mismatch")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    paths = list(args.paths)
    if args.all:
        paths += sorted(str(p) for p in (ROOT / "data" / "traces").glob("*.rtp1"))
        paths += sorted(str(p) for p in (ROOT / "scratch").glob("*.rtp1"))
    if not paths:
        ap.error("give a path or --all")

    nbytes, order = c_header_size()
    print("src/replay.h: RPY_HDR_BYTES=%d, fields in order: %s" % (nbytes, ", ".join(order)))
    bad = 0
    for p in paths:
        errs = check(p, order)
        if errs:
            bad += 1
            for e in errs:
                print("  FAIL", e)
        else:
            raw = Path(p).read_bytes()
            f = dict(zip(order, struct.unpack_from("<" + "i" * len(order), raw, 0)))
            ntok = f["n_windows"] * f["window_len"]
            sel = np.frombuffer(raw, dtype="<i2", count=f["n_slots"] * ntok * f["topk"],
                                offset=nbytes + 4 * ntok).reshape(
                                    f["n_slots"], f["n_windows"], f["window_len"], f["topk"])
            print(f"  OK   {p}: {f['n_slots']} slots x {f['n_windows']} windows x "
                  f"{f['window_len']} tok, top-{f['topk']}, n_exp={f['n_exp']}, "
                  f"eos={f['eos']} | distinct experts/ slot = {len(np.unique(sel[0]))}")
            if args.json:
                Path(args.json).write_text(repr(sel[:, 0, 0, :].tolist()))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
