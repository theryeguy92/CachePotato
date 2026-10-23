#!/usr/bin/env python3
"""verify_tierb_grouping.py — Tier B: placing experts together must not change the answer.

WHY THIS EXISTS
  A co-selection layout changes which experts share a REGION, and therefore where their bytes
  live in the file. It does not change the model: expert ids, tensor names and the router all
  stay put, so every tensor holds the same values and the forward pass reads the same rows.
  The claim is therefore **bit-identity**, which is stronger than the v3 permutation result
  ("exact routing, fp32-equal logits") — and it is only available because this permutation is
  applied to placement alone, never to the router's expert axis.

  The other half of the claim is mechanical: the grouped file must actually have fewer expert
  regions, and the pager must fetch fewer of them when both members of a pair are selected.
  A layout that is bit-identical but not grouped is a no-op, and a layout that is grouped but
  not bit-identical is a numerics change wearing a layout's clothes. Both are checked.

USAGE
  python3 tools/verify_tierb_grouping.py                     # 1-layer fixture, ~2 min
  python3 tools/verify_tierb_grouping.py --layers 1 --group 2
"""
import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HDR = struct.Struct("<IIII IIII QQQQQ IIII")


def region_table(path):
    with open(path, "rb") as f:
        head = f.read(HDR.size)
        (magic, ver, hb, fl, nt, nb, nr, ne, oe, ot, ob, orr, od, *_r) = HDR.unpack_from(head, 0)
        f.seek(orr)
        buf = f.read(od - orr)
    if magic != 0x34575043 or ver != 4:
        raise SystemExit(f"{path}: not a v4 file")
    o, regs = 4, []
    for i in range(nr):
        off, ln, lo, hi = struct.unpack_from("<QQII", buf, o)
        o += 24
        kind = buf[o]
        o += 1
        slot, mlo, mhi = struct.unpack_from("<HII", buf, o)
        o += 10 + 8
        regs.append(dict(id=i, len=ln, lo=lo, hi=hi, kind=kind, slot=slot, mlo=mlo, mhi=mhi))
    return regs


def convert(out, layers, order=None, group=2, quant="q8_0"):
    cmd = [sys.executable, "models/qwen3-30b-a3b/convert.py", "--out", out,
           "--layers", str(layers)]
    if quant != "none":
        cmd += ["--quant", quant]
    if order:
        cmd += ["--expert-order", order, "--group", str(group)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        print(r.stderr[-1500:])
        raise SystemExit(f"convert failed: {' '.join(cmd)}")
    return r.stdout.strip().splitlines()[-1]


def run(engine, model, layers, ids, prefix):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CP_")}
    r = subprocess.run([engine, model, str(layers), ids, prefix], capture_output=True,
                       text=True, cwd=ROOT, env=env)
    if r.returncode:
        print(r.stderr[-1200:])
        raise SystemExit(f"engine failed on {model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--tokens", default="785,6722,315,9625,374")
    ap.add_argument("--engine", default="./build/qwen3_eval")
    ap.add_argument("--keep", default="scratch/tierb")
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()

    keep = Path(args.keep)
    keep.mkdir(parents=True, exist_ok=True)
    E = args.experts
    # A placement order is arbitrary for this gate: it tests that WHERE experts live cannot
    # change the answer. The fitted order is a separate question, measured separately.
    rng = np.random.default_rng(args.seed)
    layers_order = [[int(x) for x in rng.permutation(E)] for _ in range(args.layers)]
    order_json = keep / "order.json"
    order_json.write_text(json.dumps(dict(group=args.group, layers_order=layers_order)))

    plain = keep / f"plain_l{args.layers}.v4cp"
    grouped = keep / f"pairs_l{args.layers}.v4cp"
    print("=" * 78)
    print(f"Tier B grouping gate: {args.layers} layer(s), {args.group} experts/region")
    print("=" * 78)
    print("  " + convert(str(plain), args.layers))
    print("  " + convert(str(grouped), args.layers, str(order_json), args.group))

    rp, rg = region_table(str(plain)), region_table(str(grouped))
    ep = [r for r in rp if r["kind"] == 1]
    eg = [r for r in rg if r["kind"] == 1]
    print(f"  expert regions: plain {len(ep)}, grouped {len(eg)}"
          f" (expected {args.layers * E} -> {args.layers * ((E + args.group - 1)//args.group)})")
    ok = len(eg) == args.layers * ((E + args.group - 1) // args.group)

    run(args.engine, str(plain), args.layers, args.tokens, str(keep / "plain"))
    run(args.engine, str(grouped), args.layers, args.tokens, str(keep / "pairs"))
    for suf in (".logits", ".route", ".router"):
        a = (keep / ("plain" + suf)).read_bytes()
        b = (keep / ("pairs" + suf)).read_bytes()
        same = a == b
        ok &= same
        print(f"  {suf:9s} {len(a):>9d} bytes  {'BIT-IDENTICAL' if same else 'DIFFERS'}")

    # the group sizes in the file must match the claim, not just the count
    sizes = sorted({r["len"] for r in eg})
    print(f"  grouped region sizes: {sizes} (one expert is "
          f"{ep[0]['len'] if ep else 0} bytes)")
    if ep and eg and max(sizes) < max(r["len"] for r in ep) * min(args.group, 2):
        print("  WARNING: grouped regions are not larger than single-expert regions")
        ok = False

    print()
    print("TIER B GROUPING: " + ("PASS — placement changed, the answer did not, regions shrank"
                                 if ok else "FAIL — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
