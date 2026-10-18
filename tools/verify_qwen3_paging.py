#!/usr/bin/env python3
"""verify_qwen3_paging.py — C1: the pager must not change the answer.

WHY THIS EXISTS
  A pager is a change in WHERE bytes come from, not in what is computed. So the gate is
  bit-identity: the same binary, the same prompt, run with the pager attached at an
  UNLIMITED budget, must produce byte-identical .logits, .route and .router to the
  unpaged run. If the answer moves at all, the attachment point is wrong.

  Bit-identity alone is too weak to catch a pager that is correct and does not count,
  so the accounting is checked against an expectation derived from the FILE, not from
  the pager's own output:
    - token 0 at an unlimited budget must fetch every region its working set touches:
      48 layers x (9 core regions + 8 experts) + embed + final_norm + lm_head = 819;
    - the bytes fetched must equal the sum of THOSE regions' declared lengths, where
      the chosen experts are read back from the .route trace the engine emitted.
  A cumulative-vs-per-token mix-up, a region count that includes evictions, or a
  granularity of tensor-instead-of-region all fail this and pass a self-check.

USAGE
  python3 tools/verify_qwen3_paging.py                    # default model/prompt
  python3 tools/verify_qwen3_paging.py --budget-mb 8192   # also check a finite budget
"""
import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HDR = struct.Struct("<IIII IIII QQQQQ IIII")
KIND_EXPERT = 1

# p0 of data/traces/qwen3_30b_a3b_full.npz ("The capital of France is")
DEFAULT_IDS = "785,6722,315,9625,374"
DEFAULT_MODEL = "weights/qwen3_30b_a3b_q8.v4cp"


class Reader:
    def __init__(self, b, off=0):
        self.b, self.o = b, off
    def u8(self):  v = self.b[self.o]; self.o += 1; return v
    def u16(self): v = struct.unpack_from("<H", self.b, self.o)[0]; self.o += 2; return v
    def u32(self): v = struct.unpack_from("<I", self.b, self.o)[0]; self.o += 4; return v
    def u64(self): v = struct.unpack_from("<Q", self.b, self.o)[0]; self.o += 8; return v
    def string(self):
        n = self.u16(); s = self.b[self.o:self.o + n].decode(); self.o += n; return s


def read_tables(path):
    """Region and tensor tables, read from the FILE. Deliberately independent of the
    pager: this is the expectation the pager is judged against."""
    with open(path, "rb") as f:
        head = f.read(88)
        (magic, ver, hbytes, flags, nt, nb, nr, ne,
         o_enc, o_tens, o_blk, o_reg, o_data, *_rest) = HDR.unpack_from(head, 0)
        assert magic == 0x34575043 and ver == 4, "not a v4 file"
        f.seek(o_tens)
        blob = f.read(o_blk - o_tens)
        r = Reader(blob)
        n = r.u32()
        assert n == nt, f"tensor table count {n} != header {nt}"
        tens = []
        for i in range(nt):
            name = r.string(); r.u32(); rank = r.u8()
            shape = tuple(r.u32() for _ in range(rank))
            r.u64(); r.u64(); r.u32(); r.u16()
            tens.append(dict(i=i, name=name, shape=shape))
        f.seek(o_reg)
        blob = f.read(o_data - o_reg)
        r = Reader(blob)
        n = r.u32()
        assert n == nr, f"region table count {n} != header {nr}"
        regs = []
        for i in range(nr):
            off = r.u64(); ln = r.u64(); lo = r.u32(); hi = r.u32()
            kind = r.u8(); slot = r.u16(); mlo = r.u32(); mhi = r.u32(); r.u64()
            regs.append(dict(id=i, off=off, len=ln, lo=lo, hi=hi, kind=kind,
                             slot=slot, mlo=mlo, mhi=mhi))
    return tens, regs


def expect_token0(tens, regs, route, k):
    """The regions token 0's working set touches, and their declared byte total.

    Core = 9 regions per layer (ln1, ln2, q_norm, k_norm, Wq, Wk, Wv, Wo, Wgate) plus
    embed, final_norm and lm_head; experts = one region per (layer, chosen expert).
    Every one is looked up by tensor NAME in the file's own tables, so a region
    granularity the engine got wrong cannot hide here."""
    by_name = {t["name"]: t["i"] for t in tens}
    reg_of = {}
    for rg in regs:
        for t in range(rg["lo"], rg["hi"]):
            reg_of[t] = rg["id"]
    L, T, K = route.shape
    core_ids, missing = set(), []
    for name in ("embed", "final_norm", "lm_head"):
        if name not in by_name:
            missing.append(name); continue
        core_ids.add(reg_of[by_name[name]])
    for l in range(L):
        for stem in ("ln1", "ln2", "q_norm", "k_norm", "Wq", "Wk", "Wv", "Wo", "Wgate"):
            nm = f"{stem}.{l}"
            if nm not in by_name:
                missing.append(nm); continue
            core_ids.add(reg_of[by_name[nm]])
    exp_ids = set()
    for l in range(L):
        for j in range(K):
            e = int(route[l, 0, j])
            nm = f"expert.{l}.{e}.Wg"
            if nm not in by_name:
                missing.append(nm); continue
            exp_ids.add(reg_of[by_name[nm]])
    ids = core_ids | exp_ids
    nbytes = sum(regs[i]["len"] for i in ids)
    return len(core_ids), len(exp_ids), len(ids), nbytes, missing


def run(engine, model, ids, prefix, env_extra):
    env = {k: v for k, v in __import__("os").environ.items()}
    env.pop("CP_PAGER", None)
    env.pop("CP_BUDGET", None)
    env.pop("CP_PAGER_SYS", None)
    env.update(env_extra)
    cmd = [engine, model, "48", ids, prefix]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stderr[-1200:])
        raise SystemExit(f"engine failed: {' '.join(cmd)}")
    return r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--engine", default="./build/qwen3_eval")
    ap.add_argument("--tokens", default=DEFAULT_IDS)
    ap.add_argument("--budget-mb", type=int, default=0,
                    help="also check this finite budget (C2's knob); 0 = unlimited only")
    ap.add_argument("--keep", default="scratch/c1")
    args = ap.parse_args()
    ids = args.tokens
    T = len(ids.split(","))
    pre = Path(args.keep)
    pre.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"C1: pager attachment — bit-identity and per-token accounting")
    print("=" * 78)
    print(f"  model {args.model}  ({T} tokens: {ids})")

    run(args.engine, args.model, ids, str(pre) + "_unpaged", {})
    run(args.engine, args.model, ids, str(pre) + "_unlimited", {"CP_PAGER": "1"})

    ok = True
    for suf in (".logits", ".route", ".router"):
        a = (Path(str(pre) + "_unpaged" + suf)).read_bytes()
        b = (Path(str(pre) + "_unlimited" + suf)).read_bytes()
        same = a == b
        ok &= same
        print(f"  {suf:9s} {len(a):>10d} bytes unpaged vs unlimited-budget pager: "
              f"{'BIT-IDENTICAL' if same else 'DIFFERS'}")

    # ---- accounting against the file, not against the pager
    tens, regs = read_tables(args.model)
    L, K = 48, 8
    route = np.fromfile(str(pre) + "_unlimited.route", dtype="<i4").reshape(L, T, K)
    n_core, n_exp, n_ids, n_bytes, missing = expect_token0(tens, regs, route, K)
    print(f"  file says token 0 touches {n_core} core + {n_exp} expert regions "
          f"= {n_ids} regions, {n_bytes/1e6:.1f} MB")
    if missing:
        print(f"  MISSING tensors: {missing[:4]}")
        ok = False

    rows = [ln.split() for ln in Path(str(pre) + "_unlimited.pager").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    num = [[int(x) for x in r] for r in rows]
    if len(num) != T:
        print(f"  per-token rows {len(num)} != {T} tokens")
        ok = False
    else:
        t0 = num[0]
        print(f"  pager says token 0 fetched {t0[1]} regions, {t0[2]/1e6:.1f} MB")
        if t0[1] != n_ids or t0[2] != n_bytes:
            print(f"  MISMATCH: expected {n_ids} regions / {n_bytes} bytes")
            ok = False
        # unlimited budget: nothing may ever be evicted, and every region is fetched once
        evict = sum(r[4] for r in num)
        distinct = sum(r[1] for r in num)
        print(f"  over {T} tokens: {distinct} region fetches, {evict} evictions "
              f"(unlimited budget: evictions must be 0)")
        if evict != 0:
            ok = False
        for r in num:
            print(f"    token {r[0]}: {r[1]:>4d} regions {r[2]/1e6:>8.1f} MB "
                  f"{r[3]:>4d} hits {r[4]:>4d} evictions")
        if num[0][3] != 0:
            print("  token 0 is the cold token: it must have no hits")
            ok = False

    if args.budget_mb:
        bm = args.budget_mb
        run(args.engine, args.model, ids, str(pre) + "_b", {"CP_PAGER": "1",
                                                           "CP_BUDGET": str(bm)})
        for suf in (".logits", ".route", ".router"):
            a = (Path(str(pre) + "_unpaged" + suf)).read_bytes()
            b = (Path(str(pre) + "_b" + suf)).read_bytes()
            same = a == b
            ok &= same
            print(f"  {suf:9s} at CP_BUDGET={bm} MB: {'BIT-IDENTICAL' if same else 'DIFFERS'}")
        rows = [ln.split() for ln in Path(str(pre) + "_b.pager").read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
        num = [[int(x) for x in r] for r in rows]
        print(f"  CP_BUDGET={bm} MB: "
              + ", ".join(f"tok {r[0]} {r[1]} regions/{r[4]} evict" for r in num))

    print()
    print("C1 PAGING: " + ("PASS — bit-identical at unlimited budget, accounting matches the file"
                           if ok else "FAIL — see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
