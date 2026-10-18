#!/usr/bin/env python3
"""c2_measure.py — C2's measurement slice: join the pager's counters with the wall clock.

WHAT IT MEASURES, AND WHY IN THIS SHAPE
  The headline is REGIONS FETCHED PER TOKEN, with bytes/token, hit rate, evictions and tok/s
  beside it. Those five numbers come from two instruments that must be cross-checked, not
  trusted separately:

    the pager   regions/bytes fetched, hits, evictions  (logical: what the policy did)
    the kernel  /proc/self/io read_bytes per step       (physical: did it really hit the disk)

  When CP_PAGER_SYS is off the page cache can serve a "miss", so the logical read count and
  the physical one diverge — and that divergence is the difference between the pager working
  and the OS masking it. Both are printed for every config.

WHAT "control" IS
  CP_PAGER=1 with no budget counts every access and never evicts: that is the OS page cache
  with instrumentation, i.e. the unpaged run. Its regions/token is the DEMAND (how much the
  token needs), while a budgeted run's regions/token is what the POLICY fetched. The two are
  different claims and the table labels them so.

USAGE
  python3 tools/c2_measure.py --gen 8 --runs control,b8192,b8192sys
  python3 tools/c2_measure.py --runs control,b8192,b8192sys,b4096,b2048 --gen 32 --out bench/results/c2
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = {
    # name: (env, label)
    "control":  ({"CP_PAGER": "1"},
                 "unpaged control (counts, never evicts)"),
    "b8192":    ({"CP_PAGER": "1", "CP_BUDGET": "8192"}, "8 GB budget"),
    "b8192sys": ({"CP_PAGER": "1", "CP_BUDGET": "8192", "CP_PAGER_SYS": "1"},
                 "8 GB budget + system cache dropped"),
    "b4096":    ({"CP_PAGER": "1", "CP_BUDGET": "4096"}, "4 GB budget"),
    "b4096sys": ({"CP_PAGER": "1", "CP_BUDGET": "4096", "CP_PAGER_SYS": "1"},
                 "4 GB budget + system cache dropped"),
    "b2048":    ({"CP_PAGER": "1", "CP_BUDGET": "2048"}, "2 GB budget"),
    "b2048sys": ({"CP_PAGER": "1", "CP_BUDGET": "2048", "CP_PAGER_SYS": "1"},
                 "2 GB budget + system cache dropped"),
}


def parse_pager(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rows.append([int(x) for x in line.split()])
    return rows           # step regions bytes hits evictions


def parse_steps(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        pos, tok, ms, kb = line.split()
        rows.append((int(pos), int(tok), float(ms), int(kb)))
    return rows


def parse_report(text):
    """The pager's own split, which is the only place expert and core are separated."""
    out = {}
    m = re.search(r"experts\s*:\s*(\d+) accesses, (\d+) hits \(([\d.]+)%\)", text)
    if m:
        out["expert_hit_pct"] = float(m.group(3))
        out["expert_accesses"] = int(m.group(1))
    m = re.search(r"core\s*:\s*(\d+) accesses, (\d+) hits \(([\d.]+)%\)", text)
    if m:
        out["core_hit_pct"] = float(m.group(3))
    m = re.search(r"(\d+) evictions, (\d+) MB evicted", text)
    if m:
        out["evictions"] = int(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/qwen3_30b_a3b_q8.v4cp")
    ap.add_argument("--engine", default="./build/qwen3_eval")
    ap.add_argument("--tokens", default="785,6722,315,9625,374")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--runs", default="control,b8192,b8192sys")
    ap.add_argument("--keep", default="scratch/c2")
    ap.add_argument("--out", default="", help="write a CSV/notes pair here")
    args = ap.parse_args()

    prompt = [int(x) for x in args.tokens.split(",")]
    T = len(prompt)
    keep = Path(args.keep)
    keep.mkdir(parents=True, exist_ok=True)   # the engine writes <prefix>.* into it
    names = [n.strip() for n in args.runs.split(",") if n.strip()]
    for n in names:
        if n not in CONFIGS:
            print(f"unknown config {n}; known: {', '.join(CONFIGS)}")
            return 2

    results = []
    print("=" * 100)
    print(f"C2 measurement: {args.model}, prompt {T} tokens, {args.gen} generated, greedy")
    print("=" * 100)
    for name in names:
        env_cfg, label = CONFIGS[name]
        env = {k: v for k, v in os.environ.items() if not k.startswith("CP_PAGER")}
        env.pop("CP_BUDGET", None)
        env.update(env_cfg)
        pre = str(keep / name)
        cmd = [args.engine, args.model, "48", args.tokens, pre, "--decode", str(args.gen)]
        print(f"\n[{name}] {label}")
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        wall = time.time() - t0
        if r.returncode:
            print(r.stderr[-1500:])
            return 1
        (keep / f"{name}.stderr").write_text(r.stderr)

        pager = parse_pager(str(keep / f"{name}.pager"))
        steps = parse_steps(str(keep / f"{name}.steps"))
        rep = parse_report(r.stderr)
        if len(steps) != args.gen:
            print(f"  expected {args.gen} steps, got {len(steps)}")
            return 1
        # Per generated token: the row charged to the position that was fed (see the note in
        # qwen3_eval.c — the final norm + head touches are charged to the position whose
        # logits they produce, one step earlier than the feed they predict).
        per = []
        for s, (pos, tok, ms, kb) in enumerate(steps):
            # pager_dump_steps() writes one row per STEP and the step index IS the position
            # (qwen3_layer_kv calls pager_step(pos0+t)), so row number == position. This
            # was off by one in the first version of this script: it read the next position
            # and zero-filled the last, which shifted every number and dropped a row.
            if pos >= len(pager):
                print(f"  no pager row for position {pos} (rows {len(pager)})")
                return 1
            row = pager[pos]
            per.append(dict(pos=pos, tok=tok, ms=ms, kb=kb,
                            regions=row[1], bytes=row[2], hits=row[3], evict=row[4]))
        n = len(per)
        # Independent expectation, so a shifted or truncated table cannot look plausible: a
        # decode position touches 48 layers x (9 core + 8 experts) + embed = 817 regions,
        # plus final_norm + lm_head charged to the position their logits come from.
        for p_ in per[:-1]:
            if not (810 <= p_["regions"] + p_["hits"] <= 825):
                print(f"  WARNING: position {p_['pos']} logged "
                      f"{p_['regions'] + p_['hits']} region touches, expected ~819")
        tot_ms = sum(p["ms"] for p in per) / 1000.0
        res = dict(name=name, label=label, per=per,
                   regions=sum(p["regions"] for p in per) / n,
                   mbytes=sum(p["bytes"] for p in per) / n / 1e6,
                   hits=sum(p["hits"] for p in per) / n,
                   evict=sum(p["evict"] for p in per) / n,
                   ms=sum(p["ms"] for p in per) / n,
                   disk_mb=sum(p["kb"] for p in per) / n / 1024.0,
                   tps=n / tot_ms if tot_ms else 0.0,
                   expert_hit=rep.get("expert_hit_pct", float("nan")),
                   core_hit=rep.get("core_hit_pct", float("nan")),
                   wall=wall)
        results.append(res)
        print(f"  regions/token {res['regions']:.1f}   MB/token {res['mbytes']:.1f}   "
              f"hits/token {res['hits']:.1f}   evict/token {res['evict']:.1f}")
        print(f"  expert hit {res['expert_hit']:.2f}%   core hit {res['core_hit']:.2f}%   "
              f"physical MB/token {res['disk_mb']:.1f}")
        print(f"  {res['ms']:.0f} ms/token -> {res['tps']:.3f} tok/s   (wall {wall:.1f}s)")

    print("\n" + "=" * 100)
    print(f"{'config':<12}{'regions/tok':>12}{'MB/tok':>9}{'hits/tok':>9}{'evict/tok':>10}"
          f"{'exp hit%':>9}{'core hit%':>10}{'phys MB/tok':>12}{'ms/tok':>8}{'tok/s':>7}")
    for r in results:
        print(f"{r['name']:<12}{r['regions']:>12.1f}{r['mbytes']:>9.1f}{r['hits']:>9.1f}"
              f"{r['evict']:>10.1f}{r['expert_hit']:>9.2f}{r['core_hit']:>10.2f}"
              f"{r['disk_mb']:>12.1f}{r['ms']:>8.0f}{r['tps']:>7.3f}")

    # Read the control row correctly: with no eviction, a region is counted once for the
    # whole run, so its regions/token and hits/token are DEMAND and reuse, not a steady-state
    # fetch count. Only a budgeted row's regions/token is "what the policy fetched".
    print("\n  control: regions/token is DEMAND (nothing is evicted, so a region is counted"
          " once ever);")
    print("           budgeted rows are what the policy fetched. Compare them as two claims,"
          " not one.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(str(out) + ".csv", "w") as f:
            f.write("config,regions_per_token,mb_per_token,hits_per_token,evictions_per_token,"
                    "expert_hit_pct,core_hit_pct,physical_mb_per_token,ms_per_token,tok_per_s\n")
            for r in results:
                f.write(f"{r['name']},{r['regions']:.2f},{r['mbytes']:.2f},{r['hits']:.2f},"
                        f"{r['evict']:.2f},{r['expert_hit']:.2f},{r['core_hit']:.2f},"
                        f"{r['disk_mb']:.2f},{r['ms']:.1f},{r['tps']:.4f}\n")
        with open(str(out) + ".txt", "w") as f:
            f.write(f"C2 measurement {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
            f.write(f"model {args.model}\nprompt {args.tokens}\ngen {args.gen}\n")
            f.write(open(str(keep / f"{names[-1]}.meta")).read())
        print(f"\nwrote {out}.csv and {out}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
