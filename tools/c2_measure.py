#!/usr/bin/env python3
"""c2_measure.py — C2's measurement: cold-start, interleaved, repeated arms.

WHY IT LOOKS LIKE THIS
  The first C2 slice (bench/results/c2/2026-10-18-slice.md) produced a 2.2x difference in
  physical bytes between the unpaged control and the 8 GB pager that could not be trusted,
  for three reasons this script removes:

  1. CACHE HISTORY. "Cold" used to be whatever ran before. Every run now starts with every
     page of the model file dropped (CP_COLD_START=1 -> posix_fadvise DONTNEED), so each arm
     begins from the same state.
  2. ORDER. Arms are interleaved *within* a rep and the order is rotated per rep (a Latin
     square), so slow drift — the T480 throttles 3100 -> 2593 MHz under load — lands on all
     arms rather than on whichever ran last.
  3. ONE RUN. Repeated, and the noisy quantities (physical MB, ms, tok/s) are reported with
     their spread and as PAIRED differences against the control. The deterministic ones
     (regions/bytes fetched, hits, evictions) must be identical across reps — the policy is a
     function of the touches, and greedy decode produces the same tokens because the pager
     does not change the arithmetic (C1). Drift there is a bug, not noise, so it fails.

WHAT IS AND IS NOT RANDOM
  regions/token, bytes/token, hits/token, evictions/token: deterministic, asserted equal.
  physical MB/token, ms/token, tok/s: noisy — mean ± sd over reps, paired against control.

USAGE
  python3 tools/c2_measure.py --runs control,b8192,b8192sys --gen 32 --reps 3
  python3 tools/c2_measure.py --runs control,b8192,b8192sys,b4096,b4096sys --gen 32 --reps 3 \
      --out bench/results/c2/2026-10-18-8gb
"""
import argparse
import os
import re
import struct
import subprocess
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
    "b8192arena": ({"CP_PAGER": "1", "CP_BUDGET": "8192", "CP_PAGER_SYS": "1",
                    "CP_ARENA": "1"}, "8 GB budget + arena (pager owns the bytes)"),
    "b11264arena": ({"CP_PAGER": "1", "CP_BUDGET": "11264", "CP_PAGER_SYS": "1",
                     "CP_ARENA": "1"}, "11 GB budget + arena"),
    "b2048arenapin": ({"CP_PAGER": "1", "CP_BUDGET": "2048", "CP_PAGER_SYS": "1",
                       "CP_ARENA": "1", "CP_PIN_CORE": "1"},
                      "2 GB budget + arena + core pinned"),
    "b4096":    ({"CP_PAGER": "1", "CP_BUDGET": "4096"}, "4 GB budget"),
    "b4096sys": ({"CP_PAGER": "1", "CP_BUDGET": "4096", "CP_PAGER_SYS": "1"},
                 "4 GB budget + system cache dropped"),
    "b11264sys": ({"CP_PAGER": "1", "CP_BUDGET": "11264", "CP_PAGER_SYS": "1"},
                  "11 GB budget + system cache dropped (near the OS's own cache size)"),
    "b13312sys": ({"CP_PAGER": "1", "CP_BUDGET": "13312", "CP_PAGER_SYS": "1"},
                  "13 GB budget + system cache dropped"),
    "b14336sys": ({"CP_PAGER": "1", "CP_BUDGET": "14336", "CP_PAGER_SYS": "1"},
                  "14 GB budget + system cache dropped (near the machine's RAM)"),
    "b11264":   ({"CP_PAGER": "1", "CP_BUDGET": "11264"}, "11 GB budget"),
    "b2048pin":  ({"CP_PAGER": "1", "CP_BUDGET": "2048", "CP_PAGER_SYS": "1",
                   "CP_PIN_CORE": "1"}, "2 GB budget + sys drop + core pinned"),
    "b4096pin":  ({"CP_PAGER": "1", "CP_BUDGET": "4096", "CP_PAGER_SYS": "1",
                   "CP_PIN_CORE": "1"}, "4 GB budget + sys drop + core pinned"),
    "b2048":    ({"CP_PAGER": "1", "CP_BUDGET": "2048"}, "2 GB budget"),
    "b2048sys": ({"CP_PAGER": "1", "CP_BUDGET": "2048", "CP_PAGER_SYS": "1"},
                 "2 GB budget + system cache dropped"),
}


def parse_pager(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rows.append([int(x) for x in line.split()])      # step regions bytes hits evictions
    return rows


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
    m = re.search(r"core\s*:\s*(\d+) accesses, (\d+) hits \(([\d.]+)%\)", text)
    if m:
        out["core_hit_pct"] = float(m.group(3))
    m = re.search(r"^prefill: (\d+) ms, (\d+) KB read", text, re.M)
    if m:
        out["prefill_ms"] = float(m.group(1))
        out["prefill_kb"] = int(m.group(2))
    return out


def cpu_mhz():
    try:
        vals = [float(m.group(1)) for m in
                re.finditer(r"^cpu MHz\s*:\s*([\d.]+)", Path("/proc/cpuinfo").read_text(), re.M)]
        return sum(vals) / len(vals) if vals else 0.0
    except OSError:
        return 0.0


def mem_available_mb():
    try:
        m = re.search(r"^MemAvailable:\s*(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
        return int(m.group(1)) // 1024 if m else 0
    except OSError:
        return 0


def run_one(engine, model, ids, pre, gen, env_cfg, cold, layers=48, threads=4):
    # CP_THREADS must NOT be stripped with the other CP_* vars: the engine defaults to ONE
    # thread, so a harness that clears the environment silently measures single-threaded
    # numbers. That is exactly what happened to the first C2 matrix (0.147 tok/s was 1 thread;
    # 4 threads is 1.68x). Threads are now explicit and recorded.
    env = {k: v for k, v in os.environ.items() if not k.startswith("CP_")}
    env["CP_THREADS"] = str(threads)
    env.update(env_cfg)
    if cold:
        env["CP_COLD_START"] = "1"
    pre = Path(pre)
    for suf in (".pager", ".steps", ".tokens", ".logits", ".meta"):
        try:
            (pre.parent / (pre.name + suf)).unlink()
        except FileNotFoundError:
            pass
    cmd = [engine, model, str(layers), ids, str(pre), "--decode", str(gen)]
    mhz0, mem0 = cpu_mhz(), mem_available_mb()
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.time() - t0
    mhz1, mem1 = cpu_mhz(), mem_available_mb()
    if r.returncode:
        print(r.stderr[-1500:])
        raise SystemExit(f"engine failed: {' '.join(cmd)}")

    pager = parse_pager(str(pre) + ".pager")
    steps = parse_steps(str(pre) + ".steps")
    rep = parse_report(r.stderr)
    if len(steps) != gen:
        raise SystemExit(f"expected {gen} steps, got {len(steps)}")

    per = []
    for pos, tok, ms, kb in steps:
        # pager_dump_steps() writes one row per STEP and the step index IS the position
        # (qwen3_layer_kv calls pager_step(pos0+t)), so row number == position. This was off
        # by one in the first version of this script: it read the next position and zero-filled
        # the last, which shifted every number and dropped a row.
        if pos >= len(pager):
            raise SystemExit(f"no pager row for position {pos} (rows {len(pager)})")
        row = pager[pos]
        per.append(dict(pos=pos, tok=tok, ms=ms, kb=kb,
                        regions=row[1], bytes=row[2], hits=row[3], evict=row[4]))
    # Independent expectation, so a shifted or truncated table cannot look plausible: a decode
    # position touches 48 x (9 core + 8 experts) + embed = 817 regions, plus the final norm and
    # head charged to the position whose logits they produce.
    touch_expect = 17 * layers + 1          # 48 x (9 core + 8 experts) + embed
    for p in per[:-1]:
        if abs(p["regions"] + p["hits"] - touch_expect) > 8:
            print(f"  WARNING: position {p['pos']} logged {p['regions'] + p['hits']} touches,"
                  f" expected ~{touch_expect}")

    toks = list(struct.unpack(f"<{gen}i", Path(str(pre) + ".tokens").read_bytes()))
    n = len(per)
    tot_ms = sum(p["ms"] for p in per) / 1000.0
    return dict(
        per=per, tokens=toks, wall=wall,
        regions=sum(p["regions"] for p in per) / n,
        mbytes=sum(p["bytes"] for p in per) / n / 1e6,
        hits=sum(p["hits"] for p in per) / n,
        evict=sum(p["evict"] for p in per) / n,
        ms=sum(p["ms"] for p in per) / n,
        disk_mb=sum(p["kb"] for p in per) / n / 1024.0,
        tps=n / tot_ms if tot_ms else 0.0,
        expert_hit=rep.get("expert_hit_pct", float("nan")),
        core_hit=rep.get("core_hit_pct", float("nan")),
        prefill_ms=rep.get("prefill_ms", float("nan")),
        prefill_mb=rep.get("prefill_kb", 0) / 1024.0,
        mhz=(mhz0 + mhz1) / 2.0, mem_mb=mem1)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/qwen3_30b_a3b_q8.v4cp")
    ap.add_argument("--engine", default="./build/qwen3_eval")
    ap.add_argument("--tokens", default="785,6722,315,9625,374")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--runs", default="control,b8192,b8192sys")
    ap.add_argument("--keep", default="scratch/c2")
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=4,
                    help="CP_THREADS for every run; the engine defaults to 1")
    ap.add_argument("--no-cold", action="store_true",
                    help="skip the page-cache drop (only to show why it is needed)")
    args = ap.parse_args()

    prompt = [int(x) for x in args.tokens.split(",")]
    T = len(prompt)
    keep = Path(args.keep)
    keep.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.runs.split(",") if n.strip()]
    for n in names:
        if n not in CONFIGS:
            print(f"unknown config {n}; known: {', '.join(sorted(CONFIGS))}")
            return 2
    cold = not args.no_cold

    print("=" * 108)
    print(f"C2 measurement: {args.model}")
    print(f"  prompt {T} tokens, {args.gen} generated (greedy), {args.reps} reps, "
          f"cold_start={cold}, interleaved with rotated order")
    print(f"  arms: {', '.join(names)}")
    print("=" * 108)
    print(f"  rough estimate {args.reps * len(names) * args.gen * 6 / 60.0:.0f} min; "
          f"every arm starts from a dropped page cache\n")

    results = {}
    tokens_ref = None
    for rep in range(args.reps):
        # Latin-square rotation: each arm occupies each position in the order once, so order
        # bias (thermal ramp, cache state left by the previous arm) is spread across arms.
        order = names[rep % len(names):] + names[:rep % len(names)]
        for name in order:
            env_cfg, label = CONFIGS[name]
            pre = keep / f"{name}_r{rep}"
            print(f"  [rep {rep}] {name:<10} {label}")
            r = run_one(args.engine, args.model, args.tokens, pre, args.gen, env_cfg, cold,
                        args.layers, args.threads)
            r["name"] = name
            r["rep"] = rep
            results[(name, rep)] = r
            print(f"           regions/tok {r['regions']:7.1f}  MB/tok {r['mbytes']:7.1f}  "
                  f"evict/tok {r['evict']:6.1f}  phys MB/tok {r['disk_mb']:7.1f}  "
                  f"{r['ms']:5.0f} ms/tok  {r['tps']:.3f} tok/s  "
                  f"(prefill {r['prefill_ms']:.0f} ms, {r['prefill_mb']:.0f} MB)")
            if tokens_ref is None:
                tokens_ref = r["tokens"]
            elif r["tokens"] != tokens_ref:
                print(f"  FAIL: {name} rep {rep} generated different tokens than the first run"
                      f"\n    {r['tokens']}\n vs {tokens_ref}\n"
                      "  The pager must not change generation (C1); this is a bug, not noise.")
                return 1

    # ---- deterministic counters must not vary at all
    bad = False
    for name in names:
        for key in ("regions", "mbytes", "hits", "evict"):
            vals = [results[(name, r)][key] for r in range(args.reps)]
            if max(vals) - min(vals) > 1e-9:
                print(f"  FAIL: {name} {key} varies across reps without the arms changing: "
                      f"{vals} — the counters are a deterministic function of the touches")
                bad = True
    if bad:
        return 1

    def agg(name, key):
        return [results[(name, r)][key] for r in range(args.reps)]

    print("\n" + "=" * 108)
    print(f"means over {args.reps} reps (deterministic counters identical in every rep)")
    print(f"{'config':<11}{'regions/tok':>12}{'MB/tok':>9}{'evict/tok':>10}{'exp hit%':>9}"
          f"{'core hit%':>10}{'phys MB/tok':>14}{'ms/tok':>10}{'tok/s':>8}")
    for name in names:
        print(f"{name:<11}{mean(agg(name,'regions')):>12.1f}{mean(agg(name,'mbytes')):>9.1f}"
              f"{mean(agg(name,'evict')):>10.1f}{mean(agg(name,'expert_hit')):>9.2f}"
              f"{mean(agg(name,'core_hit')):>10.2f}"
              f"{mean(agg(name,'disk_mb')):>9.1f}±{sd(agg(name,'disk_mb')):.1f}"
              f"{mean(agg(name,'ms')):>7.0f}±{sd(agg(name,'ms')):.0f}"
              f"{mean(agg(name,'tps')):>8.3f}")

    # ---- paired comparison against the control
    if "control" in names and len(names) > 1:
        print("\n  paired vs control, per rep (negative delta = fewer physical bytes / faster):")
        for name in names:
            if name == "control":
                continue
            d_mb, d_ms, wins_mb, wins_tps = [], [], 0, 0
            for r in range(args.reps):
                c, x = results[("control", r)], results[(name, r)]
                d_mb.append(x["disk_mb"] - c["disk_mb"])
                d_ms.append(x["ms"] - c["ms"])
                wins_mb += x["disk_mb"] < c["disk_mb"]
                wins_tps += x["tps"] > c["tps"]
            print(f"    {name:<10} d(phys MB/tok) {mean(d_mb):+8.1f}  "
                  f"d(ms/tok) {mean(d_ms):+7.0f}  "
                  f"fewer-bytes {wins_mb}/{args.reps}  faster {wins_tps}/{args.reps}")

    print("\n  control: regions/token is DEMAND (nothing is evicted, so a region is counted"
          " once ever);")
    print("           budgeted rows are what the policy fetched. Two claims, not one.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(str(out) + ".csv", "w") as f:
            f.write("config,rep,regions_per_token,mb_per_token,hits_per_token,"
                    "evictions_per_token,expert_hit_pct,core_hit_pct,physical_mb_per_token,"
                    "ms_per_token,tok_per_s,prefill_ms,prefill_mb,mhz,mem_available_mb\n")
            for name in names:
                for r in range(args.reps):
                    x = results[(name, r)]
                    f.write(f"{name},{r},{x['regions']:.2f},{x['mbytes']:.2f},{x['hits']:.2f},"
                            f"{x['evict']:.2f},{x['expert_hit']:.2f},{x['core_hit']:.2f},"
                            f"{x['disk_mb']:.2f},{x['ms']:.1f},{x['tps']:.4f},"
                            f"{x['prefill_ms']:.0f},{x['prefill_mb']:.0f},"
                            f"{x['mhz']:.0f},{x['mem_mb']}\n")
        with open(str(out) + ".txt", "w") as f:
            f.write(f"C2 measurement {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
            f.write(f"model {args.model}\nprompt {args.tokens}\ngen {args.gen}\n"
                    f"layers {args.layers}\n"
                    f"reps {args.reps}\ncold_start {cold}\nthreads {args.threads}\n"
                    f"arms {','.join(names)}\n\n")
            f.write(Path(str(keep / f"{names[0]}_r0.meta")).read_text())
            f.write("\n# generated tokens (identical in every arm and rep — asserted)\n")
            f.write("tokens " + " ".join(str(t) for t in tokens_ref) + "\n")
        print(f"\nwrote {out}.csv and {out}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
