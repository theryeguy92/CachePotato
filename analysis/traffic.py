#!/usr/bin/env python3
"""traffic.py — device-aware traffic model for a routing trace (Phase 8, T2).

WHAT IT ANSWERS
  Given how a model routes (a recorded trace), a weight layout (bytes per expert,
  experts per pageable region), a RAM budget and a measured storage curve:
    - how many cold reads does one token cost?
    - what expert hit rate does that budget buy?
    - what tok/s does the storage curve predict, and is the workload I/O- or
      compute-bound?
  It is the instrument that decides which port is viable at what budget, before any
  kernel or format work is spent on it.

WHY THE READ COUNT IS THE HEADLINE METRIC
  On the T480's SATA SSD a cold random read costs ~1.7 ms almost independently of
  size (4K: 2.4 MB/s, 4 MB: 318 MB/s). So *how many* reads bind before *how many
  bytes*. The model reports distinct cold region loads per token; bytes are secondary.

TWO COUNTERS, DELIBERATELY
  `engine_*` mirrors `pager.c` exactly: one access per expert TOUCH, a hit if that
  expert's region is resident, a miss otherwise — including when two experts of one
  token sit in the same region, which the engine charges as two accesses and (on a
  cold region) two misses while the block is pulled once. `distinct_loads` counts
  physical cold block reads. The gap between them is an over-count in the existing
  counter and only appears once regions hold more than one expert.

FIDELITY, STATED UP FRONT
  - Residency is a single GLOBAL LRU over regions shared by all slots, reset at each
    window start and at EOS, matching `pager.c` (which evicts from one flat region
    array). It is deliberately not per-slot: the engine's global pool can drift
    capacity toward whichever slot is being touched, and a per-slot partition would
    understate it.
  - Capacity is `(budget - core_bytes) / region_bytes` TOTAL regions, not per slot.
    An earlier version of this file divided by slot count and then applied the result
    per slot, inflating capacity ~10x and reporting a physically impossible 95% hit
    rate at a budget that cannot hold 4 experts per slot. Fixed; the validation table
    in the report is what caught it.
  - The budget is charged for regions only. Non-expert ("core") bytes are not
    simulated, so pass `--core-bytes` with the model's measured core footprint
    (l05: 340.3 MiB) to model the tight regime. Without it the numbers are optimistic.
  - Prefetch is NOT modelled. Prefetch marks regions resident, so it consumes budget
    and can displace useful entries; this model is therefore the no-prefetch baseline.
    Compare against engine runs with CP_PREFETCH_W=0 (the engine's default is 4).
    **Measured 2026-10-16: prefetch does not explain the discrepancy below.** With
    `CP_PREFETCH_W=0` the engine's expert hit rate at 384 MB is 44.38% (vs 44.08% at
    W=4) and at 512 MB is 88.61% (identical to W=4), and core stays at 99.99%. So the
    engine's default prefetch is inert on this workload, exactly as the committed
    sweep's `prefetch: 0 fetches` counter already showed.
  - Bytes per expert are a parameter, so the same trace answers "what if the experts
    were quantized to N bits" without recording a new trace.

STATUS: DIRECTIONAL, NOT YET QUANTITATIVE — USE ENGINE MEASUREMENTS INSTEAD
  Validated against the engine's committed expert hit rates (Q3 protocol, L05):
      384 MB  model 29.6%  vs measured 44.38%   (-14.8)
      512 MB  model 62.5%  vs measured 88.61%   (-26.1)
      736 MB  model 95.4%  vs measured 99.82%   (-4.5)
  The model UNDER-predicts everywhere by 5-26 points, so it cannot be used to make a
  quantitative claim about a budget yet. Two things are known about the gap:

    1. It is not prefetch (measured above) and not a capacity-arithmetic error: the
       core footprint is 340.3 MiB of pager-tracked regions, which is arithmetic, and
       it makes the 736 MB point (`ALL FIT`, near-100% expected) line up.
    2. Inverting the measured hit rates (the access distribution is near-uniform, so
       hit_rate ~ K/32 for K resident blocks per slot) gives an effective K of 14.1,
       22.6 and (capped at the pool) 32+ for 384/512/736 MB — i.e. **3.7x, 1.5x and
       1.0x the capacity the byte arithmetic allows**. An effective capacity larger
       than the budget permits is not something the pager can do, so the model is
       missing a mechanism, most likely the reset frequency: the trace has 488
       windows, but the Q3 measurement has 16 prompts, and at 736 MB the engine's
       total misses (323) equal almost exactly ONE expert pool (320), implying ~16
       resets, not 488.

  Next step to close it (not done here): replay the trace through the **C engine** with
  a `--pager-sim` mode instead of a second Python LRU. The engine is the ground truth
  for its own policy, it already runs this workload in seconds, and a Python per-touch
  LRU costs ~30 s per configuration — too slow to be the instrument, which is what
  stalled this line of work. Until then: **quote engine measurements, not this model.**
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpw1_trace import LatencyModel, load_trace  # noqa: E402

# Measured on l05.cpw1v3: expert_floats(768,192)*4 = 1,183,488 bytes.
L05_EXPERT_BYTES = 1_183_488
# l05's pager-tracked non-expert footprint, from the CPW1 v3 layout region by region.
L05_CORE_BYTES = int(340.3 * 1024 * 1024)


def _truncate(tr, n):
    """First n windows only — for quick interactive runs and for measuring how much a
    result depends on trace length (a proxy for the reset-frequency question above)."""
    from dataclasses import replace
    return replace(tr, tokens=tr.tokens[:n], sel=tr.sel[:, :n])


class Lru:
    """Global LRU region cache. Recency via a monotonic stamp; eviction is O(n) over
    resident regions, which is fine at analysis scale (<= a few thousand regions)."""

    __slots__ = ("cap", "res", "tick", "evictions", "peak")

    def __init__(self, cap):
        self.cap = cap
        self.res = {}
        self.tick = 0
        self.evictions = 0
        self.peak = 0

    def touch(self, region):
        """Returns True on a hit, False on a miss (and loads the region)."""
        self.tick += 1
        if region in self.res:
            self.res[region] = self.tick
            return True
        while len(self.res) >= self.cap:
            victim = min(self.res, key=self.res.get)
            del self.res[victim]
            self.evictions += 1
        self.res[region] = self.tick
        if len(self.res) > self.peak:
            self.peak = len(self.res)
        return False

    def reset(self):
        self.res.clear()


def simulate(trace, region_bytes, budget_bytes, expert_bytes, core_bytes=0):
    """Run the residency model. Returns a metrics dict."""
    g = max(1, int(region_bytes // expert_bytes))       # experts per region
    n_slots, n_exp = trace.n_slots, trace.n_experts
    n_regions_per_slot = (n_exp + g - 1) // g
    eff = budget_bytes - core_bytes
    cap = int(eff // region_bytes)                      # TOTAL resident regions
    if cap < 1:
        cap = 1                                         # degenerate; reported via diagnostics

    lru = Lru(cap)
    acc = hits = loads = 0
    tokens = 0
    for w in range(trace.n_windows):
        lru.reset()
        for t in range(trace.window_len):
            tokens += 1
            for s in range(n_slots):
                base = s * n_regions_per_slot
                for e in trace.sel[s, w, t]:
                    acc += 1
                    if lru.touch(base + e // g):
                        hits += 1
                    else:
                        loads += 1
            if trace.tokens[w, t] == trace.eos:
                lru.reset()

    per_tok = lambda x: x / tokens if tokens else 0.0       # noqa: E731
    capacity_per_slot = cap / n_slots if n_slots else 0.0
    return {
        "experts_per_region": g,
        "region_bytes": g * expert_bytes,
        "regions_per_slot": n_regions_per_slot,
        "total_regions": n_slots * n_regions_per_slot,
        "capacity_total": cap,
        "capacity_per_slot": capacity_per_slot,
        "pool_experts_per_slot": n_exp,
        "tokens": tokens,
        "engine_accesses": acc,
        "engine_hits": hits,
        "engine_hit_rate": (hits / acc * 100.0) if acc else 0.0,
        "distinct_loads": loads,
        "loads_per_token": per_tok(loads),
        "engine_misses_per_token": per_tok(acc - hits),
        "bytes_per_token": per_tok(loads) * g * expert_bytes,
        "resident_regions_max": lru.peak,
        "resident_bytes_max": lru.peak * g * expert_bytes,
        "evictions": lru.evictions,
    }


def predict(metrics, lat, region_bytes, compute_ms_per_token=0.0):
    """Storage-limited tok/s from the measured curve, plus the compute check."""
    rpt = metrics["loads_per_token"]
    t_io, moved = lat.cost(region_bytes, rpt)
    t_comp = compute_ms_per_token / 1000.0
    t_tot = t_io + t_comp
    return {
        "io_s_per_token": t_io,
        "io_share": (t_io / t_tot) if t_tot > 0 else 0.0,
        "compute_s_per_token": t_comp,
        "tok_s": (1.0 / t_tot) if t_tot > 0 else float("inf"),
        "tok_s_io_only": (1.0 / t_io) if t_io > 0 else float("inf"),
        "mb_per_token": moved,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="recorded routing trace (.npz)")
    ap.add_argument("--expert-bytes", type=int, default=L05_EXPERT_BYTES,
                    help="bytes per routed expert block (default: l05's 1,183,488)")
    ap.add_argument("--region-bytes", type=int, default=L05_EXPERT_BYTES,
                    help="bytes per pageable region (default: exactly one expert)")
    ap.add_argument("--budget-mb", type=float, default=512.0)
    ap.add_argument("--core-bytes-mb", type=float, default=L05_CORE_BYTES / 1048576,
                    help="non-expert resident footprint charged against the budget "
                         "(default: l05's measured 340.3 MiB)")
    ap.add_argument("--compute-ms", type=float, default=0.0,
                    help="measured compute-only ms/token, to separate I/O from compute")
    ap.add_argument("--sweep-budgets", default="",
                    help="comma-separated budget MB list, e.g. 384,512,736")
    ap.add_argument("--sweep-regions", default="",
                    help="comma-separated region sizes (K/M suffixes ok)")
    ap.add_argument("--sweep-expert-bits", default="",
                    help="comma-separated expert bit-widths, e.g. 16,8,4,2")
    ap.add_argument("--limit-windows", type=int, default=0,
                    help="use only the first N trace windows (0 = all); the full 488 "
                         "windows take ~30 s per configuration in this pure-Python LRU")
    ap.add_argument("--validate", action="store_true",
                    help="compare against the engine's measured expert hit rates")
    ap.add_argument("--json", default="", help="write the full result table here")
    args = ap.parse_args()

    def parse_sizes(s):
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            mult = 1
            if part[-1] in "Kk":
                mult, part = 1024, part[:-1]
            elif part[-1] in "Mm":
                mult, part = 1024 * 1024, part[:-1]
            out.append(int(float(part) * mult))
        return out

    tr = load_trace(args.trace)
    if args.limit_windows:
        tr = _truncate(tr, args.limit_windows)
    lat = LatencyModel()
    core = args.core_bytes_mb * 1024 * 1024
    print(f"trace   : {tr.summary()}")
    print(f"curve   : {lat.describe()}")

    if args.validate:
        # Measured with the Q3 protocol (prompt_T=48 gen_N=512 n_prompts=16,
        # CP_PAGER_SYS=1) at prefetch W=4, from bench/results/sweeps/2026-09-28-q3-raw.txt.
        meas = {"l05": {384: 46.06, 448: 68.46, 512: 88.61, 576: 96.88, 640: 99.04, 736: 99.82},
                "l00": {384: 35.50, 448: 56.42, 512: 80.09, 576: 94.90, 640: 98.77, 736: 99.82}}
        arm = "l05" if tr.lam > 0.25 else "l00"
        print(f"\nvalidation vs engine measurement ({arm} arm, prefetch W=4; model is no-prefetch)")
        print(f"  {'budget':>7} {'cap':>5} {'cap/slot':>9} {'model':>8} {'measured':>9} {'delta':>7}")
        rows = []
        for b, mh in meas.get(arm, {}).items():
            mm = simulate(tr, args.region_bytes, b * 1024 * 1024, args.expert_bytes, core)
            d = mm["engine_hit_rate"] - mh
            rows.append({"budget_mb": b, "measured": mh, "model": mm["engine_hit_rate"], **mm})
            print(f"  {b:7d} {mm['capacity_total']:5d} {mm['capacity_per_slot']:9.1f} "
                  f"{mm['engine_hit_rate']:8.2f} {mh:9.2f} {d:+7.2f}")
        if args.json:
            Path(args.json).write_text(json.dumps({"validation": rows}, indent=1, default=float))
            print(f"\nwrote {args.json}")
        return 0

    m = simulate(tr, args.region_bytes, args.budget_mb * 1024 * 1024, args.expert_bytes, core)
    pr = predict(m, lat, m["region_bytes"], args.compute_ms)
    print(f"\nconfig  : expert {args.expert_bytes/1024:.0f} KB | region {m['region_bytes']/1024:.0f} KB "
          f"({m['experts_per_region']} expert(s)) | budget {args.budget_mb:.0f} MB "
          f"(core {args.core_bytes_mb:.0f} MB charged)")
    print(f"          capacity {m['capacity_total']} regions total "
          f"({m['capacity_per_slot']:.1f}/slot; pool {m['pool_experts_per_slot']} experts/slot)")
    print(f"  expert hit rate    : {m['engine_hit_rate']:.2f}%  (engine counter semantics)")
    print(f"  distinct loads/tok : {m['loads_per_token']:.2f}   "
          f"(engine would count {m['engine_misses_per_token']:.2f} misses/token"
          + (f" -- over-count x{m['engine_misses_per_token']/m['loads_per_token']:.2f}"
             if m['loads_per_token'] and m['engine_misses_per_token'] > m['loads_per_token'] else "") + ")")
    print(f"  bytes/token        : {m['bytes_per_token']/1e6:.2f} MB   "
          f"({m['resident_bytes_max']/1e6:.0f} MB peak resident, {m['evictions']} evictions)")
    print(f"  predicted          : {pr['tok_s']:.2f} tok/s   "
          f"(I/O {pr['io_s_per_token']*1e3:.1f} ms = {100*pr['io_share']:.0f}%"
          + (f", compute {pr['compute_s_per_token']*1e3:.1f} ms" if args.compute_ms else "") + ")")

    results = {"trace": str(tr.path), "lam": tr.lam, "single": {**m, **pr}}

    if args.sweep_budgets:
        print(f"\nbudget sweep  (region {m['region_bytes']/1024:.0f} KB, expert {args.expert_bytes/1024:.0f} KB)")
        print(f"  {'budget':>8} {'cap':>6} {'cap/slot':>9} {'hit%':>7} {'loads/tok':>10} {'MB/tok':>8} {'tok/s':>8}")
        rows = []
        for b in [float(x) for x in args.sweep_budgets.split(",")]:
            mm = simulate(tr, args.region_bytes, b * 1024 * 1024, args.expert_bytes, core)
            pp = predict(mm, lat, mm["region_bytes"], args.compute_ms)
            rows.append({"budget_mb": b, **mm, **pp})
            print(f"  {b:8.0f} {mm['capacity_total']:6d} {mm['capacity_per_slot']:9.1f} "
                  f"{mm['engine_hit_rate']:7.2f} {mm['loads_per_token']:10.2f} "
                  f"{mm['bytes_per_token']/1e6:8.2f} {pp['tok_s']:8.2f}")
        results["budget_sweep"] = rows

    if args.sweep_regions:
        print(f"\nregion sweep  (budget {args.budget_mb:.0f} MB, expert {args.expert_bytes/1024:.0f} KB)")
        print(f"  {'region':>9} {'exp/reg':>8} {'cap':>6} {'hit%':>7} {'loads/tok':>10} {'MB/tok':>8} {'tok/s':>8}")
        rows = []
        for rb in parse_sizes(args.sweep_regions):
            mm = simulate(tr, rb, args.budget_mb * 1024 * 1024, args.expert_bytes, core)
            pp = predict(mm, lat, mm["region_bytes"], args.compute_ms)
            rows.append({"region_bytes": rb, **mm, **pp})
            print(f"  {rb/1024:8.0f}K {mm['experts_per_region']:8d} {mm['capacity_total']:6d} "
                  f"{mm['engine_hit_rate']:7.2f} {mm['loads_per_token']:10.2f} "
                  f"{mm['bytes_per_token']/1e6:8.2f} {pp['tok_s']:8.2f}")
        results["region_sweep"] = rows

    if args.sweep_expert_bits:
        print(f"\nquantization sweep  (region = 1 expert, budget {args.budget_mb:.0f} MB)")
        print(f"  {'bits':>5} {'exp KB':>7} {'cap':>6} {'hit%':>7} {'loads/tok':>10} {'MB/tok':>8} {'tok/s':>8}")
        rows = []
        for bits in [float(x) for x in args.sweep_expert_bits.split(",")]:
            eb = int(L05_EXPERT_BYTES * bits / 16.0)
            mm = simulate(tr, eb, args.budget_mb * 1024 * 1024, eb, core)
            pp = predict(mm, lat, mm["region_bytes"], args.compute_ms)
            rows.append({"bits": bits, "expert_bytes": eb, **mm, **pp})
            print(f"  {bits:5.1f} {eb/1024:7.0f} {mm['capacity_total']:6d} "
                  f"{mm['engine_hit_rate']:7.2f} {mm['loads_per_token']:10.2f} "
                  f"{mm['bytes_per_token']/1e6:8.2f} {pp['tok_s']:8.2f}")
        results["quant_sweep"] = rows

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
