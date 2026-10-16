# Phase 8 T2 — traffic model + C replay: VALIDATED

**Date:** 2026-10-16 · **Branch:** `feat/universal-ports-tier-b`
**Status: T2 deliverable met.** The replay reproduces the engine's own expert hit rate
across the whole budget curve. Two real bugs had to be found to get here; both are recorded
because each looked like a different problem.

**Tools:** `analysis/cpw1_trace.py`, `analysis/traffic.py`, `src/replay.{c,h}`,
`cachepotato --pager-sim`, `cachepotato --bench … --record-trace`,
`tools/npz_to_rtp1.py`, `tools/verify_rtp1.py`.

## Result

Traces are now recorded **by the bench run itself** (`--record-trace`), so the trace and the
pager counters describe the *same* execution — 16 prompts × (48 prompt + 512 generated) = 560
tokens, `CP_PAGER_SYS=1`, `CP_PREFETCH_W=0`. Replaying that trace:

| budget | engine measured | replay | delta |
|---:|---:|---:|---:|
| 384 MB | 44.38% | 41.25% | −3.13 |
| 448 MB | 68.46% | 67.84% | −0.62 |
| 512 MB | 88.61% | 86.84% | −1.77 |
| 576 MB | 96.88% | 96.72% | −0.16 |
| 640 MB | 99.04% | 99.01% | −0.03 |
| 736 MB | 99.82% | **99.82%** | **+0.00** |

Access counts match **exactly**: 179,200 touches = 16 × 10 × 2 × 560. At 736 MB the replay
reports 178,880/179,200 hits and **0 evictions** — identical to the engine's own counters.
Residual is 0 to −3.1 points, worst at the tightest budget (58 regions of capacity for a
320-region pool).

## Bug 1 — the pager does not reset at a sequence boundary

This was the big one, and it is a fact about `pager.c` that no document stated: the region
cache is **process-wide and persists across prompts**. It resets its *locality set* per
sequence, not its residency.

Evidence, unambiguous: over 16 prompts the engine records **320 expert misses** — which is
exactly the number of distinct `(slot, expert)` regions in the pool — where a per-prompt
reset would give 16 × ~300 ≈ 4,800. The trace confirms this independently: it touches 320
distinct regions, all 32 experts in every slot, so with capacity 350 nothing can ever be
evicted.

My model reset its LRU at every window boundary. Over the same routing data that made it
report ~35× the engine's miss count, and it is the reason the earlier note wrongly concluded
the model was broken. `--ps-reset 1` / `--reset-on-eos` now exists to *test* the EOS case;
the default is off.

## Bug 2 — the core footprint charged is ~317.7 MiB, not 340.3 MiB

The region list sums to 340.3 MiB of non-expert regions, and I used that as the budget charge.
The engine's own resident report says otherwise:

```
resident : core 74 regions/317.7 MB | experts 58/320 regions/65.5 MB | total 383.1 MB of budget 384 MB
```

When the budget binds, not every non-expert region is resident. Using the computed 340.3
under-provisioned expert capacity and made tight budgets read ~10 points low. The defaults in
`--ps-core` and `L05_CORE_BYTES` are now **317.7 MiB, the measured value**, and
`pager_report()` prints the resident split so this is observed rather than inferred.

## Bug 3 (caught by tooling, no effect on results) — RTP1 header size

`ReplayHeader` is nine 4-byte fields = **36 bytes**, written as a raw struct. The Python
writer assumed 32, so it placed the payload 4 bytes early and every expert id came back in
the hundreds for a 32-expert model. I initially "fixed" it by reordering fields for a 32-byte
pack — which was wrong, because the struct has no padding at all (9 × 4 = 36). Corrected by
setting the size to 36 and adding `_Static_assert(sizeof(ReplayHeader) == RPY_HDR_BYTES)` so
the C side fails to compile rather than writing a file its reader disagrees with. The
misalignment turned out not to change the hit counts (garbage ids still partition into
distinct regions), but that was luck, not design.

`tools/verify_rtp1.py` now checks any RTP1 file against the layout parsed out of
`src/replay.h`, the exact file size, and the expert-id range. It catches the old broken files.

## Method notes worth keeping

- **Record the trace from the run you measure.** Everything before this compared a trace
  recorded over independent windows against a bench with a different structure; two
  candidate explanations were indistinguishable until the trace came from the bench itself.
- **The instrument that resolved it was a second implementation, not more fidelity in the
  first.** C and Python now agree to the decimal on every configuration tested, which is what
  localised the disagreement to the engine rather than the model — and then to the reset
  model rather than the arithmetic.
- **A model of a system can be validated to within rounding.** 0.00 delta at the budget where
  capacity exceeds the pool, and −3.1 at the tightest, is enough to choose a port with.

## What this does NOT validate

The replay validates the *residency model* against the engine's **hit counters**. It does not
validate that a miss costs a real disk read: `pager.c` reports its own tracking, and whether
`POSIX_FADV_DONTNEED` / `MADV_DONTNEED` actually drop the pages on btrfs is still the open
methodological question recorded in the vault (the ~227 µs vs ~1705 µs 4 KB discrepancy).
Any absolute tok/s figure derived from the latency curve inherits that uncertainty.

## Reproducing

```sh
# record a trace from a real bench run (trace and counters, same execution)
CP_BUDGET=736 CP_PAGER_SYS=1 CP_PREFETCH_W=0 ./build/cachepotato \
    weights/moe-scale-run/l05.cpw1v3 --bench data/fineweb_edu_valid.bin 0 48 512 16 \
    --record-trace scratch/bench_l05.rtp1

python3 tools/verify_rtp1.py scratch/bench_l05.rtp1
./build/cachepotato --pager-sim scratch/bench_l05.rtp1 --ps-sweep 384,448,512,576,640,736
python3 analysis/traffic.py --trace data/traces/l05_fp32.npz --budget-mb 512
```

## Gates re-run after all changes

`verify_toy` PASS (4.27e-07); `verify_cpw1_v3` on l05 PASS (structure + exact round-trip +
recorded traces); `--step-check` GATE 1 PASS (0 routing differ); permutation gate PASS
(routing exact, logits fp32-equal). `pager_report()` gained a resident line; no existing
counter changed.
