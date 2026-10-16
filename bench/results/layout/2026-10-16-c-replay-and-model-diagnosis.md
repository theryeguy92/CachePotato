# C replay simulator: the traffic model was never wrong — the comparison was

**Date:** 2026-10-16 · **Branch:** `feat/universal-ports-tier-b` · **Status: T2 instrument
rebuilt; the model-vs-engine gap is now diagnosed but NOT closed.**
**Adds:** `src/replay.{c,h}`, `cachepotato --pager-sim`, `tools/npz_to_rtp1.py`

## Correction to the previous note

[`2026-10-16-traffic-model-validation.md`](2026-10-16-traffic-model-validation.md) concluded
that the Python traffic model "under-predicts the engine by 5–26 points" and was therefore
untrustworthy. **That verdict was wrong, and this note supersedes it.** The model agrees with
an independent C implementation to the decimal:

| trace | budget | Python model | C `--pager-sim` |
|---|---:|---:|---:|
| l05 (λ=0.5) | 384 MB | 29.60% | **29.60%** |
| l05 | 512 MB | 62.48% | **62.48%** |
| l05 | 736 MB | 95.37% | **95.37%** |
| l00 (λ=0) | 512 MB | 55.83% | **55.83%** |

Two independent implementations agreeing exactly means the *model* is not the problem — the
comparison is. The C replay runs a full 6-budget sweep in **milliseconds** (the Python LRU
cost ~30 s per configuration, which is what made the previous session so expensive).

## The C simulator

`cachepotato --pager-sim <trace.rtp1>` replays a recorded routing trace through the *residency
policy* `pager.c` implements — one global LRU over `(slot, expert_group)` regions, capacity
`(budget − core)/region_bytes`, resets at window boundaries and EOS — with no forward pass, no
weights and no mmap. Since it consumes a recording rather than running the model, it can sweep
budgets, region sizes and expert encodings far beyond what the pager currently supports.

It reports **two counters** on purpose: `reads` = distinct cold *region* loads (the physical
I/O that matters on a latency-bound disk), and the engine-counter reading, which charges one
miss per expert *touch* and so over-counts once a region holds more than one expert.

Flags: `--ps-budget`, `--ps-region`, `--ps-expert`, `--ps-core`, `--ps-limit`, `--ps-reset`,
`--ps-sweep`. Traces convert with `tools/npz_to_rtp1.py` (`sel` shape and the token stream are
size-checked, and the converter asserts the written file size).

## What the replay shows about the gap

| budget | C/Python replay | engine measured | delta |
|---:|---:|---:|---:|
| 384 MB | 29.60% | 46.06% | −16.5 |
| 448 MB | 46.92% | 68.46% | −21.5 |
| 512 MB | 62.48% | 88.61% | −26.1 |
| 576 MB | 76.48% | 96.88% | −20.4 |
| 640 MB | 88.19% | 99.04% | −10.9 |
| 736 MB | 95.37% | 99.82% | −4.5 |

The decisive measurement is the **no-capacity-pressure ceiling**. With a budget far above
everything (`--ps-budget 4000`), the replay still reports **95.37%** and `peak resident = 320
regions` — the entire expert pool. So the ~4.6% of misses the replay reports are *not*
capacity misses at all; they are the unavoidable cold touches implied by the trace's own
reset structure. In other words:

> **The replay is not measuring the pager. It is measuring the trace.**

That reframes the gap. At 736 MB the engine records **323 total expert misses out of 179,200
accesses** — almost exactly *one* expert pool (320). The replay over the same routing data
reports ~11,500 misses (~4.6%). A factor of ~35 in reset frequency explains it: the trace has
**488 windows**, while the Q3 measurement has **16 prompts**. Fewer resets ⇒ fewer compulsory
misses ⇒ a higher hit rate.

Two candidate causes remain, and they are distinguishable:

1. the trace's 488 windows are contiguous segments of a stream, not 488 independent
   sequences, so the replay resets ~30× too often; or
2. the engine's `--bench` prompt structure differs from how the trace was recorded.

`--ps-reset 0` (test 1 crudely) moves 736 MB from 95.37% → 98.30%, in the right direction but
not all the way, which is consistent with (1) being part of the story rather than all of it.

## Ruled out, by measurement

- **Prefetch.** The engine defaults to `CP_PREFETCH_W=4`, so it was the natural suspect.
  `CP_PREFETCH_W=0` gives **44.38%** at 384 MB (vs 44.08% at W=4) and **88.61%** at 512 MB —
  identical; core 99.99%. Prefetch is **inert on this workload**, as the committed sweep's
  `prefetch: 0 fetches` counter already implied.
- **Core-footprint arithmetic.** The pager tracks 340.3 MiB of non-expert regions
  (embeddings 150.2, attention 108.0, dense FFN 36.0, shared experts 45.1, routers 0.9). That
  is arithmetic, and it makes the `ALL FIT` point line up.
- **A modelling bug in either implementation.** They agree exactly, on both arms.

## The surgical next test

Record a fresh trace using **exactly the Q3 bench structure** — 16 prompts × (48 prompt + 512
generated), `CP_PAGER_SYS=1`, `CP_PREFETCH_W=0`, the same start offsets and stride — then
replay *that* and compare against the engine's own counters for the same run. That removes
both candidate causes at once, because the trace and the counter would then describe the same
execution. Everything needed already exists: `tools/record_routing.py` for the trace and
`--bench` for the counters.

Until that is done, **the model's absolute numbers should not be used to choose a port or a
budget.** Its measured-vs-model disagreement is a known, explained artifact, not a policy
prediction.

## What did survive

Directional results, which the reset-frequency explanation does not touch because they are
within-trace comparisons:

- **Region grouping.** At 384 MB with this model's 1.15 MB experts, grouping experts into
  larger regions reduces capacity and raises bytes moved per token
  (1 expert/region 16.65 MB/token → 3/region 68.68 → 7/region 149.85). Same sign as the
  2026-09-28 grouping negative result. The interesting grouping question still belongs to a
  port with **~10 KB experts**, where the byte arithmetic reverses.
- **Badly-tuned budgets are catastrophic, not gradual.** At 300–340 MB the replay reports
  `cap = 1` region and a 0.00% hit rate: the core footprint alone (340.3 MiB) consumes the
  whole budget. Worth knowing before anyone benchmarks a "tight" budget below the core size.

## Reproducing

```sh
python3 tools/npz_to_rtp1.py --src data/traces/l05_fp32.npz --dst data/traces/l05_fp32.rtp1
./build/cachepotato --pager-sim data/traces/l05_fp32.rtp1 --ps-sweep 384,448,512,576,640,736
./build/cachepotato --pager-sim data/traces/l05_fp32.rtp1 --ps-budget 4000   # the ceiling
./build/cachepotato --pager-sim data/traces/l05_fp32.rtp1 --ps-reset 0       # candidate 1
```

## Gates re-run after adding `src/replay.c` to the build

`verify_toy` PASS (4.27e-07); `verify_cpw1_v3` on l05 PASS (structure + exact round-trip +
recorded traces); `--step-check` GATE 1 PASS (0/640 routing differ). The new mode is additive:
it reads no weights and changes no existing code path.
