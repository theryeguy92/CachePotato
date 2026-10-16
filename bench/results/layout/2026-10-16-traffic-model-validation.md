> **SUPERSEDED 2026-10-16 (same day).** The verdict below — that the model
> "under-predicts the engine" and is therefore untrustworthy — is **wrong**. The model
> agrees with the independent C replay simulator (`cachepotato --pager-sim`) to the
> decimal on both arms, so the model was never the problem; the *comparison* was (the
> trace's reset structure differs from the bench's). Read
> [`2026-10-16-c-replay-and-model-diagnosis.md`](2026-10-16-c-replay-and-model-diagnosis.md)
> for the corrected conclusion. The measurements and the two ruled-out hypotheses below
> remain valid and are still the supporting evidence.

# Phase 8 T2 — traffic model: built, and its first validation FAILED

**Date:** 2026-10-16 · **Branch:** `feat/universal-ports-tier-b` · **Status: partial.**
**Tools:** `analysis/cpw1_trace.py`, `analysis/traffic.py`

## What was built

A device-aware traffic model for a recorded routing trace: given a weight layout (bytes
per expert, experts per pageable region), a RAM budget and the measured storage curve, it
reports expert hit rate, **distinct cold region loads per token**, bytes per token, and a
storage-limited tok/s prediction. Tooling is complete and usable: budget, region-size and
expert-bit-width sweeps, `--limit-windows` for interactive runs, JSON output.

The design decision that matters: **distinct region loads per token is the headline metric,
not bytes.** On this SSD a cold random read costs ~1.7 ms almost independently of size, so
read *count* binds before byte *count*. The model therefore counts physical cold block reads
separately from the engine's own counter, which charges one miss per expert *touch* and so
over-counts once a region holds more than one expert.

## The validation, and why it is a failure

Against the engine's committed expert hit rates (`bench/results/sweeps/2026-09-28-q3-raw.txt`,
Q3 protocol, L05 arm, prefetch W=4) plus a fresh `CP_PREFETCH_W=0` run made for this note:

| budget | capacity (regions) | model | engine | delta |
|---:|---:|---:|---:|---:|
| 384 MB | 38 | 29.6% | **44.38%** | −14.8 |
| 512 MB | 152 | 62.5% | **88.61%** | −26.1 |
| 736 MB | 350 | 95.4% | **99.82%** | −4.5 |

The model under-predicts everywhere by 5–26 points. **It cannot answer the question T2
exists to answer** ("which port, at what budget") until the gap is closed.

Two intermediate versions of the model were worse and are worth recording as traps:

- The first divided capacity by slot count and then applied it *per slot*, inflating
  capacity ~10× and reporting a physically impossible **95.4% hit rate at a budget that
  cannot hold 4 experts per slot**. Caught only because the number looked too good next to
  a near-uniform access distribution.
- The second charged a wrongly-derived core footprint (222 MB instead of 340.3 MB) and so
  under-counted eviction pressure.

## What is ruled out

**Prefetch is not the cause.** The engine defaults to `CP_PREFETCH_W=4`, which marks regions
resident and consumes budget, so it was the natural suspect. Measured directly at 384 MB:

| config | expert hit | core hit | prefetch counter |
|---|---:|---:|---|
| W=4 (default) | 44.08% | 98.41% | `0 fetches, 267756 already resident` |
| W=0 | **44.38%** | **99.99%** | `0 fetches, 0 already resident` |

At 512 MB, W=0 gives 88.61% — identical to W=4 — at 9.81 tok/s (CI [9.35, 10.27]).
Prefetch is **inert on this workload**, which the committed sweep's `prefetch: 0 fetches`
counter already implied. The 384 MB number also reproduces (W=0: 7.1297 tok/s,
CI [6.583, 7.676], against the README's 7.19).

**Core arithmetic is not the cause.** The pager tracks 340.3 MiB of non-expert regions
(token+pos embeddings 150.2, attention 108.0, dense FFN 36.0, shared experts 45.1, routers
0.9). That is arithmetic, and it makes the 736 MB point (`ALL FIT`) line up: at 736 MB the
engine records 323 total expert misses — almost exactly **one expert pool (320)**.

## The open lead

Inverting the measured hit rates is informative. The access distribution is near-uniform
(top 4 of 32 experts ≈ 14% of accesses, against 12.5% uniform), so for the engine's global
LRU `hit_rate ≈ K/32`, where K is the effective resident blocks per slot:

| budget | measured hit | implied K | K allowed by byte arithmetic | ratio |
|---:|---:|---:|---:|---:|
| 384 MB | 44.38% | 14.1 | 3.8 | **3.7×** |
| 512 MB | 88.61% | 22.6 | 15.2 | **1.5×** |
| 736 MB | 99.82% | 32+ (pool) | 35.0 | 1.0× |

An effective capacity *larger than the budget allows* is not something a pager can do, so
the model is missing a mechanism rather than mis-parameterised. The strongest candidate is
**reset frequency**: the trace has 488 windows, but the Q3 measurement has only 16 prompts,
and the 736 MB miss count (323 ≈ one pool) implies ~16 resets, not 488. If the 488 trace
windows are contiguous segments of one stream rather than independent sequences, the model
resets ~30× too often and converts an unknown number of hits into compulsory misses.

This was not settled here because the way to settle it is **not** another Python LRU: at
~2.5M touches per configuration, a single replay costs ~30 s, which made the investigations
above take far longer than the result warranted. The right instrument is the **C engine
itself** — replay the trace through `pager.c` with a `--pager-sim` mode. The engine is the
ground truth for its own policy, it already runs this workload in seconds, and it removes
the modelling error entirely.

## What this means for the direction

Nothing here threatens the Tier-B result (`bench/results/layout/`), which is measured, not
modelled. It does mean T2's deliverable is **not yet met**, and that the budget/port
question stays open.

One directional signal survived, and it has the expected sign: at 384 MB with this model's
1.15 MB experts, grouping experts into larger regions *reduces* capacity, increases bytes
moved per token, and lowers predicted throughput (3 experts/region: 9.08 → 16.65 MB/token;
7 experts/region: 149.85 MB/token). That is the same sign as the 2026-09-28 grouping
negative result, and it is why the interesting grouping question belongs to a port with
**small** (~10 KB) experts, not this one. It is a sign, not a number — the model
under-predicts hit rate, so its absolute throughput figures are not usable.

## Reproducing

```sh
# validation table (slow: ~30 s per budget in the pure-Python LRU)
python3 analysis/traffic.py --trace data/traces/l05_fp32.npz --validate

# quick directional runs
python3 analysis/traffic.py --trace data/traces/l05_fp32.npz --limit-windows 40 --budget-mb 384
python3 analysis/traffic.py --trace data/traces/l05_fp32.npz --budget-mb 384 \
    --sweep-regions 1152K,4M,8M --sweep-expert-bits 16,8,4,2

# the engine measurements quoted above
CP_BUDGET=384 CP_PAGER_SYS=1 CP_PREFETCH_W=0 ./build/cachepotato \
    weights/moe-scale-run/l05.cpw1v3 --bench data/fineweb_edu_valid.bin 0 48 512 16
```
