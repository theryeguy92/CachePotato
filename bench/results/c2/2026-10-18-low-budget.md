# C2 — 4 GB and 2 GB: the anchor thrashes completely, and the pager has no policy left

**Date:** 2026-10-18 · **Branch:** `feat/qwen3-30b-a3b` · **Status: measurement.**

Same protocol as `2026-10-18-8gb-cold.md`: cold start per run, arms interleaved with a rotating
order, 3 reps, 32 greedy tokens, deterministic counters asserted identical across reps and the
token sequence asserted identical across arms. Raw output: `2026-10-18-low-budget.{csv,txt}`.

## The numbers

| config | regions/tok | MB/tok fetched | evict/tok | expert hit | **core hit** | physical MB/tok | ms/tok | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control (unpaged) | 38.1 *(demand)* | 190.8 | 0.0 | 82.59% | 97.30% | 411.4 ± 26.6 | 5706 ± 301 | 0.176 |
| 4 GB + sys drop | 255.5 | 1260.8 | 256.7 | 36.20% | **95.01%** | 1633.1 ± 28.6 | 8406 ± 276 | 0.119 |
| 2 GB + sys drop | **818.9** | 3589.8 | 815.8 | **4.68%** | **10.77%** | 3509.5 ± 34.4 | 11738 ± 267 | 0.085 |

Reproducibility: regions/token, bytes/token, hits and evictions are **identical in all three reps**
(the script fails if they are not); physical bytes ±2% and ms ±4%.

## What it establishes

1. **At 2 GB there is no caching at all.** 818.9 regions fetched per token is the whole working set
   (819). Every token re-reads 3.6 GB from disk: expert hit 4.68%, **core hit 10.77%**, 815.8
   evictions/token against 819 touches. The pager is a pass-through.
2. **Anchor thrash is real and it was predicted.** The pre-registered expectation was core hit <90%
   and expert hit <40% at 2 GB; measured 10.77% and 4.68%. Core is 1.60 GB of a 2 GB budget, so
   there is not even room for the hot tier: LRU evicts the attention weights it will need again
   within the same token.
3. **At 4 GB the hot tier survives but the experts do not.** Core hit 95.01% (fine), expert hit
   36.20%, and the pager still reads 3× the control's bytes and runs 32% slower.
4. **The pager never beats the OS in this range** — 0.119 and 0.085 tok/s against the control's
   0.176 — because the OS keeps ~12 GB for free and the pager is capped below it. The 2 GB row is
   the clearest possible statement of that: a 2 GB cap on a machine with 12 GB available is 6×
   worse, and no policy can recover it.

## The decision this was for: core pinning

**Yes, test it — but only where it can pay: 4 GB and 2 GB.** The trigger was "core hit collapses",
and it collapses at 2 GB (10.77%) while staying at 4 GB (95.01%). The arithmetic the numbers now
support: with the 435 core regions pinned (1.597 GB), a 2 GB budget leaves ~0.4 GB (≈80 of 6144
expert regions), so core hit → ~100% and expert hit → ~20%; regions/token would fall from 819 to
roughly 384 × 0.8 + 0 ≈ 310, i.e. **~2.6× fewer reads**, and it costs the experts 0.4 GB they do not
currently get either. At 4 GB pinning would give the experts 2.4 GB with core protected — a smaller
but still testable change. At 8 GB and 11 GB pinning is irrelevant: core is never evicted (97.3%
hit).

**Not implemented in this pass** — it is a policy change and gets its own gate and measurement. It
is the next experiment after the amplification is settled, because the two interact: pinning at
2 GB makes the pager read 310 regions × ~5 MB per token, and the read amplification determines what
those regions actually cost.

## Honest limits

- The 2 GB row is a statement about the *deployment cap*, not about the policy: it says a 2 GB cap
  is unusable for a 3.6 GB/token working set, which was already likely.
- The interesting comparison for pinning is against the *4 GB unpinned* row and the *2 GB unpinned*
  row, not against the control, because it is a same-budget policy change.
- These numbers are for one prompt (p0) and 32 tokens. Expert selection is prompt-dependent; the
  earlier 8 GB table showed reuse is high (73–83%), and a layout with less reuse would move every
  row.
