# C2 results, and the thesis they actually support

**Date:** 2026-10-19 · **Branch:** `feat/qwen3-30b-a3b` · **Status: conclusion of the paging experiment.**

This is the summary record for C1 and C2. The detailed measurements are in
`bench/results/c2/`; this file says what they mean, including the part where the original
target was the wrong one.

## What was measured

**C1 — the pager is attached to the Qwen3 path and changes nothing about the answer.** At an
unlimited budget and at a budget tight enough to evict, the 48-layer output is **bit-identical** to
the unpaged run; the accounting is checked against the file's own region table (819 regions /
3600.2 MB for one token: 435 core + 384 experts), not against itself. 20 gates.

**C2 — the paging numbers.** Qwen3-30B-A3B Q8, 32.5 GB against 16 GB, 32 greedy tokens, cold start,
interleaved arms, deterministic counters identical across reps.

> **Thread-count correction (2026-10-19).** The table below was measured **single-threaded**:
> `c2_measure.py` stripped every `CP_*` variable, including `CP_THREADS`, and the engine defaults
> to one thread. Re-measured at 4 threads with everything else identical
> (`bench/results/c2/2026-10-19-threaded*`): the honest 8 GB arm is **3182 ms/token = 0.314 tok/s**
> (not 0.147) and the unpaged control is **0.521** (not 0.176) — a **2.1–3.0×** understatement. 8
> threads is worse than 4. The page-count columns are unaffected (deterministic, thread-independent),
> so the paging findings stand; the speed conclusions were too pessimistic, and the corrected
> envelope is in `docs/2026-10-19-tok-s-levers.md`.

| budget | policy | regions/token | physical MB/token | tok/s |
|---|---|---:|---:|---:|
| unpaged control | OS page cache | 38.1 *(demand)* | 418 | 0.176 |
| 11 GB | pager | 39.6 | 383 | 0.195 |
| 8 GB | pager | 108.3 | 1014 | 0.147 |
| 4 GB | pager | 255.5 | 1633 | 0.119 |
| 4 GB | pager + core pinned | 244.0 | 1597 | 0.124 |
| 2 GB | pager | 818.9 | 3510 | 0.085 |
| 2 GB | pager + core pinned | **384.0** | 2110 | **0.100** |

Three findings that survived every check:

1. **At equal memory the pager ties the OS page cache** (54.8 vs 54.7 regions/token at 11 GB). The
   OS is very good; the pager is not faster at the same RAM.
2. **Below the working set, the OS wins**, because the OS keeps ~12 GB for free and the budget is a
   cap. Pinning the hot tier recovers 2.13× of the reads at 2 GB — a real policy win, and the only
   one — but it cannot make a 2 GB cap competitive with 12 GB of RAM.
3. **Two exact layout/read experiments were slower**: co-selection expert grouping (pairs chosen
   together only 11.2% of the time, so the 2× over-fetch dominates) and a pager-owned region arena
   (1.0× bytes on a single tensor, but a region is not an access unit — the 330 MB `embed` region is
   touched for 4 KB).

## What this says about the target, not the engine

The engine works: a 32.5 GB model runs on a 16 GB machine, exactly, with a hard budget and
per-token counters. **The problem is that at 0.15 tok/s nobody can use it.** A 300-token edit takes
~34 minutes; a 1,000-token answer ~1.9 hours. That is not an assistant and not an agent; it is a
demonstration that the mechanism is sound.

The original plan assumed that this was a bandwidth problem and would improve with better layout.
The measurement says otherwise: **4.3 s of the 6.8 s token is compute, not I/O**, and the compute
scales with *active* parameters. That reframes the whole question.

## The envelope this machine actually supports

Measured anchors (Q8 unless stated):

| model | active/token | tok/s | source |
|---|---:|---:|---|
| Qwen2.5-0.5B dense | 0.5 B | **5.4** | measured |
| Qwen3-30B-A3B | 3.35 B | **0.147 @ 8 GB**, 0.176 unpaged | measured |

Fitting `tok/s = a · active^-b` between those two anchors gives b ≈ 1.75 (the 30B is worse than
linear because its 3.6 GB/token working set streams from disk rather than RAM). Extrapolating, and
flagging it as extrapolation:

| model class | active/token | fits 16 GB? | expected tok/s | usable for |
|---|---:|---|---:|---|
| 1.5 B dense, Q8 | 1.5 B | yes | **1.5–2** | interactive single-turn edits |
| 3 B dense, Q8 | 3 B | yes | **~1** | assisted coding, a few minutes per answer |
| 7 B dense, Q4/Q8 | 7 B | yes | **0.3–0.4** | batch: write a function, walk away |
| 14 B dense, Q4 | 14 B | yes | ~0.2 | not worth it over 7B |
| 30B-A3B MoE | 3.35 B | **no** | 0.147 | demonstration only |

**The two goals pull in opposite directions on this laptop.** Paging needs *total ≫ RAM*; usable
speed needs *few active parameters and a model that fits*. The only models that satisfy both are
MoE with a large total and a tiny active set — and those exist (30B-A3B, Laguna XS), but their
working set still exceeds a tight budget, so they land at the same 0.15 tok/s. A model that fits has
no paging to do; a model that pages is too slow to use.

So the honest thesis statement is narrower than the project started with:

> On a 16 GB, 4-core laptop, a model larger than RAM can be run correctly under a hard budget with
> full per-token accounting, at ~0.15 tok/s — and the OS page cache is as fast at equal memory. The
> *usable* target on this hardware is a model that fits, run by an engine whose per-token compute
> cost is the real limit.

## What to port, then

The goal changes from "a model that does not fit" to **"the best coding model that fits, at ~1
tok/s or better"**. Filter, in order: few active parameters → fits 16 GB → architecture the engine
already has → coding/agentic quality → license.

Configs read from the model cards, not from memory:

| candidate | config | active | Q8 | engine status | license | expected |
|---|---|---:|---:|---|---|---:|
| **Qwen2.5-Coder-3B** | 36 L, d 2048, 16/2 heads, d_ff 11008, theta 1e6, **tied head**, 32K ctx, vocab 151936 | 3 B | 3.2 GB | **drop-in for `qwen2.c`** | Apache 2.0 | ~1 tok/s |
| Qwen2.5-Coder-7B | 28 L, d 3584, 28/4, d_ff 18944, theta 1e6, **untied head**, 32K | 7 B | 7.6 GB | needs untied-head support in `qwen2.c` (~20 lines + gate) | Apache 2.0 | ~0.35 tok/s |
| Qwen2.5-Coder-1.5B | 28 L, d 1536, 12/2, tied | 1.5 B | 1.6 GB | drop-in | Apache 2.0 | ~1.5–2 tok/s |
| Seed-Coder-8B | Llama-3.1 shape, untied | 8 B | 8.5 GB | needs untied head | MIT | ~0.3 tok/s |
| Qwen3-Coder-30B-A3B | verified above | 3.3 B | 32.5 GB | drop-in, does not fit | Apache 2.0 | 0.147 — demonstration only |

**The 3B is the only true drop-in.** `qwen2.c` assumes a tied unembedding (it is what the 0.5B
has), and Qwen2.5-Coder-3B ties while the 7B and Seed-Coder-8B do not. Everything else is already
parametric in that file — head counts, GQA ratio, `rope_theta` and eps all come from the v4 block
params, so 16/2 (8:1) and d_ff 11008 need no code change.

**Recommendation: port `Qwen2.5-Coder-3B-Instruct` first**, then 7B as the quality tier. Both go
through the *existing* `engine/blocks/qwen2.c` path, so the work is a converter + MODEL.md + a gate
against a recorded HuggingFace baseline — days, not weeks — and neither needs a new block. That is
the cheapest route to a model this machine can actually be used with.

Two things to fix on the way:
- **`rope_theta` is hardcoded** (`1000000.0f`) in the Qwen3 loader even though the v4 attention
  block carries it. Any port with a different theta silently rotates positions wrong. Fix it with a
  gate, before it bites.
- **Dequant is the throughput ceiling.** The engine's ~1.28 ms per 1B active elements is scalar Q8
  dequant, and the vault already measured Q8 *slower* than fp32 for in-cache matrices. An AVX2
  dequant pass would roughly double the numbers in the envelope table, and it is the only change
  that lifts *every* model at once.

## What is not claimed

- Not that 0.15 tok/s is a failure of the thesis — the thesis was "can it run exactly", and it does.
- Not that the pager is useless: it enforces a budget and counts what it does, which is what makes
  the measurement possible at all, and pinning gives a real 2.1× at 2 GB.
- Not that a 3B coding model is a *good* coding model. It is the best this machine can serve at a
  usable speed, and that is a different sentence from "a good coding model".
- No number in the envelope table is measured above 3.35 B active; the 1.5/3/7 B rows are
  extrapolated from two anchors and must be replaced by measurement as each model is ported.
