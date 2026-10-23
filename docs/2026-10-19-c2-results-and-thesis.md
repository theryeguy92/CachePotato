# C2 results, and the thesis they actually support

**Date:** 2026-10-19 · **Branch:** `feat/qwen3-30b-a3b` · **Status: conclusion of the paging experiment.**

This is the summary record for C1 and C2. The detailed measurements are in `bench/results/c2/`;
this file says what they mean, including the part where the original target turned out to be the
wrong one.

> **Revised 2026-10-23.** The first version of this note drew its conclusions from single-threaded
> measurements (0.147 tok/s) that a harness bug had produced. The table below was already the
> corrected 4-thread one; the text after it now matches it too. The 4-bit results that came after
> this note are summarised at the end.

## What was measured

**C1: the pager is attached to the Qwen3 path and changes nothing about the answer.** At an
unlimited budget, and at a budget tight enough to force evictions, the 48-layer output is
**bit-identical** to the unpaged run. The accounting is checked against the file's own region table
(819 regions / 3,600.2 MB for one token: 435 core + 384 experts), not against itself.

**C2: the paging numbers.** Qwen3-30B-A3B at Q8_0, a 32.5 GB file on a 16 GB machine. 32 greedy
tokens, cold start, interleaved arms, **4 threads**, the region-chunked `embed` file, 2 reps (1 rep
for 13 and 14 GB; the deterministic counters are identical in every rep). "Honest disk" means
`CP_PAGER_SYS=1`: evicted regions are also dropped from the OS page cache, so a miss is a real disk
read.

| arm | regions/token | fetched MB/tok | physical MB/tok | ms/token | tok/s |
|---|---:|---:|---:|---:|---:|
| unpaged control | 38.3 *(demand)* | 191.3 | 401.3 | 1417 | 0.709 |
| 8 GB, page cache on | 102.2 | 511.4 | 374.1 | 1357 | 0.737 |
| 8 GB, honest disk | 102.2 | 511.4 | 926.2 | 2753 | 0.363 |
| 11 GB, honest disk | 39.2 | 196.2 | 362.0 | 1304 | 0.767 |
| **13 GB, honest disk** | 38.3 | 191.3 | 348.5 | **1252** | **0.799** |
| 14 GB, honest disk | 38.3 | 191.3 | 347.3 | 1266 | 0.790 |

Two things to read off it:

- **At 11–13 GB the pager is 8–13% faster than the unpaged control** (0.767–0.799 against 0.709)
  while running under a hard cap. It fetches about as many regions as the unpaged run needs (the
  38.3 demand figure), so it is not choosing better regions to keep; it reads fewer physical bytes
  for the same regions.
- **The budget curve saturates at ~13 GB.** 38.3 regions/token is the demand: the misses every
  token must take because it selects experts the previous tokens did not. A larger budget cannot
  remove them.

**The cost model**, fitted from the two 8 GB arms and checked against the control:
`t = 374 ms compute + physical MB ÷ 0.389 GB/s`. It predicts 1,405 ms for the control against
1,417 ms measured. At 13 GB, **disk is ~895 ms of the token (~71%)** and compute is 374 ms, so the
engine is disk-bound at every budget. At Q8_0 the ceiling on this machine is therefore
**~0.8 tok/s**, set by the compulsory 191 MB/token of newly selected experts times ~1.8× of block
rounding.

Three findings that held up:

1. **The pager does not keep a better working set than the OS.** At 11 GB, in an earlier
   single-threaded 16-token arm, it fetched 54.8 regions/token against a demand of 54.7: a tie on
   regions. Its small speed edge near the working set comes from reading fewer physical bytes.
2. **Below the working set, the OS wins**, because the OS uses all ~12 GB of free RAM while the
   budget is a cap. Pinning the core (non-expert) tier with `CP_PIN_CORE=1` cuts the regions
   fetched 2.13× at 2 GB. That is a real policy win and the only one, but it cannot make a 2 GB cap
   compete with 12 GB of RAM.
3. **Two exact layout/read experiments were slower.** Co-selection expert grouping lost because
   the paired experts were chosen together only 11.2% of the time, so fetching both doubled the
   bytes. A pager-owned region arena lost because a region is not an access unit: it read exactly
   the requested bytes on a single tensor, but the 330 MB `embed` region is touched for 4 KB.

## What this says about the target, not the engine

The engine works: a 32.5 GB model runs on a 16 GB machine, exactly, under a hard budget, with
per-token counters. At 0.8 tok/s, a 300-token answer takes about 6 minutes and a 1,000-token answer
about 21. That is usable for one self-contained question at a time, not for conversation or for an
agent loop.

Compute is only 374 ms of the 1,252 ms token, and Round 4 of
[Where tok/s goes](2026-10-19-tok-s-levers.md) shows that even the compute term is bytes streamed
from RAM. Both halves of the token scale with bytes, so the lever is fewer bytes per token: a
lower-precision encoding, or a model with fewer active parameters.

**The two goals still pull in opposite directions on this laptop.** Paging needs a model much larger
than RAM; speed needs few bytes per token. A model that fits has no paging to do, and a model that
pages pays for its compulsory expert reads.

The thesis statement this supports:

> On a 16 GB, 4-core laptop, a model twice the size of RAM can be run correctly under a hard memory
> budget with full per-token accounting, at ~0.8 tok/s at Q8_0. At equal memory the pager fetches
> about as much as the OS page cache, and is slightly faster near the working set. Speed beyond that
> comes from fewer bytes per token, not from the paging policy.

## What to port next (as recommended on 2026-10-19)

The recommendation at the time was to change the goal from "a model that does not fit" to **"the
best coding model that fits, at ~1 tok/s or better"**, filtering in order: few active parameters,
fits 16 GB, an architecture the engine already has, coding quality, license.

Configs as read from the model cards:

| candidate | config | active | Q8 size | engine status | license |
|---|---|---:|---:|---|---|
| **Qwen2.5-Coder-3B** | 36 L, d 2048, 16/2 heads, d_ff 11008, theta 1e6, **tied head**, 32K ctx, vocab 151936 | 3 B | 3.2 GB | **drop-in for `qwen2.c`** | Apache 2.0 |
| Qwen2.5-Coder-7B | 28 L, d 3584, 28/4, d_ff 18944, theta 1e6, **untied head**, 32K | 7 B | 7.6 GB | needs untied-head support in `qwen2.c` (~20 lines + gate) | Apache 2.0 |
| Qwen2.5-Coder-1.5B | 28 L, d 1536, 12/2, tied | 1.5 B | 1.6 GB | drop-in | Apache 2.0 |
| Seed-Coder-8B | Llama-3.1 shape, untied | 8 B | 8.5 GB | needs untied head | MIT |
| Qwen3-Coder-30B-A3B | same geometry as the model above | 3.3 B | 32.5 GB | drop-in apart from `rope_theta`; does not fit | Apache 2.0 |

`qwen2.c` assumes a tied output head, as the 0.5B has. Qwen2.5-Coder-3B ties; the 7B and Seed-Coder-8B
do not. Everything else in that file is already parametric (head counts, GQA ratio, `rope_theta` and
eps come from the v4 block parameters), so the 3B's 16/2 heads and d_ff 11008 need no code change.

The expected speeds for these models are in
[What this laptop can run](2026-10-19-what-can-run.md), which redoes the envelope with the 4-thread
cost model. The single-threaded envelope that stood here has been removed.

**Status:** none of these ports has been started. The 4-bit conversion of the existing model came
first (below).

Two things to fix on the way, both still open:

- **`rope_theta` is hardcoded** to 1e6 in the Qwen3 loader (`qwen3_map`), even though the v4
  attention block carries it. Any model with a different theta (Qwen3-Coder uses 1e7) would rotate
  positions wrongly without any error. Fix it with a gate before it bites.
- **Top-k and the expert-table size are hardcoded too** (8 and 128).

## What came after: 4-bit

The bytes argument above predicted that Q4_0 would roughly halve the token. Measured at 13 GB,
4 threads (details in [Where tok/s goes](2026-10-19-tok-s-levers.md), Rounds 5–7):

| file | tok/s | teacher-forced agreement with Q8_0 |
|---|---:|---|
| Q8_0 | 0.80 | reference |
| Q4_0 experts | 1.14 | 90.8% / 98.3% (120 tokens), pass |
| **Q4_0 experts + attention** | **1.22** | **93.3% / 97.5% (120 tokens), pass** |
| Q4_0 everywhere | 1.40 | 70.8% on the factual probe (24 tokens only), rejected |

The 1.22 tok/s file is the shipped configuration. Its paging is not separately gated; the
bit-identity gates run on the Q8_0 file.

## What is not claimed

- Not that 0.8 tok/s is a failure of the thesis. The thesis was "can it run exactly", and it does.
- Not that the pager is useless. It enforces a budget and counts what it does, which is what makes
  these measurements possible, and pinning gives a real 2.13× at 2 GB.
- Not that a 3B coding model is a *good* coding model. It would be the best this machine can serve
  quickly, which is a different statement.
- No speed is measured for any model between 0.5 B and 3.35 B active; those figures are
  extrapolated and must be replaced by measurements as each model is ported.
