# Where tok/s actually goes, and what to do about it

**Date:** 2026-10-19 · **Branch:** `feat/qwen3-30b-a3b` · **Status: review + the first measured lever.**

Reviewed for this note: `engine/kernels/kernels.c` (Q8 inner loop, matmul, rmsnorm), `engine/kernels/
parallel.c` (pool, dispatch, thresholds), `engine/blocks/qwen2.c` and `qwen3_moe.c` (bind, layer
loop), `experiments/2026-10-17-gemv-ceiling`, the vault's threading and quantisation entries, and
every C2 measurement on this branch.

## The measurement bug that came out of the review

`cp_threads()` in `engine/kernels/parallel.c` **defaults to one thread** and lazily starts the pool:
`int n = 1; if (CP_THREADS is set) n = atoi(...)`. The engine never sets it, and
`tools/c2_measure.py` built each run's environment by **stripping every `CP_*` variable** — so
`CP_THREADS` went with them.

**Every speed number on this branch was single-threaded.** Re-measured properly — 32 tokens, 2 reps,
cold start, same protocol, only `CP_THREADS` differing:

| arm | 1 thread | 4 threads | factor |
|---|---:|---:|---:|
| unpaged control | 5706 ms · 0.176 tok/s | **1932 ms · 0.521 tok/s** | **2.95×** |
| 8 GB (page cache on) | 5877 ms · 0.171 | **1752 ms · 0.572** | **3.35×** |
| 8 GB + `CP_PAGER_SYS=1` (honest disk) | 6805 ms · 0.147 | **3182 ms · 0.314** | **2.14×** |

8 threads is *worse* than 4 (the machine has 4 physical cores, as the vault recorded). The
deterministic columns are unchanged (108.3 regions/token, 542.9 MB fetched, 104.4 evictions): the
policy is thread-independent, so C1's bit-identity result and the C2 page/byte columns stand. Only
the *speed* columns were wrong — by **2.1–3.4×**, not the 1.68× that a 3-token spot check suggested
(that check was cold, 3 tokens, cache-on, and therefore measured the wrong thing; a reminder that
the protocol matters even for a 10-minute question).

Fix: `c2_measure.py` now sets and records `CP_THREADS` (default 4). The drivers should default to
`nproc` too; the engine's own default of 1 exists so recorded benchmarks do not move silently, which
is right for the engine and wrong for a measurement harness.

## Where the 3.18 s goes (30B at 8 GB, honest disk, 4 threads)

Per generated token, from the io counter and the deterministic page counts:

| term | time | share | notes |
|---|---:|---:|---|
| disk reads | ~2.5 s | **79%** | 977 MB physical at ~0.39 GB/s — and 1.9× of that is *fault-around* over-read, not the regions |
| compute | ~0.7–0.9 s | 22–28% | 3.35 B active elements at 4 threads (≈0.27 s per 1B) |
| serial per-token overhead | ~0.2 s | 6% | 48 × `qwen3_bind_layer`, each walking 18,867 tensor records with an `sscanf` per expert (18,432 of them) |

So on the honest arm the token is **I/O-bound by ~4:1** — the opposite of what the single-threaded
numbers suggested, and it reorders the levers: the disk is now the thing to fix, and the kernel
matters mainly for models that fit.

## The levers, in value order

**1. Threads (done, free, 1.68×).** 0.147 → 0.21 tok/s. Also raises the dense models, which are
purely compute-bound.

**2. Overlap I/O with compute (≈1.3–1.9×, medium cost).** Disk and compute are serial today: 3.2 s
of reading, then ~1.4 s of arithmetic, per token. But the *next* layer's regions are known as soon as
the router for this layer runs, and the *next token's* experts overlap the current token's 71–83% of
the time. One prefetch thread issuing `pread` into the arena while the main threads compute would
hide most of the 3.2 s. This is the v3 `CP_PREFETCH_W` idea with the arena behind it, and it is the
single largest remaining factor for the paging regime. Gate: bit-identity (values do not change).

**3. Stop over-reading (≈2×, low cost, already half-built).** The 1.27 GB/token of disk traffic is
~1.9× the 0.55 GB the regions actually need: `mmap` fault-around reads past the region. The arena's
`pread` fixes it (measured 1.00× on a single tensor) but currently loses because a *region* is not an
*access unit* — the 330 MB `embed` region is touched for 4 KB. Declaring `embed` as row-chunks and
using the arena gives ~0.55 GB/token → I/O ~1.4 s. Combined with (2): I/O hidden behind compute.

**4. Bind once, not 48 times (≈4% now, more after 2–3, small cost).** `run_layers` calls
`qwen3_bind_layer(m, l)` on **every token**, and each call walks all 18,867 tensor records and
`sscanf`s 18,432 expert names. The per-layer geometry and offsets never change between tokens; only
the arena's data pointers do, and those are resolved per touch now. Bind all 48 layers once into an
array and the walk disappears — along with ~41 MB/token of tensor-table reads.

**5. Q8 inner loop (≈1.2–1.6×, small cost, but a numerics change).** `wx_q8_slice` does
`pmovsxbd → cvtdq2ps → mul(scale) → fma(x)` four times per 32-element block, one accumulator per row.
Two changes: factor the per-block scale out of the sum (`Σ qᵢxᵢ` then `×d` — one multiply per 32
elements instead of 32), and keep 2–4 output rows in flight so the FMA chains overlap. The first is
**not bit-identical** (different rounding), so it needs approval and must pass the existing
`q8_kernels` gate (1e-5 relative) plus B5's tolerance. The second is bit-identical.

**6. Q4 for the experts (2× bytes, not yet justified).** The GEMV bench says 4-bit reaches 9.56
GFLOP/s against fp32's 6.3 at 8 threads, but the engine measured Q4 *slower* in-cache and its quality
is Tier C. Only worth revisiting after (2) and (3), and it does nothing for a model that fits.

**Not levers, verified:** dispatch cost (~0.25 µs × ~1,150 matmuls = 0.3 ms/token); the attention
maths at T=1; AVX-512 (absent on this CPU: `avx2 fma sse4.2`); more than 4 threads.

## What each target gets

Anchored on the measured 0.27 s per 1B active elements at 4 threads and 0.39 GB/s on the honest arm:

| target | fits? | compute | I/O | tok/s | useful for |
|---|---|---:|---:|---:|---|
| 30B-A3B @ 8 GB (measured) | no | 0.9 s | 2.5 s | **0.314** | batch coding, ~16 min per 300 tokens |
| 30B-A3B @ 8 GB after levers 2+3 | no | 0.9 s | ~0.7 s (overlapped) | **~0.6–1.0** | batch coding, ~5–8 min per 300 tokens |
| Qwen2.5-Coder-3B | yes | 0.8 s | 0.5 s (RAM) | ~0.8 | faster but far less capable |
| Qwen2.5-Coder-7B | yes | 1.9 s | 0.5 s | ~0.4 | same speed as the MoE, less capable |
| Qwen2.5-0.5B | yes | 0.14 s | 0.05 s | ~5 (10–15 with threads per README) | snappy chat, weak |

**The comparison that matters: a dense 7B is no faster than the 30B-A3B** (7B active vs 3.35B), and
much less capable. That is the whole argument for MoE on this machine, and it means the best model to
run is the *same shape as today* — which is exactly what a Qwen3-Coder-30B-A3B port would be.

The dense numbers are extrapolations from the two measured anchors — flag them as such until each
model is ported and measured. The point is the ordering: **threads are free, the kernel is cheap, and
the paging levers (2 and 3) are what the 30B needs.**

## Proposed order

1. ~~**Threads everywhere**~~ — **done and re-measured** (`bench/results/c2/2026-10-19-threaded*`):
   0.147 → **0.314** on the honest arm, 0.521 on the unpaged control. Harness now pins
   `CP_THREADS=4`; the drivers should default to `nproc`.
2. **Bind once** — small, bit-identical, and it removes a per-token cost that will look bigger once
   the I/O is hidden.
3. **Region-sized reads: `embed` by row + arena** — the converter change both negative results
   pointed at. Then re-measure: expect disk traffic 1.27 → ~0.55 GB/token.
4. **Overlap I/O with compute** — prefetch thread into the arena; the largest single factor for the
   paging regime.
5. **Kernel work (scale hoist + multi-row)** — the only lever that helps the *dense* target too, and
   the scale hoist needs sign-off because it is not bit-identical.

Steps 1–2 are hours. Steps 3–4 are the paging story and worth a day. Step 5 is the one that makes a
ported 3B feel usable, and it is also the only one with a numerics decision in it.

## Measured outcomes (2026-10-19, in the agreed order)

**Step 4 — bind once: DONE, +11%.** 3182 → **2861 ms/token (0.350 tok/s)** at 8 GB, 4 threads,
honest disk. Bit-identical (all five paging arms re-gated).

**Step 2 — embed by row: code done and gated, full conversion pending.** `--embed-rows 1024`
gives 149 embed regions instead of one 330 MB region; the engine resolves the chunk for a token's
row and the pager fetches ~2 MB instead of the whole table. Verified **bit-identical** on a 1-layer
file against the one-region version (unpaged and with the arena). The 32.5 GB re-conversion was
started twice and abandoned both times because it competes with every measurement for CPU/disk —
it needs a clean window, and its result is the next number to get.

**Step 5 — kernel: DONE (bit-identical part), and the numerics part was measured and rejected.**

| change | 1 thread | 4 threads | verdict |
|---|---|---|---|
| vector path reachable without the thread pool | 768×2048: 4.63 → **0.94 ms** (4.9×) | n/a (already parallel) | **keep** — `CP_THREADS=1` was silently running the *scalar* loop, because the AVX2 kernel lived only inside the parallel dispatch |
| 4 rows per pass, shared x loads | — | 0.49 → 0.37 ms (contended run); clean: 13.1 GB/s | **keep** — bit-identical, at the DRAM ceiling |
| hoist the block scale out of the dot | 0.94 → 0.47 ms (2×) | 768×2048 **0.13 → 0.19 ms (worse)**, 2048×768 +12%, 4096×2048 +9% | **rejected** — a wash, and it costs bit-identity |

**But at 8 GB none of it shows:** 2861 → 2915 ms (within noise), because 2.5 s of the 2.9 s token
is disk. The kernel lever is real and will show once the bytes come down (step 2) or at a budget
where the token is compute-bound. That is the ordering lesson: **I/O first, compute second.**

**Step 3 — prefetch: rejected before building it, from the data already on disk.** The plan was to
prefetch the previous token's expert set on the assumption that 71–83% of a token's experts recur.
The pager's own hit rate says the opposite of what the plan needs: **86.8% of touches are already
hits** (710 of 819 regions), so the previous token's 384 regions are *resident* when the next token
starts — queuing them reads nothing. The 108 regions that miss are, by definition, the experts the
previous token did *not* select, and no prefetch can know them before the router runs. At 8 GB the
I/O is demand-driven on unpredictable experts; there is nothing to overlap it with. The only real
way to cut it is fewer bytes (step 2), a bigger budget (4→11 GB halves the fetches), or a layout
that raises reuse (Tier B, measured and negative).
