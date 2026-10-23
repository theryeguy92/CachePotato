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

"Honest disk" means `CP_PAGER_SYS=1`: evicted regions are also dropped from the OS page cache, so a
miss is a real disk read rather than a page-cache hit.

Per generated token, from the io counter and the deterministic page counts:

| term | time | share | notes |
|---|---:|---:|---|
| disk reads | ~2.5 s | **79%** | 977 MB physical at ~0.39 GB/s — and 1.9× of that is *fault-around* over-read, not the regions |
| compute | ~0.7–0.9 s | 22–28% | 3.35 B active elements at 4 threads (≈0.27 s per 1B) |
| serial per-token overhead | ~0.2 s | 6% | 48 × `qwen3_bind_layer`, each walking 18,867 tensor records with an `sscanf` per expert (18,432 of them) |

The three terms are separate estimates, and they add up to slightly more than the measured 3.18 s.
So on the honest arm the token is **I/O-bound by roughly 3:1** — the opposite of what the single-threaded
numbers suggested, and it reorders the levers: the disk is now the thing to fix, and the kernel
matters mainly for models that fit.

## The levers, in value order

**1. Threads (done, free, 2.14×).** 0.147 → 0.314 tok/s on the honest arm. Also raises the dense
models, which never touch the disk.

**2. Overlap I/O with compute (≈1.3–1.9×, medium cost).** Disk and compute are serial today: ~2.5 s
of reading, then ~0.7–0.9 s of arithmetic, per token. But the *next* layer's regions are known as soon as
the router for this layer runs, and the *next token's* experts overlap the current token's 71–83% of
the time. One prefetch thread issuing `pread` into the arena while the main threads compute would
hide most of the 2.5 s. This is the v3 `CP_PREFETCH_W` idea with the arena behind it, and it is the
single largest remaining factor for the paging regime. Gate: bit-identity (values do not change).

**3. Stop over-reading (≈2×, low cost, already half-built).** The ~0.98 GB/token of disk traffic
is ~1.8× the 0.54 GB the regions actually need: `mmap` fault-around reads past the region. The arena's
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

**6. Q4 for the experts (2× bytes, not yet justified).** *(Update: measured in Rounds 5–7 below.
In the disk-bound regime it paid, and it is now the shipped configuration.)* The GEMV bench says 4-bit reaches 9.56
GFLOP/s against fp32's 6.3 at 8 threads, but the engine measured Q4 *slower* in-cache and its quality
is Tier C. Only worth revisiting after (2) and (3), and it does nothing for a model that fits.

**Not levers, verified:** dispatch cost (~0.25 µs × ~1,150 matmuls = 0.3 ms/token); the attention
maths at T=1; AVX-512 (absent on this CPU: `avx2 fma sse4.2`); more than 4 threads.

## What each target gets

*(Superseded: Round 3 below refits compute at 374 ms for 3.35 B active, i.e. ~0.11 s per 1B rather
than 0.27 s, and [What this laptop can run](2026-10-19-what-can-run.md) redoes this table. Kept as
the estimate made at the time.)*

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
   pointed at. Then re-measure: expect disk traffic ~0.98 → ~0.55 GB/token.
4. **Overlap I/O with compute** — prefetch thread into the arena; the largest single factor for the
   paging regime.
5. **Kernel work (scale hoist + multi-row)** — the only lever that helps the *dense* target too, and
   the scale hoist needs sign-off because it is not bit-identical.

Steps 1–2 are hours. Steps 3–4 are the paging story and worth a day. Step 5 is the one that makes a
ported 3B feel usable, and it is also the only one with a numerics decision in it.

## Measured outcomes (2026-10-19, in the agreed order)

The headings below use the lever numbers from the list above, not the order of the plan.

**Lever 4, bind once: DONE, +11%.** 3182 → **2861 ms/token (0.350 tok/s)** at 8 GB, 4 threads,
honest disk. Bit-identical (all five paging arms re-gated).

**Lever 3, embed by row: code done and gated, full conversion pending.** `--embed-rows 1024`
gives 149 embed regions instead of one 330 MB region; the engine resolves the chunk for a token's
row and the pager fetches ~2 MB instead of the whole table. Verified **bit-identical** on a 1-layer
file against the one-region version (unpaged and with the arena). The 32.5 GB re-conversion was
started twice and abandoned both times because it competes with every measurement for CPU/disk —
it needs a clean window, and its result is the next number to get.

**Lever 5, kernel: DONE (bit-identical part), and the numerics part was measured and rejected.**

| change | 1 thread | 4 threads | verdict |
|---|---|---|---|
| vector path reachable without the thread pool | 768×2048: 4.63 → **0.94 ms** (4.9×) | n/a (already parallel) | **keep** — `CP_THREADS=1` was silently running the *scalar* loop, because the AVX2 kernel lived only inside the parallel dispatch |
| 4 rows per pass, shared x loads | — | 0.49 → 0.37 ms (contended run); clean: 13.1 GB/s | **keep** — bit-identical, at the DRAM ceiling |
| hoist the block scale out of the dot | 0.94 → 0.47 ms (2×) | 768×2048 **0.13 → 0.19 ms (worse)**, 2048×768 +12%, 4096×2048 +9% | **rejected** — a wash, and it costs bit-identity |

**But at 8 GB none of it shows:** 2861 → 2915 ms (within noise), because 2.5 s of the 2.9 s token
is disk. The kernel lever is real and will show once the bytes come down (lever 3) or at a budget
where the token is compute-bound. That is the ordering lesson: **I/O first, compute second.**

**Lever 2, prefetch: rejected before building it, from the data already on disk.** The plan was to
prefetch the previous token's expert set on the assumption that 71–83% of a token's experts recur.
The pager's own hit rate says the opposite of what the plan needs: **86.8% of touches are already
hits** (710 of 819 regions), so the previous token's 384 regions are *resident* when the next token
starts — queuing them reads nothing. The 108 regions that miss are, by definition, the experts the
previous token did *not* select, and no prefetch can know them before the router runs. At 8 GB the
I/O is demand-driven on unpredictable experts; there is nothing to overlap it with. The only real
way to cut it is fewer bytes (lever 3), a bigger budget (8→11 GB halves the fetches), or a layout
that raises reuse (Tier B, measured and negative).

## Lever 3 measured: chunked embed helps, the arena still does not, and why

Full re-conversion with 149 embed regions (6,727 regions, 19,015 tensors, `V4 FORMAT: PASS`), and
the 48-layer output is **bit-identical** to the one-region file. Measured at 8 GB / 4 threads /
32 tokens / cold start:

| file | read path | regions/tok | fetched MB/tok | physical MB/tok | ms/tok | tok/s |
|---|---|---:|---:|---:|---:|---:|
| one `embed` region | mmap | 108.3 | 542.9 | 986 | 2861 | 0.350 |
| **149 embed regions** | **mmap** | 102.2 | 511.4 | **866** | **2668** | **0.375** |
| 149 embed regions | arena | 102.3 | 512.0 | 1059 | 3233 | 0.310 |

**Chunked embed is the best configuration so far: 0.375 tok/s** (+7% on bind-once, +155% on the
0.147 this branch started at). The arena is *still* worse, and instrumenting the pread volume
explains both facts at once.

### Why the arena cannot win on this filesystem

The pager requested **11,424 MB via `pread` over 13 steps** — exactly the region bytes it fetched —
and the process read 1.3–2× that from the block device. The excess is not the pager and not
fault-around: **btrfs/kernel issue I/O in up-to-4 MiB blocks** (`max_sectors_kb=4096`), and an expert
region is 4.78 MB, so it always spans two of them. A fetch of 4.78 MB therefore reads 8–12 MB.

Under **mmap that is not waste**: the extra block bytes land in the page cache and the *neighbouring*
region — the next expert of the same layer, or the same expert next token — is served from them.
Under the **arena** the over-read is thrown away by the `POSIX_FADV_DONTNEED` that makes the
measurement honest, so the arena pays for the same blocks and then discards what the page cache
would have reused. That is why 866 MB under mmap becomes 1059 MB under the arena.

Which corrects the earlier framing: the "1.9× amplification" is **block rounding plus readahead that
the cache reuses**, not pure waste. It only becomes waste when the pager owns the bytes and drops
them. The read path the measurements favour is therefore **mmap for the data + chunked regions for
the accounting**, which is exactly the 0.375 tok/s row.

### What is left

- **Total for this round: 0.147 → 0.375 tok/s (2.55×)** at 8 GB, with page counts, file format and
  numerics unchanged (bit-identity gated at every step).
- The remaining I/O is demand-driven: **86.8% of touches are already hits**, so there is nothing to
  prefetch (lever 2), and the bytes are block-rounded (lever 3). Both levers are now closed by
  measurement, not by opinion.
- What remains is **compute** (the token is ~1.4 s compute at 4 threads, ~0.7 s I/O now) and **budget**:
  the 11 GB arm halves the fetches through reuse alone (~0.45–0.5 tok/s expected) without touching a
  line of code. That is the next measurement, not the next implementation.

## Round 3: the consolidated table, the cost model, and why the compute levers are capped

Final measurement (chunked file, 4 threads, cold start, 32 tokens):

| arm | regions/token | physical MB/tok | ms/token | tok/s |
|---|---:|---:|---:|---:|
| unpaged control | 38.3 | 401.3 | 1417 | 0.709 |
| 8 GB, cache on | 102.2 | 374.1 | 1357 | 0.737 |
| 8 GB, honest disk | 102.2 | 926.2 | 2753 | 0.363 |
| 11 GB, honest disk | 39.2 | 362.0 | 1304 | 0.767 |
| **13 GB, honest disk** | 38.3 | 348.5 | **1252** | **0.799** |
| 14 GB, honest disk | 38.3 | 347.3 | 1266 | 0.790 |

**A two-parameter model fits all of it:** `t = 374 ms + physical_MB / 0.389 GB/s` (control predicted
1405 against 1417 measured; the 8 GB arms are exact by construction). Two consequences:

1. **Compute is 374 ms — 30% of the token at 13 GB, 14% at 8 GB.** The int8-activation dot would
   attack that 374 ms and nothing else: its ceiling is `1252 → 895 ms`, i.e. **1.4×**, and it costs
   the q8 gate's tolerance and a quality question that Tier C would have to re-answer. Same for f16
   weights (a format change) and for any further kernel work. **Not worth it** — the compute levers
   are now capped by arithmetic, not by implementation effort.
2. **Disk is 895 ms of 1252 ms at 13 GB**, and the curve saturates because 38.3 regions/token is the
   *compulsory* demand. So the wall is **0.8 tok/s**, set by the new experts a token selects.

### What that leaves

- **Fewer bytes per compulsory region**: Q4 halves each expert to ~2.4 MB, but block rounding
  (4 MiB) then dominates, so expect ~160 MB/token → ~1.0 tok/s rather than 2×. Tier C quality, and
  the Q4 kernel measured slower in the vault's earlier work.
- **Fewer compulsory regions**: a top-4 model (half the active experts) would halve the bytes;
  that is a model choice, not an engine one.
- **More reuse**: Tier B measured negative at 11.2% co-selection.
- **A faster disk**: 0.389 GB/s is the engine's effective read rate from this SATA SSD (an
  `O_DIRECT` probe saturates at 370–400 MB/s).

Time on this machine: **~0.8 tok/s ceiling for this model at Q8_0**, and compute is only ~30% of
the token. The next useful thing is not a kernel — it is a *smaller-active* model (the Qwen3-Coder
port at the same geometry, or a 3B for speed), measured with this same harness.

## Round 4: the engine is a bandwidth machine — both halves of the token are bytes

Adding up what a token must stream, from the file's own region accounting (13 GB arm):

| term | bytes/token | rate | time | from the fit |
|---|---:|---:|---:|---:|
| **RAM**: core regions (incl. `lm_head`) 1,597 MB + 384 expert regions 1,925 MB | **3,523 MB** | 9.4 GB/s (4-thread effective) | 375 ms | 374 ms |
| **disk**: physical reads after reuse | **348 MB** | 0.389 GB/s (SSD) | 895 ms | 895 ms |
| | | | **1,270 ms** | measured token: **1,252 ms** |

Each term agrees with the fit to within 0.3%, and the sum is within 1.5% of the measured token. **There is no arithmetic ceiling here — the engine is limited by
bytes/bandwidth on both sides**, so no kernel technique can move it. This closes the whole
technique list:

| technique | verdict | why |
|---|---|---|
| more CPU threads | **no** | measured 4 / 6 / 8 threads = 1921 / 1964 / 4467 ms on the 13 GB arm; the part has 4 physical cores |
| CPU affinity (`CP_PIN_CORES=1`) | **+7%, committed as opt-in** | 1478 → 1374 ms/token over alternating runs; migration between HT siblings was costing the L1/L2 hot set |
| deeper I/O queue (pre-issue all 8 experts) | **no** | O_DIRECT probe: the SSD does 288 MB/s at depth 1 and saturates at **370–400 MB/s from depth 2 up**; the engine already measures 389, so page-fault readahead is already pipelining |
| int8-activation dot (`vpmaddubsw`) | **no** | it reduces ALU work, and ALU is not the limit |
| speculative decoding | **≤1.4×, and only without Q4** | exact under greedy, but a batch of k positions multiplies the *RAM* term by k while saving only part of the disk term — and Q4 shrinks the disk term, making speculation less valuable, not more |
| prefetch / arena / Tier B grouping | **already closed** | measured in rounds 2–3 |
| AVX-512 / VNNI | **absent** | this CPU is `avx2 fma sse4.2` |

### The one lever left, and what it is worth

**Halve the bytes.** Both terms scale with bytes, so Q4 halves the token: `3,523/2 GB ÷ 9.4 GB/s`
(187 ms) + `348/2 MB ÷ 0.389 GB/s` (447 ms) = **634 ms → 1.58 tok/s**, from 0.80.

The caveat that makes this a measurement rather than a promise: the vault's earlier "Q4 is slower"
result came from an **in-cache, ALU-bound** regime (the 0.5B model, matrix resident in L2). At 13 GB
on the 30B the regime is the opposite — DRAM-bandwidth-bound — so nibble-unpacking may hide entirely
under memory latency. Nobody has measured Q4 in *this* regime, and that is the next thing to do.

A cheaper variant if a full Q4 is too lossy: **Q4 only the core** (1,597 MB of the 3,523 MB) or only
`lm_head` (330 MB). Core-only Q4 saves ~85 ms/token (~7%) at much lower quality risk than the experts.

## Round 5: Q4_0 measured — 1.40 tok/s, and the quality gate fails

Whole-model Q4_0 (17,220 MB, 6,727 regions) at 13 GB / 4 threads / 32 tokens, 2 reps:

| encoding | regions/tok | physical MB/tok | ms/token | tok/s |
|---|---:|---:|---:|---:|
| Q8_0 | 38.3 | 348.5 | 1252 | 0.799 |
| **Q4_0** | 32.9 | **120.4** | **710** | **1.40** |

**The bytes model is confirmed**: halving the encoding made the token 1.76× faster (1,252 →
710 ms). The model predicted 1.58 tok/s (634 ms) and the run measured 1.40. Two errors partly
cancel: disk came in *under* the prediction (120 vs ~174 MB/token, because smaller regions keep more
of the working set resident), while compute came in ~215 ms *over* it (~400 ms against the halved
187 ms), which is the cost of unpacking 4-bit values. Disk traffic fell further than half
(348 → 120 MB) because smaller regions mean more of the working set is resident at the same budget.

**But it does not pass the quality gate**, which was pre-registered before measuring:

| probe | teacher-forced agreement | free-running | top-5 overlap |
|---|---:|---:|---:|
| code continuation | **91.7%** | 29.2% | 30.8% |
| factual continuation | **70.8%** | 4.2% | 9.2% |

The free-running numbers (29%, 4%) are *snowball*: one flipped token changes every later context,
so they collapse for any lossy encoding and are diagnostic only. The teacher-forced number — the
Q4 model scoring Q8's own trajectory on identical contexts — is the sound one, and 70.8% on the
factual probe is a material loss. **Whole-model Q4_0 is rejected** at 4.5 bpw for a 30B on this
evidence.

### Where the bytes actually are, and the variant that follows

| part | size at Q8 | share of per-token disk | share of per-token RAM |
|---|---:|---:|---:|
| experts (6,144 regions) | 30.0 GB | **100%** | 55% |
| attention + norms + embed + lm_head | 2.5 GB | ~0% (97% resident) | 45% |

The disk traffic is *entirely* experts, and the experts are also the safest thing to quantise
less aggressively than the logits path: `lm_head` produces the logits themselves, and attention
is where a 4-bit weight error propagates most directly. So the variant to measure is **Q4 experts,
Q8 everywhere else** (`--quant-experts q4_0`): same disk saving, no unpack cost on the core, and
the quality risk confined to the expert path. Predicted ~1.6-1.7 tok/s.

## Round 6: the variant that works — Q4 experts, Q8 everywhere else

`--quant-experts q4_0` (17,984 MB, 6,727 regions, `V4 FORMAT: PASS`), same 13 GB / 4 threads /
32 tokens / 2 reps:

| encoding | regions/tok | physical MB/tok | ms/token | tok/s | quality (teacher-forced) |
|---|---:|---:|---:|---:|---|
| Q8_0 | 38.3 | 348.5 | 1252 | 0.799 | exact |
| Q4_0 experts, **Q8 rest** | 53.1 | 184.4 | **874** | **1.14** | **91.7% / 91.7% PASS** |
| Q4_0 whole model | 32.9 | 120.4 | 710 | 1.40 | 91.7% / 70.8% FAIL |

**1.14 tok/s with the pre-registered quality gate passing on both probes** — 43% faster than Q8,
on the model we already run, with no new architecture. That is the configuration to ship.

Two things worth understanding about the numbers:

- **Expert-only Q4 is slower than whole-model Q4 (1.14 vs 1.40) despite the same expert encoding.**
  The core stays at 1.6 GB, so at the same 13 GB budget fewer expert regions stay resident: expert
  hit rate 79.3% vs 84.0%, so it fetches 53 regions/token instead of 33. The disk cost of keeping
  attention and `lm_head` exact is the budget they occupy.
- **Physical bytes per token rose to 184 MB because the regions got small.** An expert at Q4 is
  2.65 MB, which is smaller than the filesystem's 4 MiB I/O unit, so block rounding costs ~1.58x
  per fetch. This is the same rounding documented under lever 3, now biting at a smaller region size:
  quantisation shrinks the payload but not the minimum read. If a Q4 file were laid out with two
  experts per region (5.3 MB), the rounding would amortise — but that is Tier B grouping, which
  costs cache efficiency, and it measured negative at 11 GB.

**The trade is now explicit:** 0.80 (exact) → 1.14 (Q4 experts, ~91.7% next-token agreement on a
matched context) → 1.40 (whole-model Q4, which drops to 70.8% on the factual probe).

The remaining candidate, if the last 20% matters: Q4 on the attention matrices *and* the experts,
with `lm_head` and `embed` left at Q8 — the failure of whole-model Q4 looks like it comes through
the logits path, and `lm_head` is 330 MB of the 1.6 GB core. Predicted ~1.3 tok/s at similar
quality; worth measuring before accepting 1.14 as final.

## Round 7: the gate was underpowered — Q4 attention + experts wins at 1.22 tok/s

The 24-token gate put expert-only Q4 at 91.7/91.7 and Q4-attention at 95.8/**87.5** — a *one
token* difference, and at n=24 one token is 4.2 points. That is not a measurement, it is noise.
Re-run at **120 generated tokens** (one token = 0.8%), with the same 90% threshold:

| encoding | teacher-forced, code | teacher-forced, fact | verdict |
|---|---:|---:|---|
| Q4 experts | 90.8% | 98.3% | PASS |
| **Q4 experts + attention** | **93.3%** | **97.5%** | **PASS** |

The n=24 numbers were not just imprecise, they were *wrong in direction*: expert-only's factual
probe moved from 91.7% to 98.3% when measured properly, and the configuration that "failed" at
87.5% is the one that scores 93.3% on the code probe. Both survivors pass; they are
statistically equivalent, and the faster one wins.

| configuration | regions/tok | physical MB/tok | ms/token | tok/s | quality (120 tok) |
|---|---:|---:|---:|---:|---|
| Q8_0 | 38.3 | 348.5 | 1252 | 0.799 | exact |
| Q4 experts | 53.1 | 184.4 | 874 | 1.14 | 90.8 / 98.3 PASS |
| **Q4 experts + attention, Q8 embed + lm_head** | **32.4** | **125.7** | **822** | **1.22** | **93.3 / 97.5 PASS** |
| Q4 whole model | 32.9 | 120.4 | 710 | 1.40 | 91.7 / **70.8** (n=24 only) |

**The last 20% delivered 7%**, not the 14% the bytes model suggested: quantising attention
(Wq/Wk/Wv/Wo, 0.9 GB) saves ~453 MB of file (17,984 → 17,531 MB) but only 52 ms/token, because attention is the part of
the working set that was already 97% resident — it halves RAM bytes, not disk bytes.

### The open question this leaves

**Whole-model Q4 (1.40 tok/s) was rejected on a 24-token measurement, and 24-token measurements
have now been shown unreliable.** Its factual probe read 70.8% against the survivors' 97.5%+ — a
gap far larger than the ±4 points of noise, so the rejection is probably right — but "probably" is
not the standard the rest of this file holds. Re-testing it needs the file again (17.2 GB, ~30 min)
and an `--gen 120` gate run. Until then, 1.22 tok/s is the fastest configuration that has passed a
gate with enough power to mean something.

The same suspicion applies to the *speed* side: with `lm_head` and `embed` at Q4 the token was
710 ms; the ~110 ms between that and the 822 ms of the 1.22 tok/s file is what keeping `embed`
and `lm_head` at Q8_0 costs.
