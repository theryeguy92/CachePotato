# The 1.9× read amplification, explained: fault-around, not policy

**Date:** 2026-10-18 · **Branch:** `feat/qwen3-30b-a3b` · **Status: closed — cause identified, fix direction stated, not implemented.**

The C2 tables showed the engine reading ~1.8–1.9× the bytes its regions declare, in the pager *and*
in the unpaged control, at every budget. That blocked any absolute bytes/token claim. It is not the
pager, not evictions, not threading, not the file layout, and not the storage. It is the kernel's
fault-around and readahead on the `mmap`, and it is a deliberate trade the engine was making without
saying so.

## The isolating measurement

One `matmul_wx_q8` over a 1.67 MB expert weight tensor, cold (`scratch/matmul_probe.c`):

| threads | read | ratio | with `MADV_RANDOM` |
|---:|---:|---:|---:|
| 1 | 8.38 MB | 5.01× | **1.68 MB (1.00×)** |
| 4 | 9.64 MB | 5.77× | **1.68 MB (1.00×)** |
| 8 | 5.45 MB | 3.26× | **1.68 MB (1.00×)** |

`MADV_RANDOM` disables fault-around, so every fault maps exactly one page and the read equals the
matrix at every thread count. That is the proof: the excess is speculative page-ins, not data the
compute needs.

## At engine level (8 GB, 16 tokens, cold, same deterministic counters)

| mode | physical MB/token | ms/token | tok/s | logical region MB/token |
|---|---:|---:|---:|---:|
| default (fault-around on) | **1011.8** | 6500 | 0.154 | 748.6 |
| `CP_NO_READAHEAD=1` (`MADV_RANDOM`) | **527.6** | 15442 | 0.065 | 748.6 |

The pager's own counters are identical in both (748.6 MB fetched/token, 80.28% hit): the policy does
not change. Only the read mechanism does. **So the 1.9× is readahead, and it buys 2.4× speed** — on
this disk 4 KB reads are latency-bound (~1.7 ms each, the project's own measurement), so fault-around
is what makes a 5 MB region one or two large reads instead of 1224 small ones.

What was ruled out, with the test that ruled it out:
- **file readahead policy** — `CP_FADV_RANDOM=1` (`posix_fadvise(POSIX_FADV_RANDOM)`, ra_pages = 0):
  928 MB vs 915 MB on one token, no effect. It is fault-around, not the file's ra state.
- **evictions** — the amplification is present in the prefill, which evicts nothing, and at 11 GB
  (6.9 evictions/token).
- **threads** — 1, 4 and 8 threads all amplify; `MADV_RANDOM` fixes all three.
- **kernel reclaim of "resident" pages** — RSS peaked at 7.9 GB against a pager belief of 8.0 GB
  with 12 GB available, and at 11 GB the same 1.8× appears.
- **storage/filesystem** — btrfs here has `compress=zstd:3`, but this file is stored uncompressed
  (`st_blocks` ≈ size), and a cold sequential 64 MB read measures 394 MB/s logical vs 419 MB/s
  physical, i.e. 1.0×. A page-by-page read of six expert regions costs 1.24× (fault-around boundary
  effect only).

`max_sectors_kb` is 4096, which is why the over-read lands in 4 MiB chunks: one `MADV_NORMAL` stream
can pull up to 4 MiB per request regardless of how much of it the matrix uses.

## What this means for reporting

- **"Bytes fetched" (logical, region lengths) is the policy metric** and is exact: 748.6 MB/token at
  8 GB, 1260.8 at 4 GB, 3589.8 at 2 GB.
- **"Bytes read" (physical) must name its mode.** With fault-around it is ~1.9× the fetched bytes
  and the run is 2.4× faster; without it, ~0.7× the fetched bytes (some re-fetches are served from
  cache) and much slower. Neither number alone is "the I/O".
- The comparison between arms stands either way, because both arms pay the same mechanism.

## The fix direction (a design change, not a patch)

Fault-around is doing two jobs badly: it suppresses latency *and* over-reads. A pager that owns its
bytes can do both correctly:

- the pager keeps a **region arena** inside the budget, and a fetch is one `pread` of exactly the
  region into a free slot (large sequential read, zero overshoot, no page faults in the compute);
- eviction recycles a slot instead of `madvise`+`fadvise`, so the budget is the arena — a real cap
  rather than a belief the kernel can overrule;
- the block reads through the arena pointer, which means the pointer must be re-resolved per touch
  (`pager_fetch(id)`) instead of baked into `m->exp[]` at bind time.

That is the architecture the "pager as product" claim actually needs, it would make absolute bytes
and the budget both trustworthy, and it is a bigger change than this branch's remaining scope. It is
recorded here so the next step is a decision rather than a rediscovery: **measure pinning first**
(core pinning is cheaper and independent), then decide whether the arena is worth it.
