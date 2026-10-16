# Phase 8 T3 — host calibration: measured hierarchy profile, and a correction

**Date:** 2026-10-16 · **Branch:** `feat/universal-ports-tier-b`
**Status: T3 deliverable met.** `build/calibrate --file <big> --out bench/profiles/<machine>.json`
emits a machine profile. Profile: [`bench/profiles/t480-2026-10-16.json`](../../../bench/profiles/t480-2026-10-16.json).

## Result

T480 · i5-8250U · SanDisk SD8SB8U SATA SSD · btrfs · 16.5 GB RAM · page 4096 B ·
device logical block 512 B · 8 CPUs. Measured over a **30 GB file (larger than RAM)**;
`fadvise` = buffered `pread` with `POSIX_FADV_DONTNEED` before each read, `odirect` =
`pread` through `O_DIRECT`:

| read size | fadvise p50 µs | fadvise p95 | MB/s | odirect p50 µs | MB/s |
|---:|---:|---:|---:|---:|---:|
| 4 KB | 334.3 | 441.2 | 14.0 | 387.7 | 12.1 |
| 16 KB | 209.7 | 489.6 | 65.4 | 196.2 | 92.9 |
| 64 KB | 246.5 | 290.5 | 296.0 | 231.5 | 340.4 |
| 256 KB | 347.9 | 401.8 | 819.9 | 341.9 | 836.1 |
| 1 MB | 749.5 | 1031.7 | 1349.9 | 856.2 | 1311.3 |
| 4 MB | 4412.8 | 8140.8 | 898.3 | 4064.7 | 1122.2 |

The two access methods agree closely, which is the point of measuring both: neither is
being fooled by the page cache. Sequential (1 MB, O_DIRECT) reads at 2443.6 MB/s — that
number is **not** a disk figure (it exceeds the SATA link and includes cached regions of
the tail of the file); treat it as an upper bound, and prefer the random-read rows.

Compute, in the engine's own kernel shape (`out[o] = Σ x[i]·W[o][i]` at 192×768):
**1.76 GFLOP/s single-threaded, 11.67 GFLOP/s on 8 threads.**

## The correction: my own previous number was the wrong one

The vault recorded 4K random reads at **p50 ≈ 227 µs**, flagged as needing a `drop_caches`
run. Earlier today I measured **~1705 µs** and wrote that into the README as a 130× read-size
gradient. **The 1705 µs figure was an artifact.** It came from a 0.2 GB file with
`POSIX_FADV_DONTNEED` per read — and 0.2 GB cannot defeat a 16 GB page cache on btrfs, so
what varied was cache behaviour, not the device:

| method | 4 KB p50 | verdict |
|---|---:|---|
| 0.2 GB file + fadvise (wrong) | 1705 µs | cache-contended artifact |
| 30 GB file, fadvise | 334 µs | honest |
| 30 GB file, O_DIRECT | 388 µs | honest |

The vault's 227 µs was substantially right and my replacement was wrong. Corrected in
`README.md`, and the latency table used by `analysis/` now carries the measured values.
The read-size gradient is real but **64×, not 130×** (14.0 → 1349.9 MB/s at 1 MB).

Also checked, since it was a T4 concern: `pager.c`'s `#define PAGE 4096` is correct *here*
(page size 4096, logical block 512). It is still a portability problem — a 10 KB expert
region on a 16 KB-page host would over-evict up to 60% — which is exactly what deriving
`PAGE` from the profile fixes.

## Sanity check: the profile predicts the engine's throughput

The compute figure is independent of the pager, so it can be checked against the engine's
all-resident measurement. `l05` at 736 MB, `CP_PREFETCH_W=0`: the replay reports ~0.036
reads/token, so the run is compute-bound, and the engine measured **11.11 tok/s**.

| estimate | tok/s |
|---|---:|
| profile compute, 1 thread (1.76 GFLOP/s ÷ 0.179 GFLOP/token) | **9.82** |
| profile compute, 8 threads | 65.15 |
| engine measured | **11.11** |

Single-threaded prediction is within 12% of the engine. Two conclusions:

- The profile is coherent enough to predict throughput, which is what T2/T3 were for.
- **The engine is single-threaded and leaves ~6.6× on the table.** The loop is serial
  (`tokens → layers → experts`), so it uses ~1 core; 8-thread compute is 11.67 GFLOP/s.
  This is now the largest measured headroom in the engine, and it is a *compute* lever,
  not a paging one — relevant because at 736 MB paging is already ~0.

## Method notes worth keeping

- **Never measure read latency on a file smaller than RAM.** The probe refuses to run in
  that case rather than emitting numbers. `sync` before measuring: dirty writeback from
  creating the file inflated the very first cold read to ~197 ms.
- **Measure both access methods.** They should agree; when they do not, the page cache is
  in the measurement.
- **`O_DIRECT` requires length and offset aligned to the device logical block size** (512 B
  here), which is why 4 KB is the smallest measured read.

## Reproducing

```sh
make calibrate
./build/calibrate --file /path/to/big.bin --bytes-mb 30720 \
    --out bench/profiles/<machine>.json --note "<hardware, fs, file size vs RAM>"
```
