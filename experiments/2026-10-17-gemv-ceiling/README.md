# What limits single-token decode on the T480?

**Question.** The engine runs a 4.3 GFLOP/token model at ~0.3 tok/s. Is that a FLOP wall, a
memory wall, or an implementation artifact — and what actually lifts it?

**Answer, measured:** single-token decode is a **GEMV**, so each weight byte is used once. In
fp32 that caps it at ~6–7 GFLOP/s on this machine no matter how good the SIMD is, because
0.5 FLOP/byte × ~13 GB/s DRAM ≈ 6.5 GFLOP/s. **The lever is bytes, so quantization is the fix**,
not vectorisation on its own.

## Reproduce

```
cc -O3 -march=native -o gemv gemv.c -lm -lpthread && ./gemv
```

Variants: `naive` (one accumulator), `ilp4` (four partial sums), `q4` (4-bit packed, scalar
dequant). Work is accounted correctly: each rep streams the matrix once *total* across threads.

## Results (i5-8250U, DDR4-2400 dual channel, 4 cores + HT)

| variant | 1 thread | 8 threads | 8t GB/s |
|---|---:|---:|---:|
| naive fp32, 768×3072 | 1.36 GFLOP/s | 6.63 | 13.3 |
| ilp4 fp32, 768×3072 | **2.23** | 6.31 | 12.6 |
| q4 4-bit, 768×3072 | 2.63 | **9.56** | 3.0 |
| naive fp32, 4096×4096 (out of L3) | — | 5.12 | 10.3 |
| q4 4-bit, 4096×4096 | — | 8.83 | 2.8 |

## Three findings

1. **fp32 is bandwidth-capped, not FLOP-capped.** Eight threads reach only 10–13 GB/s, the
   DDR4 wall. At 0.5 FLOP/byte that is ~6.5 GFLOP/s, full stop.
2. **The 1.4 GFLOP/s single-thread number was a latency artifact.** The naive loop has a serial
   dependency on one accumulator; four partial sums give **1.6× for free**.
3. **4-bit moves the ceiling ~8×** (fewer bytes per FLOP) and is already at 9.56 GFLOP/s with
   only a *scalar* dequant loop running at 3 GB/s — i.e. compute-bound, so AVX2 dequant + FMA
   has real headroom before bandwidth bites again.

## Projection for a 4.3 GFLOP/token model

| engine | GFLOP/s | tok/s |
|---|---:|---:|
| scalar fp32, 1 thread (today) | 1.4 | 0.31 |
| threaded fp32 (8t) | ~6 | 1.4 |
| threaded + 4-bit scalar dequant | ~9.5 | 2.2 |
| threaded + 4-bit + AVX2 FMA (est.) | ~30–40 | 7–9 |

The last row is an **estimate**, not a measurement; the byte budget allows it but it needs its
own benchmark. The first three are measured.

**Consequence for T6:** a port that lands at 0.3 tok/s would be measuring the scalar fp32
kernel, not the hardware. Quantization is a prerequisite for the port to be meaningful, and
this is why it comes before it.
