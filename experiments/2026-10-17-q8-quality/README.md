# Does Q8_0 hurt gpt2-moe? (it is not free)

**Question.** Q8_0 is 8.5 bpw vs 32 for fp32 — **3.8× fewer bytes**, which by the GEMV
arithmetic lifts the decode ceiling from ~6.5 to ~24 GFLOP/s. Before wiring it into the
engine: does it change the model?

**Answer: measurably, and more than "8-bit is lossless" folklore suggests.** Q8_0 on the big
weight matrices (attention, dense FFN, shared expert, router, routed experts; embeddings and
norms stay fp32), over 2 windows × 512 tokens:

| measure | result |
|---|---|
| routing vs fp32 | **20313/20480 same (99.1846%)** — 167 selections change |
| routing vs the recorded PyTorch trace | 20313/20480 (same rate, as expected) |
| logits max\|Δ\| | **1.69** (max \|logit\| ≈ 20, so ~8% of full scale) |
| per-position argmax | 1013/1024 (98.9%) |

## Why this matters more than it looks

The project's Tier-A/Tier-B claim is *exact routing, fp32-equal logits*. Quantization cannot
preserve that: it is a **lossy** encoding, so a quantized model is a different model, the way an
int4 checkpoint differs from its fp16 parent. That is expected — but it means quantization is
**Tier-C behaviour**, not a storage detail, and must be gated as an approximation with a stated
budget rather than as an equality.

The 1.69 logit delta is too large to be explained by 8-bit rounding alone (0.37% max per-weight
error). Two candidates, not yet separated:

1. **The router.** `Wr` is quantized, and a 0.37% perturbation of router logits directly moves
   near-tie selections. `Wr` is only 0.25M of ~185M params, so keeping it fp32 costs ~0.13% of
   the bytes.
2. **Error accumulation** across 12 layers in a small, lightly-trained research model (1M
   training tokens), plausibly far more sensitive than a mature model — llama.cpp's Q8_0 is
   widely reported as near-lossless on well-trained models.

Separating them needs a per-matrix sweep, which the numpy oracle is too slow for (minutes per
forward). The engine, once wired, does it in seconds.

## What this changes

- Q8_0 must be gated on **quality** (routing agreement, argmax agreement, loss on held-out
  data), not only decode exactness. Decode exactness is necessary and already proven:
  `tools/verify_q8_decode.py` shows C == Python bit for bit on 5 cases including subnormals and
  a 1000× dynamic range.
- Keeping the **router in fp32** is the obvious first mitigation and costs ~0.13% of the bytes.
- "Near-lossless" should be **measured on this model**, not imported from another project's
  experience with a different one.

## Reproduce

`models/gpt2-moe/reference.py` applied to weights round-tripped through `format/q8.py`, compared
against fp32 on `data/traces/l05_fp32.npz` windows. No engine changes involved.

---

# And what the Q8 kernel actually buys: only out of cache

`matmul_xw` (fp32) vs `matmul_xw_q8` on the shapes gpt2-moe actually uses. Q8 reads **3.76×
fewer bytes**, so if the kernel were bandwidth-bound it should win nearly 4×. It does not.

| shape | fp32, 1t | q8, 1t | q8/fp32 | fp32, 4t | q8, 4t | q8/fp32 |
|---|---:|---:|---:|---:|---:|---:|
| expert W1 768×192 | 28.8 µs | 46.5 | **0.62×** | 62.2 | 65.6 | 0.95× |
| expert W2 192×768 | 30.9 | 43.9 | **0.71×** | 24.2 | 36.4 | 0.67× |
| attention 768×768 | 123.8 | 178.4 | **0.69×** | 102.8 | 132.9 | 0.77× |
| dense W1 768×3072 | 815.8 | 719.7 | 1.13× | 611.9 | 347.7 | **1.76×** |

**Q8 is a net loss on every matrix that fits in cache, and a win only on the largest one that
does not.** The reason is that the scalar dequant costs ~3 operations per element
(`int8→float`, `× scale`, `FMA`) against 1 for fp32, so at small sizes the kernel is
compute-bound on the decode and the 3.76× byte saving buys nothing. Only at 768×3072 does the
fp32 kernel hit real DRAM bandwidth (12 GB/s) and the byte saving dominate.

## What this changes about the plan

The received idea was "quantize, and bytes buy speed". **False as implemented.** The ordering
that follows from the measurement:

1. **A vectorised dequant is a prerequisite**, not an optimisation. With AVX2 the decode is
   ~4–8 instructions per 32 elements (`vpmovsxbd` → `vcvtdq2ps` → `vfmadd`), which is the only
   way the byte saving reaches the FMA loop. Scalar dequant cannot be expected to beat fp32.
2. **Quantization only pays for weights larger than cache.** For gpt2-moe that is almost
   nothing; for a 4-bit open-weight MoE it is everything, because every expert matrix is
   DRAM-bound. So Q8/4-bit is a *large-model* win, and measuring it on gpt2-moe understates it.
3. Byte savings still pay **off-disk** immediately and unconditionally: a 3.76× smaller file is
   3.76× less SSD traffic, which the pager reads whether or not the kernel is fast.

Separately: the first version of these Q8 kernels was **not threaded at all**, which made them
3× slower than fp32 at 4 threads. Threading is now split by group (a group's 32 outputs share
one scale, so it is the smallest unit that can move between threads without re-reading it).
That is why the 4-thread column above is not catastrophic — but it is still not a win.
