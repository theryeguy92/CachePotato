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
