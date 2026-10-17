# Q8_0 on gpt2-moe: what it costs, what it buys, and what it does not

Answers asked in order. Each was measured; two of the three answers contradicted the plan.

## 1. Is Q8_0 lossless? No.

Big matrices quantized, 2 windows × 512 tokens: routing **99.1846%** same as fp32, logits
`max|Δ| 1.69`, argmax 1013/1024. With experts quantized too (128 tokens): routing **99.4922%**,
logits 0.807, argmax 127/128.

Quantization is **Tier-C behaviour**: a quantized file is a different model, so it is gated on
routing/argmax agreement and never on equality.

## 2. Is the router the culprit? No.

| configuration (512 tokens) | routing vs fp32 | logits | argmax |
|---|---:|---:|---:|
| everything quantized | 99.2188% | 1.693 | 507/512 |
| router kept in fp32 | 99.4336% | 1.788 | 507/512 |

Keeping `Wr` fp32 recovers 22 of 10240 selections, changes **no argmax**, and the logit delta
rises. The cost is the accumulated effect of 8-bit weights through 12 layers in a small,
lightly-trained model — not any single matrix.

## 3. Does Q8 buy speed? Only out of cache.

`matmul_xw` vs `matmul_xw_q8`, q8/fp32 time ratio:

| shape | scalar 1t | AVX2 1t | AVX2 4t |
|---|---:|---:|---:|
| 768×192 | 0.62× | 0.60× | 1.22× |
| 192×768 | 0.71× | 0.71× | 0.81× |
| 768×768 | 0.69× | 0.72× | 0.75× |
| **768×3072** | 1.13× | 0.90× | **2.55×** |

Q8 reads 3.76× fewer bytes and still loses wherever the matrix fits in cache, because **fp32 is
already auto-vectorised by the compiler** and the decode adds work per element. Only where fp32
is genuinely DRAM-bound does the byte saving dominate. Every expert matrix of a real 4-bit MoE
is that case; gpt2-moe's are not.

## 4. End to end: a quantized file the engine loads

`convert.py --quant q8_0` writes a v4 file with Q8 attention/dense/shared/router *and* Q8
expert weights (W1|W2 per expert, as one tensor) while biases move to a per-slot f32 tensor.

| | fp32 v3 | Q8 everything |
|---|---:|---:|
| file | 735.8 MB | **312.4 MB** (2.36×) |
| validator | — | `f32 tensors EXACT; 329 quantized within bound (worst 0.39%, limit 1.57%)` |
| routing vs fp32 | — | **99.4922%** |
| argmax (128 tokens) | — | **127/128** |
| speed (4 threads) | 26.9 tok/s | 26.9 tok/s |
| **peak RSS** | 457 MB | **763 MB** ← worse |

## The honest conclusion: the file shrank, nothing else improved

The expert pool is no longer materialized (that was 379 MB), but the **non-expert tensors are
still copied into a fp32 layout** (340 MB) while the 312 MB file stays mapped, so resident
memory went *up*. Speed is unchanged because gpt2-moe's matrices are cache-resident, where Q8
loses (§3), and because the copy path is still taken.

**The fix is specific and small:** allow *partial* in-place loading — f32 tensors point into
the file, Q8 expert tensors are read on the fly — instead of the current all-or-nothing rule
that any quantized tensor disables. Then RSS ≈ the file's touched pages, and the expert region
stops being copied at all.

This is why the measurement matters: "quantize and the bytes buy speed" was the plan, and on
this model it is false in three separate ways. It remains true that **every one of these limits
is a small-model limit** — on a 4-bit MoE the expert matrices are DRAM-bound (§3), the expert
pool is ~95% of the bytes rather than 51%, and the file is far larger than RAM so the paging
behaviour is what matters, not the copy.
