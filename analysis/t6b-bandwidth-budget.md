# T6b: can this machine run an MoE fast enough for a paging win to be measurable?

**Answer: yes, at ~2.5 tok/s today and up to ~4.5 tok/s if the kernels reach the measured
bandwidth ceiling.** That is slow but *measurable* — 0.4 s/token, a 200-token answer in ~80 s —
which is what a paging experiment needs. The thesis is not dead on this hardware.

More importantly, the analysis redirects the work: **single-token decode is a GEMV, so the binding
resource is bytes/token, not FLOPs.** Read count is precisely what Tier B claims to improve, so the
thesis and the bottleneck are the same quantity. That is a better position than it looked.

## The measurement that grounds it

Qwen2.5-0.5B, from the v4 file, 4 threads, decode:

| quantity | value |
|---|---:|
| parameters (incl. tied head counted once each side) | 0.630 B |
| **bytes read per token** (fp32: every weight, once) | **2.52 GB** |
| measured throughput | 2.90 tok/s |
| **achieved bandwidth** | **7.3 GB/s** |
| ceiling, from `experiments/2026-10-17-gemv-ceiling` (ilp4 fp32, out of L3, multi-thread) | ~13 GB/s |
| share of ceiling actually used | **56 %** |

So the engine is already at 56 % of the best bandwidth this measurement bench could produce. The
headroom in *utilization* is ~1.8×, not 8×. Chasing FLOPs is the wrong lever; the right ones are
(a) fewer bytes and (b) a higher share of the ceiling.

## What that implies for the MoE candidates

Decode reads every **active** weight once per token. At Q8 (8.5 bpw ≈ 1.0625 B/param):

| model | active | GB/token @Q8 | tok/s @ 7.3 GB/s | @ 13 GB/s |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 2.75 B | 2.92 | **2.50** | 4.45 |
| gpt-oss-20b | 3.39 B | 3.60 | **2.03** | 3.61 |

And for reference, Qwen2.5-0.5B itself: fp32 2.52 GB/token → 2.9 tok/s measured; at Q8 it would be
0.67 GB/token → **10.9 tok/s at today's 7.3 GB/s**, 19.4 at the ceiling. Quantization is a **3.8×
lever on bytes**, which is larger than anything micro-optimising the inner loop will produce.

## What this changes about the T6b plan

The plan was "AVX2 on the fp32 paths". Measured, that is the *second* lever:

1. **Wire Q8 into the Qwen path** (the loader refuses non-f32 today; the Q8 stack, in-place loading
   and on-the-fly decode all exist and are gated for gpt2-moe). Expected ~3.8× on bytes → the single
   largest available win, and it is mostly *wiring*, not new kernels.
2. **Raise bandwidth utilization from 56 %** toward the ceiling: prefetch, avoid re-reading matrices
   the batch path touches once per token, and check that `matmul_wx`'s row-per-output access is
   actually streaming (it is the per-token hot loop and it has never been profiled).
3. **Only then** micro-optimise FLOPs, which this analysis says are not binding.

## The numbers to beat, recorded before the work

- Qwen2.5-0.5B, fp32, 4 threads: **2.90 tok/s** (7.3 GB/s, 56 % of ceiling).
- Target after Q8 wiring: **~10 tok/s** if bandwidth-bound behaviour holds.
- A 3B-active MoE at Q8 must clear **~2 tok/s** to make a paging experiment readable; the arithmetic
  says it can, at 2.5–4.5 tok/s.

## Caveats, stated

- The 13 GB/s ceiling is from a synthetic GEMV on 4096×4096, not from this model's shapes. Real
  ceilings for 896-wide matvecs may be lower, which would lower every projection here.
- Expert weights in a MoE come from **SSD** unless cached, at ~300 MB/s. The bandwidth arithmetic
  above assumes they come from RAM, i.e. it measures the *post-paging* ceiling — which is exactly the
  number the paging experiment needs as its upper bound. Paging performance decides how often the
  model achieves it, not how fast it goes when it does.
- Two arithmetic errors were made and caught writing this note (dividing instead of multiplying
  bytes/token by tok/s, twice). The numbers above were recomputed from the definition.
