<sub>[**Overview**](../README.md) · [**Progress & results**](PROGRESS.md) · [**Experiments**](EXPERIMENTS.md) · [**Roadmap**](ROADMAP.md)</sub>

---

## Phase 8 — the engine meets someone else's model

Phase 8 asks whether the engine can be *model-agnostic*: can it run models it was not written for,
under the same exactness discipline? Three ports now run.

### What a port is

[`models/CONTRACT.md`](../models/CONTRACT.md) defines it. Each port ships a `MODEL.md` whose
architecture facts are marked **VERIFIED** (from a published config or, better, the actual tensor
shapes) or **ASSUMED**, a `convert.py`, and an **independent** `reference.py` written without
importing the framework that produced the model. Independence is the point: on 2026-09-25 two
implementations agreed with each other, disagreed with the frozen spec, and every gate passed. A
reference that shares code with the thing it checks cannot catch that.

The contract also asks for a `verify/gate.py`. Only `gpt2-moe` has one; the two Qwen ports keep
their checks in `verify/` and `tools/`, and `tools/run_gates.sh` runs them.

### The three ports

The tier is the one defined in the [Roadmap](ROADMAP.md): **A** reproduces the base model exactly,
**B** changes only the file layout and is still exact, **C** changes the numbers (for example by
quantising) and is judged on quality instead.

| port | tier | size | what it shows |
|---|---|---|---|
| [`gpt2-moe`](../models/gpt2-moe/MODEL.md) | A, MoE | 736 MB | the contract works end to end on a MoE with recorded routing |
| [`qwen2.5-0.5b`](../models/qwen2.5-0.5b/MODEL.md) | A at f32, C at Q8_0 | 1,976 MB f32 / 525 MB Q8_0 | the engine can run a foreign architecture and hold a conversation |
| [`qwen3-30b-a3b`](../models/qwen3-30b-a3b/MODEL.md) | C (Q8_0 and Q4_0) | 32.5 GB Q8_0 / 17.5 GB 4-bit | a model twice the size of RAM runs, paged, with output bit-identical to the unpaged run |

**gpt2-moe** is the project's own model, ported through the contract. Its independent numpy
reference reproduces the routing recorded from the trained model in **20,480 of 20,480 selections**,
and the C engine matches that reference with **0 of 1,280** mismatches.

**Qwen2.5-0.5B-Instruct** needed four blocks the engine did not have: **RMSNorm** (no mean, no
bias), **RoPE**, **grouped-query attention** (14 query heads sharing 2 KV heads) and a **gated FFN**
(SwiGLU, three matrices). It also needed the Qwen2 tokenizer, which differs from GPT-2's: a
Unicode-aware pre-tokenizer, then BPE, then special tokens.

**Qwen3-30B-A3B** (48 layers, 128 experts, 8 active per token) needs the same kinds of block plus
QK-norm and top-8 routing, implemented in its own block file on the shared kernels. It does not fit in the T480's 16 GB, so it is the first model that has to be
paged. See [The 30B](#the-30b-a-model-larger-than-ram) below.

### Verification, in the order it was built (Qwen2.5-0.5B)

Each step is a gate in [`tools/run_gates.sh`](../tools/run_gates.sh), which runs 21 gates in all.

| step | result |
|---|---|
| HuggingFace baseline recorded as a fixed target | bit-identical across two recordings |
| per-layer HF intermediates recorded (175 arrays) | 0 missing sites |
| independent numpy reference vs HF logits | **rel 1.3e-06** |
| C primitives vs that reference | RMSNorm 2.4e-07, RoPE 6.8e-08, SiLU 4.0e-08 |
| C forward pass of the whole model vs HF logits | **rel 1.9e-05, argmax 5/5** |
| the `.v4cp` file contains the published weights | 14/14 tensors byte-identical |
| the engine runs **from the file**, not from a test blob | identical to the blob path |
| Qwen2 tokenizer ids vs HF | **16/16 exact** |
| `q8_decode` / `q4_decode` (C vs Python) | bit-identical |

Recording the per-layer intermediates *before* writing any C was the most useful single decision.
With only the final logits, a wrong RoPE looks the same as a wrong attention block twenty-four
layers later. With intermediates, every failure names its block.

### What it measured, and what changed

**The limit is bytes, not arithmetic.** Generating one token is a matrix-vector product: every
weight is read once and used once. Qwen2.5-0.5B at f32 reads **2.52 GB per token** (the 1,976 MB
file plus the tied embedding read a second time as the output head) and ran at **2.90 tok/s, i.e.
7.3 GB/s**. That is **56%** of the ~13 GB/s this machine reached in a dedicated GEMV benchmark. So
the lever is fewer bytes and a larger share of that ceiling, not more instructions. Full
arithmetic: [`analysis/t6b-bandwidth-budget.md`](../analysis/t6b-bandwidth-budget.md).

That changed the plan. The intended first step was AVX2 on the f32 kernels; what actually paid was
**quantisation and two loop bugs**:

| change | effect |
|---|---|
| Q8_0 weights (8.5 bits per weight), loaded in place, decoded on the fly | 1,976 MB → 925 MB, 3.1 → 5.6 tok/s |
| Q8_0 on the embedding / tied head as well | 925 → 525 MB, 5.6 → 9.1 tok/s |
| one horizontal sum per row instead of one per 32-element group | 9.1 → 13.5 tok/s |
| keep the thread-pool workers awake across a token (spin instead of condvar) | 13.5 → 15.5 tok/s |

Both bugs were found by profiling, not guessing. The extra horizontal sums held the kernel 5.7 GB/s
below the 9.3 GB/s it reaches now. Waking sleeping workers cost **9 ms per token**: a dispatch takes
2.0 µs when the pool is awake and **63 µs** after it has slept.

### Three bugs worth naming

All three are the same kind of mistake: **a quantity whose meaning depends on an index or a unit,
with nothing forcing the two to be computed together.** All three produced a model that still ran.

1. `exp_hid` was derived from `len / 4`. For a block encoding `len` is in **bytes**, so it came out
   as 50 instead of 192 and every expert bias pointed at the wrong place. Routing fell to 40% with no
   error.
2. A multi-layer quantised group is **one tensor**, so layer `l` starts `l` blocks in. Passing the
   tensor base for every layer made every layer read layer 0. Routing fell to 10.7%, silently.
3. `dtype` in a tensor record is an **index into the file's encoding table**. Assuming `0 == f32`
   held only while f32 happened to be written first; quantising the embedding first made every f32
   tensor read as quantised.

A fourth kind of error appeared in the *checks* rather than the code: a check that cannot tell
"the model is wrong" from "I compared the wrong things". HF hooks return outputs before the residual
is added; comparing a 1-D array with a 2-D one makes numpy **broadcast** instead of raising; and each
table in the file starts with a `u32` count that two scratch parsers skipped, mis-reading every name.
Each was caught only because the gates report *where* they failed.

### Negative results, kept

- **Canonical accumulation was deferred, not adopted.** Permuting expert weights leaves routing
  exactly unchanged but moves the logits by ~5.7e-06, because the router's **softmax** sums over the
  expert axis. Making it bit-exact is cheap, but it would move the engine ~1 ulp *away* from PyTorch
  and put the 100% recorded-routing match at risk, for a property nothing yet needs.
- **Two accumulators in the Q8 dot product made no measurable difference.** Interleaved A/B over
  8 pairs: B won 4 of 8, median −0.72 ms. Not shipped.
- **The measurement itself needed a rig.** The same binary measured 15.5 tok/s and 12.7 tok/s an
  hour apart, because this laptop throttles from 3,100 MHz idle to **2,593 MHz** under a 4-thread
  load. [`tools/ab.py`](../tools/ab.py) therefore interleaves A and B, compares per-pair differences
  with a sign test, and refuses to name a winner inside the noise. It was validated both ways:
  identical binaries must produce no winner, and a known 3.7× difference must be detected.

### Can you talk to it?

```sh
make build/qwen_chat
CP_THREADS=4 ./build/qwen_chat scratch/qwen_q8e.v4cp 64
```

```
user> What is the capital of France?
assistant> Paris.
[prefill 18 tok in 1.60s (11.3 tok/s) | 2 tok in 0.20s (9.97 tok/s)]
```

It answers at ~10–15 tok/s on four threads. It is a 0.5B model: right on simple facts, confidently
wrong on detail. The engine is right and the model is small, and keeping those two claims separate
is what the verification structure is for.

### The 30B: a model larger than RAM

Qwen3-30B-A3B at Q8_0 is a 32.5 GB file; the T480 has 16 GB. Full records:
[C1](2026-10-18-c1-pager-wiring.md), [C2](2026-10-19-c2-results-and-thesis.md),
[where tok/s goes](2026-10-19-tok-s-levers.md).

**It is exact under paging.** With the pager attached, the 48-layer output is **bit-identical** to the
unpaged run, both at an unlimited budget and at a 2 GB budget that forces evictions. The per-token
counters were checked against the file's own region table (token 0: 819 regions, 3,600.2 MB), and
token-by-token decoding is bit-identical to processing the prompt in one pass. These gates run on
the Q8_0 file.

**Measured speed, Q8_0, 4 threads, cold start:**

| arm | regions/token | physical MB/token | tok/s |
|---|---:|---:|---:|
| unpaged (the OS page cache decides) | 38.3 | 401.3 | 0.709 |
| 8 GB budget, real disk reads | 102.2 | 926.2 | 0.363 |
| 11 GB budget | 39.2 | 362.0 | 0.767 |
| 13 GB budget | 38.3 | 348.5 | 0.799 |

- **Near the working set the pager is slightly faster than the OS.** At 11–13 GB it fetches about
  as many regions as the unpaged run needs, reads fewer physical bytes, and runs 8–13% faster.
- **Below the working set the OS wins**, because it can use all the free RAM while the budget is a
  hard cap.
- **Pinning the core tier is the one policy win.** `CP_PIN_CORE=1` keeps the non-expert regions
  resident and cuts the regions fetched 2.13× at a 2 GB budget. It has no effect from 8 GB up.
- **Prefetch, a pager-owned read arena and co-selection expert grouping did not help.** Prefetch was
  ruled out before it was built (86.8% of touches already hit); the other two were built, measured
  and were slower.

**Both halves of a token are bytes.** A two-term model fits every arm:
`t = 374 ms + physical MB ÷ 0.389 GB/s`. The 374 ms is ~3.5 GB streamed from RAM; the rest is the
experts each token newly selects, read from disk. At 13 GB, 71% of the token is disk. The planning
estimate in [`MODEL.md`](../models/qwen3-30b-a3b/MODEL.md) (2.0–3.6 tok/s) counted only the RAM
side, which is why the measured 0.80 is lower.

**So the remaining lever was fewer bytes per weight:**

| file | tok/s at 13 GB | quality vs Q8_0 (teacher-forced next-token agreement) |
|---|---:|---|
| Q8_0 | 0.80 | reference |
| Q4_0 experts, Q8_0 elsewhere | 1.14 | 90.8% / 98.3%, pass |
| **Q4_0 experts and attention, Q8_0 embed and `lm_head`** | **1.22** | **93.3% / 97.5%, pass** |
| Q4_0 everywhere | 1.40 | 70.8% on the factual probe, rejected on a 24-token run only |

The shipped configuration is the 1.22 tok/s file. Quality was measured on two probes of 120
generated tokens against a 90% threshold set before measuring. An earlier 24-token version of the
gate was too noisy to separate the configurations, and the whole-model Q4 rejection still rests on
it.

### What Phase 8 has not done

- **One engine, two forward paths.** `engine/blocks/qwen2.c` and `qwen3_moe.c` have their own
  loaders and forward passes beside the GPT-2 `Model`, so `cachepotato -i` speaks only GPT-2. The
  Qwen3 loader still hardcodes `rope_theta` (1e6) and top-8 and sizes its tables for 128 experts.
- **No model that is both useful and fast.** The 30B runs at ~1.2 tok/s. The coding-model ports
  proposed in [What this laptop can run](2026-10-19-what-can-run.md) have not been started.
- **Whole-model Q4 has not been re-measured** with the 120-token gate.

## Results

These are the Phase 7 results. Measured on a ThinkPad T480 (16 GB RAM, SATA SSD, btrfs), single-threaded, on AC power, against a
**dense GPT-2 small** control. Full data and the pre-registered decision rules:
[`bench/results/sweeps/2026-09-28-q3-analysis.md`](../bench/results/sweeps/2026-09-28-q3-analysis.md).

### In short: sparse + paging beats dense exactly where dense cannot fit

| RAM budget | dense fits? | MoE (λ=0.5) | MoE (λ=0) | dense GPT-2 | verdict |
|---:|:---:|---:|---:|---:|---|
| 384 MB | no | **7.19** | 5.75 | 2.88 | **+149%** for sparse, p < 0.0001 |
| 512 MB | yes | 10.71 | 9.21 | 10.80 | tie |
| 736 MB | yes | **12.91** | 11.07 | 10.69 | **+21%** for sparse, p < 0.0001 |

*Tokens/sec, greedy, 48-token prompt + 512 generated.* At 384 MB the dense model evicts **5.3× more
bytes per token** (99.30 vs 18.65 MB), and that difference in volume explains the 2.5× gap. The
claim is **conditional**: at 512 MB the 475 MB dense model fits while the 736 MB sparse model does
not, and the sparse model's compute advantage (+20.3%) is given back almost exactly to paging
(−17.6%).

### The mechanism: locality-aware routing cuts paging I/O

The router is trained with a bonus for reusing the previous token's experts at the same layer
(λ=0.5), with λ=0 as a required control on an identical data stream. Routing is deterministic, so
these byte counts reproduce to four significant figures across sessions.

| RAM budget | bytes fetched, λ=0.5 | bytes fetched, λ=0 | reduction |
|---:|---:|---:|---:|
| 384 MB | 152,767 MB | 251,319 MB | −39.2% |
| 512 MB | 23,350 MB | 40,595 MB | −42.5% |
| 576 MB | 6,638 MB | 10,640 MB | −37.6% |
| 736 MB | 679 MB | 679 MB | **0%** |

At 736 MB both arms are fully resident, so the reduction is exactly zero, as it should be. In the
512 MB mechanism test (core tensors ~99.9% resident in both arms) the expert cache hit rate is
**84.6% at λ=0.5 vs 71.4% at λ=0**, a **46% cut in expert bytes fetched**.

### Does the training make the difference?

Paired, interleaved runs at 512 MB, with arm order counterbalanced:

| comparison | effect | significance |
|---|---|---|
| λ=0.5 vs λ=0 (co-designed vs bolted-on) | **+1.50 tok/s (+16.3%)** | t=2.674, p ≈ 0.018 |
| λ=0 vs dense (untrained router, same budget) | **−17.3%** | p = 0.0009 |

At 512 MB, router training decides whether the sparse architecture is a net gain or a net loss. Two
caveats are recorded in the analysis: the machine's noise floor is 8–17%, and the effect varies
across the three document blocks sampled.

### What was established along the way

- **Dense weights have no locality to exploit.** Instrumentation showed ~2.47 GB of weight reads
  per 5-token pass. Forced cold, GPT-2 small drops from **7.79 to 0.70 tok/s** while pulling
  23.26 GB off disk. Evicting dense weights is pure loss, which is the case for sparsity.
  ([LRU residency record](../bench/results/cache/2026-09-14-lru-residency.md))
- **The paging machinery works.** RSS follows the budget (545 MB → 74 MB) while the output stays
  correct, checked against a from-scratch Python reference.
- **The export is faithful at the routing level**, not merely loadable: the C engine reproduces the
  recorded PyTorch routing **100% (0 of 20,480 mismatches)**, with logits identical to
  **0.000e+00**.
- **A negative result on expert size, specific to this model and disk.** Pairing 1.18 MB experts
  into 2.36 MB pages roughly *doubles* the bytes moved for at most an 8% cut in read count, so
  32 × 1.18 MB is the right layout here. It is not a general rule: a 1.18 MB expert is already
  large enough for efficient reads. (Co-selection grouping was later tried on the 30B, whose
  experts are 2.65–5 MB, and also lost.) The same run found that the MoE's ~134 MB/s against
  dense's ~286 MB/s is caused by **access order**, not read size.
  ([expert-grouping sweep](../bench/results/sweeps/2026-09-28-expert-grouping.txt))

## Status

**Phase 7 (Integration): done.** All three pre-registered gates pass, and the end-to-end demo
exists: a trained sparse model served by the paging engine, benchmarked against a dense baseline,
with the locality hit rate reported.

**Phase 8 (Universal ports): in progress.** Three ports run through one contract
([`models/CONTRACT.md`](../models/CONTRACT.md)) and one self-describing
[format](../format/SPEC.md) with three encodings (f32, Q8_0, Q4_0), checked by 21 gates. The paging
experiment on a model that does not fit is complete: exact, 0.80 tok/s at Q8_0 and 1.22 tok/s with
the gated 4-bit file at a 13 GB budget. That is usable for single questions, not for interactive
work. The options for what comes next are in
[What this laptop can run](2026-10-19-what-can-run.md).

| Phase | Scope | Status |
|---|---|:---:|
| 0 — Foundations | mmap/`madvise` primitives, SSD latency measurement | ✅ |
| 1 — Dense baseline | mmap-backed GPT-2 forward pass in C, verified | ✅ |
| 2 — Tokenizer | GPT-2 byte-level BPE, raw-byte CPT1 format | ✅ |
| 3 — Memory-hierarchy paging | Region tracker, LRU eviction, cache-ceiling experiments | ✅ |
| 4 — Sparse/MoE architecture | 32 experts × 1.18 MB + shared, top-2, locality bias | ✅ |
| 5 — Training loop | C trainer with gradient-checked backprop | ✅ |
| 6 — Scale training + data | PyTorch MoE trainer, FineWeb-Edu, λ=0/0.5 ablation | ✅ |
| 7 — Integration | Paging engine + trained sparse model, Q1/Q2/Q3 gates | ✅ |
| 8 — Universal ports | `engine/` split, `format/SPEC.md`, three ports, Q8_0 + Q4_0, threads, 21 gates | 🔄 in progress |
| 9+ — Scale, agentic workloads | Tool calling, domain SFT, longer-context tiers | ⏳ later |
