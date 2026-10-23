<sub>[**Overview**](../README.md) · [**Progress & results**](PROGRESS.md) · [**Experiments**](EXPERIMENTS.md) · [**Roadmap**](ROADMAP.md)</sub>

---

## Phase 8 — the engine meets someone else's model

Phase 8 asked whether the engine can be *model-agnostic*: can the same binary run a model it was
not written for, with the same exactness discipline? Two ports now run, and the answer is yes —
with the caveats stated below.

### What a port is

[`models/CONTRACT.md`](../models/CONTRACT.md). Every port ships a `MODEL.md` whose architecture facts
are each marked **VERIFIED** (from a published config or, better, from the actual tensor shapes) or
**ASSUMED**; a `convert.py`; an **independent** `reference.py` written without importing the
framework that produced the model; and a `verify/gate.py`. The independence is the point: on
2026-09-25 two implementations agreed with each other, disagreed with the frozen spec, and every
gate passed. A reference that shares code with the thing it checks cannot catch that.

### The three ports

| port | tier | size | what it proves |
|---|---|---|---|
| [`gpt2-moe`](../models/gpt2-moe/MODEL.md) | A (as-is), MoE | 736 MB | the contract works end to end on a MoE with recorded routing |
| [`qwen2.5-0.5b`](../models/qwen2.5-0.5b/MODEL.md) | A (as-is), dense | 525 MB (Q8) | the engine can meet a foreign architecture and **talk** |
| `qwen3-30b-a3b` | B (paged), MoE | 32.5 GB Q8 / 17.5 GB Q4 | the paging thesis on a model **4x larger than RAM**: bit-identical under a budget, 0.80 tok/s at Q8, **1.22 tok/s** at Q4 |

**gpt2-moe** is the project's own model, ported through the contract. Its independent numpy
reference reproduces the routing recorded from the trained model **20480/20480 selections
(100.0000%)**, and the C engine reproduces that reference with **0/1280** mismatches.

**Qwen2.5-0.5B-Instruct** required four blocks the engine did not have — **RMSNorm** (no mean, no
bias), **RoPE**, **grouped-query attention** (14 query heads against 2 KV heads) and a **gated
FFN** (SwiGLU, three matrices). It also required a Qwen2 tokenizer, which is not GPT-2's: it is a
Unicode-aware pre-tokenizer plus BPE plus special tokens.

### Verification & the order it was built

Every step is a gate in [`tools/run_gates.sh`](../tools/run_gates.sh) (20 gates, all passing).

| step | result |
|---|---|
| HuggingFace baseline recorded as a fixed target | bit-identical across two recordings |
| per-layer HF intermediates recorded (175 arrays) | 0 missing sites |
| independent numpy reference vs HF logits | **rel 1.3e-06** |
| C primitives vs that reference | RMSNorm 2.4e-07, RoPE 6.8e-08, SiLU 4.0e-08 |
| C forward of the whole model vs HF logits | **rel 1.9e-05, argmax 5/5** |
| the `.v4cp` file contains the published weights | 14/14 tensors byte-identical |
| the engine runs **from the file**, not a test blob | identical to the blob path |
| Qwen2 tokenizer ids vs HF | **16/16 exact** |
| `q8_decode` / `q4_decode` (C vs Python) | bit-identical |

Recording the per-layer intermediates *before* writing any C was the single most useful decision:
with only final logits, a wrong RoPE is indistinguishable from a wrong attention block twenty-four
layers later. With intermediates, every failure names its block.

### What it measured, and what changed

**The binding resource is bytes, not FLOPs.** Single-token decode is a GEMV: every weight is read
once and used once. Qwen2.5-0.5B at fp32 reads **2.52 GB per token** and ran at **2.90 tok/s =
7.3 GB/s**, which is **56%** of the ~13 GB/s this machine could produce in a dedicated GEMV
benchmark. So the lever is fewer bytes and a higher share of the ceiling — not more instructions.
Full arithmetic: [`analysis/t6b-bandwidth-budget.md`](../analysis/t6b-bandwidth-budget.md).

That redirected the plan. The intended first step was AVX2 on the fp32 kernels; the measured step
was **quantization and two loop bugs**:

| change | effect |
|---|---|
| Q8_0 weights (8.5 bpw), in-place loading, on-the-fly decode | 1976 MB → 925 MB, 3.1 → 5.6 tok/s |
| Q8_0 on the embedding / tied head as well | 925 → 525 MB, 5.6 → 9.1 tok/s |
| one horizontal sum per row instead of per 32-element group | 9.1 → 13.5 tok/s |
| keep pool workers awake across a token (spin vs condvar) | 13.5 → 15.5 tok/s |

The last two were found by profiling, not by guessing. The horizontal-sum bug was costing
5.7 GB/s of the 9.3 GB/s the kernel now reaches, and the worker wake-up was costing **9 ms per
token** (dispatch costs 2.0 µs when the pool is hot but **63 µs** after it has slept).

### Three bugs worth naming

All three are the same species: **a quantity whose meaning depends on an index or a unit, with
nothing forcing the two to be computed together.** All three produced a model that ran.

1. `exp_hid` derived from `len / 4` — for a block encoding, `len` is **bytes**, so it computed 50
   instead of 192 and mis-pointed every expert bias. Routing fell to 40% with no error.
2. A multi-layer quantized group is **one tensor**, so layer `l` starts `l` blocks in — passing the
   tensor base for every layer made every layer read layer 0. Routing fell to 10.7%, silently.
3. `dtype` in a tensor record is an **index into the file's encoding table**. Assuming `0 == f32`
   held only while f32 happened to be written first; quantizing the embedding first made every f32
   tensor read as quantized.

A fourth class appeared in the *measurement* rather than the code: a check that cannot distinguish
"the model is wrong" from "I compared the wrong things". HF hooks return pre-residual outputs; a
1-D array compared against a 2-D one makes numpy **broadcast** instead of raising; each table is
preceded by a `u32` count that two scratch parsers skipped, mis-reading every name. Each was
caught only because the gates report *where* they failed.

### Negative results, kept

- **Canonical accumulation was deferred, not adopted.** Permuting expert weights leaves routing
  exactly invariant but moves logits ~5.7e-06, because the router's **softmax normalisation** sums
  over the expert axis. Making it bit-exact is cheap; it would move the engine ~1 ulp *away* from
  PyTorch, risking the 100% recorded-routing match for a property nothing yet needs.
- **Two accumulators in the Q8 dot product is noise.** Interleaved A/B, 8 pairs: 4/8, median
  −0.72 ms. Not a result, not shipped. The loop is neither latency- nor instruction-bound.
- **The measurement itself needed a rig.** The same binary measured 15.5 tok/s and 12.7 tok/s an
  hour apart: this laptop throttles from 3100 MHz idle to **2593 MHz** under 4-thread load. So
  [`tools/ab.py`](../tools/ab.py) interleaves A and B, compares per-pair deltas with a sign test, and
  refuses to name a winner inside the noise. It was validated in both directions — identical
  binaries must produce no winner, and a known 3.7× difference must be detected.

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

It answers, at ~10–15 tok/s on four threads. It is a 0.5B model: correct on simple facts and
confidently wrong on detail. The engine is right; the model is small — and keeping those two
claims separable is what the whole verification structure is for.

### What Phase 8 has not done

- **The paging thesis HAS now been exercised on a model that does not fit — and the answer is a
  tie.** Qwen3-30B-A3B (32.5 GB at Q8, 48 layers, 128 experts, top-8) runs on the T480 from a CPW1
  v4 file with the output **bit-identical** to the unpaged run at every budget, and per-token
  counters checked against the file's own region table. On four threads it runs at **1.22 tok/s with 4-bit experts and attention at a 13 GB budget** (0.80 at Q8), 0.77 at 11 GB Q8, and **0.36 at 8 GB with
  real disk reads** — and the first
  measurements said 0.147 only because the harness was silently running one thread. At 13 GB it also
  *beats* the unpaged control (0.71), so the region-LRU is better than the OS's paging at equal
  memory —
  and at equal memory the pager only *ties* the OS page cache (54.8 vs 54.7 regions/token at 11 GB).
  Core pinning is the one policy win (2.13x fewer reads at 2 GB). See
  [`docs/2026-10-19-c2-results-and-thesis.md`](2026-10-19-c2-results-and-thesis.md).
- **That measurement moved the target, not the claim.** 4.3 s of the 6.8 s token is compute, not
  I/O, and it scales with *active* parameters — so a model larger than RAM is provably runnable but
  not usable here. The usable target on this hardware is a model that *fits* with few active
  parameters, which is a different port than the one this section once proposed.
- **The engine runs two forward paths.** `engine/blocks/qwen2.c` has its own loader and forward
  beside the GPT-2 `Model`; merging them is a refactor with no numerical payoff yet, and until it
  happens `cachepotato -i` speaks only GPT-2.

The T6b estimate above predicted **2–4 tok/s** for a 3B-active MoE. Measured: **0.80 tok/s** at a
13 GB budget, 0.36 at 8 GB. A two-parameter fit (`t = 374 ms compute + bytes/0.389 GB/s`) says
**71% of the token is disk even at 13 GB**, so the gap is bandwidth and compulsory reads, not
compute. The
gap is not bandwidth — 3.6 GB/token at the disk's 0.39 GB/s would be 9 s, and the run is 6.8 s. It
is **compute**, dominated by scalar Q8 dequant, at ~1.28 ms per 1B active elements. That is the
number to attack if this engine is to become usable, and it is the same number for every model.
## Results

Measured on a ThinkPad T480 (16 GB RAM, SATA SSD, btrfs), single core, AC power, against
a **dense GPT-2** small control. Full data and pre-registered decision rules:
[`bench/results/sweeps/2026-09-28-q3-analysis.md`](../bench/results/sweeps/2026-09-28-q3-analysis.md).

### The TLDR: sparse + paging beats dense exactly where dense cannot fit

| RAM budget | dense fits? | MoE (λ=0.5) | MoE (λ=0) | dense GPT-2 | verdict |
|---:|:---:|---:|---:|---:|---|
| 384 MB | no | **7.19** | 5.75 | 2.88 | **+149%** for sparse, p < 0.0001 |
| 512 MB | yes | 10.71 | 9.21 | 10.80 | tie |
| 736 MB | yes | **12.91** | 11.07 | 10.69 | **+21%** for sparse, p < 0.0001 |

*Tokens/sec, greedy, 48-token prompt + 512 generated.* At 384 MB the dense model evicts
**5.3× more bytes per token** (99.30 vs 18.65 MB) — that volume difference is the causal
chain behind the 2.5×. The claim is **conditional, not blanket**: at 512 MB a 475 MB dense
model fits while the 736 MB sparse model does not, and the sparse model's compute advantage
(+20.3%) is given back almost exactly to paging (−17.6%).

### The mechanism: locality-aware routing cuts paging I/O

The router is trained with a bonus for reusing the previous token's experts at the same
layer (λ=0.5), with λ=0 as a mandatory ablation on an identical data stream. Routing is
deterministic, so these byte counts reproduce to four significant figures across sessions.

| RAM budget | bytes fetched, λ=0.5 | bytes fetched, λ=0 | reduction |
|---:|---:|---:|---:|
| 384 MB | 152,767 MB | 251,319 MB | −39.2% |
| 512 MB | 23,350 MB | 40,595 MB | −42.5% |
| 576 MB | 6,638 MB | 10,640 MB | −37.6% |
| 736 MB | 679 MB | 679 MB | **0%** |

At 736 MB both arms are fully resident, the reduction going to exactly zero is the
mechanism behaving as designed. At the 512 MB mechanism test (core tensors ~99.9% resident
in both arms) expert cache hit rate is **84.6% at λ=0.5 vs 71.4% at λ=0**, a **46% cut in
expert bytes fetched**.

### Does the training make the difference?

Paired, interleaved, arm-order-counterbalanced runs at 512 MB:

| comparison | effect | significance |
|---|---|---|
| λ=0.5 vs λ=0 (co-design vs bolted-on) | **+1.50 tok/s (+16.3%)** | t=2.674, p ≈ 0.018 |
| λ=0 vs dense (unbiased router at same budget) | **−17.3%** | p = 0.0009 |

At 512 MB, router training is the difference between the sparse architecture being a net
positive and a net negative. Both caveats are recorded in the analysis: the machine's noise
floor is 8–17%, and the effect is content-dependent across the three document blocks sampled.

### What was proven along the way

- **Dense weights have no exploitable locality.** Instrumentation showed ~2.47 GB of weight
  reads per 5-token pass; forced-cold, GPT-2 small drops from **7.79 → 0.70 tok/s** while
  pulling 23.26 GB off disk. Evicting dense weights is pure loss, the justification
  for sparsity. ([`experiment-2.md`](../bench/results/cache/2026-09-14-lru-residency.md))
- **The paging machinery works.** RSS tracks the budget exactly (545 MB → 74 MB) while
  correct output is preserved, verified against a from-scratch Python reference.
- **The export is faithful at the routing level**, not merely loadable: the C engine
  reproduces the recorded PyTorch routing **100% (0/20,480 mismatches)** with logits
  identical to **0.000e+00**.
- **A negative result on expert geometry — and it is specific to this geometry.** Pairing
  1.18 MB experts into 2.36 MB pages roughly *doubles* bytes moved for at most an 8%
  read-count reduction, so the 32 × 1.18 MB layout is correct **for this model on this
  disk**. It is not a general rule: a 1.18 MB expert already clears the sequential-read
  floor, so the conclusion does not transfer to a quantized MoE whose experts are ~10 KB.
  The same run found the MoE's ~134 MB/s against dense's ~286 MB/s is an **access-order**
  problem, not a read-size one.
  ([`expert-grouping-2026-09-28.txt`](../bench/results/sweeps/2026-09-28-expert-grouping.txt))

## Status

**Phase 7 (Integration) — done.** All three pre-registered gates pass, and the end-to-end
demo exists: a trained sparse model served by the paging engine, benchmarked against a dense
baseline, with the locality hit rate reported.

**In progress: universal ports (Phase 8).** The engine is now model-agnostic: three ports run
through one contract ([`models/CONTRACT.md`](../models/CONTRACT.md)), a self-describing
[format](../format/SPEC.md) with three encodings (f32, Q8_0, Q4_0), and 20 gates. See
[Phase 8](#phase-8--the-engine-meets-someone-elses-model) for what was measured and what is still
missing. The paging experiment itself is now done on a model that does not fit — exact, and at
0.80 tok/s at a 13 GB budget — batch-usable, not interactive. The next step is not a bigger
model but a smaller, faster one: see
[`docs/2026-10-19-c2-results-and-thesis.md`](2026-10-19-c2-results-and-thesis.md).

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
| 8 — Universal ports | `engine/` split, `format/SPEC.md`, two ports running (Tier A), Q8_0 encoding, 16 gates | 🔄 in progress |
| 9+ — Scale, agentic workloads | Tool calling, domain SFT, longer-context tiers | ⏳ later |
