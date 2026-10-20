<div align="center">

<img src="assets/cachepotato_image.png" alt="CachePotato mascot — a low-poly potato with a circuit-pattern slice" width="190">

# CachePotato

**Running a sparse language model on a "potato". Co-designing a sparse language model and the memory-hierarchy engine.**

[![Language](https://img.shields.io/badge/language-C11-555555?style=flat-square&logo=c&logoColor=white)](#build--run)
[![Dependencies](https://img.shields.io/badge/third--party%20deps-none-brightgreen?style=flat-square)](#build--run)
[![Build](https://img.shields.io/badge/build-make%20%C2%B7%20zero--config-orange?style=flat-square)](#build--run)
[![Status](https://img.shields.io/badge/phase-8%20in%20progress%20%C2%B7%202%20ports%20running-blue?style=flat-square)](#status)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

<sub>SSD, RAM and CPU cache treated as one addressable memory space. </sub><br>
<sub>Allowing a larger model to run on a machine, even a potato.</sub><br>
<sub>A model-agnostic engine with one port per open-weight model. Two ports run today — see [Phase 8](#phase-8--the-engine-meets-someone-elses-model).</sub>

[**Results**](#results) · [**Where this is going**](#where-this-is-going) · [**Build & run**](#build--run) · [**Status**](#status) · [**Reproduce**](#reproducing-the-benchmarks) · [**Experiment log**](bench/results/)

</div>

---

## Influence and credit for the project

First I would like to thank contributors and developers for `Colibri` and `Hummingbird`, they influenced
this project and inspired me to work on this project. Please check them out.

The idea of treating SSD, RAM and VRAM as one managed hierarchy and streaming MoE expert
weights from disk is **not original to this project**.

- **[Colibri](https://github.com/JustVugg/colibri)** — a highly specialized C engine that runs
  very large MoE models (hundreds of billions of parameters) on consumer machines by streaming
  expert weights from SSD. It is the clearest existing demonstration that the memory hierarchy,
  not the model size, is the thing to engineer.
- **[Hummingbird](https://github.com/prayangshuuu/hummingbird)** — a C17 runtime in the same
  space, which explicitly builds on Colibri's ideas with a generalized, adapter-based design. It
  unifies SSD/RAM/VRAM, keeps always-needed weights resident, streams the expert pool on demand,
  and ships a CLI, an OpenAI-compatible server and CPU/CUDA/Metal backends.

**What is shared.** The core thesis — a zero-dependency-ish C runtime that pages expert weights
under a RAM budget so a model larger than memory can still run — is shared with both.

**What this project adds.** CachePotato bets on a different
*mechanism*: it treats the **on-disk layout of the weights** as a compilation target (expert
ordering, region grouping, alignment) rather than only as a runtime caching policy, and it
insists that such a reordering be **proved a symmetry**: routing bit-identical, logits exact —
before it is relied on. Alongside that, the project's port contract requires every model to ship
an *independent* reference implementation and a recorded baseline that must never be regenerated
to make a failing test pass. Those two things: layout-as-compilation with a proven symmetry, and
verification structured to catch a shared misunderstanding are the contributions worth
defending.

---

## Question: Can a LLM run on a "potato"?

Running a frontier-scale model normally means buying enough RAM to hold it. CachePotato
asks the opposite question: **what if the engine and the model were designed for each
other from the start?** Instead of bolting a storage-aware runtime onto a model that
assumes it lives in memory.

Sparsity (Mixture-of-Experts) and SSD offloading both already exist. What is unproven is
the **co-design loop**:

- the **engine** treats SSD → RAM → CPU cache as one hierarchy, paging weight *regions*
  under a RAM budget instead of full-loading or blindly `mmap`-ing; and
- the **model** is trained with a routing bias toward *memory locality*, so a token tends
  to pick experts whose weights the engine already has resident, rather than scattering
  reads across the whole weight file.

**Phase 7 tested the second half of that loop on a hand-built model and got a conditional
result** (§[Results](#results), and read the caveats in §[Limitations](#limitations--non-goals)).
The project is now extending the **first** half — the engine — to open-weight models it
did not train. That pivot is described in §[Where this is going](#where-this-is-going).

### Terminology

| Symbol | Meaning |
|---|---|
| **λ** (lambda) | The strength of the locality bonus during training. A knob on the router that rewards reusing recent experts. λ=0.5 means the bonus is on; λ=0 means it is off. |
| **MoE (λ=0.5)** vs **MoE (λ=0)** | The same sparse model trained two ways, differing only in that one knob. Comparing them isolates the effect of the training from the effect of the architecture. |
| **Dense GPT-2** | The ordinary, non-sparse baseline model, used as the control to beat. |
| **tok/s** | Tokens per second. How fast text is generated. Higher is better. |
| **RAM budget** | The ceiling you give the engine, e.g. `CP_BUDGET=512` means it may use 512 MB of RAM and must page the rest from disk. |
| **RSS** | Resident Set Size. The RAM the process is actually using. |
| **LRU** | Least Recently Used. The eviction rule: when out of room, discard whatever was touched longest ago. |
| **KV cache** | Saved per-token attention state during generation, so earlier tokens are not recomputed. |
| **Prefetch** | Loading something from disk *before* it is needed, to hide the wait. |
| **p < 0.05** | A significance test. It means a difference this large is unlikely to be luck. |

## Where this is going

Phase 7 answered the question it set out to answer for **one hand-built model on one
machine**. Phase 8 makes the *engine* model-agnostic and adds one **port** per open-weight
model.

The originally-planned first port was [Nemotron 3.5
Bolt 74B-A9B](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/). It was
deferred, deliberately: a ~2.4 GB/token expert-traffic model is a *format stress case*, not a
first port, and the blocks a port needs (RMSNorm, RoPE, grouped-query attention, gated FFNs)
cannot be debugged for the first time on a model nobody can afford to be wrong about. The first
ports were chosen instead for the opposite property — small enough that a wrong implementation is
visible to a human reading the output.

The idea is for a port to be a **compilation** step, not a training run. A port is
classified into exactly one tier, and the tier decides which gate it must pass:

| Tier | What it changes | Router training | Numerics vs. the base model | Expected paging win |
|---|---|---|:---:|---|
| **A — as-is** | file format, kernels, block types | none | **exact** (`max\|Δ\| = 0`) | whatever the base model's routing already gives |
| **B — layout** | expert file order, region grouping, prefetch, alignment | **none** | **exact** (a permutation) | large — it targets *read count* |
| **C — co-design** | router weights (λ fine-tune), possibly top-K | required | exact only against *its own* recorded reference | largest; unproven at scale |

**Tier B is the direction**, and it is a re-ordering of the original thesis rather than a
rejection of it: the engine half stays the project's spine, while the trained-routing half
(λ) goes from prerequisite to optional upgrade. It does not weaken any earlier claim —
"sparse + paging wins only where dense does not fit" carries over verbatim.

**Why layout, specifically.** On the T480's SATA SSD, cold random reads cost a few
hundred microseconds, and *how many* reads you issue binds well before *how many bytes*
you move — a 64× throughput spread is available purely by choosing a larger read:

| read size | µs/read (p50) | MB/s |
|---:|---:|---:|
| 4 KB | 334 | 14.0 |
| 16 KB | 210 | 65.4 |
| 64 KB | 247 | 296.0 |
| 256 KB | 348 | 819.9 |
| 1 MB | 750 | 1349.9 |
| 4 MB | 4413 | 898.3 |

That is a 64× throughput spread available purely by choosing a larger read — which is why
expert ordering and region grouping, not just byte volume, are the lever. Measured by
`build/calibrate` over a **30 GB file (larger than RAM — the only honest protocol here,
since a smaller file is served from page cache on btrfs and fadvise does not reliably
evict it)**. An earlier version of this table used a 0.2 GB file and overstated 4 KB
latency ~5×; the vault's older ~227 µs figure was substantially right. Full profile and
method: [`bench/profiles/t480-2026-10-16.json`](bench/profiles/t480-2026-10-16.json).

Two constraints carry over unchanged into every port:

- **Verification-first, exact.** A port's C output must reproduce its Python reference
  bit-for-bit (`max|Δ| = 0`), and a Tier-A/B port must also reproduce the *base model's*
  recorded routing. No tolerance is added for ports.
- **Bit-identical pager byte counts apply to refactors, not to layout experiments.**
  Reordering experts or changing region size changes what is resident, so byte counts
  differ by design and need their own pre-registered protocol.

Tier B does **not** claim model co-design, and it is conditional on the regime: if the
model fits in RAM, paging has nothing to do and the engine should say so rather than
pretend. Full statement, evidence, and open questions:
[`docs/2026-10-16-universal-ports.md`](docs/2026-10-16-universal-ports.md).

## Phase 8 — the engine meets someone else's model

Phase 8 asked whether the engine can be *model-agnostic*: can the same binary run a model it was
not written for, with the same exactness discipline? Two ports now run, and the answer is yes —
with the caveats stated below.

### What a port is

[`models/CONTRACT.md`](models/CONTRACT.md). Every port ships a `MODEL.md` whose architecture facts
are each marked **VERIFIED** (from a published config or, better, from the actual tensor shapes) or
**ASSUMED**; a `convert.py`; an **independent** `reference.py` written without importing the
framework that produced the model; and a `verify/gate.py`. The independence is the point: on
2026-09-25 two implementations agreed with each other, disagreed with the frozen spec, and every
gate passed. A reference that shares code with the thing it checks cannot catch that.

### The two ports (More to come soon)

| port | tier | size | what it proves |
|---|---|---|---|
| [`gpt2-moe`](models/gpt2-moe/MODEL.md) | A (as-is), MoE | 736 MB | the contract works end to end on a MoE with recorded routing |
| [`qwen2.5-0.5b`](models/qwen2.5-0.5b/MODEL.md) | A (as-is), dense | 525 MB (Q8) | the engine can meet a foreign architecture and **talk** |

**gpt2-moe** is the project's own model, ported through the contract. Its independent numpy
reference reproduces the routing recorded from the trained model **20480/20480 selections
(100.0000%)**, and the C engine reproduces that reference with **0/1280** mismatches.

**Qwen2.5-0.5B-Instruct** required four blocks the engine did not have — **RMSNorm** (no mean, no
bias), **RoPE**, **grouped-query attention** (14 query heads against 2 KV heads) and a **gated
FFN** (SwiGLU, three matrices). It also required a Qwen2 tokenizer, which is not GPT-2's: it is a
Unicode-aware pre-tokenizer plus BPE plus special tokens.

### Verification & the order it was built

Every step is a gate in [`tools/run_gates.sh`](tools/run_gates.sh) (16 gates, all passing).

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
Full arithmetic: [`analysis/t6b-bandwidth-budget.md`](analysis/t6b-bandwidth-budget.md).

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
  [`tools/ab.py`](tools/ab.py) interleaves A and B, compares per-pair deltas with a sign test, and
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
  [`docs/2026-10-19-c2-results-and-thesis.md`](docs/2026-10-19-c2-results-and-thesis.md).
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
[`bench/results/sweeps/2026-09-28-q3-analysis.md`](bench/results/sweeps/2026-09-28-q3-analysis.md).

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
  for sparsity. ([`experiment-2.md`](bench/results/cache/2026-09-14-lru-residency.md))
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
  ([`expert-grouping-2026-09-28.txt`](bench/results/sweeps/2026-09-28-expert-grouping.txt))

## How it works

- **Weight classes.** Tensors needed for every token (embeddings, attention, LayerNorm)
  stay resident; the large routed-expert pool lives on disk and moves up on demand. One
  expert is one contiguous 1,183,488-byte block, so a miss is single sequential read.
- **Regions, not pages.** The pager tracks `(tensor, layer)` regions for dense tensors and
  one region per routed expert, updating recency on every touch and evicting LRU with
  `madvise(MADV_DONTNEED)`, optionally `POSIX_FADV_DONTNEED` (`CP_PAGER_SYS=1`) to model a
  model larger than the page cache.
- **Prefetch from routing history.** Because routing is locally correlated, the engine
  prefetches the union of the previous `W` tokens' expert sets at the same layer
  (`CP_PREFETCH_W`, default 4). Prefetched regions consume budget, so the policy is scored
  honestly rather than getting free residency.
- **Locality-aware routing.** The router scores each expert and adds a λ·(previously
  selected) bonus; the locality set lives in sequence state alongside the KV cache and
  resets at window start **and** at EOS. λ is stored in the weight file header, because
  serving at a different λ than training routes differently.
- **Verification-first numerics.** The C forward pass is written to mirror
  `experiments/forward_by_hand.py` op-for-op, and a Python sidecar gates every export.

## Build & run

The engine needs only a C11 compiler, `libm` and **POSIX threads** — no BLAS, no Python at
runtime, no third-party libraries. (pthread was added on 2026-10-17 for the bit-identical
threaded kernels; it is a platform C library, not a package, but it is a real link
dependency and the rule says so here rather than leaving the claim quietly false.)

```sh
make          # builds build/cachepotato (engine) and build/train (C trainer)
make run      # interactive REPL against weights/toy.bin
```

Weights are gitignored; generate them from source:

```sh
pip install -r tools/requirements.txt

python3 tools/export_toy_weights.py          # tiny 1 KB fixture -> weights/toy.bin
python3 tools/convert_tokenizer.py           # GPT-2 BPE        -> weights/tokenizer.bin
python3 tools/convert_weights.py             # HF GPT-2  ~500MB -> weights/gpt2.bin
```

Then run the engine:

```sh
# forward pass on raw token ids (reference/verification mode)
./build/cachepotato weights/gpt2.bin 5 1 7

# encode -> forward -> next token
./build/cachepotato weights/gpt2.bin -p "Alan Turing theorized that computers"

# streaming interactive REPL (nucleus sampling by default)
./build/cachepotato weights/gpt2.bin -i --max-new 256 --temp 0.8 --top-p 0.95

# same REPL under a 512 MB pager budget, and forced-cold for the worst case
CP_BUDGET=512 ./build/cachepotato weights/gpt2.bin -i
CP_BUDGET=512 CP_COLD=1 ./build/cachepotato weights/gpt2.bin -i
```

| Flag | Purpose |
|---|---|
| `-p "text"` | Encode a prompt and report the next token |
| `-i` | Interactive REPL, streaming to stdout, tok/s · RSS · disk to stderr |
| `--temp` `--top-p` `--max-new` `--seed` | Decoding controls (defaults 0.8 / 0.95 / 128) |
| `--greedy` | Restore greedy decoding |
| `--bench <bin> <start> <T> <N> <P>` | Pre-registered throughput protocol with 95% CI |
| `--gen-check` / `--step-check` / `--moe-sel` | Verification gates against recorded reference data |

| Env var | Purpose |
|---|---|
| `CP_BUDGET=<MB>` | Pager budget; unset disables the pager |
| `CP_PAGER_SYS=1` | Also drop pages from the system page cache |
| `CP_COLD=1` | Force full eviction every token (worst case) |
| `CP_PREFETCH_W=<n>` | Prefetch window in tokens (default 4, 0 disables) |
| `CP_INSTR=<path>` | Write a `tensor,layer,token,bytes` touch log as CSV |
| `CP_LOGITS=1` | Print the full logits block for verify scripts |

### Reproducing the benchmarks

```sh
tools/run_q3_bench.sh          # resumable 19-config sweep -> bench/results/sweeps/2026-09-28-q3-raw.txt
tools/run_q3_recheck.sh        # paired/interleaved arm comparison
python3 tools/verify_forward.py weights/gpt2.bin        # C vs. Python reference
python3 tools/verify_cpw1_v3.py                         # export + routing gate
```

## Layout

| Path | Purpose |
|---|---|
| `engine/` | The C runtime, model-agnostic. `core/` (mmap loader, CPW1 v2/v3, forward + incremental decode, trace replay), `pager/` (regions, LRU, prefetch), `kernels/`, `tokenizers/`, `blocks/` (**empty** — splitting `model.c`'s block types out of `core/` is a pending refactor) |
| `models/` | One port per model: [`CONTRACT.md`](models/CONTRACT.md) (what a port must ship) and `gpt2-moe/` — the project's own model ported through it, with an independent numpy reference, a converter and the port gate |
| `format/` | [`SPEC.md`](format/SPEC.md) — the self-describing weight format (CPW1 v4). **Implemented and gated**: a v3→v4 round-trip reproduces the recorded PyTorch routing 100% and survives a shuffled physical payload, i.e. tensors are addressed by offset |
| `src/` | `main.c` (CLI/REPL/bench/gates) and `train.c` (the C trainer) — the two entry points, now that the runtime lives in `engine/` |
| `experiments/` | Phase 0 scratch programs — mmap/`madvise` probes, SSD latency. Learning artifacts, not engine code |
| `docs/` | Dated direction notes — currently the universal-ports pivot and its evidence |
| `bench/results/` | Committed benchmark output, including the pre-registered Q3 analysis |
| `tools/` | Python sidecar: HF/reference verification, weight + tokenizer conversion, trace analysis, bench drivers. **Never a runtime dependency** |
| `tests/` | C unit tests for the numerical kernels — currently a placeholder; verification today runs through `tools/` and [`tools/run_gates.sh`](tools/run_gates.sh) |
| `weights/` | Model files — gitignored, regenerate with `tools/` |
| `data/` | Training corpora — gitignored |

## Status

**Phase 7 (Integration) — done.** All three pre-registered gates pass, and the end-to-end
demo exists: a trained sparse model served by the paging engine, benchmarked against a dense
baseline, with the locality hit rate reported.

**In progress: universal ports (Phase 8).** The engine is now model-agnostic: two ports run
through one contract ([`models/CONTRACT.md`](models/CONTRACT.md)), a self-describing
[format](format/SPEC.md) with two block encodings, and 16 gates. See
[Phase 8](#phase-8--the-engine-meets-someone-elses-model) for what was measured and what is still
missing. The paging experiment itself is now done on a model that does not fit — exact, and at
0.80 tok/s at a 13 GB budget — batch-usable, not interactive. The next step is not a bigger
model but a smaller, faster one: see
[`docs/2026-10-19-c2-results-and-thesis.md`](docs/2026-10-19-c2-results-and-thesis.md).

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

## Limitations & non-goals


- **One machine, one model.** Everything was measured on a single ThinkPad T480 (SATA SSD,
  16 GB RAM) against one dense control. There is no multi-device, NUMA, or GPU story.
- **The result is conditional.** Sparse + paging wins **only where the dense model does not
  fit**. At 512 MB the two are statistically tied.
- **The co-design effect is the weaker of the two results.** +16.3% at p ≈ 0.018 against a
  measured machine noise floor of 8–17%, and it is content-dependent across document blocks.
  The 384 MB re-measurement (+38.2%) is significant but **flagged**: λ=0's access pattern
  exhausts the hot tier, so that figure is a whole-hierarchy effect, not pure expert paging.
- **Usable speed is not a claim.** The 30B port runs exactly and at 1.14 tok/s with 4-bit experts
  at a 13 GB budget (0.80 at Q8, 0.36 at 8 GB with real disk reads). 4-bit experts + attention is a
  Tier-C choice and costs ~5% next-token agreement with Q8 on a matched context (gated at 120
  generated tokens; 93.3% / 97.5%). 4-bit on `lm_head`/`embed` as well reads 1.40 tok/s but is
  unvalidated — its rejection rested on a 24-token measurement, which later work showed to be
  too noisy to trust. The measured envelope on this machine is set by
  **compulsory expert traffic, not compute**: compute is 374 ms/token, disk is 895 ms at 13 GB, and
  the curve saturates because the remaining reads are exactly the experts each token selects. A
  model with fewer active parameters is the lever, not a faster kernel.
- **Model quality is not a claim.** This is a 12-layer, 736 MB research artifact at
  GPT-2-small scale, pretrained on FineWeb-Edu. It proves an architecture; it is not a useful
  assistant. Quality was checked only against a pre-registered repetition-rate gate.
- **Single-threaded, scalar, fp32.** No SIMD kernels, no threading, no quantization. The
  ~134 MB/s effective read bandwidth on small scattered expert regions is known headroom.
- **No test target yet.** `make` builds the engine and trainer; `tests/` is a placeholder.
  Correctness is currently enforced by the `tools/verify_*.py` gates against recorded
  reference data rather than by a `make test`.
- **Not a production serving system.** No batching, No paged KV cache, No HTTP server. The
  REPL is the interface.

## License

Released under the [MIT License](LICENSE).
