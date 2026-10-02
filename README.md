<div align="center">

<img src="assets/cachepotato_image.png" alt="CachePotato mascot — a low-poly potato with a circuit-pattern slice" width="190">

# CachePotato

**Running a spare language model on a "potato". Co-designing a sparse language model and the memory-hierarchy engine.**

[![Language](https://img.shields.io/badge/language-C11-555555?style=flat-square&logo=c&logoColor=white)](#build--run)
[![Dependencies](https://img.shields.io/badge/runtime%20deps-none-brightgreen?style=flat-square)](#build--run)
[![Build](https://img.shields.io/badge/build-make%20%C2%B7%20zero--config-orange?style=flat-square)](#build--run)
[![Status](https://img.shields.io/badge/phase-7%20done%20%C2%B7%20gates%20pass-blue?style=flat-square)](#status)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

<sub>SSD, RAM and CPU cache treated as one addressable memory space. </sub><br>
<sub>Allowing a larger model to run on a machine, even a potato.</sub>

[**Results**](#results) · [**Build & run**](#build--run) · [**Status**](#status) · [**Reproduce**](#reproducing-the-benchmarks) · [**Experiment log**](bench/results/)

</div>

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
- **A negative result on expert geometry.** Pairing 1.18 MB experts into 2.36 MB pages
  roughly *doubles* bytes moved for at most an 8% read-count reduction. The 32 × 1.18 MB
  layout is correct. ([`expert-grouping-2026-09-28.txt`](bench/results/sweeps/2026-09-28-expert-grouping.txt))

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

The engine needs only a C11 compiler and `libm` — no BLAS, no Python at runtime.

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
| `src/` | The engine, in C. `model` (forward + incremental decode), `weights` (mmap loader, CPW1 v2/v3), `pager` (regions, LRU, prefetch), `tokenizer` (GPT-2 BPE), `kernels`, `instr`, `train` |
| `experiments/` | Phase 0 scratch programs — mmap/`madvise` probes, SSD latency. Learning artifacts, not engine code |
| `tools/` | Python sidecar: HF/reference verification, weight + tokenizer conversion, trace analysis, bench drivers. **Never a runtime dependency** |
| `bench/results/` | Committed benchmark output, including the pre-registered Q3 analysis |
| `tests/` | C unit tests for the numerical kernels — currently a placeholder; verification today runs through `tools/` |
| `weights/` | Model files — gitignored, regenerate with `tools/` |
| `data/` | Training corpora — gitignored |

## Status

**Phase 7 (Integration) — done.** All three pre-registered gates pass, and the end-to-end
demo exists: a trained sparse model served by the paging engine, benchmarked against a dense
baseline, with the locality hit rate reported.

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
| 8+ — Scale, agentic workloads | Tool calling, domain SFT, longer-context tiers | ⏳ planned |

## Limitations & non-goals


- **One machine, one model.** Everything was measured on a single ThinkPad T480 (SATA SSD,
  16 GB RAM) against one dense control. There is no multi-device, NUMA, or GPU story.
- **The result is conditional.** Sparse + paging wins **only where the dense model does not
  fit**. At 512 MB the two are statistically tied.
- **The co-design effect is the weaker of the two results.** +16.3% at p ≈ 0.018 against a
  measured machine noise floor of 8–17%, and it is content-dependent across document blocks.
  The 384 MB re-measurement (+38.2%) is significant but **flagged**: λ=0's access pattern
  exhausts the hot tier, so that figure is a whole-hierarchy effect, not pure expert paging.
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
