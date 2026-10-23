<div align="center">

<img src="assets/cachepotato_image.png" alt="CachePotato mascot — a low-poly potato with a circuit-pattern slice" width="190">

# CachePotato

**Running a sparse language model on a "potato". Designing a sparse language model and the memory-hierarchy engine.**

[![Language](https://img.shields.io/badge/language-C11-555555?style=flat-square&logo=c&logoColor=white)](#build--run)
[![Dependencies](https://img.shields.io/badge/engine%20third--party%20deps-none-brightgreen?style=flat-square)](#build--run)
[![Build](https://img.shields.io/badge/build-make%20%C2%B7%20zero--config-orange?style=flat-square)](#build--run)
[![Status](https://img.shields.io/badge/phase-8%20in%20progress%20%C2%B7%203%20ports%20running-blue?style=flat-square)](docs/PROGRESS.md#status)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

<sub>SSD, RAM and CPU cache treated as one addressable memory space. </sub><br>
<sub>Allowing a larger model to run on a machine, even a potato.</sub><br>
<sub>A model-agnostic engine with one port per open-weight model. Three run today — GPT-2 small, Qwen2.5-0.5B, and Qwen3-30B-A3B (32.5 GB on a 16 GB laptop) — see [Progress &amp; results](docs/PROGRESS.md#phase-8--the-engine-meets-someone-elses-model).</sub>

[**Start here**](#quick-start) · [**Build & run**](#build--run) · [**Progress & results**](docs/PROGRESS.md) · [**Experiments**](docs/EXPERIMENTS.md) · [**Roadmap**](docs/ROADMAP.md)

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

**The shared idea:** A zero-dependency-ish C runtime that **pages** expert weights
under a RAM budget so a model larger than memory can still run.

**What this project adds.** CachePotato bets on a different mechanism: 
- It treats the **on-disk layout of the weights** as a compilation target (expert
ordering, region grouping, alignment) rather than only as a runtime caching policy.
- **Reordering is proved a symmetry**: routing bit-identical and logits exact before it is relied on.

---

---

## Quick Start

> **CachePotato is a from-scratch C inference engine — no CUDA, no ggml, no Python at runtime —
> built to answer one question honestly: what does running a model larger than RAM actually cost
> on a laptop, in bytes, latency, and correctness?** It runs a 32.5 GB Qwen3-30B-A3B on a 16 GB
> ThinkPad at **1.2 tok/s** from a 17 GB 4-bit file, and the paged output is gated
> **bit-identically** against the unpaged model.

Most "run an LLM locally" work optimises throughput on hardware that already fits the model. This
starts from the other end: a model that does *not* fit, a hard memory budget, and a custom paged
file format in which the unit of paging (a region) is also the unit of measurement.

### Capabilities & Limitations

| | |
|---|---|
| **Run a 30B on a 16 GB laptop** | `cachepotato run qwen3-30b-a3b` — ~20 s for the first prompt (cold disk), then ~0.8 s/token. Follow-ups stay in the KV cache and run faster (2.2 tok/s measured on turn 2). |
| **Something interactive** | the 0.5B chat at ~10–15 tok/s on four threads — the one that feels like a normal assistant. |
| **See *why* it is slow** | per-token counters: regions fetched, MB read from disk, hit rates, evictions. This is the part a general-purpose runtime will not show you. |
| **Experiment with quantisation safely** | the format carries f32 / Q8_0 / Q4_0, and any lossy encoding has to pass a pre-registered quality gate. Q4 experts+attention passes at 93.3% / 97.5% next-token agreement with Q8; whole-model Q4 failed it (70.8%). |
| **Trust the output** | paging is bit-identical to not paging; decode is bit-identical to prefill. Same machine, same bytes — every artifact carries a machine stamp. |

### What it is not

- **Not fast.** 1.2 tok/s is one self-contained question, one file, walk away. Not a chat partner,
  and not an agent: a write → run → fix → repeat loop needs 10–30x this.
- **Not a sampler.** Greedy only, deliberately — the gate structure rests on the output being a
  deterministic function of the weights. A temperature knob is the first thing that would erode
  that, so it is absent rather than half-supported.
- **Not production.** One user, one model, one machine. No GPU/NUMA story, no concurrency, no API server.
- **Not a speed champion against mature runtimes.** The token is bytes divided by bandwidth on both
  the disk and the RAM side, measured to 2%. A runtime with more mature kernels may well be faster
  on the same model; the claim here is *repeatable measurement and exactness*, not throughput.
- **One port per architecture family.** A new family means a new block implementation and its own gates.

### Three commands

```bash
git clone https://github.com/theryeguy92/CachePotato && cd CachePotato
cargo install --path cli          # builds the C engine into a `cachepotato` binary
cachepotato run qwen3-30b-a3b     # downloads 17.5 GB on first run, then chat
```

Linux on x86-64 with AVX2, a C compiler and Rust. `cachepotato doctor` checks the machine
first: CPU, free memory, disk type and space, and what each model would get. The weights come
pre-converted from [Hugging Face](https://huggingface.co/ryanlevey/cachepotato-qwen3-30b-a3b)
and are checked against the repo's SHA-256 sums; an interrupted download resumes.

| | |
|---|---|
| `qwen3-30b-a3b` | 4-bit experts + attention, 17.5 GB, 1.2 tok/s on the T480 — the default |
| `qwen3-30b-a3b:q8` | 8-bit, 32.5 GB, 0.8 tok/s — the exact reference |

The pager budget is set from available memory (leaving 2 GB for everything else) and the
threads from the physical cores; `--budget <MB>` and `--threads <n>` override them. A budget
that makes the machine swap looks exactly like a broken model (0.36 tok/s), which is why the
default is conservative. `--pin-cores` is the measured setting on this Intel part. Models live
in `~/.local/share/cachepotato` (`CACHEPOTATO_HOME` moves them). Details:
[Build & run](#build--run).

### Why it is different

The engine is the instrument, not only the product. Every design choice here is backed by a
measurement that can be reproduced and, where it lost, recorded as a loss: the pager ties the OS
page cache at equal memory and beats it at 13 GB; core pinning recovers 2.1x of the reads at a 2 GB
budget; prefetch was disproved before it was built; the region arena and expert grouping are
documented negative results; and the 4-bit path exists because the bytes model predicted a 2x win
and the quality gate decided which variant was allowed to have it. If you want to know what a
paging scheme costs on *this* machine, the numbers are in [`docs/`](docs/) and
[`bench/results/`](bench/results/).

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
result** (§[Results](docs/PROGRESS.md#results), and read the caveats in §[Limitations](#limitations--non-goals)).
The project is now extending the **first** half — the engine — to open-weight models it
did not train. That pivot is described in §[Where this is going](docs/ROADMAP.md).

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

The user-facing `cachepotato` binary is a Rust crate (`cli/`) that compiles the same C engine
in through `engine/api/cachepotato.h`. The crates (`clap`, `reqwest`, `sha2`, `indicatif`) are
used for the command line and the download only; none of them touches the arithmetic, and a
gate (`tools/verify_cli_chat.sh`) checks that the Rust binary emits exactly the bytes the C
driver emits.

```sh
make          # build/cachepotato (GPT-2 engine), build/train, build/qwen3_chat, build/qwen_chat
make run      # interactive REPL against weights/toy.bin
cd cli && cargo build --release   # cli/target/release/cachepotato
```

`build/qwen3_chat <model.v4cp> [max_new]` is the C chat driver: the same API without the
downloader, reading `CP_BUDGET` / `CP_THREADS` / `CP_PIN_CORES` when they are set. The layer
count comes from the file, and the tokenizer is looked up next to the model, in its
`tokenizer/` folder, then in `weights/`.

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
| `engine/api/` | `cachepotato.h` — the public C API (open, chat with a token callback, stats). The chat drivers and the Rust CLI are clients of it |
| `cli/` | The `cachepotato` command (Rust): `run`, `pull`, `list`, `doctor`. Compiles `engine/` in via `build.rs` |
| `src/` | `main.c` (CLI/REPL/bench/gates) and `train.c` (the C trainer) — the two entry points, now that the runtime lives in `engine/` |
| `experiments/` | Phase 0 scratch programs — mmap/`madvise` probes, SSD latency. Learning artifacts, not engine code |
| `docs/` | Dated direction notes — currently the universal-ports pivot and its evidence |
| `bench/results/` | Committed benchmark output, including the pre-registered Q3 analysis |
| `tools/` | Python sidecar: HF/reference verification, weight + tokenizer conversion, trace analysis, bench drivers. **Never a runtime dependency** |
| `tests/` | C unit tests for the numerical kernels — currently a placeholder; verification today runs through `tools/` and [`tools/run_gates.sh`](tools/run_gates.sh) |
| `weights/` | Model files — gitignored, regenerate with `tools/` |
| `data/` | Training corpora — gitignored |

## Limitations & non-goals

**Scope of every number here**

- **One machine.** Everything was measured on a single ThinkPad T480: i5-8250U (4 cores, AVX2/FMA,
  no AVX-512), 16 GB RAM, a SATA SSD on btrfs that the engine reads at an effective ~0.39 GB/s
  (an O_DIRECT probe saturates at 370–400 MB/s). There is no GPU, NUMA or
  multi-device story. Bit-identity is **per machine**: the same source on a different CPU emits
  different vector code and different bytes, so cross-machine checks compare function, not bytes.
- **The speed is a property of this disk and this RAM.** On the 30B, a token is bytes divided by
  bandwidth on both sides, and the cost model fits to ~2% (at Q8, 13 GB: ~3.5 GB from RAM in
  374 ms + 348 MB from disk in 895 ms). Faster storage or more RAM changes the numbers; a better
  kernel mostly does not.

**The 30B port (Qwen3-30B-A3B)**

- **Usable speed is not a claim.** 1.22 tok/s at a 13 GB budget with the 4-bit file, 0.80 at Q8,
  0.36 at Q8 with an 8 GB budget and real disk reads. At Q8 the curve saturates at ~13 GB because
  the remaining reads are the compulsory ones: the experts each token newly selects (the 4-bit
  file has not had a budget sweep). Good for one self-contained question; agentic
  write → run → fix loops would need an estimated 10–30x this.
- **The fast file is lossy.** `qwen3_q4attn.v4cp` stores the experts and attention matrices as Q4_0
  (4.5 bpw), with `embed`/`lm_head` at Q8_0 and norms at f32. Against Q8 it scores 93.3% / 97.5%
  teacher-forced next-token agreement on two probes of 120 generated tokens each. That clears a
  pre-registered 90% gate, but it is two probes and not a task benchmark. Free-running agreement
  is far lower, because one flipped token changes every later context; it is reported but is not
  the criterion.
- **Whole-model Q4_0 is unsettled.** It reads 1.40 tok/s but was rejected (70.8% on the factual
  probe) on a 24-token run, a gate length later shown too noisy to decide anything. The rejection
  is probably right and has not been re-measured at 120 tokens.
- **The paging gates run on the Q8 file.** Paged vs unpaged bit-identity and decode vs prefill
  bit-identity are gated on `qwen3_30b_a3b_q8.v4cp`. The 4-bit files pass the format validator
  (structure, regions, digests) but are not in the paging gate suite.
- **The pager's edge over the OS is small, and only near the working set.** At 11 GB it fetches the
  same regions the unpaged run demands (54.8 vs 54.7 per token); at 11–13 GB it runs 0.77–0.80
  tok/s against the unpaged control's 0.71 at Q8, while reading fewer physical bytes. Below the
  working set the OS page cache wins, because it gets all the free RAM while the budget is a hard
  cap. Pinning the core tier (`CP_PIN_CORE=1`, not the `CP_PIN_CORES` CPU-affinity switch) is the
  one eviction-policy win, and only at budgets of ≤4 GB. A pager-owned read arena and co-selection
  expert grouping were measured and lost; prefetch was ruled out before it was built, because 86.8%
  of touches already hit.
- **Model quality is the base model's.** Nothing here measures Qwen3-30B-A3B's quality beyond its
  agreement with Q8. Greedy decoding only, and a 2,048-token context: when it fills, the chat
  driver says so and starts a new conversation.

**The co-design result (gpt2-moe, Phase 7)**

- **It is conditional.** Sparse + paging wins **only where the dense model does not fit**. At
  512 MB the two are statistically tied.
- **The co-design effect is the weaker of the two results.** +16.3% at p ≈ 0.018 against a
  measured machine noise floor of 8–17%, and it is content-dependent across document blocks.
  The 384 MB re-measurement (+38.2%) is significant but **flagged**: λ=0's access pattern
  exhausts the hot tier, so that figure is a whole-hierarchy effect, not pure expert paging.
  Locality-aware routing has not been tried on any model the project did not train.
- **The model is a research artifact.** 12 layers, 736 MB, GPT-2-small scale, pretrained on
  FineWeb-Edu. It proves an architecture; it is not a useful assistant, and its quality was checked
  only against a pre-registered repetition-rate gate.

**The engine**

- **"Model-agnostic" means one port per architecture family.** The Qwen paths
  (`engine/blocks/qwen2.c`, `qwen3_moe.c`) have their own loaders and forward passes beside the
  GPT-2 `Model`, so `cachepotato -i` speaks only GPT-2. The Qwen3 v4 loader still hardcodes
  `rope_theta` (1e6) and top-8, and sizes its expert tables for 128 experts, so a Qwen3-family
  model with different values needs a loader change and its own gate.
- **Converting a model yourself needs Python.** The Qwen3-30B-A3B files are published
  pre-converted on [Hugging Face](https://huggingface.co/ryanlevey/cachepotato-qwen3-30b-a3b).
  Producing any other file means downloading the published checkpoint (61 GB bf16 for the 30B)
  and running the converters in `tools/` and `models/` (`pip install -r tools/requirements.txt`:
  numpy, torch, transformers).
- **No `make test`.** `tests/` is a placeholder. Correctness is enforced by
  [`tools/run_gates.sh`](tools/run_gates.sh) against recorded reference data. On a fresh clone only
  the four kernel and decoder gates can pass: the Qwen and v4 gates skip themselves when their
  gitignored weight files are absent, and the GPT-2/gpt2-moe gates fail, because they expect the
  generated `weights/toy.bin`, the trained gpt2-moe checkpoint and the corpus to be present.
- **Not a production serving system.** One user, one model, one process: no batching, no paged KV
  cache, no HTTP or OpenAI-compatible server. The REPL and the chat drivers are the interface.

## Documentation

| page | what is in it |
|---|---|
| [Progress & results](docs/PROGRESS.md) | Phase 8: the ports, the port contract and the gate suite; the measured results; status by phase |
| [Experiments](docs/EXPERIMENTS.md) | every investigation on record — including the ones that lost |
| [Roadmap](docs/ROADMAP.md) | where this is going, and what it would take |
| [bench/results/](bench/results/) | raw sweep data, the C2 tables and their CSVs |
| [format/SPEC.md](format/SPEC.md) · [models/CONTRACT.md](models/CONTRACT.md) | the file format, and what a port must ship |

## License

Released under the [MIT License](LICENSE).
