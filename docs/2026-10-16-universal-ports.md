# Direction: universal ports and Tier-B co-design

**Status:** agreed 2026-10-16, not yet implemented.
**Branch:** `feat/universal-ports-tier-b`.

This note is the checked-in source of truth for a scope decision made on 2026-10-16:
extend CachePotato from *one hand-built model* to *a model-agnostic engine plus one
port per open-weight model*, and treat **weight layout** — not router training — as
the primary port mechanism.

It exists so the README, the vault, and the code agree about what has been proven,
what is being pursued, and what is explicitly *not* being claimed.

---

## 1. What this supersedes

The original project thesis, as stated in the vault, has two halves:

1. **the engine** treats SSD → RAM → CPU as one hierarchy and pages weight *regions*
   under a RAM budget instead of full-loading or blindly `mmap`-ing; and
2. **the model** is trained with a routing bias toward memory locality (λ), so a token
   tends to pick experts already resident.

The pivot keeps (1) as the project's spine and *re-orders* (2) from prerequisite to
optional upgrade. It does not contradict (2).

## 2. Three tiers of "port"

Every port is classified into exactly one of these, and the classification is part of
the port's contract because it determines which gate the port must pass.

| Tier | What it changes | Router training | Numerics vs. base model | Expected paging win |
|---|---|---|---|---|
| **A — as-is** | file format, kernels, block types | none | **exact** (`max\|Δ\| = 0`) | whatever the base model's routing already gives |
| **B — layout** | expert file order, region grouping, prefetch policy, alignment | **none** | **exact** (a permutation) | large — targets *read count*, see §4 |
| **C — co-design** | router weights (λ fine-tune), possibly top-K | required | exact only against *its own* recorded reference | largest; unproven at scale |

**Tier B is the one being pursued.** Its appeal for a universal engine is that a port
becomes a *compilation* step rather than a training run: no GPU, no training budget, no
per-model expertise. Anyone's open weights can be taken and laid out for the measured
storage hierarchy of the machine they will run on.

## 3. What Tier B preserves, from the original thesis

| Original claim | Preserved? |
|---|---|
| SSD/RAM/cache as one hierarchy; page *regions* under a budget, not blind `mmap` | **Yes — this is Tier B** |
| Region granularity, LRU, and prefetch policy are the engine's contribution | Yes, unchanged |
| Prefetch from routing history is valid because routing is locally correlated | Yes; only the definition of a prefetch target changes |
| "Sparse + paging wins only where dense does not fit" | Yes — carried over verbatim, not weakened |
| The model is trained for memory locality (λ) | **No.** Tier B never touches the router |

The engine half of the thesis is also the half with the strongest existing evidence:
Phase 3's "paging machinery works" result (RSS tracks budget 545 MB → 74 MB, correct
output verified against a Python reference) was a Tier-B result, established *before*
any λ work and independently of it.

## 4. Why layout is the lever (measured)

Measured on the T480 with `POSIX_FADV_DONTNEED` + random `pread` on a never-cached file
(the first run on `weights/gpt2.bin` was partly cached and was discarded as too
optimistic):

| read size | µs/read | MB/s |
|---:|---:|---:|
| 4 KB | 1705 | 2.4 |
| 64 KB | 1939 | 33.8 |
| 256 KB | 2958 | 88.6 |
| 1 MB | 4819 | 217.6 |
| 4 MB | 13171 | 318.4 |

Read latency is nearly independent of size (~1.7 ms), so on this device **read count,
not byte count, is the binding constraint** — and a 130× spread in throughput is
available purely by choosing a larger read. That is the quantitative basis for Tier B.

Two consequences that must not be lost:

- The existing Q3 conclusion that "pairing experts into 2.36 MB pages roughly doubles
  bytes moved, so the 32 × 1.18 MB geometry is correct" **is not universal.** It holds
  because a 1.18 MB expert already clears the sequential-read floor. For a real
  quantized MoE whose experts are ~10 KB, grouping is a throughput optimization, not a
  size optimization.
- The same Q3 analysis observed that the MoE's ~134 MB/s versus the dense model's
  ~286 MB/s is an **access-order** problem, not a read-size one. Tier B is the direct
  answer to that.

## 5. What Tier B does not claim

- **It is not model co-design.** The engine compiles a model's *layout* for the
  measured hierarchy; the model's *routing* is the model's business. If the project
  later claims "we trained the model to fit the engine," that sentence needs a Tier C
  port to point at.
- **Bit-identical pager byte counts do not carry over.** Reordering experts and
  changing region size changes which regions are resident, so bytes fetched differ by
  design for the same model at the same budget. The existing "refactors must be
  bit-identical in pager byte counts" rule applies to *refactors*, not to layout
  experiments; a layout change needs its own pre-registered protocol.
- **It is conditional on the regime.** If the model fits in RAM, Tier B has nothing to
  do, and the engine's job is to notice that honestly rather than pretend to page.

## 6. Known risk: exactness under permutation

The router's top-k scan breaks ties by lower expert id (`if (p[e] > p[a])`, strict `>`;
the C trainer and PyTorch use the same form). Permuting experts in the file requires
permuting the router's columns consistently. The *score set* — and therefore the
selected *set* — is unchanged, but:

- where two experts **tie**, the id-based tie-break can resolve to a different expert,
  changing the selected set; and
- the routed output is accumulated in **selection order**, so a reordering can change
  summation order and therefore the last bits.

For the current top-2 model this should be stable (the order is score-determined and
`S = p[a] + p[b]` sums two distinct scores), but a general top-k path — e.g. Nemotron
3.5 Bolt's top-9 with `norm_topk_prob` and a `routed_scaling_factor` — is where
layout-dependent bits can appear.

The fix, if it bites, is to make the summation order canonical (sum selected experts in
a canonical order after selection), which makes the numerics depend only on the selected
set and never on file order. **That is a numerics change and requires explicit
approval**; it should be decided in the format spec rather than discovered mid-port.

The cheap experiment that settles it: permute the experts of
`weights/moe-scale-run/l05.cpw1v3`, apply the matching router-column permutation, and
run the existing gates. Expected `max|Δ| = 0.000e+00` and identical routing.

## 7. Target repository layout

```
engine/{core,pager,kernels,blocks,tokenizers}/
format/SPEC.md                  # self-describing: block list, tensor table, region table
models/<port>/{MODEL.md,convert.py,reference.py,train/,reshape/}
analysis/  verify/  bench/protocols/  bench/results/<model>/  tests/
```

Two properties matter more than the directory names:

- **The encoding/op boundary stays clean.** `kernels/` is driven by the quantization a
  checkpoint ships (f32, q8_0, mxfp4, iq2_xxs); `blocks/` is driven by the architecture
  (mamba2, gqa, routed MoE). A block must never learn its own dtype. If that leaks,
  every new quantization touches every block.
- **Every port declares its tier.** A `BIT_IDENTICAL` reshape must pass the base
  model's existing reference; a `ROUTING_CHANGED` reshape needs its own recorded trace
  (as λ=0/0.5 do today). Without this, a faithful port and a modified model are
  indistinguishable by inspection.

## 8. Open questions

- Is λ still worth its complexity at 2 bits? The byte-rate and per-token-traffic models
  disagree about the sign of its value for a real quantized MoE.
- Text LLM only, or full VLM, for a vision-capable port?
- Are quantized encodings first-class in the format now, or fp32/bf16 only?
- Where do machine calibration profiles live (`bench/profiles/<machine>.json`)?
- Does the format need a canonical summation order to keep permutation exactness?
- What is the required expert cache hit rate for interactive throughput at a given
  budget? (First estimate: ≥66% for a 30B-A3B-class model on this CPU/disk pair.)
