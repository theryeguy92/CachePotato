<sub>[**Overview**](../README.md) · [**Progress & results**](PROGRESS.md) · [**Experiments**](EXPERIMENTS.md) · [**Roadmap**](ROADMAP.md)</sub>

---

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
method: [`bench/profiles/t480-2026-10-16.json`](../bench/profiles/t480-2026-10-16.json).

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
[`docs/2026-10-16-universal-ports.md`](2026-10-16-universal-ports.md).

