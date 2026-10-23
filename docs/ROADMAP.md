<sub>[**Overview**](../README.md) · [**Progress & results**](PROGRESS.md) · [**Experiments**](EXPERIMENTS.md) · [**Roadmap**](ROADMAP.md)</sub>

---

## Where this is going

Phase 7 answered its question for **one hand-built model on one machine**. Phase 8 makes the
*engine* model-agnostic and adds one **port** per open-weight model.

The first port was originally going to be [Nemotron 3.5
Bolt 74B-A9B](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/). It was deferred
on purpose. A model that moves ~2.4 GB of expert weights per token is a stress test for the format,
not a first port, and the blocks every port needs (RMSNorm, RoPE, grouped-query attention, gated
FFNs) should not be debugged for the first time on a model where mistakes are expensive. The first
ports were chosen for the opposite property: small enough that a wrong implementation shows up in
the output.

The intent is for a port to be a **compilation** step, not a training run. Each port is classified
into exactly one tier, and the tier decides which gate it must pass:

| Tier | What it changes | Router training | Numerics vs. the base model | Expected paging win |
|---|---|---|:---:|---|
| **A — as-is** | file format, kernels, block types | none | **exact** (`max\|Δ\| = 0`) | whatever the base model's routing already gives |
| **B — layout** | expert file order, region grouping, prefetch, alignment | **none** | **exact routing**; bit-identical if only placement changes, fp32-equal logits if experts are renumbered | large, if read count is the limit |
| **C — co-design** | router weights (λ fine-tune), possibly top-K, or a lossy encoding | λ: required | exact only against *its own* recorded reference | largest; unproven at scale |

Quantising a model (Q8_0, Q4_0) also puts it in Tier C, because it changes the numbers and must be
judged on quality rather than exactness.

**Tier B was the chosen direction**, as a re-ordering of the original thesis rather than a
rejection of it: the engine half stays the project's spine, and the trained-routing half (λ) goes
from prerequisite to optional upgrade. "Sparse + paging wins only where dense does not fit" carries
over unchanged.

**What happened when it was tried.** On Qwen3-30B-A3B the layout machinery works: expert
placement in the file can be recompiled without changing the output. But grouping experts that are often
chosen together was *slower*, because the paired experts were selected together only 11.2% of the
time and the extra bytes outweighed the saved reads
([record](../bench/results/c2/2026-10-18-tierb.md)). On this machine the limit turned out to be
bytes rather than read count, so the lever that paid was quantisation (Tier C), not layout.

**Why layout looked like the lever.** On the T480's SATA SSD a small cold random read costs a few
hundred microseconds, so the number of reads can matter more than the number of bytes:

| read size | µs/read (p50) | MB/s |
|---:|---:|---:|
| 4 KB | 334 | 14.0 |
| 16 KB | 210 | 65.4 |
| 64 KB | 247 | 296.0 |
| 256 KB | 348 | 819.9 |
| 1 MB | 750 | 1349.9 |
| 4 MB | 4413 | 898.3 |

On this table, choosing a larger read spans a ~96× range in throughput (4 KB to 1 MB). Measured by
`build/calibrate` over a **30 GB file, larger than RAM**, because a smaller file is served from the
page cache on btrfs and `fadvise` does not reliably evict it. An earlier version of this table used
a 0.2 GB file and overstated 4 KB latency ~5×. Full profile and method:
[`bench/profiles/t480-2026-10-16.json`](../bench/profiles/t480-2026-10-16.json).

**Treat the large-read rows with suspicion.** The drive is a SATA SSD (SanDisk SD8SB8U), and SATA
III tops out around 550–600 MB/s, so the 820–1,350 MB/s rows cannot be pure disk reads; some of
those reads were probably served from cache. Under the real workload the engine reads the disk at
an effective ~0.39 GB/s, and a separate `O_DIRECT` probe saturates at 370–400 MB/s.

Two constraints carry over unchanged into every port:

- **Verification first, and exact.** A port's C output must reproduce its Python reference
  bit for bit (`max|Δ| = 0`), and a Tier A or B port must also reproduce the *base model's*
  recorded routing. No tolerance is added for ports. The one stated exception: renumbering experts
  reorders the router's softmax sum, so it moves the logits slightly (max |Δ| 5.7e-06 measured)
  while leaving routing exact, and that comparison carries an explicit tolerance.
- **Bit-identical pager byte counts apply to refactors, not to layout experiments.** Reordering
  experts or changing the region size changes what is resident, so byte counts differ by design and
  need their own pre-registered protocol.

Tier B does **not** claim model co-design, and it only matters when the model does not fit: if the
model fits in RAM, paging has nothing to do, and the engine should say so. Full statement and
evidence: [`docs/2026-10-16-universal-ports.md`](2026-10-16-universal-ports.md). Current options
for the next port: [What this laptop can run](2026-10-19-what-can-run.md).
