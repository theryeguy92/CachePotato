<sub>[**Overview**](../README.md) · [**Progress & results**](PROGRESS.md) · [**Experiments**](EXPERIMENTS.md) · [**Roadmap**](ROADMAP.md)</sub>

---

## Experiments on record

Each entry is a self-contained measurement with its data, its decision rule and — where it lost — the
reason. Nothing here is regenerated to make a gate pass.

| record | question it answers |
|---|---|
| [Phase 7 sweeps](../bench/results/README.md) | does a co-designed sparse model beat a dense one when the model does not fit? (the original Q1/Q2/Q3 result) |
| [C1 — wiring the pager into the 30B](2026-10-18-c1-pager-wiring.md) | is a paged run *bit-identical* to an unpaged one? (yes, at every budget) |
| [C2 — results and thesis](2026-10-19-c2-results-and-thesis.md) | what does paging this model actually cost, and what is the claim that survives it? |
| [C2 tables — 8 GB cold](../bench/results/c2/2026-10-18-8gb-cold.md), [low budget](../bench/results/c2/2026-10-18-low-budget.md), [slice](../bench/results/c2/2026-10-18-slice.md) | the per-budget numbers: regions fetched, physical bytes, tok/s |
| [Read amplification](../bench/results/c2/2026-10-18-read-amplification.md) | why does an mmap'd pager read 1.9x the bytes it asks for? |
| [Core pinning](../bench/results/c2/2026-10-18-pinned.md) | the one policy win: 2.13x fewer reads at a 2 GB budget |
| [Region arena](../bench/results/c2/2026-10-18-arena.md) | does owning the bytes (pread) beat the page cache? (no — and why) |
| [Tier B grouping](../bench/results/c2/2026-10-18-tierb.md) | does co-selection-aware expert grouping help? (no — negative result) |
| [Where tok/s goes](2026-10-19-tok-s-levers.md) | the token as bytes/bandwidth, and every lever tried since |
| [What this laptop can run](2026-10-19-what-can-run.md) | the speed envelope, and which 2026 models fit inside it |
| [The port contract](2026-10-16-universal-ports.md) | what a new model port must ship before it counts |

## Re-running any of it

```sh
./tools/run_gates.sh                       # the 20 gates
python3 tools/c2_measure.py --help         # the C2 arms (cold start, interleaved, 4 threads)
python3 tools/verify_q4_quality.py --gen 120   # 4-bit vs Q8 on matched contexts
```

Every measurement writes its own CSV/metadata next to the record, and the metadata carries the
machine stamp — so a number can be traced to the hardware and the commit that produced it.
