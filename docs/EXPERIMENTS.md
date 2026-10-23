<sub>[**Overview**](../README.md) · [**Progress & results**](PROGRESS.md) · [**Experiments**](EXPERIMENTS.md) · [**Roadmap**](ROADMAP.md)</sub>

---

## Experiments on record

Each entry is a self-contained measurement with its data, its decision rule and, where it lost, the
reason. Nothing here is regenerated to make a gate pass.

| record | question it answers |
|---|---|
| [Phase 7 sweeps](../bench/results/README.md) | does a co-designed sparse model beat a dense one when the model does not fit? (the original Q1/Q2/Q3 result) |
| [C1: wiring the pager into the 30B](2026-10-18-c1-pager-wiring.md) | is a paged run *bit-identical* to an unpaged one? (yes, at every budget tested) |
| [C2: results and thesis](2026-10-19-c2-results-and-thesis.md) | what does paging this model actually cost, and which claim survives? |
| [C2 tables: 8 GB cold](../bench/results/c2/2026-10-18-8gb-cold.md), [low budget](../bench/results/c2/2026-10-18-low-budget.md), [slice](../bench/results/c2/2026-10-18-slice.md) | the per-budget numbers: regions fetched, physical bytes, tok/s (single-threaded; see the note below) |
| [Read amplification](../bench/results/c2/2026-10-18-read-amplification.md) | why does the pager read ~1.9× the bytes it asks for? (later traced to 4 MiB block rounding, most of which the page cache reuses) |
| [Core pinning](../bench/results/c2/2026-10-18-pinned.md) | the one policy win: 2.13× fewer regions fetched at a 2 GB budget |
| [Region arena](../bench/results/c2/2026-10-18-arena.md) | does owning the bytes (`pread`) beat the page cache? (no, and why) |
| [Tier B grouping](../bench/results/c2/2026-10-18-tierb.md) | does grouping experts that are often chosen together help? (no) |
| [Where tok/s goes](2026-10-19-tok-s-levers.md) | the token as bytes over bandwidth, every lever tried since, and the Q4_0 quality gate |
| [What this laptop can run](2026-10-19-what-can-run.md) | the speed envelope, and which current models fit inside it |
| [The port contract](2026-10-16-universal-ports.md) | the scope decision behind Phase 8, and what a new port must ship |

The 2026-10-18 C2 tables were measured before a harness bug was found: every run was silently
single-threaded. Their region and byte counts are unaffected; their tok/s are 2–3× too low. The
4-thread numbers are in the C2 summary and in *Where tok/s goes*.

## Re-running any of it

```sh
./tools/run_gates.sh                             # the 21 gates (those whose weight files are missing skip or fail)
python3 tools/c2_measure.py --help               # the C2 arms (cold start, interleaved, 4 threads)
python3 tools/verify_q4_quality.py --gen 120     # 4-bit vs Q8_0 on matched contexts
```

Each C2 measurement writes its CSV and metadata next to the record. The metadata carries a machine
stamp (host, CPU, instruction set, compiler, git commit), so a number can be traced to the hardware
and the commit that produced it. The Phase 7 sweeps predate the stamp.
