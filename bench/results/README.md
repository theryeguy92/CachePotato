# Benchmark results

Committed output for every number quoted in the root `README.md`. Each file records the
machine state and software revision it was produced on; where a figure in the README is a
summary, this directory holds the raw evidence and the pre-registered decision rule behind it.

Measurements were taken on a ThinkPad T480 (16 GB RAM, SanDisk SATA SSD, btrfs),
single core, AC power unless the log notes otherwise.

## Layout

| Directory | Contents |
|---|---|
| `cache/` | Memory-hierarchy paging experiments — eviction, residency, tensor-touch traces |
| `routing/` | The locality-routing mechanism: cache-hit behaviour and output quality |
| `sweeps/` | Multi-config sweeps. The Q3 sweep is the headline end-to-end result |
| `layout/` | Weight-layout experiments: is a compiled layout a symmetry of the model? |
| `logs/c-trainer/` | C trainer loss curves (`build/train`, TinyStories, no checkpoints saved) |
| `logs/scale-run/` | The Phase 6 PyTorch scale run that produced the served MoE checkpoints |

Files are named `YYYY-MM-DD-topic.ext` so they sort chronologically.

## `cache/`

| File | What it shows |
|---|---|
| [`2026-09-14-lru-residency.md`](cache/2026-09-14-lru-residency.md) | LRU eviction under `CP_BUDGET`. Includes Experiment 1 (the cache ceiling: 7.79 → 0.70 tok/s forced-cold) and Experiment 2 (RSS 545 → 74 MB with no locality knee on dense GPT-2) |
| [`touches_gpt2.csv`](cache/touches_gpt2.csv), [`touches_toy.csv`](cache/touches_toy.csv) | Raw `CP_INSTR` tensor-touch logs: which weights a forward pass touches, and how often. The data the hot/cold policy was designed from |

**Takeaway:** dense weights have no exploitable locality, which is the empirical
justification for sparsity.

## `routing/`

Both files compare the two trained arms (`λ=0.5` vs `λ=0`) on held-out FineWeb-Edu at
step 180,000, identical corpus.

| File | What it shows |
|---|---|
| [`2026-09-25-routing-locality.md`](routing/2026-09-25-routing-locality.md) | Does λ change *temporal* routing, and does that buy cache hits? Measured over 488 windows × 512 tokens per arm. Marginal expert utilization is blind to λ; temporal reuse rate is not |
| [`2026-09-25-quality-q2.md`](routing/2026-09-25-quality-q2.md) | Is the λ=0.5 arm usable? The apparent "collapse" in the training-loop samples was a logging-protocol artifact. Primary metric: repetition rate over 24 held-out prompts |

**Takeaway:** the routing bias is real, and it does not cost output quality.

## `sweeps/`

| File | What it shows |
|---|---|
| [`2026-09-28-q3-analysis.md`](sweeps/2026-09-28-q3-analysis.md) | **Start here.** Complete analysis of the 19-config Q3 sweep against the pre-registered rules: headline PASS, flat-curve rule, the mechanism, and both caveats |
| [`2026-09-28-q3-raw.txt`](sweeps/2026-09-28-q3-raw.txt) | Raw per-prompt output for the matrix, with model checksums in the header |
| [`2026-09-28-q3-recheck.txt`](sweeps/2026-09-28-q3-recheck.txt) | Paired/interleaved 512 MB arm recheck (the clean mechanism test, core tier ~99.9% resident) |
| [`2026-09-28-q3-recheck-384.txt`](sweeps/2026-09-28-q3-recheck-384.txt) | Paired, arm-order-counterbalanced 384 MB recheck — the whole-hierarchy effect, flagged |
| [`2026-09-28-expert-grouping.txt`](sweeps/2026-09-28-expert-grouping.txt) | Offline replay of recorded traces at matched byte budgets: expert-grouping granularity. A **negative result** — pairing 1.18 MB experts into 2.36 MB pages roughly doubles bytes moved |

## `layout/`

| File | What it shows |
|---|---|
| [`2026-10-16-permutation-exactness.md`](layout/2026-10-16-permutation-exactness.md) | Does reordering experts in the weight file (with the router's columns permuted to match) leave the model unchanged? **PASS** — routing exactly invariant, logits fp32-equivalent (`max\|Δ\| = 5.722e-06`, relative 2.85e-07), C engine cross-check 0/1280. Records the correction that a permuted layout is **not** bit-identical, and the permutation-direction trap that made a correct file look broken twice |
| [`2026-10-16-l05-permutation.json`](layout/2026-10-16-l05-permutation.json) | The exact per-layer permutations used (seed `20261016`) and the permuted file's sha256, so the experiment is byte-reproducible |
| [`2026-10-16-t2-validated.md`](layout/2026-10-16-t2-validated.md) | **The Phase 8 traffic model and C replay, validated:** the replay reproduces the engine's expert hit rate to 0.00 points at 736 MB and −3.1 at 384 MB across six budgets, with access counts matching exactly. Records the two bugs that had to be fixed — the pager does **not** reset at a sequence boundary, and the charged core footprint is 317.7 MiB not 340.3 — plus an RTP1 header-size bug caught by `tools/verify_rtp1.py` |
| [`2026-10-16-traffic-model-validation.md`](layout/2026-10-16-traffic-model-validation.md) | The Phase 8 traffic model's **failed first validation**: it under-predicts the engine's expert hit rate by 5–26 points, so it is directional only. Records that prefetch and core arithmetic are ruled out, and that the effective capacity implied by measurement is up to 3.7× what the bytes allow — an open mechanism, most likely trace reset frequency |

**Takeaway:** layout compilation is a verified symmetry at the routing level, which is what
the universal-ports direction (`docs/2026-10-16-universal-ports.md`) depends on.

## `logs/`

Raw training output, kept for loss-curve provenance. These are large; they are evidence,
not summaries.

| Directory | What it is |
|---|---|
| `c-trainer/` | The C trainer (`build/train`) on TinyStories: `dense-sanity-300` / `dense-2000` and the two MoE arms at 300 steps. Ends with "checkpoint saving deferred (CPW1 v3 is Phase 7)" — these were loss-curve milestones, not the exported model |
| `scale-run/` | The Phase 6 PyTorch run (`train_moe.py`) that produced `weights/moe-scale-run/l0{0,5}.cpw1v3`, the checkpoints the engine serves. `-console` variants are the driver's console transcript; the plain `.log` is the run log |

## Reproducing

The drivers in `tools/` write back to these paths, so re-running them appends to the
existing evidence rather than creating a parallel copy.

```sh
tools/run_q3_bench.sh      # resumable 19-config sweep  -> sweeps/2026-09-28-q3-raw.txt
tools/run_q3_recheck.sh    # paired/interleaved recheck -> sweeps/2026-09-28-q3-recheck.txt
python3 tools/verify_forward.py weights/gpt2.bin     # C vs. Python reference
```

Both drivers are resumable: a config that already has a `BENCH_RESULT` line is skipped, so
an interrupted run costs only the config that was in flight.
