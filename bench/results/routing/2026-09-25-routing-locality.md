# Routing locality — Phase D mechanism measurement (2026-09-25)

**Question:** does the λ locality bias actually change *temporal* routing, and does
that buy cache hits? Phase 6 could not answer this: it logged aggregate expert
utilization, a *frequency* statistic that is blind to a temporal bias.

**Answer: yes, decisively — on the mechanism.** Whether it buys *usable* quality
is still open (see Caveats).

## Setup

- Traces: `tools/record_routing.py`, per-token top-2 expert selections for all 10
  MoE layers, recorded through the *same* `moe_ffn` used in training (λ bias
  active, locality set reset at window start and at EOS).
- Data: held-out FineWeb-Edu, 488 windows × 512 tokens = **249,856 tokens** per
  arm (2,491,020 valid adjacent pairs per layer).
- Arms: `l00` (λ=0) and `l05` (λ=0.5), both at step 180,000, identical corpus.
- Precision: **fp32 primary**, bf16 sensitivity (the two agree to 3 decimals on
  every headline).
- Analysis: `tools/analyze_routing.py`. LRU via stack distances, so every cache
  capacity is evaluated from one pass. Block = one routed expert = 1,183,488 B.

**Instrumentation is not a confound:** the `TRACE` hook added to `train_moe.py`
is opt-in and gated bit-identical against the un-instrumented path
(`tools/verify_trace_hook.py`: max|Δlogits| = 0.000e+00, |Δloss| = 0.000e+00,
hook both off and on). The scale run itself executed with `TRACE = None`.

## 1. Marginal utilization — the metric that could not see λ

| | l00 (λ=0) | l05 (λ=0.5) |
|---|---|---|
| max share over layers | 0.036 | 0.043 |

Uniform is 0.031. Both are near-uniform and nearly identical — this is exactly
why the Phase 6 collapse-watch said nothing about whether λ works. It cannot.

## 2. Temporal reuse rate — `P(sel(t) ∩ sel(t-1) ≠ ∅)`

| layer | l00 | l05 | Δ (pp) |
|---|---|---|---|
| 1 | 0.140 | 0.283 | +14.27 |
| 2 | 0.169 | 0.421 | +25.21 |
| 3 | 0.213 | 0.282 | +6.93 |
| 4 | 0.253 | 0.414 | +16.10 |
| 5 | 0.241 | 0.320 | +7.90 |
| 6 | 0.287 | 0.553 | +26.52 |
| 7 | 0.250 | 0.486 | +23.65 |
| 8 | 0.274 | 0.528 | +25.41 |
| 9 | 0.310 | 0.506 | +19.59 |
| 10 | 0.247 | 0.545 | +29.77 |
| **ALL** | **0.238** | **0.434** | **+19.53** |

95% CI over 488 windows (bootstrap, windows as the independent unit):
l00 [0.236, 0.241], l05 [0.430, 0.438], **Δ [+19.34, +19.75] pp**.

Two secondary observations:
- **Natural locality is real** (l00 = 0.238 ≫ chance), confirming Mixtral §5 —
  so λ=0+LRU is a genuine competitor, not a strawman, and the correct control.
- **Locality strengthens with depth** (both arms), also as Mixtral predicts;
  this depth gradient is why MoE-ing middle/late layers was the right call.

## 3. Top-1 stickiness and transition structure

`P(top1(t) = top1(t-1))`, random baseline = 0.031:

| | l00 | l05 |
|---|---|---|
| ALL layers | 0.075 | 0.166 |
| identical top-2 sets | 0.021 | 0.065 |

So λ=0.5 is ~5.4× chance on top-1 stickiness vs ~2.4× for λ=0. Per-layer
transition-matrix difference ‖Δ‖_F averages **1.748** (0 would mean inert),
uniform across depth (1.44–1.91). **λ reshapes the routing distribution, not just
its marginals.**

## 3b. Prefetch coverage — the number the engine actually gets

Token-level reuse **overstates** what a prefetcher can use: a token with a partial
overlap counts as a "reuse" but only covers part of its accesses. The metric that
matters is *access-level coverage* — the share of the 2 expert accesses per token
per layer already covered by prefetching the union of the previous W tokens' sets
at that layer. `W=1` is the spec's stated contract ("same experts as the previous
token at this layer").

| W | l00 (λ=0) | l05 (λ=0.5) | Δ (pp) |
|---|---|---|---|
| 1 (spec contract) | 0.130 | **0.249** | +11.94 |
| 2 | 0.200 | 0.303 | +10.36 |
| 3 | 0.260 | 0.351 | +9.15 |
| 4 | 0.313 | 0.393 | +8.01 |

Two design consequences, both load-bearing for Phase 7:

1. **`W=1` leaves most of the benefit on the table.** Widening to `W≈3–4` lifts
   λ=0.5 coverage from 0.249 to ~0.35–0.39 (and λ=0 from 0.130 to ~0.31). The
   `[P7-axis]` recency window is not a refinement — it is worth roughly 1.5× the
   coverage of the spec's contract.
2. **A wider serving-time window substitutes for part of the training-time
   bias.** The λ advantage shrinks as W grows (+11.94 pp → +8.01 pp), because the
   window captures more of the natural locality λ was amplifying. **Any Phase 7
   comparison must give both arms the same W**, or the thesis overstates λ by
   handing the control a deliberately weak prefetcher.

## 4. LRU replay (global (layer, expert) budget, 20 accesses/token)

| capacity | l00 hit | l05 hit | Δ (pp) | l00 B/tok | l05 B/tok | Δ bytes |
|---|---|---|---|---|---|---|
| 32 blk / 38MB | 13.27% | 25.93% | **+12.65** | 20.53 | 17.53 | −14.59% |
| 48 / 57MB | 20.57% | 32.03% | +11.46 | 18.80 | 16.09 | −14.43% |
| 64 / 76MB | 27.14% | 37.82% | +10.69 | 17.25 | 14.72 | −14.67% |
| 80 / 95MB | 33.00% | 42.82% | +9.82 | 15.86 | 13.53 | −14.65% |
| 96 / 114MB | 38.27% | 47.48% | +9.22 | 14.61 | 12.43 | −14.93% |
| 128 / 151MB | 48.54% | 56.61% | +8.08 | 12.18 | 10.27 | −15.69% |
| 160 / 189MB | 58.51% | 65.35% | +6.84 | 9.82 | 8.20 | −16.48% |
| 192 / 227MB | 68.06% | 73.69% | +5.63 | 7.56 | 6.23 | −17.63% |
| 224 / 265MB | 77.26% | 81.66% | +4.40 | 5.38 | 4.34 | −19.34% |
| 256 / 303MB | 85.92% | 88.99% | +3.07 | 3.33 | 2.61 | −21.79% |
| 288 / 341MB | 93.80% | 95.34% | +1.54 | 1.47 | 1.10 | −24.81% |
| 320 / 379MB | 99.99% | 99.99% | 0.00 | 0.00 | 0.00 | 0.00% |

λ=0.5 wins at **every** budget below full residency, and the advantage is largest
when RAM is tightest — the regime the project actually targets. The two converge
only when everything fits, as they must.

## Verdict against the committed Q1 bar

Committed 2026-09-25 *before measuring*: ≥10 pp hit rate **or** ≥25% fewer
bytes/token, at a fixed budget, vs λ=0+LRU.

- best hit-rate delta: **+12.65 pp** at 38MB, 95% CI [+12.46, +12.88] → **PASS**
- best bytes/token delta: −24.81% at 341MB (just short of −25%; the OR is met on hit rate)

**Q1: PASS. Mechanism: λ raises temporal reuse by 19.53 pp.**

## Caveats — what this does NOT establish

1. **Quality is unresolved.** The λ=0.5 arm degenerated into token loops under
   the logged greedy sampling protocol (every sample from step 120k to 180k),
   while λ=0 stayed coherent. If that collapse is real, **the engine build is
   wasted regardless of this hit-rate win.** The pre-registered Q2 sampled check
   (temperature ≈0.8, multi-prompt) is still the other half of the go/no-go.
2. **Simulated, not real.** This is LRU over recorded traces, not the Phase 3
   pager. The real pager has prefetch, madvise semantics, and system-cache
   effects (Exp 2) that this does not model.
3. **Single corpus slice**, 250k tokens, one seed. Worth re-running on a second
   shard before publishing.
4. **No tok/s claim.** The T480 is compute-bound at this size; a hit-rate win need
   not produce a tok/s win (this is why Q3 exists).
5. **The λ=0.5 quality gap on val loss is small** (+0.84% ppl final), so the
   quality cost of the mechanism is cheap on teacher forcing — the risk is
   entirely in free-running generation.

## Reproduce

```sh
# on the server (GPU), per arm:
python tools/record_routing.py --ckpt ckpt/l05/model_only.pt \
  --data data/fineweb_edu_valid.bin --out traces/l05_fp32.npz \
  --tokens 250000 --precision fp32
# on the T480:
python tools/analyze_routing.py --traces data/traces --precision fp32
```
