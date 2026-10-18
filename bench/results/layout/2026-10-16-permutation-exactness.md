# Tier-B permutation gate — expert reordering is a valid layout operation

**Date:** 2026-10-16
**Branch:** `feat/universal-ports-tier-b`
**Gate:** `tools/verify_permutation_cpw1_v3.py`
**Artifact:** `weights/moe-scale-run/l05.cpw1v3` (λ=0.5, seed `20261016`, 10 per-layer permutations)
**Verdict:** **PASS** — with one correction to the claim it was expected to support.

## Why this was run

The direction taken on 2026-10-16 (`docs/2026-10-16-universal-ports.md`) makes weight
**layout** the port mechanism: a port reorders expert blocks in the file and re-emits the
router with its expert dimension permuted to match. Everything downstream — region
grouping, per-machine compilation, co-activation ordering — assumes that operation leaves
the model unchanged. That assumption was asserted nowhere, so it was tested on an artifact
that already exists and is already gated, before any format or engine work depends on it.

## What the operation is

With `perm` a permutation of the `N=32` experts at one MoE layer:

```
experts'[k] = experts[perm[k]]        Wr'[:,k] = Wr[:,perm[k]]        br'[k] = br[perm[k]]
```

This is the only form that is a symmetry: permuting the expert blocks **without** permuting
the router's columns changes which weights each expert id refers to, which is a different
model, not a layout change.

## Result

| # | Check | Result |
|---|---|---|
| 1 | Experts + router columns equal the checkpoint under `perm`, all 10 MoE layers | **exact** |
| 2 | Router scores equivariant at a controlled input (`s'[k] == s[perm[k]]`) | **max\|Δ\| = 0.0** |
| 2 | Top-k selection equivariant (`perm[idx_new] == idx_base`) | **exact**, all 10 layers |
| 3 | Full-sequence logits, base vs permuted (23-token fixture, EOS@8) | max\|Δ\| **5.722e-06**, relative **2.85e-07**, argmax **23/23 identical** |
| 4 | Routing traces, `perm[permuted ids]` vs base ids | **0 / 460 differ** |
| 5 | C engine routing, `perm[C(permuted)]` vs `C(base)` | **0 / 1280 differ** |

Supporting measurements: hidden state entering the first MoE layer is **bit-identical**
between the two files; the MoE layer output differs by `1.19e-07`; the C engine reproduces
its own Python reference **0/1280** on both the base and the permuted file (via
`tools/verify_cpw1_v3.py` and the `--moe-sel` comparison), so check 5 is a cross-file
symmetry test on an engine that is independently correct.

## The correction: float32-equivalent, not bit-identical

The plan predicted `max|Δ| = 0.000e+00`. The measured value is `5.722e-06`, i.e. ~2.4×
float32 epsilon relative to the largest logit. It is not noise from the harness — check 2
proves the router weights and scores are *exactly* equivariant, so the difference enters
through the **summation order inside the router matmul**: permuting `Wr`'s columns changes
the order in which the 768 products are accumulated, and the routed output accumulates in
selection order.

Consequence for how results are stated:

- **Routing is exactly invariant** under a layout permutation — no tolerance needed.
- **Logits are float32-equivalent, not bit-identical.** Argmax and every downstream
  discrete decision agreed on this fixture, but the correct claim is
  *"exact routing, fp32-equal logits"*, and any cross-layout logits comparison must use a
  stated tolerance (relative < 1e-6 used here) rather than demanding `0.0`.
- A *runtime* layout change (which regions are pageable, how they are grouped, in what
  order they are faulted) does not touch weight order at all, so it remains bit-identical.
  The tolerance applies only to permuting weights in the file.

## The trap that cost the most time: permutation direction

With `experts'[k] = experts[perm[k]]`, and `idx`/`idx'` the argmax of the original/permuted
scores,

```
perm[idx'] == idx          <-- correct
perm[idx]  == idx'         <-- WRONG, and fails on a correct file
```

This was got backwards twice while building the gate — once in a scratch harness, once in
check 5 of the gate itself — and both times it looked exactly like a real failure: 6.24%
agreement (chance level for 32 experts) on the recorded-trace comparison, and 1147/1280
disagreements in the C cross-check. It was neither a model bug nor an engine bug. The
cheap discriminator, now the first thing to do when this gate looks red:

```python
# synthetic, 5 lines, settles the direction in seconds
W = rng.standard_normal((7, 32)); perm = rng.permutation(32); x = rng.standard_normal((6, 7))
so, sn = x @ W, x @ W[:, perm]
assert np.abs(sn - so[:, perm]).max() == 0.0
assert np.array_equal(perm[np.argsort(-sn, axis=1)[:, :2]], np.argsort(-so, axis=1)[:, :2])
```

The gate's docstring records this so the next person does not spend the afternoon on it.
A related harness error made while debugging: generating probe inputs from a *shared* RNG
so that two runs silently compared different data. The gate now uses an RNG-free
deterministic input.

## What this licenses

- Expert reordering with a matching router permutation is a **valid, verified layout
  operation** on this model, exactly at the routing level and to fp32 rounding on logits.
- Region grouping, alignment and per-machine compilation may proceed on this basis.
- It does **not** license claiming bit-identical logits across two different weight
  orderings, and it does not license skipping the gate for a new port: a port with a
  different router (top-9, different normalization, tied scores) must re-run it, since
  tie-breaking is where exactness could still break.

## Reproducing

```sh
python3 tools/permute_experts_cpw1_v3.py \
    --src weights/moe-scale-run/l05.cpw1v3 \
    --dst scratch/l05_permuted.cpw1v3 --seed 20261016 \
    --report scratch/l05_permutation.json

python3 tools/verify_permutation_cpw1_v3.py \
    --base weights/moe-scale-run/l05.cpw1v3 \
    --permuted scratch/l05_permuted.cpw1v3 \
    --ckpt weights/moe-scale-run/l05/model_only.pt \
    --perm-json scratch/l05_permutation.json \
    --engine ./build/cachepotato --window 64
```

Both tools are deterministic (per-slot `default_rng(seed + slot)`); the permuted file is
byte-reproducible from the seed and its sha256 is recorded in the report.


## Correction (2026-10-17): the mechanism was misattributed

This note originally said the logits moved "because permuting `Wr`'s columns changes the order
in which the 768 products are accumulated". **That is wrong.** The router dot product sums over
the *feature* axis, which a permutation of experts does not touch; instrumenting the engine
showed the hidden state is **identical after layers 1 and 2**.

The layout-dependent reduction is the **router softmax normalisation**, which sums `exp(score)`
over the **expert axis** — exactly the axis the permutation reorders. Measured over 20,000
random permutations, the denominator changes by ~1 fp32 ulp in **68%** of cases. In the engine
the divergence first appears late (layer 8: probabilities `0.0407722071` vs `0.0407722108`),
growing to `1.3e-05` in the logits. The routed-expert accumulation is already canonical: it
sums in descending weight order, which is a function of the values rather than the layout.

The measurements and the verdict in this note stand unchanged — routing exactly invariant,
logits fp32-equivalent. Only the explanation was wrong. Making it bit-exact is **deferred**
(see `format/SPEC.md` §10), because the engine currently matches PyTorch's routing 100% and a
canonical softmax sum would shift it ~1 ulp away from PyTorch, risking that property for a
benefit nothing yet consumes.
