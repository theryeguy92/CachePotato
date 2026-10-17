# Port contract — what `models/<port>/` must contain

A **port** is everything needed to run one model on the engine. The engine knows nothing about
any specific model; a port supplies the model-specific parts. This document is the contract,
and it exists because writing it revealed three things that a port *must* state which nothing
in the format records.

**A port is classified into exactly one tier**, and the tier decides which gates apply
(`docs/2026-10-16-universal-ports.md`):

| Tier | Changes | Numerics |
|---|---|---|
| **A — as-is** | conversion only | exact vs the base model |
| **B — layout** | expert order, regions, alignment; **no retraining** | exact vs the base model |
| **C — co-design** | router weights | exact only vs its own recorded reference |

## Required files

```
models/<port>/
  MODEL.md        architecture facts, each VERIFIED (source named) or ASSUMED — required
  convert.py      source checkpoint -> CPW1 v4 via format/v4_writer.py — required
  reference.py    independent forward; the numerical oracle — required
  verify/gate.py  this port's gates — required
  reshape/        optional; every script declares BIT_IDENTICAL or ROUTING_CHANGED
```

`MODEL.md` must state the **tier**, the **fact table**, and — this is the part that is easy to
skip — the **assumptions**, with the concrete consequence of each being wrong.

## What "independent reference" means, and why it is not negotiable

`reference.py` is a second implementation, written from `MODEL.md` and the block descriptions
in `format/SPEC.md`. It must not import the trainer that produced the checkpoint.

The reason is a failure this project has already had: on 2026-09-25 the two implementations
(F trainer, C engine) **agreed with each other, disagreed with the frozen architecture spec,
and every gate passed**. A gate that compares two copies of one understanding cannot detect a
misreading. Only a separately written implementation, anchored to the recorded behaviour of
the trained model, can.

The port's strongest single number is therefore **reference vs recorded routing**, which must
be exact. For gpt2-moe it is **20480/20480 (100.0000%)**.

## The three things this contract had to add after doing it once

Writing the first port exposed three facts that are *needed to load a model correctly* and
that **neither the checkpoint nor the format records**. Every future port must state each one
explicitly in `MODEL.md`, because getting any of them wrong does not crash — it produces a
model that runs and emits fluent garbage.

1. **The layer split may not be in the checkpoint.** gpt2-moe's `moe_from` is absent from the
   checkpoint config; the as-built value is the trainer's default. A raw checkpoint is
   therefore *not* authoritative about its own architecture. The exported CPW1 file is (the v3
   header carries `moe_from`; a v4 file carries it as the block table). A port reading a raw
   checkpoint must mark the split **ASSUMED** and cross-check it — gpt2-moe does so twice, via
   the v3 header and by the reference reproducing the recorded routing only with split 1.

2. **Dead weights may be present and must be dropped.** gpt2-moe's checkpoint carries a dense
   FFN on *every* layer (12), including the MoE layers where those weights were never trained;
   the file carries the 2 dense layers. 68.6M params are stripped. A converter must know which
   side of that difference it is on, and `convert.py` now refuses a checkpoint whose dense-FFN
   group does not have one slot per layer, because "already compacted" and "never compacted"
   look the same in a file and differ by a silent 10× misindex.

3. **A model-level scalar may have no home in the format.** `lam` (the λ locality bias) is part
   of gpt2-moe's behaviour: serving it at a different λ routes differently. CPW1 v3 stores it
   in the header. **CPW1 v4 has no field for it** — an open spec question, deliberately not
   patched by reusing a spec'd block parameter (that collision was a bug once already). Until
   the format has a home for it, a v4 file must be driven with `CP_V4_LAMBDA`, and a port whose
   behaviour depends on a scalar in this position must say so in `MODEL.md`.

## Gates

A port passes when all of these hold. Nothing here is relaxed for ports.

| Gate | Requirement |
|---|---|
| Reference vs recorded routing | **exact** — 0 mismatches on held-out windows |
| C engine vs reference | **exact** on routing; logits to a stated tolerance |
| Format acceptance | v3→v4 (or source→v4) round-trip identical, including a shuffled payload |
| Tier-B permutation | expert reorder + router permutation leaves routing invariant |
| Refusal | the loader refuses an unsupported block kind or encoding rather than approximating |

## What a port does NOT have to do

- Ship the C trainer. Only gpt2-moe has one, and it exists to verify numerics, not to train.
- Support every architecture. A port that needs a block the engine lacks is **refused loudly**
  at load (`engine/core/weights_v4.c` lists what is implemented: MHA with
  `head_dim = d_model/n_heads`, gelu-tanh dense FFN, softmax top-2 routed MoE with 1 shared
  expert; `moe_latent`, `ssm`, GQA, sigmoid/group-limited routing and quantized encodings are
  specified but not implemented).
- Use the engine's strided layout. `convert.py` in gpt2-moe emits engine order so the loader
  can use the payload **in place**; any other order loads through the copy path and produces
  identical output. That is a performance choice, not a correctness one.
