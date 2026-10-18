# reshape/ — model-side edits for this port

**Empty, by design.** gpt2-moe is a **Tier A (as-is)** port: the weight file carries the model
as trained, and no reshaped variant is shipped.

Every script added here must declare its class on the first line of its module docstring:

| Class | Meaning | Gate |
|---|---|---|
| `BIT_IDENTICAL` | a permutation or re-layout that provably does not change the model (expert reorder, alignment) | must pass the base model's reference unchanged — routing exact, logits exactly 0 |
| `ROUTING_CHANGED` | changes what the model computes (λ fine-tune, top-k change) | cannot be checked against the base model; needs its own recorded routing trace as reference |

Without that declaration a faithful port and a modified model are indistinguishable by
inspecting the tree — which is why the class is a requirement and not a convention.

Tier-B capability is still gated for this model: `tools/verify_permutation_cpw1_v3.py` proves
expert reordering with a matching router permutation leaves routing exactly invariant, so the
model stays able to express a layout change even though none is applied.
