# qwen3-30b-a3b — the first sparse open-weight port

**Tier: not yet assigned.** It cannot be Tier A or B until the tier's gate can be run, and that
requires the blocks below to exist. Declaring a tier now would be a promise, not a fact.

This is the port that tests the project's actual thesis: a 30B-class model that **cannot fit in
16 GB of RAM**, whose per-token cost depends on *which experts* a token selects. Paging is the
only reason it can run here at all — and read count is exactly what Tier B claims to improve.

**Model:** `Qwen/Qwen3-30B-A3B`, Apache-2.0. Not yet downloaded.

## 1. Architecture facts — VERIFIED from config.json and the safetensors headers

Read on 2026-10-17. "Header" means the JSON header of a shard, read via an HTTP range request —
so the **shapes are from the published weights**, not from a config field that claims them.

| Fact | Value | Status |
|---|---|---|
| architecture | `Qwen3MoeForCausalLM`, `model_type: qwen3_moe` | VERIFIED — config |
| layers | 48, **every one a MoE layer** (`mlp_only_layers: []`, `decoder_sparse_step: 1`) | VERIFIED — config + header: layer 0 has `mlp.gate` + `mlp.experts.*`, and no layer-level dense `down_proj` |
| `hidden_size` | 2048 | VERIFIED |
| attention | 32 heads / **4 KV heads**, `head_dim` 128 → **GQA 8:1** | VERIFIED — `q_proj [4096,2048]`, `k/v_proj [512,2048]` |
| **QK-norm** | **`q_norm` and `k_norm`, `[128]` each** — per-head RMSNorm on q,k | **VERIFIED from the WEIGHTS** — config has no `qk_norm` field at all (see §3) |
| experts | **128**, `top-8`, `norm_topk_prob: true`, softmax scoring (no `scoring_func` key) | VERIFIED — config + `mlp.gate [128, 2048]` |
| expert FFN | SwiGLU: `gate [768,2048]`, `up [768,2048]`, `down [2048,768]` | VERIFIED — header |
| **shared expert** | **none** — no `shared_expert` keys anywhere in the index | VERIFIED — index scan |
| dense FFN | none executed; `intermediate_size: 6144` appears unused | VERIFIED — no layer-level dense projections in the header |
| `lm_head` | **untied** (`tie_word_embeddings: false`), separate tensor in the last shard | VERIFIED — index |
| norm | RMSNorm, `rms_norm_eps = 1e-06` | VERIFIED |
| position | RoPE, `rope_theta = 1e6`, `rope_scaling: null`, **full** rotary (no partial factor) | VERIFIED |
| sliding window | `null`, `use_sliding_window: false` → full causal | VERIFIED |
| vocab | 151936, `max_position_embeddings` 40960 | VERIFIED |
| checkpoint | **61.1 GB** bf16, 16 shards, **18 867 tensors** | VERIFIED — index |
| expert tensors | weights only, **no biases** | VERIFIED — header |

## 2. Size and speed arithmetic, done before writing code

Active parameters per token (attention + router + 8 of 128 experts + head):

| piece | params/token |
|---|---:|
| attention (q,k,v,o) × 48 | 0.91 B |
| router × 48 | 0.013 B |
| **8 experts × 3 matrices × 48 layers** | **1.81 B** |
| embed + untied lm_head | 0.62 B |
| **total active** | **≈ 3.35 B** |

That is **3.35 B × 1.0625 B/param ≈ 3.56 GB read per token at Q8**. At the measured 7.3 GB/s that
is **2.0 tok/s**; at the ~13 GB/s ceiling of the GEMV bench, **3.6 tok/s**. Slow, and *readable* —
which is all a paging experiment needs.

Storage, and why paging is mandatory rather than optional:

| encoding | size | fits in 16 GB? |
|---|---:|---|
| bf16 (published) | 61.1 GB | no |
| Q8_0 | ~32 GB | no |
| Q4-class (≈4.5 bpw) | ~17 GB | **no** — marginal, and no room for KV cache |
| 2-bit | ~8 GB | yes, but no engine support |

So this model **cannot be run without paging**, by construction. That is the point.

## 3. What the engine must learn — and the third config that lied

Missing blocks, in dependency order:

1. **QK-norm** — per-head RMSNorm applied to q and k *before* RoPE. Small, and absent from the
   config; found only by reading tensor names.
2. **top-8 of 128 experts** — the engine's MoE block is hard-wired to top-2. Generalising it is
   mechanical but touches the tie-break rule, the weights normalisation and the trace format.
3. **No shared expert** — the block assumes exactly one. Needs `n_shared = 0` to be expressible,
   which the spec's `n_shared` parameter already allows, but the block does not implement.
4. **Untied `lm_head`** — the engine assumes tied unembedding everywhere.
5. **All 48 layers MoE** — the engine's `dense_slot` machinery assumes some layers are dense.
6. **Pager integration in the Qwen path** — `engine/blocks/qwen2.c` currently just `mmap`s and
   lets the OS page cache decide. For this model the pager *is* the product; without it there is
   no experiment, only a very slow mmap.

**The config lied about QK-norm, and this is the third time.** `config.json` has no field naming
it, yet every layer carries `q_norm.weight` and `k_norm.weight`. The pattern is now well
established: **gpt2-moe** (no `moe_from` in the checkpoint), **Qwen2.5** (no `attention_bias`, but
q/k/v biases exist), **Qwen3** (no `qk_norm`, but q/k norms exist). A config describes intent; the
weights describe reality. Every fact in §1 is therefore cross-checked against a tensor.

## 4. Plan, in the order that has worked

1. Record the HuggingFace baseline (logits for a fixed prompt, bit-identical across two runs) and
   **per-layer intermediates**, so each new block can be verified alone.
2. Write an independent numpy reference; it must reproduce those logits. Four blocks were found
   this way for Qwen2.5 and none of them were found by reading code.
3. Port the blocks, gated against the reference.
4. Wire the **pager** into the Qwen path, then measure:
   - tok/s for a model that does **not** fit,
   - **regions fetched per token**, with and without a Tier-B layout.
5. Only then assign a tier.

## 5. Known limits

- **16 shards, 61 GB.** The download alone is significant, and every conversion is another 30-60
  GB on disk. Needs planning before it starts, not during.
- **No shared expert** means the expert pool is the entire FFN — 6144 expert blocks across 48
  layers. That is a much harder paging problem than gpt2-moe's 320, and it is the real test.
- **Q4-class encoding is not settled.** A symmetric group-32 scheme was tried on the 0.5B model and
  was not good enough there; this model needs a *better* 4-bit scheme (or 2-bit) to fit, and the
  quality question has to be re-asked at 30B rather than assumed from 0.5B.

## 6. Tie-breaking is not an edge case here — it is the norm (measured 2026-10-18)

The baseline validator found that **every one of the 240 (layer, token) positions** in the recorded
routing trace has at least one tie at the 8th/9th expert boundary. Router logits are recorded in
bf16, which carries ~8 mantissa bits, so among 128 experts two probabilities land on *exactly* the
same value routinely. "The top 8" therefore has no unique answer at the boundary, and `torch.topk`
and `argsort` may legitimately choose different members.

This is not a nuisance to work around; it constrains what the port can claim:

1. **The C block must have a documented tie-break rule, and it must be deterministic.** The
   gpt2-moe block uses "strict `>`, lower expert id wins", which is a fine rule — but it has to be
   stated, because it decides 8 of 128 experts at almost every token.
2. **Tier B's `routing exactly invariant` claim is unaffected** — it compares our engine to *itself*
   under two layouts, so a fixed own-rule suffices. That is the strong claim and it survives.
3. **C-vs-HF routing agreement must be tie-aware.** Comparing exact index sets against this trace
   would report mismatches that are not errors. The comparison must be "the selected set is a valid
   top-8 given ties" — every chosen expert at least as likely as the 8th largest, and no unchosen
   expert strictly more likely. `tools/check_qwen3_baseline.py` implements exactly this rule and is
   the reference for what "agreement" means.
4. **Recorded weights are unambiguous where recorded indices are not.** They matched
   softmax+renormalise to 1.79e-07, which is what proved the trace's indices correct in the first
   place. Any future routing check should use *both*: the set must be valid given ties, and the
   weights must match.

Practical consequence for B5: expected routing agreement with HF is not a percentage to be maximised
— it should be **exact under the tie-aware rule**, and anything less is a real bug rather than
rounding.
