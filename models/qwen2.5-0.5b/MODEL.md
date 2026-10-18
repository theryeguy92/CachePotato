# qwen2.5-0.5b-instruct — a real chat model

**Tier: A (as-is)** for the architecture; **Tier C** if quantized (a quantized port is a
different model and is gated on quality, never exactness).

This is the **first open-weight port**, chosen to be the smallest model that exercises the
blocks the engine was missing while needing **nothing else**. It is the opposite of gpt2-moe:
gpt2-moe proved the *port contract* on an architecture built to fit the engine; this proves the
engine can meet someone else's architecture.

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`, Apache-2.0. Downloaded to
`weights/qwen2.5-0.5b-instruct/` (gitignored).

## 1. Architecture facts — all VERIFIED from `config.json`

Read directly from the published config on 2026-10-17; nothing here is from memory.

| Fact | Value | Status |
|---|---|---|
| architecture | `Qwen2ForCausalLM`, decoder-only | VERIFIED — `architectures` |
| `vocab_size` | 151936 | VERIFIED |
| `hidden_size` (d_model) | 896 | VERIFIED |
| `num_hidden_layers` | 24 | VERIFIED |
| `num_attention_heads` | 14 → `head_dim` = 896/14 = **64** | VERIFIED |
| `num_key_value_heads` | **2** → **GQA 7:1** | VERIFIED |
| `intermediate_size` | 4864 | VERIFIED |
| `hidden_act` | **`silu`** (SwiGLU: `down(silu(gate(x)) * up(x))`) | VERIFIED |
| normalization | **RMSNorm**, `rms_norm_eps = 1e-6`, **no mean subtraction, no bias** | VERIFIED |
| positional | **RoPE**, `rope_theta = 1000000.0`, `rope_scaling` absent | VERIFIED |
| sliding window | `sliding_window: 32768` but **`use_sliding_window: false`** → full causal attention | VERIFIED |
| `max_position_embeddings` | 32768 | VERIFIED |
| embeddings | **tied** (`tie_word_embeddings: true`), no separate lm_head | VERIFIED |
| attention bias | **`q_proj`, `k_proj`, `v_proj` HAVE bias (896 / 128 / 128); `o_proj` does NOT** | VERIFIED from the **state_dict** — the config's `attention_bias` field is absent and misleading (see §3) |
| MLP bias | **none** on all three projections | VERIFIED from the state_dict |
| QK-norm | **none** (Qwen3 has it, Qwen2.5 does not) | VERIFIED — absence in config *and* state_dict |
| eos / bos | 151645 `<|im_end|>` / 151643 | VERIFIED — `tokenizer_config.json` |
| chat template | `<|im_start|>role\n … <|im_end|>` | VERIFIED — `tokenizer_config.json` |

## 2. What the engine must learn (and what it must not)

Four blocks, all new to this engine:

1. **RMSNorm** — `x / sqrt(mean(x²) + eps) * g`. No mean, no bias. Different from the
   LayerNorm the engine implements, and silently different results if confused with it.
2. **RoPE** — rotate q/k in 2-D pairs before attention, `theta = 1e6`. Applied to q and k per
   head, not to v.
3. **GQA** — 14 query heads against **2** key/value heads; query head `h` uses kv head
   `h / 7`. The KV cache is therefore 7× smaller than MHA, which is a real paging win.
4. **SwiGLU** — `W2 @ (silu(W1 x) * W3 x)`: **three** matrices per FFN, not two. The engine's
   dense FFN has `W1|W2`; this shape is different, not a setting.

What it does **not** need, which is why it was chosen: no MoE, no QK-norm, no rope scaling, no
sliding window, no biases at all, no separate lm_head. Every one of those was screened out
against a real config rather than assumed.

## 3. The bias question — the config lies, the weights do not

`config.json` has no `attention_bias` key, and a port that trusted that would conclude "no
biases". **The state_dict disagrees:** every layer carries `self_attn.q_proj.bias` `(896,)`,
`k_proj.bias` `(128,)` and `v_proj.bias` `(128,)`, while `o_proj` and all three MLP projections
have no bias at all. In Qwen2's modelling code the q/k/v biases are unconditional; the config
flag only gates the MLP (and is unset here).

This is the second model-level fact in this project that a config field got wrong, and the
lesson is the same one as the layer split in gpt2-moe: **a config describes intent, the weights
describe reality.** Every fact in §1 is therefore cross-checked against a real tensor shape.

Consequence for the port: the engine's `matmul_xw` takes a bias and Qwen2.5 needs biases on
q/k/v but not on o/MLP. The converter materialises **zeros** where a bias is absent (~0.4 MB per
layer for the MLP, negligible against 988 MB) rather than adding a NULL-bias branch to every
kernel. The gate proves the values are identical either way, so the choice costs nothing but a
few megabytes.

## 4. Files this port ships

```
models/qwen2.5-0.5b/
  MODEL.md        this file
  convert.py      safetensors -> CPW1 v4, with RoPE precomputation and tied embeddings
  reference.py    independent numpy forward (RMSNorm, RoPE, GQA, SwiGLU)
  verify/gate.py  reference vs the HF transformers logits, then C vs reference
```

## 5. Verification plan

The gpt2-moe port could be gated against routing recorded from its own trainer. This model is
dense, so there is no routing to record — but the reference is stronger in one respect: **the
published weights are the target**, checked through a library that is not ours.

**The target is recorded, not recomputed.** `tools/record_hf_baseline.py` wrote
`data/traces/qwen25_05b_hf.npz` (21.6 MB, gitignored like the other traces):

| field | contents |
|---|---|
| `raw_logits` | (5, 151936) fp32 for `"The capital of France is"` |
| `chat_logits` | (34, 151936) fp32 for the `<\|im_start\|>` templated form |
| `*_greedy` | a 16-token greedy continuation from those logits |
| `*_top_ids/_vals` | top-32 per position, readable without numpy |
| `meta` | torch version, dtype, thread count, **weights sha256** |

Verified **bit-identical across two independent recordings** (max|Δ| = 0.000e+00), which is the
property that makes it a target: a baseline that drifts between runs lets a drifting port pass.

Recorded behaviour: the raw prompt continues `" Paris. It is the largest city in Europe…"`; the
chat form answers `"The capital of France is Paris.<|im_end|>"`. So the model is credible and
both the plain and the templated paths have a target.

**The gate order, and why it cannot be reordered:**

1. `reference.py` (numpy) vs `raw_logits` — a real cross-implementation check, since HF is not
   ours. **If this fails, the reference is wrong, not HF.**
2. The C engine vs `reference.py` — argmax exact, logits within the 5e-4 fp32-accumulation
   bound established on gpt2-moe. **If this fails, the engine is wrong, not the reference.**

**And the rule written into the recorder's own output:** *do not regenerate this file to make a
failing test pass.* Re-recording to match a buggy implementation is the one failure mode that
would make every downstream gate meaningless.

## 6. Speed expectation, stated before measuring

0.98 GFLOP/token (0.49B params, dense, incl. the 151936-wide tied head). At the measured
effective 2.6 GFLOP/s that is **0.38 tok/s**; at the Q8 kernel's 13 GFLOP/s it is **13 tok/s**.
The truth is in between and is the point of measuring. The head is unusually wide (136M MACs of
the 491M total), so tokenizer/vocab cost matters more here than in a MoE.

## 7. Known limits

- **0.5B is small.** It chats, but it is a demonstration of the engine, not an assistant to
  rely on. Quality is Qwen's, not ours.
- **No MoE.** This port deliberately does not exercise the project's central claim (expert
  paging). That is the *next* port's job, and this one exists to make that port's new blocks
  already proven.
- **Chat template is data, not code.** The REPL will need `--chat-template` support to wrap
  prompts in `<|im_start|>…<|im_end|>`; a raw completion prompt will produce a raw completion.
