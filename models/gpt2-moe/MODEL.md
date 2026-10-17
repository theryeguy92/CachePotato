# gpt2-moe — the first CachePotato port

**Tier: A (as-is).** The weight file carries this model unchanged; no reshape is applied and
no router training is involved. `reshape/` exists but is empty — see §7.

This is the project's own research model (Phase 4–7), ported through the port contract to
prove the contract is sufficient *before* an open-weight model is attempted. It is the only
model here with a complete Python reference, recorded routing traces and a served v3 file, so
it is the cheapest way to find out what a port actually needs.

**Source checkpoint:** `weights/moe-scale-run/l05/model_only.pt` (λ=0.5 arm, step 180,000).
The λ=0 arm (`l00`) is the ablation and is identical apart from `lam`.

## 1. Architecture facts

Marked **VERIFIED** (read out of the artifact or the code, source named) or **ASSUMED**
(needed, but not recorded anywhere authoritative). A port must not fill a gap from memory.

| Fact | Value | Status |
|---|---|---|
| Architecture family | decoder-only transformer, GPT-2 semantics | VERIFIED — `train_moe.py` docstring + this repo's spec |
| `vocab` | 50257 | VERIFIED — checkpoint `config` |
| `d_model` | 768 | VERIFIED — checkpoint `config` |
| `n_heads` | 12, `head_dim` = 768/12 = **64** | VERIFIED — `config` + tensor shapes |
| `d_ff` (dense FFN hidden) | 3072 | VERIFIED — `config` |
| `n_layer` | 12 | VERIFIED — `config` |
| `n_ctx` | 1024 | VERIFIED — `config` |
| `n_exp` (routed experts/layer) | 32 | VERIFIED — `config` |
| `exp_hid` (expert hidden) | 192 | VERIFIED — `config` |
| `topk` | 2 | VERIFIED — `config`; the engine implements top-2 only |
| `lam` (locality bias) | 0.5 (this arm) | VERIFIED — `config` + CPW1 v3 header |
| **layer split** | MoE on 1–10, dense on 0 and 11 | **ASSUMED** — see §3 |
| Normalization | LayerNorm, `eps = 1e-5`, learnable gain+bias | VERIFIED — `train_moe.EPS`, tensor shapes |
| Activation | GELU **tanh approximation** (`0.5x(1+tanh(√(2/π)(x+0.044715x³)))`) | VERIFIED — `train_moe.gelu_tanh` |
| Attention | multi-head, causal, `head_dim = d_model/n_heads`, no GQA, no qk-norm | VERIFIED — tensor shapes `(L,d,d)` |
| Positional encoding | learned absolute, `(n_ctx, d_model)` | VERIFIED — `pos_emb` present |
| Unembedding | **tied** with `token_emb` (no `lm_head` tensor) | VERIFIED — state_dict has none; `matmul_wf(token_emb)` in the engine |
| Shared expert | 1 per MoE layer, hidden = `d_model` (not `exp_hid`) | VERIFIED — `W1s (L,d,d)` |
| Router | linear `d_model -> n_exp`, **softmax** over scores (not sigmoid) | VERIFIED — engine `moe_ffn` |
| Top-k tie-break | **lower expert id wins** (strict `>`) | VERIFIED — `model.c` and `train.c` use the same scan |
| Routing weight | `p[sel] / sum(p[sel])` (softmax probs renormalized) | VERIFIED — engine `moe_ffn` |
| Locality set | previous token's selection at the same layer; **reset at EOS** | VERIFIED — spec decision 2026-09-15, both implementations |
| EOS token id | 50256 | VERIFIED — `train_moe.EOS` |
| Weight layout | `(in, out)` everywhere — no transposes | VERIFIED — `train_moe.py` docstring; matches C |
| Expert block layout | `W1e(d,eh)`, `b1e(eh)`, `W2e(eh,d)`, `b2e(d)` concatenated | VERIFIED — `tools/cpw1_v3.py:expert_slices` |

## 2. Tensors

27 logical tensors (the routed pool is one tensor in v3 and 320 per-expert tensors in v4):

```
token_emb (50257,768)  pos_emb (1024,768)
ln1_g/b (12,768)  Wq,bq,Wk,bk,Wv,bv,Wo,bo (12,768,768)/(12,768)
ln2_g/b (12,768)
W1,b1,W2,b2 (2,768,3072)/(2,3072)/(2,3072,768)/(2,768)     <- DENSE SLOTS ONLY
Wr (10,768,32)  br (10,32)
W1s,b1s,W2s,b2s (10,768,768)/(10,768)                      <- shared expert
experts (10,32,295872)                                     <- W1e|b1e|W2e|b2e
lnf_g/b (768)
```

**Dense-FFN arrays are indexed by DENSE SLOT, not by layer** (2 slots, not 12). The dead
dense-FFN weights that the trainer allocates on MoE layers are **not in the file** — 68.6M
params stripped. A loader that indexes `W1` by layer will read the wrong FFN.

## 3. The layer split is ASSUMED, and this is a real hazard

`moe_from` **is not in the checkpoint's config dict.** The as-built layout is the trainer's
`Config` default of 1. The exported CPW1 v3 header does carry it (`moe_from=1`), so the FILE
is authoritative and a raw checkpoint is not.

Consequence, recorded because it is exactly the class of bug this project has been bitten by:
a port that reads only a raw checkpoint has to assume the split, and assuming it *wrong* does
not crash — layer 11's dense weights are present but never trained, so loading them under a
different split emits fluent garbage. Full background: Decision-Log 2026-09-25.

For this port the assumption is safe because it is cross-checked twice: the v3 header says 1,
and `reference.py` reproduces the recorded routing 100% only with split 1.

## 4. Files a port ships

```
models/gpt2-moe/
  MODEL.md        this file — facts, tier, assumptions
  convert.py      checkpoint (+ optional v3 file) -> CPW1 v4
  reference.py    independent numpy forward; the numerical oracle
  verify/gate.py  this port's gates
  reshape/        empty: Tier A applies no reshape
```

## 5. The oracle is independent on purpose

`reference.py` is a **second implementation** in numpy, written from §1 and the block
descriptions in `format/SPEC.md`, with **no import of `train_moe`**. Every other gate in this
repo compares the C engine against the trainer, which cannot catch a misreading of the
architecture: both sides share it. That failure mode is not hypothetical here — on 2026-09-25
two implementations agreed with each other, disagreed with the spec, and every gate passed.

The result that makes this reference trustworthy: it reproduces the **routing recorded from
the trained model** 30720/30720 selections (100.0000%, 3 windows of 512 tokens). An
independently written forward pass reaching the same discrete decisions is a much stronger
statement than agreeing with the code it was derived from.

## 6. Gates this port must pass

| Gate | Requirement | Where |
|---|---|---|
| Reference vs recorded routing | **exactly** 0 mismatches | `verify/gate.py` |
| C engine vs reference | **exactly** 0 routing mismatches; logits to a stated tolerance | `verify/gate.py` |
| v4 round-trip | v3 → v4 → identical routing, including a shuffled payload | `tools/verify_v4_roundtrip.py` |
| Tier-B permutation | router permutation leaves routing invariant | `tools/verify_permutation_cpw1_v3.py` |

## 7. Reshape status

None. This is a Tier-A port: the file carries the model as trained. `reshape/` is empty by
design, and the expert-permutation gate is still re-run because Tier-B capability exists and
the model must remain able to express it.

## 8. Known limits

- **Not a useful assistant.** 736 MB research artifact at GPT-2-small scale; proves an
  architecture, not a model. Quality was checked only against a repetition-rate gate.
- **As-built layer split** deviates from the frozen architecture spec (see §3).
- **top-2 only**, softmax routing only, no latent MoE, no GQA — the engine refuses those
  rather than mis-loading them (see `format/SPEC.md` §7 and `engine/core/weights_v4.c`).
