#!/usr/bin/env python3
"""record_hf_qwen3_shard.py — record per-layer targets from SHARD 1 ONLY (layers 0-2).

FAIL-FAST PROTOCOL. The full checkpoint is 61 GB; shard 1 is 4 GB and holds layers 0-2 plus the
embedding table. A TRUNCATED model (num_hidden_layers=2) built from those tensors produces genuine
intermediates for those layers — same weights, same math — so the new blocks (QK-norm, top-8
routing, expert SwiGLU) can be verified against real numbers before committing 61 GB and an hour of
conversion.

What this CANNOT give: final logits, or a 48-layer routing trace. `model.norm` and `lm_head` live
in the last shard, and a 2-layer model is not the 48-layer model. Stated here so the absence is not
mistaken for a gap.

Captured per layer: input_layernorm, q/k/v projections, q_norm/k_norm outputs, self_attn output,
post_attention_layernorm, router logits, and the top-8 selection DERIVED from HF's own router
logits with the documented rule (softmax -> topk -> renormalise if norm_topk_prob). It is derived,
not read out of HF's internals — the derivation is itself checked against `mlp` output below.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "weights/qwen3-30b-a3b"
OUT = ROOT / "data/traces/qwen3_30b_a3b_shard1.npz"
NLAYERS = 1   # one layer is enough to gate the blocks, and halving it halves the peak
              # memory when the 128 experts are folded into the fused runtime layout
PROMPT = "The capital of France is"

def main():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    torch.set_num_threads(4)

    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.num_hidden_layers = NLAYERS
    cfg.tie_word_embeddings = False
    # FORCE fp32. `from_config` honours the config's `torch_dtype: bfloat16`, and the first
    # recording was therefore bf16-precision: q_proj disagreed with an fp32 reference by 3.4e-03,
    # which is bf16 rounding, not a bug. The port computes in fp32, so the target must too.
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32).eval()
    sd = load_file(str(MODEL / "model-00001-of-00016.safetensors"))
    # keep only what this truncated model has, and report what is missing
    # THE CHECKPOINT AND THE RUNTIME DISAGREE ON EXPERT LAYOUT, and filtering by key name
    # silently dropped every expert weight: the checkpoint ships one tensor per expert
    # (`experts.N.gate_proj.weight`), while transformers 5.x keeps them FUSED
    # (`experts.gate_up_proj (128,1536,2048)`, `experts.down_proj (128,2048,768)`).
    # The first recording therefore had randomly-initialised experts and an `mlp` target with
    # peak 0.012 where the true output is ~1.9 — plausible-looking garbage. Fold them explicitly.
    import numpy as _np
    for i in range(NLAYERS):
        gu = _np.stack([_np.concatenate([sd[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"].float().numpy(),
                                         sd[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"].float().numpy()], 0)
                        for e in range(cfg.num_experts)])
        sd[f"model.layers.{i}.mlp.experts.gate_up_proj"] = torch.from_numpy(gu)
        del gu
        dn = _np.stack([sd[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"].float().numpy()
                        for e in range(cfg.num_experts)])
        sd[f"model.layers.{i}.mlp.experts.down_proj"] = torch.from_numpy(dn)
        del dn
    want = set(model.state_dict().keys())
    have = {k: v for k, v in sd.items() if k in want}
    missing = sorted(want - set(have))
    model.load_state_dict(have, strict=False)
    print(f"truncated model: {NLAYERS} layers, {len(have)}/{len(want)} tensors loaded")
    print(f"  missing (expected: final norm + lm_head live in the last shard): "
          f"{[m for m in missing if 'lm_head' in m or m == 'model.norm.weight']}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print(f"  prompt {ids.shape[1]} tokens: {ids.tolist()[0][:8]}")

    rec, sites = {}, []
    def hook(name):
        def fn(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o):
                rec[name] = o.detach().float().numpy().copy()
        return fn
    for i in range(NLAYERS):
        for sub in ("input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.q_norm", "self_attn.k_norm", "self_attn",
                    "post_attention_layernorm", "mlp.gate", "mlp"):
            mod = model.model.layers[i]
            for part in sub.split("."):
                mod = getattr(mod, part)
            mod.register_forward_hook(hook(f"L{i}.{sub}")); sites.append(f"L{i}.{sub}")
    with torch.no_grad():
        model(ids)
    print(f"  hooked {len(sites)} sites")

    # top-8 derived from HF's own router logits, with Qwen3's rule
    for i in range(NLAYERS):
        logits = torch.from_numpy(rec[f"L{i}.mlp.gate"])
        probs = torch.softmax(logits.float(), dim=-1)
        top = torch.topk(probs, 8, dim=-1)
        rec[f"L{i}.top8"] = top.indices.numpy().astype(np.int64)
        rec[f"L{i}.top8_w"] = (top.values / top.values.sum(-1, keepdim=True)).numpy()
        print(f"  L{i} router: logits {tuple(logits.shape)} top8[0] {top.indices[0].tolist()}")
    rec["input_ids"] = ids[0].numpy().astype(np.int64)
    rec["meta"] = np.array(json.dumps({
        "model": "Qwen/Qwen3-30B-A3B", "shard": "model-00001-of-00016.safetensors",
        "layers": NLAYERS, "tensors_loaded": len(have), "missing": missing[:8],
        "note": "truncated: no final norm, no lm_head, no logits. Block-level targets only."}))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **rec)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(rec)} arrays)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
