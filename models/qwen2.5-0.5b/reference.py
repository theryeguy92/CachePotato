#!/usr/bin/env python3
"""reference.py — independent numpy forward for Qwen2.5-0.5B-Instruct.

Written from models/qwen2.5-0.5b/MODEL.md and the recorded HF intermediates, NOT by calling
transformers. The whole point is that it is a second implementation: if it and HF disagree, the
disagreement is real and one of them is wrong. A reference that wraps the library cannot
disagree with it.

Gated block by block against data/traces/qwen25_05b_intermediates.npz:
    RMSNorm  -> L{i}.input_layernorm / L{i}.post_attention_layernorm
    RoPE     -> rotary.0 (cos) / rotary.1 (sin)
    GQA      -> L{i}.self_attn
    SwiGLU   -> L{i}.mlp
    full     -> logits in qwen25_05b_hf.npz

Run `python3 models/qwen2.5-0.5b/reference.py --check` to gate every block at once.
"""
import numpy as np

EPS = 1e-6          # rms_norm_eps, VERIFIED from config.json


def rmsnorm(x, g, eps=EPS):
    """x / sqrt(mean(x^2) + eps) * g.  No mean subtraction, no bias — this is NOT LayerNorm,
    and confusing the two produces a model that runs and is subtly wrong."""
    var = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * g


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_tables(T, head_dim, theta=1_000_000.0):
    """cos/sin in HF's layout: [c_0..c_{d/2-1}, c_0..c_{d/2-1}], float32."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / head_dim))
    pos = np.arange(T, dtype=np.float64)
    emb = np.outer(pos, inv)                      # (T, half)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return np.concatenate([cos, cos], axis=-1), np.concatenate([sin, sin], axis=-1)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rope(q, k, cos, sin):
    """q,k: (T, heads, head_dim). cos/sin are (T, head_dim) and broadcast over heads."""
    c = cos[:, None, :]
    s = sin[:, None, :]
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


def attn_block(x, w, cfg):
    """GQA: n_heads query heads against n_kv key/value heads; head h uses kv head h//rep."""
    D, H = cfg["d_model"], cfg["n_heads"]
    KV, HD = cfg["n_kv"], cfg["head_dim"]
    T = x.shape[0]
    xn = rmsnorm(x, w["ln1"])
    q = xn @ w["Wq"].T + w["bq"]          # (T, H*HD)
    k = xn @ w["Wk"].T + w["bk"]          # (T, KV*HD)
    v = xn @ w["Wv"].T + w["bv"]
    q = q.reshape(T, H, HD)
    k = k.reshape(T, KV, HD)
    v = v.reshape(T, KV, HD)
    cos, sin = rope_tables(T, HD, cfg["rope_theta"])
    q, k = apply_rope(q, k, cos, sin)
    rep = H // KV
    k = np.repeat(k, rep, axis=1)          # (T, H, HD)
    v = np.repeat(v, rep, axis=1)
    # (H, T, T) scores
    qh = q.transpose(1, 0, 2)
    kh = k.transpose(1, 0, 2)
    vh = v.transpose(1, 0, 2)
    scores = np.einsum("htd,hsd->hts", qh, kh) / np.sqrt(HD)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask[None], -np.inf, scores)
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    attn = e / e.sum(axis=-1, keepdims=True)
    ao = np.einsum("hts,hsd->htd", attn, vh).transpose(1, 0, 2).reshape(T, H * HD)
    return ao @ w["Wo"].T


def attention(x, w, cfg):
    """Full sub-layer. HF's module hook returns the PRE-residual output, so the two are kept
    separate: comparing x + block(x) against block(x) looks like a 13% error and is not one."""
    return x + attn_block(x, w, cfg)


def mlp_block(x, w):
    """down(silu(gate(x)) * up(x)) — three matrices, not two. Pre-residual."""
    xn = rmsnorm(x, w["ln2"])
    return (silu(xn @ w["Wg"].T) * (xn @ w["Wu"].T)) @ w["Wd"].T


def swiglu(x, w):
    return x + mlp_block(x, w)


def forward(tokens, w, cfg):
    x = w["embed"][tokens]
    for l in range(cfg["n_layers"]):
        x = attention(x, w["L%d" % l], cfg)
        x = swiglu(x, w["L%d" % l])
    x = rmsnorm(x, w["norm"])
    return x @ w["embed"].T              # tied embeddings


def load(model_dir):
    """Load weights. torch is used ONLY as the container reader: the published safetensors are
    bfloat16, which numpy cannot read, and bf16 -> fp32 is exact. The architecture below never
    calls into torch — that separation is what lets this file disagree with transformers."""
    import json
    import torch
    from safetensors.torch import load_file
    sd = {k: v.to(torch.float32).numpy() for k, v in load_file(f"{model_dir}/model.safetensors").items()}
    cfg = json.load(open(f"{model_dir}/config.json"))
    c = dict(d_model=cfg["hidden_size"], n_heads=cfg["num_attention_heads"],
             n_kv=cfg["num_key_value_heads"], n_layers=cfg["num_hidden_layers"],
             head_dim=cfg["hidden_size"] // cfg["num_attention_heads"],
             rope_theta=cfg.get("rope_theta", 10000.0))
    w = {"embed": sd["model.embed_tokens.weight"].astype(np.float32),
         "norm": sd["model.norm.weight"].astype(np.float32)}
    for l in range(c["n_layers"]):
        p = f"model.layers.{l}."
        w["L%d" % l] = dict(
            ln1=sd[p + "input_layernorm.weight"].astype(np.float32),
            ln2=sd[p + "post_attention_layernorm.weight"].astype(np.float32),
            Wq=sd[p + "self_attn.q_proj.weight"].astype(np.float32),
            Wk=sd[p + "self_attn.k_proj.weight"].astype(np.float32),
            Wv=sd[p + "self_attn.v_proj.weight"].astype(np.float32),
            Wo=sd[p + "self_attn.o_proj.weight"].astype(np.float32),
            bq=sd[p + "self_attn.q_proj.bias"].astype(np.float32),
            bk=sd[p + "self_attn.k_proj.bias"].astype(np.float32),
            bv=sd[p + "self_attn.v_proj.bias"].astype(np.float32),
            Wg=sd[p + "mlp.gate_proj.weight"].astype(np.float32),
            Wu=sd[p + "mlp.up_proj.weight"].astype(np.float32),
            Wd=sd[p + "mlp.down_proj.weight"].astype(np.float32))
    return c, w


def check():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    z = np.load(root / "data/traces/qwen25_05b_intermediates.npz")
    c, w = load(str(root / "weights/qwen2.5-0.5b-instruct"))
    ids = z["input_ids"]
    T = len(ids)
    print("=" * 78); print("REFERENCE: block-by-block vs HF intermediates"); print("=" * 78)
    bad = 0
    def report(name, got, want, tol=2e-5):
        nonlocal bad
        d = float(np.abs(got - np.asarray(want, dtype=np.float32)).max())
        peak = max(float(np.abs(want).max()), 1e-30)
        ok = d / peak < tol
        print(f"  {name:26} max|d| {d:.3e}  rel {d/peak:.2e}  {'ok' if ok else 'FAIL'}")
        if not ok: bad += 1

    x = w["embed"][ids]
    # RMSNorm: both sites, layer 0
    report("RMSNorm L0 site1", rmsnorm(x, w["L0"]["ln1"]),
           z["L0.input_layernorm"][0])
    # RoPE tables
    cos, sin = rope_tables(T, c["head_dim"], c["rope_theta"])
    report("RoPE cos", cos, z["rotary.0"][0])
    report("RoPE sin", sin, z["rotary.1"][0])
    # full block stack
    for l in range(c["n_layers"]):
        want = z["L%d.self_attn.0" % l][0] if "L%d.self_attn.0" % l in z.files \
               else z["L%d.self_attn" % l][0]
        got = attn_block(x, w["L%d" % l], c)
        if l in (0, 11, 23):
            report(f"GQA attention L{l}", got, want)
        x_after_attn = x + got
        xn2 = rmsnorm(x_after_attn, w["L%d" % l]["ln2"])
        if l == 0:
            report("RMSNorm L0 site2", xn2, z["L0.post_attention_layernorm"][0])
        got2 = mlp_block(x_after_attn, w["L%d" % l])
        if l in (0, 11, 23):
            report(f"SwiGLU mlp L{l}", got2, z["L%d.mlp" % l][0])
        x = x_after_attn + got2
    report("final RMSNorm", rmsnorm(x, w["norm"]), z["norm"][0])
    # The end-to-end target comes from the BASELINE file, whose logits are recorded as
    # (T, vocab). The intermediates file stores a 1-D logits array, and comparing a (T, vocab)
    # against a (vocab,) does not fail — numpy BROADCASTS, and the gate compares every row
    # against one row and reports garbage. Shape-check before comparing.
    base = np.load(root / "data/traces/qwen25_05b_hf.npz")
    logits = rmsnorm(x, w["norm"]) @ w["embed"].T
    want = base["raw_logits"]
    assert logits.shape == want.shape, (logits.shape, want.shape)
    report("FINAL logits vs HF", logits, want, tol=5e-5)
    print()
    print("REFERENCE: PASS" if not bad else f"REFERENCE: FAIL ({bad} block(s))")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        raise SystemExit(check())
    print(__doc__)
