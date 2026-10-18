#!/usr/bin/env python3
"""reference.py — independent numpy forward for Qwen3-30B-A3B blocks, gated against shard 1.

Three blocks the engine has never had:
  **QK-norm**   per-head RMSNorm on q (32 heads) and k (4 heads) BEFORE RoPE. Found only in the
                weights; the config has no field naming it.
  **top-8 of 128** routing with softmax and norm_topk_prob renormalisation.
  **expert SwiGLU**  down(silu(gate(x)) * up(x)), three matrices per expert, no shared expert.

Gated block by block against data/traces/qwen3_30b_a3b_shard1.npz, which came from a TRUNCATED
2-layer model built from real weights. If a block disagrees here, the reference is wrong — not HF.

Run: python3 models/qwen3-30b-a3b/reference.py --check
"""
import numpy as np

N_HEADS, N_KV, HD = 32, 4, 128
EPS = 1e-6
THETA = 1_000_000.0
TOPK = 8

def rmsnorm(x, g, eps=EPS):
    """x / sqrt(mean(x^2) + eps) * g — no mean subtraction, no bias."""
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g

def softmax(x):
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)

def rope_tables(T, hd, theta=THETA):
    half = hd // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / hd))
    emb = np.outer(np.arange(T, dtype=np.float64), inv)
    c = np.cos(emb).astype(np.float32); s = np.sin(emb).astype(np.float32)
    return np.concatenate([c, c], -1), np.concatenate([s, s], -1)

def apply_rope(x, cos, sin):
    """x: (T, heads, hd). Rotate pairs (i, i+hd/2) — HF's rotate_half formulation."""
    half = x.shape[-1] // 2
    rh = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos[:, None, :] + rh * sin[:, None, :]

def attention(x, w):
    """GQA with QK-norm: q per query head (32), k per KV head (4), both normalised then rotated."""
    T = x.shape[0]
    q = x @ w["Wq"].T                      # (T, 4096)
    k = x @ w["Wk"].T                      # (T, 512)
    v = x @ w["Wv"].T                      # (T, 512)
    q = q.reshape(T, N_HEADS, HD)
    k = k.reshape(T, N_KV, HD)
    q = rmsnorm(q, w["q_norm"])            # per-head RMSNorm over hd
    k = rmsnorm(k, w["k_norm"])
    cos, sin = rope_tables(T, HD)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    rep = N_HEADS // N_KV
    k = np.repeat(k, rep, axis=1); v = np.repeat(v.reshape(T, N_KV, HD), rep, axis=1)
    qh, kh, vh = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
    sc = np.einsum("htd,hsd->hts", qh, kh) / np.sqrt(HD)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    sc = np.where(mask[None], -np.inf, sc)
    a = softmax(sc)
    ao = np.einsum("hts,hsd->htd", a, vh).transpose(1, 0, 2).reshape(T, N_HEADS * HD)
    return ao @ w["Wo"].T

def moe(x, w):
    """Router -> softmax -> top-8 -> renormalise (norm_topk_prob) -> weighted expert SwiGLU.
    No shared expert: every FFN read is an expert read, which is what makes paging the product."""
    logits = x @ w["Wgate"].T              # (T, 128)
    probs = softmax(logits)
    idx = np.argsort(-probs, axis=-1, kind="stable")[:, :TOPK]
    vals = np.take_along_axis(probs, idx, -1)
    wt = vals / vals.sum(-1, keepdims=True)
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            e = int(idx[t, j])
            xg = x[t] @ w["Wg"][e].T
            xu = x[t] @ w["Wu"][e].T
            h = (xg / (1.0 + np.exp(-xg))) * xu
            out[t] += wt[t, j] * (h @ w["Wd"][e].T)
    return out, logits, idx, wt

# ---------------------------------------------------------------- gate
def check():
    from pathlib import Path
    from safetensors import safe_open
    root = Path(__file__).resolve().parents[2]
    z = np.load(root / "data/traces/qwen3_30b_a3b_shard1.npz")
    shard = str(root / "weights/qwen3-30b-a3b/model-00001-of-00016.safetensors")
    f = safe_open(shard, framework="pt")
    def W(name):
        return f.get_tensor("model.layers.0." + name).float().numpy()
    ids = z["input_ids"]
    embed = f.get_tensor("model.embed_tokens.weight").float().numpy()
    x = embed[ids]
    print("=" * 78); print("QWEN3-30B-A3B REFERENCE vs shard-1 targets (layer 0)"); print("=" * 78)
    bad = 0
    def rep(name, got, want, tol=2e-5):
        nonlocal bad
        w = np.asarray(want, np.float32).reshape(np.shape(got))
        d = float(np.abs(got - w).max()); peak = max(float(np.abs(w).max()), 1e-30)
        ok = d / peak < tol
        print(f"  {name:26} max|d| {d:.3e}  rel {d/peak:.2e}  {'ok' if ok else 'FAIL'}")
        bad += not ok

    rep("RMSNorm (input)", rmsnorm(x, W("input_layernorm.weight")), z["L0.input_layernorm"][0])
    xn = rmsnorm(x, W("input_layernorm.weight"))
    rep("q_proj", xn @ W("self_attn.q_proj.weight").T, z["L0.self_attn.q_proj"][0])
    rep("k_proj", xn @ W("self_attn.k_proj.weight").T, z["L0.self_attn.k_proj"][0])
    T = len(ids)
    q = (xn @ W("self_attn.q_proj.weight").T).reshape(T, N_HEADS, HD)
    k = (xn @ W("self_attn.k_proj.weight").T).reshape(T, N_KV, HD)
    rep("q_norm", rmsnorm(q, W("self_attn.q_norm.weight")), z["L0.self_attn.q_norm"][0])
    rep("k_norm", rmsnorm(k, W("self_attn.k_norm.weight")), z["L0.self_attn.k_norm"][0])

    wts = {"Wq": W("self_attn.q_proj.weight"), "Wk": W("self_attn.k_proj.weight"),
           "Wv": W("self_attn.v_proj.weight"), "Wo": W("self_attn.o_proj.weight"),
           "q_norm": W("self_attn.q_norm.weight"), "k_norm": W("self_attn.k_norm.weight")}
    attn = attention(xn, wts)   # xn = rmsnorm(x, ln1); attention() expects the normed input
    rep("attention", attn, z["L0.self_attn"][0])
    h = x + attn
    rep("RMSNorm (post-attn)", rmsnorm(h, W("post_attention_layernorm.weight")),
        z["L0.post_attention_layernorm"][0])

    hn = rmsnorm(h, W("post_attention_layernorm.weight"))
    mw = {"Wgate": W("mlp.gate.weight"), "Wg": {}, "Wu": {}, "Wd": {}}
    for name, key in (("Wg", "gate_proj"), ("Wu", "up_proj"), ("Wd", "down_proj")):
        for e in range(128):
            mw[name][e] = f.get_tensor(f"model.layers.0.mlp.experts.{e}.{key}.weight").float().numpy()
    rep("router logits", hn @ mw["Wgate"].T, z["L0.mlp.gate"])
    out, logits, idx, wt = moe(hn, mw)
    # Compare the SET, not the order. The selection order is not a target — only which experts
    # are active, and the weights follow idx. Exact order comparison produced a spurious failure
    # when two experts' probabilities were within one ulp.
    same = all(set(a) == set(b) for a, b in zip(idx.tolist(), z["L0.top8"].tolist()))
    print(f"  {'top-8 selection':26} {'SAME SET' if same else 'DIFFERS'} "
          f"({sorted(idx[0].tolist())} vs {sorted(z['L0.top8'][0].tolist())})")
    bad += (not same)
    rep("moe output", out, z["L0.mlp"][0])
    print()
    print("REFERENCE: PASS" if not bad else f"REFERENCE: FAIL ({bad})")
    return 0 if not bad else 1

if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        raise SystemExit(check())
    print(__doc__)
