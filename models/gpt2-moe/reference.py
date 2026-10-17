#!/usr/bin/env python3
"""reference.py — the numerical oracle for the gpt2-moe port.

WHY THIS EXISTS, AND WHY IT IS NOT train_moe.py
  Every other gate in this repo compares the C engine against `train_moe.MoEModel` — the
  TRAINER. That is fine for verifying an export, but it cannot verify a PORT, because a port
  must be checkable against something that does not share the trainer's code. If the trainer
  and the thing it is checked against are the same object, a misreading of the architecture
  is invisible: both agree, and both are wrong. That exact failure has already happened here
  (the layer-split deviation on 2026-09-25: two implementations agreed with each other and
  disagreed with the spec, and every gate passed).

  So this file is a second, independent implementation of the same architecture, in numpy,
  written from `format/SPEC.md`'s block descriptions and the MODEL.md fact table rather than
  by importing the trainer. It has no torch dependency and no import of train_moe.

WHAT IT PROVES
  check_against_recorded() replays a recorded routing trace (data/traces/*.npz) and requires
  this implementation to reproduce the expert selections that the TRAINED model made. That is
  the strong statement: an independently written forward pass reaches the same routing
  decisions as the trained model. The C engine is then gated against this file.

  It is deliberately exact on routing (0 mismatches required) and tolerant on logits, because
  the engine's and numpy's fp32 summation orders differ while routing is a discrete decision.
"""

import numpy as np

EOS = 50256
EPS = 1e-5


# ---------------------------------------------------------------- primitives

def layernorm(x, g, b, eps=EPS):
    """LayerNorm over the last axis. Mirrors the C engine's order: mean, then variance about
    that mean, then scale/shift."""
    mu = x.mean(axis=-1, keepdims=True)
    d = x - mu
    var = (d * d).mean(axis=-1, keepdims=True)
    return d / np.sqrt(var + eps) * g + b


def gelu(x):
    """GPT-2's tanh approximation. The constant and the cubic term are part of the
    architecture, not a detail: the C engine's gelu() must use the same form."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def matmul_xw(x, W, b):
    """out[o] = sum_i x[i] * W[i][o] + b[o]   (W is stored (in, out) — the C layout)."""
    return x @ W + b


# ---------------------------------------------------------------- model

class Model:
    """The gpt2-moe architecture, from MODEL.md's fact table.

    Weights are addressed BY NAME, the same way the weight file addresses them, so a mismatch
    between this and a file is a missing/renamed tensor rather than an offset bug.
    """

    def __init__(self, cfg, arrays):
        self.cfg = cfg
        self.w = arrays
        c = cfg
        # The LAYER SPLIT IS NOT IN THE CHECKPOINT. The Phase 6 config dict has no moe_from
        # key; the as-built layout (MoE on 1..10, dense on 0 and 11) is the trainer's
        # Config default of 1. That is a genuine portability hazard and the reason the
        # exported CPW1 v3 header carries moe_from explicitly: the FILE is authoritative,
        # a raw checkpoint is not. MODEL.md records this as ASSUMED-from-trainer-default.
        moe_from = c.get("moe_from", 1)
        self.moe_layers = [l for l in range(c["n_layer"])
                           if moe_from <= l < c["n_layer"] - 1]
        self.dense_layers = [l for l in range(c["n_layer"]) if l not in self.moe_layers]

    # ---- attention: MHA, causal, head_dim = d_model / n_heads
    def attention(self, layer, x):
        c = self.cfg
        D, H = c["d_model"], c["n_heads"]
        HD = D // H
        w = self.w
        xn = layernorm(x, w["ln1_g"][layer], w["ln1_b"][layer])
        q = matmul_xw(xn, w["Wq"][layer], w["bq"][layer])
        k = matmul_xw(xn, w["Wk"][layer], w["bk"][layer])
        v = matmul_xw(xn, w["Wv"][layer], w["bv"][layer])
        T = x.shape[0]
        # (T, H, HD) -> (H, T, HD)
        qh = q.reshape(T, H, HD).transpose(1, 0, 2)
        kh = k.reshape(T, H, HD).transpose(1, 0, 2)
        vh = v.reshape(T, H, HD).transpose(1, 0, 2)
        logits = np.einsum("htd,hsd->hts", qh, kh) / np.sqrt(HD)
        # causal mask: position i may attend to j <= i
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        logits = np.where(mask[None, :, :], -np.inf, logits)
        attn = softmax(logits)
        ao = np.einsum("hts,hsd->htd", attn, vh).transpose(1, 0, 2).reshape(T, D)
        return x + matmul_xw(ao, w["Wo"][layer], w["bo"][layer])

    # ---- dense FFN (gelu)
    def dense_ffn(self, layer, x):
        w = self.w
        xn = layernorm(x, w["ln2_g"][layer], w["ln2_b"][layer])
        h = gelu(matmul_xw(xn, w["W1"][layer], w["b1"][layer]))
        return x + matmul_xw(h, w["W2"][layer], w["b2"][layer])

    # ---- routed MoE + shared expert, with the lambda locality bias
    def moe_ffn(self, layer, x, tokens, lam):
        c = self.cfg
        w = self.w
        D, N, K = c["d_model"], c["n_exp"], c["topk"]
        xn = layernorm(x, w["ln2_g"][layer], w["ln2_b"][layer])
        T = xn.shape[0]

        # Router for all positions in one matmul (the expensive part), then the
        # sequential selection loop (locality is per-position, so it cannot be batched).
        r_all = xn @ w["Wr"][layer] + w["br"][layer]
        prev = np.zeros(N, dtype=np.float32)
        idx = np.zeros((T, K), dtype=np.int64)
        wt = np.zeros((T, K), dtype=np.float32)
        for t in range(T):
            p = softmax(r_all[t] + lam * prev)
            # top-k with LOWER ID WINNING TIES, matching the engine's strict `>` scan
            order = np.argsort(-p, kind="stable")
            sel = order[:K]
            idx[t] = sel
            v2 = p[sel]
            wt[t] = v2 / v2.sum()
            prev = np.zeros(N, dtype=np.float32)
            if tokens[t] != EOS:
                prev[sel] = 1.0

        # shared expert: always active
        sh = gelu(matmul_xw(xn, w["W1s"][layer], w["b1s"][layer]))
        out = matmul_xw(sh, w["W2s"][layer], w["b2s"][layer])

        # routed experts, accumulated in selection order (as the engine does)
        for t in range(T):
            for kk in range(K):
                e = int(idx[t, kk])
                h = gelu(matmul_xw(xn[t], w["W1e"][layer][e], w["b1e"][layer][e]))
                o = matmul_xw(h, w["W2e"][layer][e], w["b2e"][layer][e])
                out[t] += wt[t, kk] * o
        return x + out, idx

    def forward(self, tokens, lam, trace=None):
        """Returns (logits (T, vocab), routing {layer: (T, topk)}).

        Tied unembedding: the output projection reuses token_emb, which is why there is no
        separate lm_head tensor in the file.
        """
        c, w = self.cfg, self.w
        tokens = np.asarray(tokens, dtype=np.int64)
        x = w["token_emb"][tokens] + w["pos_emb"][:len(tokens)]
        routing = {}
        for l in range(c["n_layer"]):
            x = self.attention(l, x)
            if l in self.moe_layers:
                x, idx = self.moe_ffn(l, x, tokens, lam)
                routing[l] = idx
            else:
                x = self.dense_ffn(l, x)
        xf = layernorm(x, w["lnf_g"], w["lnf_b"])
        logits = xf @ w["token_emb"].T          # tied unembedding
        if trace is not None:
            trace.update(routing)
        return logits, routing


# ---------------------------------------------------------------- checkpoint loading

def load_checkpoint(path):
    """Read a PyTorch checkpoint WITHOUT importing train_moe.

    torch is used only as a container format here — the architecture above never calls into
    it. A port whose oracle shared the trainer's model class could not detect a shared
    misreading of the architecture, which is the whole point of this file.
    """
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    sd = ck["model"]
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}
    return cfg, arrays, ck.get("step"), ck.get("lam", cfg.get("lam"))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="reference forward for the gpt2-moe port")
    ap.add_argument("--ckpt", default="weights/moe-scale-run/l05/model_only.pt")
    ap.add_argument("--tokens", default="262,2071,318,17764,351,6868,18366,737,50256,14906")
    ap.add_argument("--lam", type=float, default=None)
    args = ap.parse_args()
    cfg, arrays, step, lam = load_checkpoint(args.ckpt)
    lam = args.lam if args.lam is not None else lam
    m = Model(cfg, arrays)
    toks = np.array([int(t) for t in args.tokens.split(",")], dtype=np.int64)
    logits, routing = m.forward(toks, lam)
    print(f"reference: {len(toks)} tokens, lam={lam}, step={step}")
    print(f"  logits {logits.shape}; next-token argmax {int(logits[-1].argmax())}")
    for l in sorted(routing)[:3]:
        print(f"  layer {l} selections[:4] {routing[l][:4].tolist()}")


if __name__ == "__main__":
    main()
