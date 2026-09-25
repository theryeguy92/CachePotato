"""train_moe.py — CachePotato-MoE PyTorch trainer (Phase 6, 3090 server).

Spec-faithful to 05-Architecture-Specs/Model-Architecture-Spec:
- GPT-2 semantics (LayerNorm eps 1e-5, GELU tanh approx, tied unembedding)
- Dense FFN early/late, MoE FFN on middle layers: 1 shared + N routed, top-2
- Router in FP32 (risk-2 countermeasure); Switch aux loss; locality bias
  with reset at window start AND at EOS (spec decision 2026-09-15)
- Parameters stored in C layout (in, out) — no transposes anywhere;
  the bridge export and future CPW1 v3 map 1:1.

Modes:
  bridge  — load C-trainer export, compare forward (G1 gate)
  smoke   — TinyStories run, must match C trainer curve shape (G2 gate)
  train   — the scale run (FineWeb-Edu bins; tmux; checkpoint/resume)
"""

import argparse
import hashlib
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

EOS = 50256
EPS = 1e-5

# Optional instrumentation hook (added 2026-09-25, AFTER the Phase 6 scale run).
# When an external driver sets this to a dict, moe_ffn records the top-k expert
# indices it selected at each position. Strictly opt-in: while it is None the
# selection loop below is the training path unchanged. The scale run itself used
# train_moe.py md5 b9e3c31ff8f58aa74d64575506b3f60a (no hook).
TRACE = None


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


# ---------------------------------------------------------------- model

class Config:
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layer, n_ctx,
                 n_exp, exp_hid, lam, aux_alpha=0.01, topk=2, moe_from=1):
        self.vocab, self.d_model, self.n_heads = vocab, d_model, n_heads
        self.d_ff, self.n_layer, self.n_ctx = d_ff, n_layer, n_ctx
        self.n_exp, self.exp_hid, self.topk = n_exp, exp_hid, topk
        self.lam, self.aux_alpha = lam, aux_alpha
        # First layer whose FFN is sparse; MoE on moe_from .. n_layer-2.
        #   1 = AS BUILT by the Phase 6 scale run (MoE on 1..10, dense on 0 and 11)
        #   2 = what Model-Architecture-Spec asks for (MoE on 2..10, dense 0-1,11)
        # The default MUST stay 1: the existing checkpoints were trained with 1,
        # and layer 1's dense FFN weights are at random init (never trained), so
        # loading them under moe_from=2 would silently emit garbage.
        # See Decision-Log 2026-09-25. Set 2 only for a fresh training run.
        self.moe_from = moe_from

    @staticmethod
    def sanity(lam=0.5, moe_from=1):
        return Config(50257, 128, 4, 512, 4, 256, 8, 32, lam, moe_from=moe_from)

    @staticmethod
    def spec(lam=0.5, moe_from=1):
        return Config(50257, 768, 12, 3072, 12, 1024, 32, 192, lam,
                      moe_from=moe_from)

    def moe_layers(self):
        return [l for l in range(self.n_layer)
                if self.moe_from <= l < self.n_layer - 1]

    def dense_layers(self):
        return [l for l in range(self.n_layer) if l not in self.moe_layers()]


class MoEModel(torch.nn.Module):
    """All weights in C layout (in, out). MoE layers: moe_from <= l < n_layer-1."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D, Ff, L, V, C = cfg.d_model, cfg.d_ff, cfg.n_layer, cfg.vocab, cfg.n_ctx
        N, EH = cfg.n_exp, cfg.exp_hid
        P = torch.nn.Parameter

        def p(*shape):
            return P(torch.randn(*shape) * 0.02)

        self.token_emb = p(V, D)
        self.pos_emb = p(C, D)
        for name, shape in [
            ("ln1_g", (L, D)), ("ln1_b", (L, D)),
            ("Wq", (L, D, D)), ("bq", (L, D)), ("Wk", (L, D, D)), ("bk", (L, D)),
            ("Wv", (L, D, D)), ("bv", (L, D)), ("Wo", (L, D, D)), ("bo", (L, D)),
            ("ln2_g", (L, D)), ("ln2_b", (L, D)),
            ("W1", (L, D, Ff)), ("b1", (L, Ff)), ("W2", (L, Ff, D)), ("b2", (L, D)),
            ("Wr", (L, D, N)), ("br", (L, N)),
            ("W1s", (L, D, D)), ("b1s", (L, D)), ("W2s", (L, D, D)), ("b2s", (L, D)),
            ("W1e", (L, N, D, EH)), ("b1e", (L, N, EH)),
            ("W2e", (L, N, EH, D)), ("b2e", (L, N, D)),
        ]:
            self.register_parameter(name, p(*shape))
        self.lnf_g = p(D)
        self.lnf_b = p(D)
        with torch.no_grad():
            for n in ["ln1_g", "ln2_g"]:
                getattr(self, n).fill_(1.0)
            self.lnf_g.fill_(1.0)
            for n in ["ln1_b", "ln2_b", "bq", "bk", "bv", "bo", "b1", "b2",
                      "br", "b1s", "b2s", "b1e", "b2e"]:
                getattr(self, n).zero_()

    def moe_layer(self, l):
        return self.cfg.moe_from <= l < self.cfg.n_layer - 1

    def ln(self, x, g, b):
        return F.layer_norm(x, (x.shape[-1],), g, b, EPS)

    def attention(self, l, x):
        cfg = self.cfg
        B, T, D = x.shape
        H, HD = cfg.n_heads, cfg.d_model // cfg.n_heads
        xn = self.ln(x, self.ln1_g[l], self.ln1_b[l])
        q = torch.matmul(xn, self.Wq[l]) + self.bq[l]
        k = torch.matmul(xn, self.Wk[l]) + self.bk[l]
        v = torch.matmul(xn, self.Wv[l]) + self.bv[l]
        q = q.view(B, T, H, HD).transpose(1, 2)  # (B,H,T,HD)
        k = k.view(B, T, H, HD).transpose(1, 2)
        v = v.view(B, T, H, HD).transpose(1, 2)
        # SDPA (flash kernel): identical math to masked softmax(qk^T/sqrt(HD))v,
        # fused — no (B,H,T,T) materialization (memory) and much faster
        ao = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ao = ao.transpose(1, 2).reshape(B, T, D)
        return x + torch.matmul(ao, self.Wo[l]) + self.bo[l]

    def dense_ffn(self, l, x):
        xn = self.ln(x, self.ln2_g[l], self.ln2_b[l])
        h = gelu_tanh(torch.matmul(xn, self.W1[l]) + self.b1[l])
        return x + torch.matmul(h, self.W2[l]) + self.b2[l], 0.0, None

    def moe_ffn(self, l, x, tokens):
        """Two phases: (1) SEQUENTIAL router selection over T — required for
        locality (prev-token bias + EOS reset), tiny fp32 ops only;
        (2) BATCHED expert compute via gather/scatter once selections are
        known. Semantics identical to the naive per-position dispatch."""
        cfg = self.cfg
        B, T, D = x.shape
        N, EH, K = cfg.n_exp, cfg.exp_hid, cfg.topk
        xn = self.ln(x, self.ln2_g[l], self.ln2_b[l])

        # ---- phase 1: router selection (fp32, even under bf16 autocast)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xnf = xn.float()
            # the expensive part — router logits for ALL positions in ONE matmul;
            # only the tiny bias/softmax/topk ops stay in the sequential loop
            r_all = torch.matmul(xnf, self.Wr[l].float()) + self.br[l].float()  # (B,T,N)
            prev = torch.zeros(B, N, device=x.device)
            counts = torch.zeros(N, device=x.device)
            probs_all = torch.zeros(B, T, N, device=x.device)
            sel_idx = torch.zeros(B, T, K, dtype=torch.long, device=x.device)
            sel_w = torch.zeros(B, T, K, device=x.device)
            for t in range(T):
                p = F.softmax(r_all[:, t] + cfg.lam * prev, dim=-1)
                probs_all[:, t] = p
                v2, i2 = p.topk(K, dim=-1)
                sel_idx[:, t] = i2
                if TRACE is not None:
                    TRACE.setdefault(l, []).append(i2.detach().to("cpu", torch.int16))
                sel_w[:, t] = v2 / v2.sum(dim=-1, keepdim=True)
                onehot = torch.zeros_like(prev)
                onehot.scatter_(1, i2, 1.0)
                counts += onehot.sum(dim=0)
                prev = onehot * (tokens[:, t] != EOS).unsqueeze(1).float()

        # ---- phase 2: batched expert compute
        xf = xn.reshape(B * T, D)
        sha = gelu_tanh(torch.matmul(xf, self.W1s[l]) + self.b1s[l])
        out = torch.matmul(sha, self.W2s[l]) + self.b2s[l]      # shared (B*T,D)
        flat_idx = sel_idx.reshape(B * T, K)
        flat_w = sel_w.reshape(B * T, K).to(out.dtype)
        for s in range(K):
            for e in range(N):
                mask = flat_idx[:, s] == e
                if mask.any():
                    xe = xf[mask]
                    he = gelu_tanh(torch.matmul(xe, self.W1e[l][e]) + self.b1e[l][e])
                    oe = torch.matmul(he, self.W2e[l][e]) + self.b2e[l][e]
                    out[mask] += flat_w[mask, s].unsqueeze(1) * oe
        out = out.reshape(B, T, D)

        fe = counts / (B * T * K)
        pe = probs_all.mean(dim=(0, 1))
        aux = cfg.aux_alpha * N * torch.dot(fe, pe)
        stats = {"counts": counts.detach(), "total": float(B * T * K)}
        return x + out, aux, stats

    def forward(self, tokens, targets=None):
        cfg = self.cfg
        B, T = tokens.shape
        x = self.token_emb[tokens] + self.pos_emb[:T]
        aux_total = 0.0
        stats = {}
        for l in range(cfg.n_layer):
            x = self.attention(l, x)
            if self.moe_layer(l):
                x, aux, st = self.moe_ffn(l, x, tokens)
                aux_total = aux_total + aux
                stats[l] = st
            else:
                x, _, _ = self.dense_ffn(l, x)
        xf = self.ln(x, self.lnf_g, self.lnf_b)
        logits = torch.matmul(xf, self.token_emb.t())  # tied unemb
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), targets.reshape(-1))
            loss = loss + aux_total
        return logits, loss, stats


# ---------------------------------------------------------------- bridge (G1)

def load_c_export(path):
    with open(path, "rb") as f:
        hd = np.fromfile(f, dtype=np.int32, count=9)
        vocab, d_model, n_heads, d_ff, n_layer, n_ctx, n_exp, exp_hid, T = hd
        n_params, n_moe = np.fromfile(f, dtype=np.int64, count=2)
        loss = float(np.fromfile(f, dtype=np.float32, count=1)[0])
        params = np.fromfile(f, dtype=np.float32, count=n_params)
        moe = np.fromfile(f, dtype=np.float32, count=n_moe)
        logits = np.fromfile(f, dtype=np.float32, count=T * vocab).reshape(T, vocab)
    cfg = Config(vocab, d_model, n_heads, d_ff, n_layer, n_ctx, n_exp, exp_hid, 0.5)
    return cfg, loss, params, moe, logits, T


def c_layout_slices(cfg):
    """Yield (name, size) in the exact C params/moe buffer order."""
    D, Ff, L, V, C = cfg.d_model, cfg.d_ff, cfg.n_layer, cfg.vocab, cfg.n_ctx
    N, EH = cfg.n_exp, cfg.exp_hid
    dense = [("token_emb", V * D), ("pos_emb", C * D),
             ("ln1_g", L * D), ("ln1_b", L * D),
             ("Wq", L * D * D), ("bq", L * D), ("Wk", L * D * D), ("bk", L * D),
             ("Wv", L * D * D), ("bv", L * D), ("Wo", L * D * D), ("bo", L * D),
             ("ln2_g", L * D), ("ln2_b", L * D),
             ("W1", L * D * Ff), ("b1", L * Ff), ("W2", L * Ff * D), ("b2", L * D),
             ("lnf_g", D), ("lnf_b", D)]
    moe = [("Wr", L * D * N), ("br", L * N),
           ("W1s", L * D * D), ("b1s", L * D), ("W2s", L * D * D), ("b2s", L * D),
           ("W1e", L * N * D * EH), ("b1e", L * N * EH),
           ("W2e", L * N * EH * D), ("b2e", L * N * D)]
    return dense, moe


def bridge(args):
    cfg, ref_loss, params, moe, ref_logits, T = load_c_export(args.export)
    model = MoEModel(cfg).eval()
    dense_names, moe_names = c_layout_slices(cfg)
    off = 0
    with torch.no_grad():
        for name, size in dense_names:
            getattr(model, name).copy_(torch.from_numpy(params[off:off + size].reshape(getattr(model, name).shape)))
            off += size
        off = 0
        for name, size in moe_names:
            getattr(model, name).copy_(torch.from_numpy(moe[off:off + size].reshape(getattr(model, name).shape)))
            off += size

    tokens = torch.tensor([[464, 3139, 286, 4881, 318]])
    targets = torch.tensor([[3139, 286, 4881, 318, 262]])
    with torch.no_grad():
        logits, loss, _ = model(tokens, targets)
    # NOTE: the C export's "logits" are actually softmax PROBABILITIES —
    # forward_loss() softmaxes the buffer in place. Compare probabilities.
    ref = torch.from_numpy(ref_logits).unsqueeze(0)
    dlog = (F.softmax(logits, dim=-1) - ref).abs().max().item()
    dloss = abs(loss.item() - ref_loss)
    print(f"bridge: max abs prob diff {dlog:.3e} | loss {loss.item():.6f} vs C {ref_loss:.6f} (diff {dloss:.3e})")
    ok = dlog < 1e-3 and dloss < 1e-3
    print("G1 PASS" if ok else "G1 FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------- data

class TokenStream:
    """Deterministic random-window sampler over a uint16 bin."""

    def __init__(self, path, T, seed):
        self.ids = np.memmap(path, dtype=np.uint16, mode="r")
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.cursor = 0  # windows drawn so far (for resume)

    def batch(self, B, device):
        xs, ys = [], []
        for _ in range(B):
            i = self.rng.integers(0, len(self.ids) - self.T - 2)
            w = self.ids[i:i + self.T + 1].astype(np.int64)
            xs.append(w[:-1])
            ys.append(w[1:])
        self.cursor += 1
        return (torch.tensor(np.array(xs), device=device),
                torch.tensor(np.array(ys), device=device))

    def stream_hash(self, B=8):
        h = hashlib.sha256()
        save = self.rng.bit_generator.state
        for _ in range(B):
            i = self.rng.integers(0, len(self.ids) - self.T - 2)
            h.update(self.ids[i:i + self.T + 1].tobytes())
        self.rng.bit_generator.state = save
        return h.hexdigest()[:16]


# ---------------------------------------------------------------- train loop (G2 smoke / scale)

def run_train(args, cfg):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.bf16 and dev == "cuda"
    torch.manual_seed(args.seed)
    model = MoEModel(cfg).to(dev)
    stream = TokenStream(args.data, args.T, args.seed)
    print(f"stream hash: {stream.stream_hash()}  (must match across lambda variants)",
          flush=True)
    print(f"layout: MoE {cfg.moe_layers()}  dense {cfg.dense_layers()}", flush=True)
    val_ids = np.memmap(args.val, dtype=np.uint16, mode="r") if args.val else None

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim == 1 or n.startswith(("ln", "b")) else decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.wd},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        stream.rng.bit_generator.state = ck["rng"]
        stream.cursor = ck["cursor"]
        torch.set_rng_state(ck["torch_rng"])
        print(f"resumed from {args.resume} at step {start_step}", flush=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (args.lr_floor + (1 - args.lr_floor) * 0.5 * (1 + math.cos(math.pi * t)))

    os.makedirs(args.ckpt_dir, exist_ok=True)
    log = open(args.log, "a", buffering=1)
    prompt_ids = torch.tensor([stream.ids[stream.ids.tolist().index(EOS) + 1:][:16].tolist()] if False else [[464, 1, 616]], device=dev)  # "The a ..." tiny prompt

    def sample(max_new=30):
        model.eval()
        ids = prompt_ids.clone()
        with torch.no_grad():
            for _ in range(max_new):
                logits, _, _ = model(ids[:, -cfg.n_ctx:])
                nxt = logits[0, -1].argmax()
                if nxt.item() == EOS:
                    break
                ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
        model.train()
        return ids[0, prompt_ids.shape[1]:].tolist()

    model.train()
    t0 = time.time()
    cum_counts = {}  # cumulative utilization per MoE layer (C-trainer-comparable)
    for step in range(start_step + 1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = stream.batch(args.B, dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            _, loss, stats = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        for l, st in stats.items():
            cum_counts[l] = cum_counts.get(l, torch.zeros(cfg.n_exp)) + st["counts"].cpu()

        if step == 1 or step % 20 == 0:
            msg = f"step {step:6d} | loss {loss.item():.4f}"
            for l in sorted(stats):
                cum_mx = (cum_counts[l].max() / cum_counts[l].sum()).item()
                bat_mx = stats[l]["counts"].max().item() / stats[l]["total"]
                msg += f" | L{l} share {cum_mx:.2f}/{bat_mx:.2f}"
            msg += f" | {time.time() - t0:.1f}s"
            t0 = time.time()
            print(msg, flush=True)
            log.write(msg + "\n")

        if val_ids is not None and step % 500 == 0:
            model.eval()
            with torch.no_grad():
                vx = torch.tensor(np.array([val_ids[i:i + args.T] for i in range(0, 8 * args.T, args.T)]).astype(np.int64), device=dev)
                vy = torch.tensor(np.array([val_ids[i + 1:i + args.T + 1] for i in range(0, 8 * args.T, args.T)]).astype(np.int64), device=dev)
                _, vloss, _ = model(vx, vy)
            print(f"step {step:6d} | val loss {vloss.item():.4f}", flush=True)
            log.write(f"step {step:6d} | val loss {vloss.item():.4f}\n")
            model.train()

        if step % 1000 == 0 or step == args.steps:
            for old in ["ckpt2.pt"]:
                p = os.path.join(args.ckpt_dir, old)
                if os.path.exists(p):
                    os.replace(p, p.replace("ckpt2", "ckpt3"))
            for pair in [("ckpt1.pt", "ckpt2.pt")]:
                p1 = os.path.join(args.ckpt_dir, pair[0])
                if os.path.exists(p1):
                    os.replace(p1, os.path.join(args.ckpt_dir, pair[1]))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "rng": stream.rng.bit_generator.state,
                        "cursor": stream.cursor, "torch_rng": torch.get_rng_state(),
                        "config": vars(cfg)},
                       os.path.join(args.ckpt_dir, "ckpt1.pt"))
            if step % 2500 == 0:
                gen = sample()
                log.write(f"step {step} sample ids: {gen}\n")
                print(f"step {step:6d} | sample: {gen}", flush=True)

    log.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bridge", "smoke", "train"])
    ap.add_argument("--export", default="/tmp/bridge_moe.bin")
    ap.add_argument("--data", default="data/tinystories_train.bin")
    ap.add_argument("--val", default="")
    ap.add_argument("--log", default="logs/train.log")
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--spec", action="store_true", help="use the spec (12L/768d/32exp) config")
    ap.add_argument("--moe-from", type=int, default=1,
                    help="first sparse-FFN layer (1=as-built Phase 6, 2=spec)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--lr-floor", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    if args.mode == "bridge":
        return bridge(args)
    cfg = (Config.spec(args.lam, args.moe_from) if args.spec
           else Config.sanity(args.lam, args.moe_from))
    return run_train(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
