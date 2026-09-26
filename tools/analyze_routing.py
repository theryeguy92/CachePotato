#!/usr/bin/env python3
"""analyze_routing.py — Phase D: does the λ locality bias actually change
temporal routing, and does it buy cache hits?

Reads routing traces produced by tools/record_routing.py and reports:

  1. marginal expert utilization per layer  (frequency control — blind to λ)
  2. temporal reuse rate P(sel(t) ∩ sel(t-1) ≠ ∅) per layer  (the decisive number)
  3. top-1 stickiness + per-layer transition matrices (the mechanism fingerprint)
  4. LRU replay under a global (layer, expert) byte budget, λ=0.5 vs λ=0+LRU

LRU is evaluated via stack distances: one pass yields the hit rate at every
capacity, since an access hits an LRU cache of C blocks iff its stack distance
is < C.

Usage:
  python tools/analyze_routing.py --traces data/traces --precision fp32
"""
import argparse
import math
from pathlib import Path

import numpy as np

EXPERT_BYTES = 1_183_488          # 295,872 params x fp32 (as built)
N_EXP = 32
EOS_DEFAULT = 50256


# ----------------------------------------------------------------- stack LRU
def stack_distances(access):
    """Return stack distance per access (-1 = not resident at any capacity).

    Stack distance = number of *distinct* blocks used more recently, which is
    exactly the capacity at which an LRU cache would still hold the block.
    """
    dist = np.empty(access.shape[0], dtype=np.int32)
    stack = []
    idx = stack.index
    for i, b in enumerate(access.tolist()):
        try:
            p = idx(b)
        except ValueError:
            p = -1
        else:
            del stack[p]
        stack.insert(0, b)
        dist[i] = p
    return dist


def reuse_stats(sel, tokens, eos, K):
    """Reuse / stickiness / transition matrices over adjacent valid pairs.

    A pair (t-1, t) is excluded when token[t-1] == EOS, because the model's
    locality set is reset there — reuse across that boundary is not the bias.
    """
    L, W, T, _ = sel.shape
    prev = sel[:, :, :-1, :]
    cur = sel[:, :, 1:, :]
    valid = (tokens[:, :-1] != eos)                     # (W, T-1)

    overlap = np.zeros((L, W, T - 1), dtype=bool)
    for k in range(K):
        overlap |= (prev[:, :, :, k:k + 1] == cur).any(axis=-1)

    both = np.ones((L, W, T - 1), dtype=bool)
    for k in range(K):
        both &= (prev[:, :, :, k:k + 1] == cur).any(axis=-1)

    top1_prev = prev[:, :, :, 0]
    top1_cur = cur[:, :, :, 0]
    stick = top1_prev == top1_cur

    v = np.broadcast_to(valid, (L, W, T - 1))
    reuse = (overlap & v).sum(axis=(1, 2)) / v.sum(axis=(1, 2))
    same = (both & v).sum(axis=(1, 2)) / v.sum(axis=(1, 2))
    sticky = (stick & v).sum(axis=(1, 2)) / v.sum(axis=(1, 2))

    # per-layer transition matrix on top-1, row-normalised
    tm = np.zeros((L, N_EXP, N_EXP), dtype=np.float64)
    li, wi, ti = np.nonzero(v)
    np.add.at(tm, (li, top1_prev[li, wi, ti], top1_cur[li, wi, ti]), 1.0)
    rowsum = tm.sum(axis=2, keepdims=True)
    tmn = np.divide(tm, rowsum, out=np.zeros_like(tm), where=rowsum > 0)
    return reuse, same, sticky, tmn, int(v.sum())


def per_window_reuse(sel, tokens, eos, K):
    """Reuse rate per (layer, window) so CIs can be taken over documents."""
    L, W, T, _ = sel.shape
    prev, cur = sel[:, :, :-1, :], sel[:, :, 1:, :]
    valid = tokens[:, :-1] != eos
    ov = np.zeros((L, W, T - 1), dtype=bool)
    for k in range(K):
        ov |= (prev[:, :, :, k:k + 1] == cur).any(axis=-1)
    v = np.broadcast_to(valid, ov.shape)
    num = (ov & v).sum(axis=2)                     # (L, W)
    den = valid.sum(axis=1)                        # (W,)
    return num.mean(axis=0) / den                  # (W,) mean over layers


def boot_ci(x, n=2000, seed=0):
    """95% CI of the mean, resampling windows (the independent unit)."""
    rng = np.random.default_rng(seed)
    W = len(x)
    means = x[rng.integers(0, W, size=(n, W))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def prefetch_coverage(sel, tokens, eos, K, W):
    """Access-level coverage of prefetching the union of the previous W tokens'
    expert sets at the same layer.

    W=1 is the spec's stated contract ("same experts as the previous token at
    this layer"). This is the number the engine's prefetcher actually gets: it
    can be much lower than the token-level reuse rate, because a token with a
    partial overlap scores as a "reuse" but only covers part of its accesses.

    A pair contributes only if none of the W preceding tokens is an EOS, since
    the locality set is cleared there.
    """
    L, Wn, T, _ = sel.shape
    cur = sel[:, :, W:, :]                                  # (L, Wn, T-W, K)
    hist = [sel[:, :, W - w:-w, :] for w in range(1, W + 1)]  # W x (L,Wn,T-W,K)
    valid = np.ones((Wn, T - W), dtype=bool)
    for w in range(1, W + 1):
        valid &= (tokens[:, W - w:T - w] != eos)

    covered = np.zeros((L, Wn, T - W), dtype=np.int32)
    for j in range(K):                                      # each needed expert
        hit = np.zeros((L, Wn, T - W), dtype=bool)
        for h in hist:
            hit |= (h[:, :, :, 0] == cur[:, :, :, j])
            if K > 1:
                hit |= (h[:, :, :, 1] == cur[:, :, :, j])
        covered += hit
    v = np.broadcast_to(valid, covered.shape)
    return (covered * v).sum() / (v.sum() * K)


def marginal(sel):
    L = sel.shape[0]
    out = np.zeros(L)
    for l in range(L):
        c = np.bincount(sel[l].ravel(), minlength=N_EXP)
        out[l] = c.max() / c.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/traces")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--budgets", default="32,48,64,80,96,128,160,192,224,256,288,320")
    args = ap.parse_args()

    root = Path(args.traces)
    arms = ["l00", "l05"]
    data = {}
    for a in arms:
        f = root / f"{a}_{args.precision}.npz"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        z = np.load(f, allow_pickle=False)
        data[a] = {k: z[k] for k in
                   ("sel", "tokens", "moe_layers", "lam", "topk", "T")}
        data[a]["sel"] = data[a]["sel"].astype(np.int32)

    sel0, sel5 = data["l00"]["sel"], data["l05"]["sel"]
    tokens = data["l00"]["tokens"]
    layers = data["l00"]["moe_layers"]
    K = int(data["l00"]["topk"])
    eos = EOS_DEFAULT
    T = int(data["l00"]["T"])
    W = tokens.shape[0]
    n_tok = W * T

    print("=" * 78)
    print("PHASE D — routing locality, λ=0.5 (l05) vs λ=0 (l00)   precision=%s"
          % args.precision)
    print("=" * 78)
    print(f"windows {W}  x  T {T}  =  {n_tok:,} tokens   MoE layers {list(layers)}")
    print(f"block = one routed expert = {EXPERT_BYTES:,} bytes; "
          f"total paged tier = {len(layers)*N_EXP} blocks "
          f"= {len(layers)*N_EXP*EXPERT_BYTES/1e6:.0f} MB\n")

    # ---- 1. marginal utilization (the statistic Phase 6 logged)
    m0, m5 = marginal(sel0), marginal(sel5)
    print("-- 1. MARGINAL UTILIZATION (frequency control; uniform = %.3f) --"
          % (1 / N_EXP))
    print("  layer |  l00 maxshare |  l05 maxshare")
    for i, l in enumerate(layers):
        print("   %4d |        %.3f   |        %.3f" % (l, m0[i], m5[i]))
    print("  max over layers: l00 %.3f   l05 %.3f\n" % (m0.max(), m5.max()))

    # ---- 2/3. reuse, stickiness, transitions
    r0, s0, k0, t0, npair = reuse_stats(sel0, tokens, eos, K)
    r5, s5, k5, t5, _ = reuse_stats(sel5, tokens, eos, K)

    print("-- 2. TEMPORAL REUSE  P(sel(t) ∩ sel(t-1) ≠ ∅)  "
          "[%d valid pairs/layer] --" % npair)
    print("  layer |   l00  |   l05  |   Δ (pp)")
    for i, l in enumerate(layers):
        print("   %4d | %6.3f | %6.3f | %+7.2f" % (l, r0[i], r5[i],
                                                   100 * (r5[i] - r0[i])))
    R0, R5 = r0.mean(), r5.mean()
    print("   ALL  | %6.3f | %6.3f | %+7.2f   <-- headline"
          % (R0, R5, 100 * (R5 - R0)))
    pw0 = per_window_reuse(sel0, tokens, eos, K)
    pw5 = per_window_reuse(sel5, tokens, eos, K)
    c0, c1 = boot_ci(pw0)
    c2, c3 = boot_ci(pw5)
    dl, dh = boot_ci(pw5 - pw0)
    print("  95%% CI over %d windows:  l00 [%.3f, %.3f]   l05 [%.3f, %.3f]"
          % (W, c0, c1, c2, c3))
    print("  Δ reuse 95%% CI: [%+.2f, %+.2f] pp" % (100 * dl, 100 * dh))

    print("\n-- 3. TOP-1 STICKINESS  P(top1(t) = top1(t-1)) --")
    print("  layer |   l00  |   l05  |   Δ (pp)   | identical-set l00/l05")
    for i, l in enumerate(layers):
        print("   %4d | %6.3f | %6.3f | %+7.2f   |   %.3f / %.3f"
              % (l, k0[i], k5[i], 100 * (k5[i] - k0[i]), s0[i], s5[i]))
    print("   ALL  | %6.3f | %6.3f | %+7.2f   |   %.3f / %.3f"
          % (k0.mean(), k5.mean(), 100 * (k5.mean() - k0.mean()),
             s0.mean(), s5.mean()))
    print("  (random routing baseline for stickiness = %.3f)"
          % (1 / N_EXP))
    fro = [float(np.linalg.norm(t5[i] - t0[i])) for i in range(len(layers))]
    print("  transition-matrix ||Δ||_F per layer: "
          + " ".join("%.2f" % f for f in fro))
    print("  mean ||Δ||_F = %.3f  (0 would mean λ is inert)\n"
          % float(np.mean(fro)))

    # ---- 3b. prefetch coverage: what a prefetcher actually buys
    print("-- 3b. PREFETCH COVERAGE (access-level: share of the K expert "
          "accesses per token\n        already covered by prefetching the union "
          "of the previous W tokens' sets) --")
    print("     W |   l00  |   l05  |   Δ (pp)")
    for W in (1, 2, 3, 4):
        c0 = prefetch_coverage(sel0, tokens, eos, K, W)
        c5 = prefetch_coverage(sel5, tokens, eos, K, W)
        print("  %4d | %6.3f | %6.3f | %+7.2f" % (W, c0, c5, 100 * (c5 - c0)))
    print("  W=1 is the spec's stated contract ('same experts as the previous")
    print("  token at this layer'). Token-level reuse OVERSTATES this number.\n")

    # ---- 4. LRU replay
    def accesses(sel):
        L, Wd, Td, Kd = sel.shape
        blk = np.empty((Wd, Td, L * Kd), dtype=np.int32)
        for li in range(L):
            blk[:, :, li * Kd:(li + 1) * Kd] = li * N_EXP + sel[li]
        return blk.reshape(-1)

    print("-- 4. LRU REPLAY (global (layer,expert) budget, LRU eviction) --")
    print("  capacity      |   l00 hit%  |   l05 hit%  |  Δ (pp) | l00 B/tok |"
          " l05 B/tok |  Δbytes%")
    dist = {}
    for a, sel in (("l00", sel0), ("l05", sel5)):
        dist[a] = stack_distances(accesses(sel))

    budgets = [int(b) for b in args.budgets.split(",")]
    results = []
    window_hits = {}
    for C in budgets:
        row = {}
        wl = {}
        for a in arms:
            d = dist[a]
            ok = (d >= 0) & (d < C)
            row[a] = ok.mean()
            wl[a] = ok.reshape(W, -1).mean(axis=1)
        window_hits[C] = wl
        b0 = (1 - row["l00"]) * (len(layers) * K) * EXPERT_BYTES   # bytes/token
        b5 = (1 - row["l05"]) * (len(layers) * K) * EXPERT_BYTES
        dpp = 100 * (row["l05"] - row["l00"])
        db = 100 * (b5 - b0) / b0 if b0 else float("nan")
        results.append((C, row["l00"], row["l05"], dpp, b0, b5, db))
        print("  %4d blk %5.0fMB | %10.2f%% | %10.2f%% | %+7.2f | %9.2f |"
              " %9.2f | %+8.2f" % (C, C * EXPERT_BYTES / 1e6,
                                   100 * row["l00"], 100 * row["l05"], dpp,
                                   b0 / 1e6, b5 / 1e6, db))

    # ---- verdict vs the committed Q1 bar
    best_dpp = max(r[3] for r in results)
    best_db = min(r[6] for r in results)
    print("\n" + "=" * 78)
    print("VERDICT vs committed Q1  (≥10 pp hit rate  OR  ≥25% fewer bytes/token)")
    print("=" * 78)
    print("  best hit-rate delta   : %+.2f pp  at %d blocks (%.0f MB)"
          % (best_dpp, max(results, key=lambda r: r[3])[0],
             max(results, key=lambda r: r[3])[0] * EXPERT_BYTES / 1e6))
    Cbest = max(results, key=lambda r: r[3])[0]
    cdl, cdh = boot_ci(window_hits[Cbest]["l05"] - window_hits[Cbest]["l00"])
    print("    95%% CI on that delta : [%+.2f, %+.2f] pp over %d windows"
          % (100 * cdl, 100 * cdh, W))
    print("  best bytes/token delta: %+.2f %%  at %d blocks (%.0f MB)"
          % (best_db, min(results, key=lambda r: r[6])[0],
             min(results, key=lambda r: r[6])[0] * EXPERT_BYTES / 1e6))
    print("  reuse-rate delta      : %+.2f pp  (%.4f -> %.4f)"
          % (100 * (R5 - R0), R0, R5))
    q1 = (best_dpp >= 10.0) or (best_db <= -25.0)
    print("\n  Q1 (cache effect): %s" % ("PASS" if q1 else "FAIL"))
    if abs(R5 - R0) < 0.005:
        print("  MECHANISM: λ≈inert — reuse delta < 0.5pp; the ablation does not"
              " test the thesis. Next step is a λ sweep, not the engine build.")
    elif R5 > R0:
        print("  MECHANISM: λ raises temporal reuse by %.2f pp." % (100 * (R5 - R0)))
    else:
        print("  MECHANISM: λ %s reuse by %.2f pp — wrong direction."
              % ("lowers" if R5 < R0 else "leaves", abs(100 * (R5 - R0))))


if __name__ == "__main__":
    main()
