#!/usr/bin/env python3
"""Expert paging-granularity analysis — Phase 7 Task 2, the PAGING half.

WHY THIS EXISTS
    `05-Architecture-Specs/Expert-Sizing-Tradeoffs.md` requires Phase 7 to
    benchmark both expert geometries: 32 x 1.18MB (as built) versus the
    16 x 2.36MB fallback. Retraining at 16 experts is out of scope, but the
    *paging* half is testable offline from the routing traces already recorded.

    A live number points straight at this: on 2026-09-28 the MoE achieved
    ~134 MB/s effective bandwidth against the dense model's ~286 MB/s. The
    difference is read PATTERN — the MoE scatters 1.18MB expert blocks while
    the dense model streams ~6.4MB regions. Larger expert pages buy bigger,
    more sequential reads. They also fetch an unwanted partner on every miss.

    This tool measures that trade from traces, with no machine time.

METHOD
    A group is fetched as a unit, so LRU over groups IS LRU over the group
    sequence. Hit rate at capacity C = share of accesses whose stack distance
    is < C. Capacities are compared at MATCHED BYTES, which is the only fair
    comparison: grouping does not change how many bytes fit, it changes what
    must travel together.

GROUPINGS
    ungrouped      k=1, as built
    adjacent       k=2, contiguous pairs -> one sequential read. Pairs are
                   arbitrary with respect to what the router co-selects.
    matched        k=2, greedy max-weight matching on observed co-selection.
                   Needs the exporter to permute expert order so each pair is
                   contiguous, but requires NO retraining — this is exactly
                   the layout/routing co-design the project is about.
    random         k=2, control: how much of `matched`'s gain is just luck.

USAGE
    python3 tools/analyze_expert_grouping.py [--precision fp32] [--windows 488]
"""
import argparse
from pathlib import Path

import numpy as np

EXPERT_BYTES = 1_183_488       # one routed expert, from the CPW1 v3 layout
N_EXP = 32                     # routed experts per MoE layer


def stack_distances(seq, n_blocks):
    """Stack distance per access: number of DISTINCT blocks used more recently.

    -1 marks a first touch (compulsory miss, resident at no capacity).

    Equivalent to the list-based version in analyze_routing.py but O(n_blocks)
    vectorised per access instead of an O(stack) Python list scan, which makes
    a 5M-access trace tractable across four groupings.
    """
    last = np.full(n_blocks, -1, dtype=np.int64)
    out = np.empty(seq.shape[0], dtype=np.int32)
    for i, b in enumerate(seq.tolist()):
        cur = last[b]
        if cur < 0:
            out[i] = -1
        else:
            out[i] = int(np.count_nonzero(last > cur))
        last[b] = i
    return out


def block_seq(sel):
    """Expert-access sequence in engine order: window, token, layer, top-k."""
    L, W, T, K = sel.shape
    blk = np.empty((W, T, L * K), dtype=np.int64)
    for li in range(L):
        blk[:, :, li * K:(li + 1) * K] = li * N_EXP + sel[li]
    return blk.reshape(-1)


def co_selection(sel):
    """Per layer, counts of which expert PAIRS are selected by the same token."""
    L, W, T, K = sel.shape
    counts = np.zeros((L, N_EXP, N_EXP), dtype=np.int64)
    for li in range(L):
        s = sel[li].reshape(-1, K)                 # (W*T, K)
        for a in range(K):
            for b in range(a + 1, K):
                np.add.at(counts[li], (s[:, a], s[:, b]), 1)
    return counts


def group_map_adjacent():
    """block -> group, pairing adjacent expert indices within each layer."""
    g = np.empty(10 * N_EXP, dtype=np.int64)
    for li in range(10):
        for e in range(N_EXP):
            g[li * N_EXP + e] = li * (N_EXP // 2) + e // 2
    return g, 10 * (N_EXP // 2)


def group_map_matched(sel):
    """block -> group, pairing by greedy maximum co-selection weight.

    Tested before coding anything in the engine: if pairing is not
    co-selection-aware it over-fetches by construction.
    """
    counts = co_selection(sel)
    g = np.empty(10 * N_EXP, dtype=np.int64)
    for li in range(10):
        c = counts[li].copy()
        c = c + c.T                                # symmetric weight
        free = set(range(N_EXP))
        pair_id = 0
        while free:
            best, bw = None, -1
            for i in sorted(free):
                for j in sorted(free):
                    if i < j and c[i, j] > bw:
                        bw, best = c[i, j], (i, j)
            i, j = best
            g[li * N_EXP + i] = li * (N_EXP // 2) + pair_id
            g[li * N_EXP + j] = li * (N_EXP // 2) + pair_id
            free.discard(i)
            free.discard(j)
            pair_id += 1
    return g, 10 * (N_EXP // 2)


def group_map_random(seed=0):
    rng = np.random.default_rng(seed)
    g = np.empty(10 * N_EXP, dtype=np.int64)
    for li in range(10):
        perm = rng.permutation(N_EXP)
        for slot, e in enumerate(perm):
            g[li * N_EXP + e] = li * (N_EXP // 2) + slot // 2
    return g, 10 * (N_EXP // 2)


def evaluate(dist, n_groups, group_bytes, budgets_mb):
    """Hit%, bytes/token and read-events/token at matched byte budgets."""
    per_tok = 20                                # 10 layers x top-2
    rows = []
    for mb in budgets_mb:
        budget = mb * 1024 * 1024
        C = max(1, budget // group_bytes)
        if C > n_groups:
            C = n_groups
        ok = (dist >= 0) & (dist < C)
        hit = float(ok.mean())
        miss = 1.0 - hit
        rows.append((mb, C, hit, miss * per_tok, miss * per_tok * group_bytes))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/traces")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--windows", type=int, default=488,
                    help="subsample windows (488 = all; lower for speed)")
    ap.add_argument("--budgets", default="24,32,48,64,96,128,192,320")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    root = Path(args.traces)

    data = {}
    for arm in ("l00", "l05"):
        z = np.load(root / f"{arm}_{args.precision}.npz", allow_pickle=False)
        sel = z["sel"].astype(np.int64)
        if args.windows < sel.shape[1]:
            sel = sel[:, :args.windows]
        data[arm] = sel

    print("=" * 100)
    print("EXPERT PAGING GRANULARITY — offline replay from recorded routing "
          "traces (%s)" % args.precision)
    print("=" * 100)

    schemes = {
        "ungrouped": lambda: (np.arange(10 * N_EXP), 10 * N_EXP),
        "adjacent": group_map_adjacent,
        "matched": lambda: group_map_matched(data["l05"]),
        "random": group_map_random,
    }

    # ---- co-selection of the pairings actually available
    sel5 = data["l05"]
    counts = co_selection(sel5)
    total_tokens = sel5.shape[1] * sel5.shape[2]
    g_adj, _ = group_map_adjacent()
    co_adj = 0
    for li in range(10):
        for e in range(0, N_EXP, 2):
            co_adj += counts[li, e, e + 1] + counts[li, e + 1, e]
    print(f"\n-- pair co-selection (l05, {total_tokens:,} tokens x 10 layers) --")
    print(f"  adjacent pairs: both members chosen by the same token at the same")
    print(f"  layer in {100.0*co_adj/total_tokens/10:.2f}% of tokens "
          f"({co_adj:,} co-selections)")
    print(f"  chance if selection were independent and uniform: "
          f"{100.0*(2*(1/N_EXP)*(1/(N_EXP-1))):.2f}%")

    for arm in ("l00", "l05"):
        sel = data[arm]
        seq = block_seq(sel)
        print(f"\n{'-'*100}\nARM {arm}   ({seq.shape[0]:,} expert accesses)\n{'-'*100}")
        print(f"{'grouping':<12} {'MB budget':>10} {'cap':>6} {'hit%':>8} "
              f"{'reads/tok':>10} {'MB/tok':>9}")
        base = None
        for name, mk in schemes.items():
            g, n_g = mk()
            gseq = g[seq]
            dist = stack_distances(gseq, n_g)
            rows = evaluate(dist, n_g, EXPERT_BYTES * (1 if name == "ungrouped" else 2),
                            budgets)
            for (mb, C, hit, reads, byt) in rows:
                tag = ""
                if name == "ungrouped":
                    base = base or {}
                    base[mb] = (reads, byt)
                else:
                    r0, b0 = base[mb]
                    tag = (f"   reads {reads/r0-1:+.1%}  bytes {byt/b0-1:+.1%}")
                print(f"{name:<12} {mb:>10} {C:>6} {100*hit:>7.2f}% "
                      f"{reads:>10.2f} {byt/1e6:>9.2f}{tag}")
            print()

    print("Read: bytes/tok is the volume cost. reads/tok is what buys bandwidth —")
    print("a grouped miss fetches one 2.37MB sequential block instead of a")
    print("1.18MB one. Grouping wins only if the byte penalty is smaller than the")
    print("bandwidth gain (dense streams ~286MB/s vs MoE scattered ~134MB/s).")


if __name__ == "__main__":
    main()
