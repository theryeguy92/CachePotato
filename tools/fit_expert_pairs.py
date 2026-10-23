#!/usr/bin/env python3
"""fit_expert_pairs.py — Tier B: order each layer's experts so co-selected ones are adjacent.

WHY THIS EXISTS
  A REGION is what the pager fetches, and the file decides what a region covers. With one
  region per expert, a token pays for 8 separate ~5 MB reads per layer. If two experts are
  usually selected together, one region covering both turns two reads into one — at the cost
  of over-fetching when only one of them is chosen. The OS cannot do this: it cannot see
  regions, only pages.

PURE PLACEMENT, NOT A ROUTER PERMUTATION
  Unlike the v3 permutation experiment, this changes WHERE experts live in the file, not
  which expert a router row means: expert ids, tensor names and `Wgate` rows are untouched.
  So nothing in the arithmetic changes and the gate is BIT-IDENTITY, not "exact routing,
  fp32-equal logits". That is a stronger claim than v3 could make, and it is available here
  only because the permutation is applied to placement alone.

FIT ON A DIFFERENT PROMPT THAN YOU MEASURE ON
  Co-selection counts come from a routing trace, and a layout fitted on the measurement
  prompt would be an overfit. This tool records the trace it fitted on in its output so the
  measurement can state which workload the layout was compiled for.

USAGE
  # a routing trace comes from the engine: prefill writes <prefix>.route, decode too
  python3 tools/fit_expert_pairs.py --route scratch/fit.route --group 2 --out scratch/pairs.json
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_route(path, layers, topk):
    raw = Path(path).read_bytes()
    n = len(raw) // 4
    if n % (layers * topk) != 0:
        raise SystemExit(f"{path}: {n} ints is not a multiple of layers*topk ({layers*topk})")
    t = n // (layers * topk)
    return np.frombuffer(raw, dtype="<i4").reshape(layers, t, topk), t


def pair_layer(sel, n_exp, group):
    """Greedy max-weight matching over co-selection counts, then leftovers by use count.

    `sel` is (T, k) expert ids. Greedy matching is O(E^2) and good enough here: the point is
    to put experts that are chosen together next to each other, and the count matrix is the
    thing that decides it. Groups of `group` (>2) are formed by extending each matched pair
    with the best remaining partner of its members."""
    co = np.zeros((n_exp, n_exp), dtype=np.int64)
    use = np.zeros(n_exp, dtype=np.int64)
    for row in sel:
        for i in range(len(row)):
            use[row[i]] += 1
            for j in range(i + 1, len(row)):
                a, b = row[i], row[j]
                co[a, b] += 1
                co[b, a] += 1
    pairs = []
    for a in range(n_exp):
        for b in range(a + 1, n_exp):
            if co[a, b] > 0:
                pairs.append((int(co[a, b]), a, b))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used = [False] * n_exp
    order = []
    chosen = []
    for cnt, a, b in pairs:
        if used[a] or used[b]:
            continue
        used[a] = used[b] = True
        chosen.append((a, b, cnt))
        order.extend([a, b])
        if group <= 2:
            continue
        # extend to the requested group size with the members' best unused partners
        members = [a, b]
        while len(members) < group:
            best, best_c = None, -1
            for m in members:
                for c in range(n_exp):
                    if not used[c] and co[m, c] > best_c:
                        best, best_c = c, int(co[m, c])
            if best is None:
                break
            used[best] = True
            members.append(best)
            order.append(best)
    for e in range(n_exp):
        if not used[e]:
            used[e] = True
            order.append(e)
    return order, chosen, use, co


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True, help="routing trace: L x T x topk int32")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    route, T = read_route(args.route, args.layers, args.topk)
    print(f"fit: {args.route} — {args.layers} layers x {T} positions x top-{args.topk}")
    layers, stats = [], []
    for l in range(args.layers):
        order, chosen, use, co = pair_layer(route[l], args.experts, args.group)
        assert sorted(order) == list(range(args.experts)), (l, "not a permutation")
        layers.append([int(x) for x in order])
        if chosen:
            # what the grouping buys, in-sample: distinct groups selected per position
            distinct = 0.0
            for row in route[l]:
                gs = {order.index(int(e)) // args.group for e in row}
                distinct += len(gs)
            stats.append((l, len(chosen), sum(c for _, _, c in chosen) / max(len(chosen), 1),
                          distinct / T))
    if stats:
        mean_pair = sum(s[2] for s in stats) / len(stats)
        mean_distinct = sum(s[3] for s in stats) / len(stats)
        print(f"  pairs matched: {stats[0][1]}/layer, mean co-selection {mean_pair:.2f} of {T}"
              f" positions  ({100.0 * mean_pair / T:.1f}%)")
        print(f"  in-sample distinct regions/token: {mean_distinct:.2f} of {args.topk}"
              f"  ({100.0 * (1 - mean_distinct / args.topk):.1f}% fewer than one-region-per-expert)")
        print("  (in-sample: the fit saw these positions. The measurement must use another"
              " prompt, and its number will be worse.)")
    out = dict(group=args.group, topk=args.topk, layers=args.layers, experts=args.experts,
               source=str(args.route), positions=int(T), layers_order=layers)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
