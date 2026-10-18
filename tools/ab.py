#!/usr/bin/env python3
"""ab.py — interleaved A/B benchmark with an explicit noise floor.

Two rules this exists to enforce:

1. INTERLEAVE. Run A,B,A,B,... inside one time window. Sequential runs of the same binary
   measured 15.5 tok/s earlier and 12.7 an hour later — a 20% drift that is larger than the
   optimisations being chased, so any sequential comparison is meaningless.

2. REFUSE TO DECLARE A WINNER INSIDE THE NOISE. Report the medians AND the spread, then only
   claim a difference if it exceeds the observed noise. A harness that always names a winner is
   the same failure as a gate that always passes.

Usage: ab.py A_BINARY B_BINARY [--runs 5] [--ntok 32] [--model PATH]
Validate it first with the SAME binary twice: it must report "within noise".
"""
import argparse, re, shutil, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def one(exe, model, ntok):
    r = subprocess.run([str(exe), str(model), str(ntok)], capture_output=True, text=True,
                       cwd=ROOT, env={**__import__("os").environ, "CP_THREADS": "4"})
    m = re.search(r"ms_per_token=([0-9.]+)", r.stdout + r.stderr)
    if not m:
        print(f"  {exe.name}: no BENCH line:\n{r.stdout[-300:]}{r.stderr[-300:]}"); sys.exit(1)
    return float(m.group(1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--ntok", type=int, default=32)
    ap.add_argument("--model", default="scratch/qwen_q8e.v4cp")
    args = ap.parse_args()
    a, b = Path(args.a), Path(args.b)
    A, B = [], []
    for i in range(args.runs):
        # ALTERNATE THE ORDER. Running A then B every pair gives B a systematic advantage or
        # penalty from thermal ramp / cache state; with two IDENTICAL binaries that bias made B
        # "win" 5/5 at p=0.0625. Alternating cancels it, and the identical-binary validation is
        # what caught it.
        if i % 2 == 0:
            A.append(one(a, args.model, args.ntok))
            B.append(one(b, args.model, args.ntok))
        else:
            B.append(one(b, args.model, args.ntok))
            A.append(one(a, args.model, args.ntok))
    def stats(v):
        return statistics.median(v), min(v), max(v)
    ma, amin, amax = stats(A)
    mb, bmin, bmax = stats(B)
    spread = max(amax - amin, bmax - bmin) / min(ma, mb)
    diff = (ma - mb) / ma
    print(f"  A {a.name:22} median {ma:7.3f} ms/token  [{amin:.3f}..{amax:.3f}]")
    print(f"  B {b.name:22} median {mb:7.3f} ms/token  [{bmin:.3f}..{bmax:.3f}]")
    print(f"  unpaired spread {spread*100:5.1f}%   difference {diff*100:+6.2f}% (B vs A)")

    # PAIRED analysis. This machine throttles from 3100 MHz to 2593 MHz under sustained load,
    # so absolute times drift by ~20% within a session. Comparing per-PAIR differences (A_i
    # against the B_i measured seconds later) cancels that drift; comparing two lists does not.
    pairs = [a_i - b_i for a_i, b_i in zip(A, B)]
    pdiff = statistics.median(pairs)
    wins = sum(1 for d in pairs if d > 0)
    print(f"  paired: median delta {pdiff:+7.3f} ms  B faster in {wins}/{len(pairs)} pairs")
    consistent = min(wins, len(pairs) - wins)
    if consistent == 0:
        # All pairs agree in sign: the SIGN TEST is the statistic, not the effect size against
        # its own spread. The first version compared |median delta| against median|delta|, which
        # are the same number when the effect is consistent — so it called a 3.7x difference
        # "noise". A test that cannot see the largest effect in the suite is not a test.
        pval = 2.0 / (2 ** len(pairs))
        verdict = "faster" if pdiff > 0 else "slower"
        print(f"  VERDICT: B {'faster' if pdiff > 0 else 'slower'} in ALL {len(pairs)} pairs, "
              f"median {pdiff:+.1f} ms; sign-test p={pval:.4f}"
              + ("" if pval <= 0.05 else "  (WEAK: collect more pairs)"))
        return 0
    print(f"  VERDICT: within noise (split {wins}/{len(pairs)-wins}). Not a result — "
          f"collect more pairs.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
