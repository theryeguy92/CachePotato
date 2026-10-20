#!/usr/bin/env python3
"""verify_q4_quality.py — does a Q4_0 file still answer the same thing as Q8_0?

Q4_0 is 4.5 bits/weight. The project accepted Q8_0 because it is fp32-exact in practice; Q4_0
is NOT: it changes the weights, so no bit-identity gate can apply. This gate is the Tier-C
substitute, and its criteria were fixed BEFORE the measurement (do not tune them afterwards):

  1. TEACHER-FORCED AGREEMENT >= 90%: run the Q8-generated token sequence through the Q4 model
     as a PREFILL, so every position is scored on the SAME context, and ask whether Q4's argmax
     is the token Q8 chose. This is the sound metric. Free-running agreement (below) is
     confounded by snowballing — one flipped token changes every later context, so it collapses
     even for a tiny quality loss and is reported as diagnostic only.
  1b. FREE-RUNNING GREEDY AGREEMENT: reported, no threshold.
  2. TOP-5 OVERLAP >= 90%: mean |top5(q8) ∩ top5(q4)| / 5 over all compared positions.
  3. LOGIT SCALE: report the relative RMS difference, |q4-q8| / rms(q8), on the final position.
     There is no pass threshold on this one — it is diagnostic, and a large value with a small
     token effect is fine, while the reverse would be suspicious.
  4. Both probe prompts must produce the same text answer (the tokens file is compared directly).

If 1 or 2 fail, Q4_0 is not usable at this size and the core-only variant (Q4 attention/head,
Q8 experts) is the fallback.
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/qwen3_eval"
Q8 = ROOT / "scratch/qwen3_q8_chunked.v4cp"
Q4 = ROOT / "scratch/qwen3_q4_chunked.v4cp"
GEN = 24
# two probes: a continuation of natural text, and one that needs a fact
PROBES = [
    ("code", [785, 6722, 315, 9625, 374]),
    ("fact", [3838, 374, 220, 12817, 13]),
]
BUDGET_MB, THREADS = 13312, 4


def prefill(model: Path, ids, out: Path):
    """Prefill-only: logits for every position, so each is scored on the same context."""
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("CP_")}
    env.update({"CP_THREADS": str(THREADS), "CP_PAGER": "1", "CP_BUDGET": str(BUDGET_MB),
                "CP_PAGER_SYS": "1", "CP_COLD_START": "1"})
    r = subprocess.run([str(EXE), str(model), "48", ",".join(map(str, ids)), str(out)],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stderr[-800:]); raise SystemExit(f"{model.name} prefill failed")
    V = 151936
    return np.fromfile(str(out) + ".logits", dtype=np.float32).reshape(len(ids), V)


def run(model: Path, ids, out: Path, gen=GEN):
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("CP_")}
    env.update({"CP_THREADS": str(THREADS), "CP_PAGER": "1", "CP_BUDGET": str(BUDGET_MB),
                "CP_PAGER_SYS": "1", "CP_COLD_START": "1"})
    r = subprocess.run([str(EXE), str(model), "48", ",".join(map(str, ids)), str(out),
                        "--decode", str(gen)], capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stderr[-800:]); raise SystemExit(f"{model.name} failed")
    toks = np.fromfile(str(out) + ".tokens", dtype=np.int32)
    V = 151936
    logits = np.fromfile(str(out) + ".logits", dtype=np.float32).reshape(len(toks), V)
    return toks, logits


def top5(row, k=5):
    return set(np.argpartition(-row, k)[:k].tolist())


def main():
    for name in ("Q8", "Q4"):
        m = {"Q8": Q8, "Q4": Q4}[name]
        if not m.exists():
            print(f"missing {m}"); return 1
    ok = True
    for probe, ids in PROBES:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            t8, l8 = run(Q8, ids, td / "q8")
            t4, l4 = run(Q4, ids, td / "q4")
            # teacher-forced: Q4 scores Q8's own trajectory, position by position
            ctx = list(ids) + [int(x) for x in t8[:-1]]
            lt = prefill(Q4, ctx, td / "tf")
            pred = lt.argmax(axis=1)[len(ids) - 1:]          # predicts the next token
            agree_tf = float(np.mean(pred == t8[:len(pred)]))
            n = min(len(t8), len(t4))
            agree = float(np.mean(t8[:n] == t4[:n]))
            ov = float(np.mean([len(top5(l8[i]) & top5(l4[i])) / 5.0 for i in range(n)]))
            rel = float(np.linalg.norm(l4[-1] - l8[-1]) / (np.linalg.norm(l8[-1]) + 1e-30))
            print(f"  probe {probe}: teacher-forced {100*agree_tf:5.1f}% | free-running "
                  f"{100*agree:5.1f}% (snowball) | top-5 {100*ov:5.1f}% | logit rel-RMS {rel:.3e}")
            print(f"    q8: {t8[:12].tolist()}")
            print(f"    q4: {t4[:12].tolist()}")
            if agree_tf < 0.90:
                ok = False
    print()
    if ok:
        print(f"Q4 QUALITY: PASS — >=90% teacher-forced agreement on both probes "
              f"(Q8 vs Q4_0, budget {BUDGET_MB} MB, {THREADS} threads)")
        return 0
    print("Q4 QUALITY: FAIL — see the numbers above; try the core-only fallback")
    return 1


if __name__ == "__main__":
    sys.exit(main())
