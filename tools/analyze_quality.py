#!/usr/bin/env python3
"""analyze_quality.py — apply the PRE-REGISTERED Q2 rule to sampled output.

The rule was committed in the vault (Phase-7-Integration § Q2) before these
samples were generated:

  repetition rate R = 1 - (unique 4-grams / total 4-grams), per generation,
  averaged over prompts.

  PASS       R(λ=0.5) <= max(1.5 * R(λ=0), 0.15)
  FAIL       R(λ=0.5) >= 0.50, or >= 3 * R(λ=0) with R(λ=0.5) >= 0.20
  AMBIGUOUS  otherwise

Usage:
  python tools/analyze_quality.py --dir data/q2 --tokenizer weights/tokenizer.bin
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np


def load_vocab(path):
    """CPT1: magic, version, n_vocab, n_merges, then {len u16, bytes} per id."""
    b = Path(path).read_bytes()
    magic, _ver, n_vocab, _nm = struct.unpack_from("<4sIII", b, 0)
    assert magic == b"CPT1", magic
    off, vocab = 16, []
    for _ in range(n_vocab):
        (ln,) = struct.unpack_from("<H", b, off)
        off += 2
        vocab.append(b[off:off + ln])
        off += ln
    return vocab


def dec(vocab, ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", "replace")


def rule(r0, r5):
    if r5 <= max(1.5 * r0, 0.15):
        return "PASS"
    if r5 >= 0.50 or (r5 >= 3.0 * r0 and r5 >= 0.20):
        return "FAIL"
    return "AMBIGUOUS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/q2")
    ap.add_argument("--tokenizer", default="weights/tokenizer.bin")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    d = Path(args.dir)
    arms = {}
    for a in ("l00", "l05"):
        f = d / f"{a}.json"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        arms[a] = json.loads(f.read_text())

    print("=" * 78)
    print("Q2 — sampled quality, λ=0.5 (l05) vs λ=0 (l00)")
    print("=" * 78)
    for a in ("l00", "l05"):
        j = arms[a]
        print(f"  {a}: lam={j['lam']} step={j['step']} prompts="
              f"{len(j['modes']['sampled']['prompts'])} "
              f"max_new={j['max_new']} temp={j['temperature']} top_p={j['top_p']}")

    print("\n-- aggregate metrics (mean over prompts) --")
    print("  arm | mode    | distinct4 |  repetition R | unique-tok")
    R = {}
    for mode in ("sampled", "greedy"):
        for a in ("l00", "l05"):
            g = arms[a]["modes"][mode]["aggregate"]
            R[(a, mode)] = g["rep"]
            print("  %s | %-7s | %9.4f | %13.4f | %10.4f"
                  % (a, mode, g["distinct4"], g["rep"], g["uniq"]))

    print("\n-- pre-registered rule, sampled decoding (primary) --")
    r0, r5 = R[("l00", "sampled")], R[("l05", "sampled")]
    print("  R(λ=0)   = %.4f" % r0)
    print("  R(λ=0.5) = %.4f" % r5)
    print("  threshold: R(λ=0.5) <= max(1.5*R(λ=0), 0.15) = %.4f"
          % max(1.5 * r0, 0.15))
    verdict = rule(r0, r5)
    print("  ratio    = %.2fx" % (r5 / r0 if r0 > 0 else float("inf")))
    print("\n  Q2 VERDICT (sampled): %s" % verdict)

    g0, g5 = R[("l00", "greedy")], R[("l05", "greedy")]
    print("\n-- same rule, greedy decoding (continuity with the training logs) --")
    print("  R(λ=0)=%.4f  R(λ=0.5)=%.4f  ratio=%.2fx  -> %s"
          % (g0, g5, g5 / g0 if g0 > 0 else float("inf"), rule(g0, g5)))

    if Path(args.tokenizer).exists() and args.show:
        vocab = load_vocab(args.tokenizer)
        for mode in ("sampled", "greedy"):
            print("\n" + "=" * 78)
            print("SAMPLES — %s decoding (first %d prompts)" % (mode, args.show))
            print("=" * 78)
            for i in range(args.show):
                for a in ("l00", "l05"):
                    m = arms[a]["modes"][mode]
                    p = dec(vocab, m["prompts"][i])[-90:].replace("\n", " ")
                    g = dec(vocab, m["generations"][i]).replace("\n", " ")
                    print("\n[%s prompt %d] prompt ...%s" % (a, i, p))
                    print("  -> %s" % g[:300])
    print("\nFinal: %s" % verdict)


if __name__ == "__main__":
    main()
