"""verify_toy.py — prove the C forward pass matches the numpy ground truth.

Runs ./build/cachepotato, parses its printed logits for tokens [5,1,7],
recomputes the expected logits with the same seeded weights as
forward_by_hand.py, and reports the max absolute difference.

Exit 0 on PASS, 1 on FAIL. Run from repo root:  python3 tools/verify_toy.py
"""

import os
import subprocess
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
TOL = 1e-4  # C prints 6 decimals; float32 op-order dust is ~1e-6


def expected_logits():
    rng = np.random.default_rng(42)
    VOCAB, D, H, D_FF = 8, 4, 2, 8
    HD = D // H
    T = 3

    def rw(*s):
        return rng.standard_normal(s, dtype=np.float32) * 0.2

    W = {"token_emb": rw(VOCAB, D), "Wq": rw(D, D), "Wk": rw(D, D),
         "Wv": rw(D, D), "Wo": rw(D, D), "W1": rw(D, D_FF), "W2": rw(D_FF, D),
         "unemb": rw(VOCAB, D)}

    def ln(v, eps=1e-5):
        return (v - v.mean(axis=-1, keepdims=True)) / \
               np.sqrt(v.var(axis=-1, keepdims=True) + eps)

    x = W["token_emb"][[5, 1, 7]]
    xn = ln(x)
    q, k, v = xn @ W["Wq"], xn @ W["Wk"], xn @ W["Wv"]
    qh, kh, vh = [a.reshape(T, H, HD) for a in (q, k, v)]
    att = np.zeros((H, T, T), dtype=np.float32)
    for h in range(H):
        s = qh[:, h, :] @ kh[:, h, :].T / np.sqrt(HD)
        s = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, s)
        e = np.exp(s - s.max(axis=-1, keepdims=True))
        att[h] = e / e.sum(axis=-1, keepdims=True)
    x = x + np.concatenate([att[h] @ vh[:, h, :] for h in range(H)], axis=1) @ W["Wo"]
    x = x + np.maximum(0, ln(x) @ W["W1"]) @ W["W2"]
    return ln(x) @ W["unemb"].T


def main():
    out = subprocess.run(["./build/cachepotato"], capture_output=True,
                         text=True, cwd=REPO).stdout
    block = out.split("logits:\n")[1].strip().split("\n")
    got = np.array([[float(v) for v in line.split()] for line in block[:3]])

    exp = expected_logits()
    diff = np.abs(got - exp).max()
    print(f"max abs diff vs numpy ground truth: {diff:.2e}  (tol {TOL:.0e})")
    if diff < TOL:
        print("PASS — C forward pass matches the numpy reference")
        return 0
    print("FAIL")
    print("expected:\n", exp)
    print("got:\n", got)
    return 1


if __name__ == "__main__":
    sys.exit(main())
