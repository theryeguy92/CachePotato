"""verify_toy.py — prove the C forward pass matches the numpy ground truth.

Runs ./build/cachepotato (toy weights, tokens 5 1 7), parses its printed
logits, recomputes the expected logits with the same seeded weights as
forward_by_hand.py (v2 semantics: GELU, zero biases/pos_emb, TIED unemb),
and reports the max absolute difference.

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

    token_emb = rw(VOCAB, D)
    Wq, Wk, Wv, Wo = rw(D, D), rw(D, D), rw(D, D), rw(D, D)
    W1, W2 = rw(D, D_FF), rw(D_FF, D)

    def ln(v, eps=1e-5):
        return (v - v.mean(axis=-1, keepdims=True)) / \
               np.sqrt(v.var(axis=-1, keepdims=True) + eps)

    def gelu(v):  # tanh approximation, same as the C engine
        return 0.5 * v * (1 + np.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3)))

    x = token_emb[[5, 1, 7]]  # pos_emb is zeros in the toy file
    xn = ln(x)
    q, k, v = xn @ Wq, xn @ Wk, xn @ Wv
    qh, kh, vh = [a.reshape(T, H, HD) for a in (q, k, v)]
    att = np.zeros((H, T, T), dtype=np.float32)
    for h in range(H):
        s = qh[:, h, :] @ kh[:, h, :].T / np.sqrt(HD)
        s = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, s)
        e = np.exp(s - s.max(axis=-1, keepdims=True))
        att[h] = e / e.sum(axis=-1, keepdims=True)
    x = x + np.concatenate([att[h] @ vh[:, h, :] for h in range(H)], axis=1) @ Wo
    x = x + gelu(ln(x) @ W1) @ W2
    return ln(x) @ token_emb.T  # tied unembedding (v2)


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
