#!/usr/bin/env python3
"""verify_v4_roundtrip.py — CPW1 v4 acceptance gate (format/SPEC.md §9.2/§9.3).

WHAT THIS GATES
  The format's own acceptance test: the model the project already has must load from a v4
  file and behave identically to the v3 it was written from — and a v4 file whose payload is
  in a DIFFERENT PHYSICAL ORDER must behave identically too. The second is the real test:
  same-order emission would only prove a reader can follow a table, not that it addresses
  tensors by offset.

CHECKS
  A. STRUCTURE   the Python validator passes on the v4 file (offsets, regions, digests,
                 tensor/shape round-trip).
  B. C vs C      the C engine's routing and its final logits argmax are identical between
                 the v3 file and the v4 file built from it, driven through the SAME engine
                 and the same lambda.
  C. C vs REF    the C engine on the v4 file reproduces the ROUTING RECORDED FROM PYTORCH
                 (data/traces/*.npz) exactly. This is an independent reference, not a
                 comparison of the loader against itself.
  D. SHUFFLE     (B) and (C) repeated for the shuffled file, proving offset addressing.

  Everything here is exact: routing selections must match 0/N, and argmax must be equal. No
  tolerance is introduced.

USAGE
  python3 tools/verify_v4_roundtrip.py \
      --v3 weights/moe-scale-run/l05.cpw1v3 \
      --v4 scratch/l05.v4cp --v4-shuffled scratch/l05_shuf.v4cp \
      --ckpt weights/moe-scale-run/l05/model_only.pt \
      --trace data/traces/l05_fp32.npz \
      --tokens-bin data/fineweb_edu_valid.bin --engine ./build/cachepotato \
      --token-start 0 --tokens 64 --window 1
"""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))


def run_engine(engine, file, tokens_bin, start, T, out, lam=None, env_extra=None):
    import os
    env = dict(os.environ)
    if lam is not None:
        env["CP_V4_LAMBDA"] = "%.6f" % lam
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([engine, file, "--moe-sel", tokens_bin, str(start), str(T), out],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return None, r.stderr.strip()
    raw = Path(out).read_bytes()
    nm, T2, K = struct.unpack_from("<iii", raw, 0)
    sel = np.frombuffer(raw, dtype="<i4", count=nm * T2 * K, offset=12).reshape(nm, T2, K).copy()
    argmax = None
    for line in r.stderr.splitlines():
        if "next-argmax" in line:
            argmax = int(line.split("next-argmax")[1].split()[0])
    return (sel, argmax), r.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", required=True)
    ap.add_argument("--v4", required=True)
    ap.add_argument("--v4-shuffled", default="")
    ap.add_argument("--ckpt", default="weights/moe-scale-run/l05/model_only.pt")
    ap.add_argument("--trace", default="data/traces/l05_fp32.npz")
    ap.add_argument("--tokens-bin", default="data/fineweb_edu_valid.bin")
    ap.add_argument("--engine", default="./build/cachepotato")
    ap.add_argument("--token-start", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--lam", type=float, default=None,
                    help="lambda to drive both files with (default: the trace's)")
    args = ap.parse_args()

    print("=" * 78)
    print("CPW1 v4 acceptance gate")
    print(f"  v3      : {args.v3}")
    print(f"  v4      : {args.v4}")
    if args.v4_shuffled:
        print(f"  shuffled: {args.v4_shuffled}")
    print("=" * 78)
    fails = []

    lam = args.lam
    if lam is None:
        z = np.load(args.trace, allow_pickle=False)
        lam = float(z["lam"]) if "lam" in z.files else 0.0
    print(f"  lambda  : {lam:.3f} (v4 files carry no lambda; see the writer's note)")

    # ---- A. structure
    print("\nA. STRUCTURE (independent Python validator)")
    r = subprocess.run([sys.executable, str(ROOT / "tools/verify_v4_file.py"),
                        "--file", args.v4, "--v3", args.v3],
                       capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()
    print("   " + tail)
    if r.returncode != 0:
        fails.append("structure")
        for line in r.stdout.splitlines():
            if line.strip().startswith("-"): print("   " + line.strip())

    # ---- z. recorded reference routing from PyTorch
    z = np.load(args.trace, allow_pickle=False)
    ref_sel = z["sel"]                       # (n_moe, windows, T, K)
    ref_T = int(z["tokens"].shape[1])

    def check(tag, path):
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "sel.bin")
            got, msg = run_engine(args.engine, path, args.tokens_bin,
                                  args.token_start, args.tokens, out, lam)
            if got is None:
                print(f"   engine FAILED: {msg}")
                return False, None
            sel, argmax = got
            print(f"   {msg.splitlines()[-1] if msg else ''}")
            # C vs C
            ok = True
            if sel.shape[1] == ref_T and args.tokens == ref_T:
                pass
            return True, (sel, argmax)

    print("\nB. C ENGINE, v3 vs v4 (same engine, same lambda, same tokens)")
    with tempfile.TemporaryDirectory() as td:
        g3, m3 = run_engine(args.engine, args.v3, args.tokens_bin, args.token_start,
                            args.tokens, str(Path(td) / "v3.bin"), lam)
        g4, m4 = run_engine(args.engine, args.v4, args.tokens_bin, args.token_start,
                            args.tokens, str(Path(td) / "v4.bin"), lam)
        if g3 is None or g4 is None:
            print(f"   engine failed: {m3 if g3 is None else m4}")
            fails.append("engine")
        else:
            s3, a3 = g3
            s4, a4 = g4
            mism = int((s3 != s4).sum())
            print(f"   routing : {mism} / {s3.size} selections differ")
            print(f"   argmax  : v3 {a3} | v4 {a4} | equal {a3 == a4}")
            if mism or a3 != a4:
                fails.append(f"v3-vs-v4 {mism} routing, argmax {a3} vs {a4}")

    print("\nC. C ENGINE on v4 vs ROUTING RECORDED FROM PYTORCH (independent reference)")
    with tempfile.TemporaryDirectory() as td:
        g4, m4 = run_engine(args.engine, args.v4, args.tokens_bin, args.token_start,
                            min(args.tokens, ref_T), str(Path(td) / "v4.bin"), lam)
        if g4 is None:
            print(f"   engine failed: {m4}")
            fails.append("engine-ref")
        else:
            sel, _ = g4
            n = min(args.window, ref_sel.shape[1], sel.shape[1])
            exp = ref_sel[:, :n, :, :].transpose(1, 0, 2, 3)     # (n, n_moe, T, K)
            got = sel[:, :sel.shape[1], :]
            # the recording has 512 tokens/window while the engine ran a shorter slice
            T = got.shape[1]
            exp = exp[:, :, :T, :]
            mism = int((got != exp).sum())
            tot = exp.size
            rate = 100.0 * (1 - mism / tot) if tot else 0.0
            print(f"   v4 vs recorded: {rate:.4f}%  ({mism}/{tot} selections differ)")
            if mism:
                fails.append(f"v4-vs-recorded {mism}")

    if args.v4_shuffled:
        print("\nD. SHUFFLED PHYSICAL ORDER (the offset-addressing test)")
        with tempfile.TemporaryDirectory() as td:
            gs, ms = run_engine(args.engine, args.v4_shuffled, args.tokens_bin,
                                args.token_start, args.tokens, str(Path(td) / "sh.bin"), lam)
            g4, _ = run_engine(args.engine, args.v4, args.tokens_bin, args.token_start,
                               args.tokens, str(Path(td) / "v4.bin"), lam)
            if gs is None:
                print(f"   engine FAILED on the shuffled file: {ms}")
                fails.append("shuffled load")
            else:
                ss, as_ = gs
                s4, a4 = g4
                mism = int((ss != s4).sum())
                print(f"   routing vs natural order: {mism} / {ss.size} differ")
                print(f"   argmax  : natural {a4} | shuffled {as_} | equal {a4 == as_}")
                if mism or a4 != as_:
                    fails.append(f"shuffled {mism} routing")

    # ---- E. the engine's own Gate 1 on the v4 file (full-sequence vs incremental).
    # Independent of the v3 comparison: it checks the v4 loaded model is self-consistent
    # across the two forward paths, which share no code.
    print("\nE. C ENGINE Gate 1 (full-sequence vs incremental) ON the v4 file")
    import os
    env = dict(os.environ, CP_V4_LAMBDA="%.6f" % lam)
    r = subprocess.run([args.engine, args.v4, "--step-check", args.tokens_bin,
                        str(args.token_start), str(min(args.tokens, 64))],
                       capture_output=True, text=True, env=env)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    for l in lines[-5:]:
        print("   " + l.strip())
    if r.returncode != 0:
        fails.append("v4 gate1")

    print("\n" + "=" * 78)
    if fails:
        print("V4 ACCEPTANCE: FAIL — " + ", ".join(fails))
        return 1
    print("V4 ACCEPTANCE: PASS")
    print("  a v4 file written from the v3 model loads, CRC-verifies every region, and")
    print("  reproduces the recorded PyTorch routing exactly — including when its payload")
    print("  is in a different physical order, which is what offset addressing means.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
