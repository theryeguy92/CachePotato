#!/usr/bin/env python3
"""verify/v4_values.py — the v4 file must CONTAIN the published weights.

Structural validation says the file is well-formed; it does not say the bits are the right bits.
This compares every tensor in the file against the safetensors it was converted from, exactly.

The parser here is deliberately independent of format/v4_writer.py: a checker that shares the
writer's code cannot detect a writer bug. It caught its own bug first — each table is preceded by
a u32 count (SPEC §5), and starting at the first record instead of the count mis-parses every
name, which is why an earlier version of this check silently compared nothing.
"""
import json, struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

def read_v4_tensors(path):
    buf = Path(path).read_bytes()
    magic, ver, hb, fl = struct.unpack_from("<4I", buf, 0)
    assert magic == 0x34575043, hex(magic)
    nt, nb, nr, ne = struct.unpack_from("<4I", buf, 16)
    oe, ot, ob, og, od = struct.unpack_from("<5Q", buf, 32)
    o = ot
    count = struct.unpack_from("<I", buf, o)[0]; o += 4      # count prefix, then records
    assert count == nt, (count, nt)
    tens = {}
    for _ in range(nt):
        n = struct.unpack_from("<H", buf, o)[0]; o += 2
        name = buf[o:o + n].decode(); o += n
        dtype = struct.unpack_from("<I", buf, o)[0]; o += 4
        rank = buf[o]; o += 1
        shape = struct.unpack_from("<" + "I" * rank, buf, o); o += 4 * rank
        off, ln = struct.unpack_from("<2Q", buf, o); o += 16
        o += 4 + 2
        tens[name] = (shape, off, ln, dtype)
    return buf, tens, od

def main():
    ap_file = sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen.v4cp"
    model = ROOT / "weights/qwen2.5-0.5b-instruct"
    import torch
    from safetensors.torch import load_file
    sd = {k: v.to(torch.float32).numpy()
          for k, v in load_file(str(model / "model.safetensors")).items()}
    cfg = json.load(open(model / "config.json"))
    L = cfg["num_hidden_layers"]

    buf, tens, od = read_v4_tensors(ROOT / ap_file)
    print("=" * 78); print("V4 VALUES: does the file contain the published weights?"); print("=" * 78)
    print(f"  {ap_file}: {len(tens)} tensors")

    def got(nm):
        shape, off, ln, dt = tens[nm]
        return np.frombuffer(buf, dtype="<f4", count=ln // 4, offset=od + off).reshape(shape)

    def stack(nm):
        return np.stack([sd[f"model.layers.{l}.{nm}"].astype(np.float32) for l in range(L)])

    expect = {"embed": sd["model.embed_tokens.weight"], "norm": sd["model.norm.weight"],
              "ln1": stack("input_layernorm.weight"), "ln2": stack("post_attention_layernorm.weight"),
              "Wq": stack("self_attn.q_proj.weight"), "bq": stack("self_attn.q_proj.bias"),
              "Wk": stack("self_attn.k_proj.weight"), "bk": stack("self_attn.k_proj.bias"),
              "Wv": stack("self_attn.v_proj.weight"), "bv": stack("self_attn.v_proj.bias"),
              "Wo": stack("self_attn.o_proj.weight"),
              "Wg": stack("mlp.gate_proj.weight"), "Wu": stack("mlp.up_proj.weight"),
              "Wd": stack("mlp.down_proj.weight")}
    bad = 0
    for nm, want in expect.items():
        if nm not in tens:
            print(f"  {nm:6} MISSING"); bad += 1; continue
        ok = np.array_equal(got(nm), np.ascontiguousarray(want, dtype=np.float32))
        if not ok: bad += 1
        print(f"  {nm:6} {str(got(nm).shape):20} identical={ok}")
    # the materialised zero bias must actually be zero, not uninitialised
    if "bo" in tens:
        ok = bool(np.all(got("bo") == 0.0))
        print(f"  bo     {str(got('bo').shape):20} all-zero={ok}  (materialised; Qwen has no o bias)")
        bad += not ok
    print()
    print("V4 VALUES: PASS" if not bad else f"V4 VALUES: FAIL ({bad})")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
