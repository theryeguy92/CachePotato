#!/usr/bin/env python3
"""v4_values.py — does the .v4cp file contain the SOURCE weights, or just plausible bytes?

`verify_v4_file.py` proves structure, regions and digests are self-consistent. It cannot tell
whether the payload is the right tensor: a file whose every payload was shifted by one expert would
pass every digest check it performs.

This re-derives each payload from the checkpoint and compares it byte for byte against the file at
the offset the converter recorded. Uses the converter's manifest rather than parsing the v4 tables,
so a fault in the table writer cannot hide a fault in the payload writer.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

def load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: v4_values.py <file.v4cp> <manifest.json> [--model DIR]"); return 2
    fpath, mpath = Path(args[0]), Path(args[1])
    model = Path(args[args.index("--model") + 1]) if "--model" in args else ROOT / "weights/qwen3-30b-a3b"
    man = json.load(open(mpath))
    q8 = load("q8", ROOT / "format/q8.py")
    index = json.load(open(model / "model.safetensors.index.json"))["weight_map"]
    from safetensors import safe_open
    handles = {}
    def src_bytes(name, enc):
        sh = index[name]
        if sh not in handles:
            handles[sh] = safe_open(str(model / sh), framework="pt")
        a = np.ascontiguousarray(handles[sh].get_tensor(name).float().numpy(), dtype=np.float32)
        return q8.encode_q8_0(a)[0] if enc == "q8_0" else a.tobytes()

    print("=" * 78); print("QWEN3 V4 VALUES: file payload vs the checkpoint"); print("=" * 78)
    bad = 0; shown = {}
    with open(fpath, "rb") as f:
        for t in man["tensors"]:
            f.seek(man["off_data"] + t["offset"])
            got = f.read(t["nbytes"])
            want = src_bytes(t["src"], t["enc"])
            ok = got == want
            if not ok:
                bad += 1
            cls = ("expert" if t["name"].startswith("expert.") else
                   t["src"].rsplit(".", 2)[-2] if "self_attn" in t["src"] else
                   "embed" if "embed_tokens" in t["src"] else
                   "head" if "lm_head" in t["src"] else "norm")
            d = shown.setdefault(cls, [0, 0]); d[0] += ok; d[1] += 1
    for cls, (ok, n) in sorted(shown.items()):
        print(f"  {cls:10} {ok}/{n} payloads byte-identical")
    print()
    print(f"V4 VALUES: {'PASS' if not bad else f'FAIL ({bad})'} — "
          f"{sum(d[1] for d in shown.values())} tensors compared")
    return 0 if not bad else 1

if __name__ == "__main__":
    raise SystemExit(main())
