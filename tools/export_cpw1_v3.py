#!/usr/bin/env python3
"""export_cpw1_v3.py — export a trained CachePotato-MoE checkpoint to CPW1 v3.

Writes only the LIVE parameters (strips the dense-FFN weights the trainer
allocates on MoE layers but never uses) in a layout whose routed-expert region is
contiguous per (layer, expert), so one pager region = one sequential read.

Usage:
  python tools/export_cpw1_v3.py --ckpt weights/moe-scale-run/l05/model_only.pt \
      --out weights/moe-scale-run/l05.cpw1v3
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import cpw1_v3 as v3      # noqa: E402
import train_moe as tm    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-moe-from", type=int, default=None,
                    help="refuse to export unless the checkpoint's layout matches")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = tm.Config(**ck["config"])
    if args.expect_moe_from is not None and cfg.moe_from != args.expect_moe_from:
        raise SystemExit("checkpoint moe_from=%d, --expect-moe-from=%d"
                         % (cfg.moe_from, args.expect_moe_from))

    model = tm.MoEModel(cfg)
    model.load_state_dict(ck["model"], strict=True)   # proves the layout matches

    live = v3.payload_floats(cfg)
    alloc = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    v3.write(args.out, ck["model"], cfg)
    dt = time.time() - t0

    size = v3.HDR_SIZE + live * 4
    print("exported %s" % args.out)
    print("  step         : %s" % ck.get("step"))
    print("  layout       : moe_from=%d  MoE %s  dense %s"
          % (cfg.moe_from, cfg.moe_layers(), cfg.dense_layers()))
    print("  allocated    : %s params" % f"{alloc:,}")
    print("  live (written): %s params (%.1fM)" % (f"{live:,}", live / 1e6))
    print("  stripped     : %s dead params (dense FFN on MoE layers)"
          % f"{alloc - live:,}")
    print("  file         : %s bytes (%.0f MB), header %d bytes"
          % (f"{size:,}", size / 1e6, v3.HDR_SIZE))
    print("  expert block : %s floats = %s bytes  (one pager region)"
          % (f"{v3.expert_floats(cfg):,}", f"{v3.expert_floats(cfg) * 4:,}"))
    print("  wrote in %.1fs" % dt)


if __name__ == "__main__":
    main()
