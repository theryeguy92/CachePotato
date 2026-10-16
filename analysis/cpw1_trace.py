#!/usr/bin/env python3
"""cpw1_trace.py — shared plumbing for reading recorded routing traces.

One trace schema for every model, so analysis code is model-agnostic:

    .npz with
      tokens      (W, T)             uint16   token ids, W windows of T tokens
      sel         (L, W, T, K)       int16    top-K expert ids per window/token
      moe_layers  (L,)               int16    global layer index of each MoE layer
      topk        scalar             int16
      lam         scalar             float32
      T           scalar             int32
      eos         scalar             int64    EOS token id (resets engine state)
      precision / step / start       provenance

  `sel` is indexed by MoE SLOT (0..L-1) in the same order as `moe_layers`, which is
  also the order the weight file stores expert blocks in. That correspondence is what
  lets an analysis map (slot, expert) -> region -> byte offset without the model.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trace:
    tokens: np.ndarray      # (W, T) uint16
    sel: np.ndarray         # (L, W, T, K) int16
    moe_layers: np.ndarray  # (L,) int16
    topk: int
    lam: float
    eos: int
    path: Path

    @property
    def n_slots(self) -> int:
        return int(self.sel.shape[0])

    @property
    def n_windows(self) -> int:
        return int(self.sel.shape[1])

    @property
    def window_len(self) -> int:
        return int(self.sel.shape[2])

    @property
    def n_experts(self) -> int:
        """Expert-id space size, from the largest id present (ids are dense)."""
        return int(self.sel.max()) + 1

    def summary(self) -> str:
        return (f"{self.path.name}: lam={self.lam:.2f} top{self.topk} "
                f"{self.n_slots} MoE slots x {self.n_windows} windows x "
                f"{self.window_len} tokens | {self.n_experts} experts | eos={self.eos}")


def load_trace(path) -> Trace:
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    return Trace(
        tokens=z["tokens"].astype(np.int64),
        sel=z["sel"].astype(np.int64),
        moe_layers=z["moe_layers"].astype(np.int64),
        topk=int(z["topk"]) if "topk" in z.files else int(z["sel"].shape[-1]),
        lam=float(z["lam"]) if "lam" in z.files else float("nan"),
        eos=int(z["eos"]) if "eos" in z.files else 50256,
        path=path,
    )


# ---------------------------------------------------------------- latency model

# Measured on the T480 (i5-8250U, SanDisk SD8SB8U SATA SSD) with
# POSIX_FADV_DONTNEED + random pread on a never-cached file, 2026-10-16.
# read_size_bytes -> (microseconds per read, MB/s)
# NOTE: absolute values are contested (the vault records 227us for 4K from an earlier
# run); the SHAPE of the curve is the load-bearing part. See the file docstring of
# analysis/traffic.py and bench/results/layout/ for the discrepancy.
T480_COLD = {
    4096:    (1705.3, 2.40),
    16384:   (1756.2, 9.33),
    65536:   (1938.5, 33.81),
    262144:  (2958.0, 88.62),
    1048576: (4819.1, 217.59),
    4194304: (13171.3, 318.44),
}


class LatencyModel:
    """Cold-read cost as a function of region size.

    Deliberately a step function over measured points, not a fitted formula: the
    measurement's meaning is in the discrete sizes, and the model should not invent
    resolution it does not have. Reads larger than the largest measured point are
    charged the largest point's bandwidth (optimistic, and flagged by `extrapolated`).
    """

    def __init__(self, table=None):
        self.table = dict(table or T480_COLD)
        self.sizes = sorted(self.table)

    def extrapolated(self, region_bytes: int) -> bool:
        return region_bytes > self.sizes[-1]

    def cost(self, region_bytes: int, n_reads: float):
        """(seconds, MB moved) for `n_reads` cold reads of `region_bytes` each."""
        if n_reads <= 0:
            return 0.0, 0.0
        if region_bytes <= self.sizes[0]:
            us, mbs = self.table[self.sizes[0]]
            n_bytes = region_bytes
        elif region_bytes >= self.sizes[-1]:
            us, mbs = self.table[self.sizes[-1]]
            n_bytes = region_bytes
        else:
            lo = max(s for s in self.sizes if s <= region_bytes)
            us, mbs = self.table[lo]
            n_bytes = region_bytes
        moved = n_reads * n_bytes / 1e6
        return n_reads * us / 1e6, moved

    def describe(self) -> str:
        return ", ".join(f"{s//1024}K={self.table[s][0]:.0f}us" for s in self.sizes)
