#!/usr/bin/env python3
"""q8.py — Q8_0 encode/decode, and the reference the C decoder is checked against.

GROUP LAYOUT (34 bytes, 32 elements), matching llama.cpp's Q8_0 so published checkpoints need
no re-encoding:

    bytes 0..1   fp16 scale `d`  (little-endian)
    bytes 2..33  32 x int8

    decode: y[i] = (float)q[i] * d

WHY THE DECODER IS CHECKED AGAINST A PYTHON ONE, BIT FOR BIT
  The format spec (§6) requires it, and the reason is concrete: a decoder that is wrong by one
  bit produces confident nonsense rather than a crash. `tools/verify_q8_decode.py` compares the
  C decoder with this one on real tensors and requires EXACT equality, not a tolerance.

  The scale is fp16, which is an exact subset of fp32, so `q[i] * (float)d` has exactly one
  correct answer in fp32 arithmetic — there is no tolerance to hide in.

CUDA/AVX note: nothing here needs a dependency. fp16 decode is bit manipulation.
"""
import struct

import numpy as np

GROUP = 32
BLOCK_BYTES = 34


def _f16_bits(x):
    """fp32 -> fp16 bits (round-to-nearest-even), for writing scales."""
    return np.float16(x).view(np.uint16)


def encode_q8_0(x):
    """Vectorised: ONE numpy pass over all groups instead of a Python loop per row.

    The loop version was correct and cost ~10 minutes for one layer of Qwen3-30B-A3B, because
    18 867 expert tensors mean interpreter overhead rather than arithmetic. The group math was
    already numpy; only the iteration was not. Output is BIT-IDENTICAL — `d` is still rounded to
    fp16 FIRST and the quantisation then divides by the stored value, so the scale and the values
    agree exactly as before.
    """
    a = np.ascontiguousarray(x, dtype=np.float32)
    flat = a.reshape(-1, 32)
    amax = np.abs(flat).max(axis=1)
    d = np.where(amax > 0, amax / 127.0, 0.0).astype(np.float32)
    dh = d.astype(np.float16)                      # fp16-rounded scale, as stored
    dqf = dh.astype(np.float32)[:, None]
    live = dqf > 0
    safe = np.where(live, dqf, np.float32(1.0))
    q = np.clip(np.rint(flat / safe), -127, 127).astype(np.int8)
    q = np.where(live, q, np.int8(0)).astype(np.int8)

    out = np.empty((flat.shape[0], 34), dtype=np.uint8)
    out[:, 0:2] = dh.view(np.uint16).reshape(-1, 1).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes(), a.shape

def decode_q8_0(blocks, shape, dtype=np.float32):
    """Decode Q8_0 blocks back to floats. This is the reference the C decoder must equal."""
    n = int(np.prod(shape))
    if n % GROUP != 0:
        raise ValueError("element count is not a multiple of 32")
    ngroups = n // GROUP
    raw = np.frombuffer(blocks, dtype=np.uint8, count=ngroups * BLOCK_BYTES)
    raw = raw.reshape(ngroups, BLOCK_BYTES)
    d = raw[:, 0:2].copy().view(np.float16).astype(dtype).reshape(ngroups, 1)
    q = raw[:, 2:34].copy().view(np.int8).astype(dtype)
    return (q * d).astype(dtype).reshape(shape)


def block_count(n):
    return (n + GROUP - 1) // GROUP
