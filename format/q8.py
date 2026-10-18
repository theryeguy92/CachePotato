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
    """Quantize a float array (any shape, last axis a multiple of 32) to Q8_0 blocks.

    Returns (blocks_bytes, shape). One fp16 scale per 32 elements; q = round(x/d) clamped to
    [-127, 127] (NOT -128: Q8_0 uses symmetric range and reserves -128, and using it would
    decode 1 LSB off from every other implementation).
    """
    a = np.ascontiguousarray(x, dtype=np.float32)
    if a.shape[-1] % GROUP != 0:
        raise ValueError(f"last dim {a.shape[-1]} is not a multiple of {GROUP}")
    flat = a.reshape(-1, GROUP)
    out = bytearray()
    for row in flat:
        amax = float(np.abs(row).max())
        d = amax / 127.0 if amax > 0 else 0.0
        dh = np.float16(d)
        # quantize against the scale that will actually be STORED, not the ideal one
        dq = float(dh)
        if dq > 0:
            q = np.rint(row / dq).astype(np.int32)
            q = np.clip(q, -127, 127).astype(np.int8)
        else:
            q = np.zeros(GROUP, dtype=np.int8)
        out += np.array([dh], dtype=np.float16).tobytes()
        out += q.tobytes()
    return bytes(out), a.shape


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
