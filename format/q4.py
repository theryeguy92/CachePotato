#!/usr/bin/env python3
"""q4.py — Q4_0 encode/decode (4.5 bpw) and the reference the C decoder must match.

GROUP LAYOUT (18 bytes, 32 elements):

    bytes 0..1   fp16 scale `d` (little-endian)
    bytes 2..17  16 bytes; byte k holds element k (LOW nibble) and element k+16 (HIGH nibble)

    decode: y[i] = (float)q[i] * d    with q in [-8, 7]

    The i / i+16 pairing is llama.cpp's Q4_0 exactly. It was chosen after consecutive-pair
    packing broke the SIMD path: with pairs, every 8-element vector of nibbles needs a STRIDE-2
    gather of x, whereas i / i+16 makes the low nibbles one contiguous run (0..15) and the high
    nibbles another (16..31), matching the x layout.

Encoding: d = max|x| / 8 (fp16-rounded FIRST, then used for quantization so the stored scale and
the values agree), q = clamp(round(x/d), -8, 7). The -8..7 range matches llama.cpp's Q4_0 so a
published checkpoint needs no re-encoding — the same rationale as Q8_0. Note that -8 is reachable
here (unlike Q8_0's unused -128) because llama.cpp's Q4_0 uses it.

WHY 4-BIT NOW, beyond size: the measured Q8 inner loop is neither latency- nor instruction-bound,
so ~9 GB/s is what this machine gives for streaming weights. Fewer bytes is the only lever left —
4-bit is ~1.9x fewer bytes than Q8 and ~7x fewer than fp32.
"""
import numpy as np

GROUP = 32
BLOCK_BYTES = 18

def encode_q4_0(x):
    a = np.ascontiguousarray(x, dtype=np.float32)
    if a.shape[-1] % GROUP:
        raise ValueError(f"last dim {a.shape[-1]} is not a multiple of {GROUP}")
    flat = a.reshape(-1, GROUP)
    out = bytearray()
    for row in flat:
        amax = float(np.abs(row).max())
        d = amax / 8.0 if amax > 0 else 0.0
        dh = np.float16(d)
        dq = float(dh)
        if dq > 0:
            q = np.clip(np.rint(row / dq), -8, 7).astype(np.int32)
        else:
            q = np.zeros(GROUP, dtype=np.int32)
        q = q & 0xF                                   # two's complement in 4 bits
        packed = ((q[:16].astype(np.int32) | (q[16:].astype(np.int32) << 4))
                  & 0xFF).astype(np.uint8)   # byte k = element k | element k+16
        out += np.array([dh], dtype=np.float16).tobytes()
        out += packed.tobytes()
    return bytes(out), a.shape

def decode_q4_0(blocks, shape, dtype=np.float32):
    n = int(np.prod(shape))
    if n % GROUP:
        raise ValueError("element count is not a multiple of 32")
    ng = n // GROUP
    raw = np.frombuffer(blocks, dtype=np.uint8, count=ng * BLOCK_BYTES).reshape(ng, BLOCK_BYTES)
    d = raw[:, 0:2].copy().view(np.float16).astype(dtype).reshape(ng, 1)
    b = raw[:, 2:18].copy()
    q = np.empty((ng, GROUP), dtype=np.int32)
    q[:, :16] = (b & 0x0F).astype(np.int32)      # elements 0..15
    q[:, 16:] = (b >> 4).astype(np.int32)        # elements 16..31
    q = np.where(q > 7, q - 16, q).astype(dtype)        # sign-extend the nibble
    return (q * d).astype(dtype).reshape(shape)
