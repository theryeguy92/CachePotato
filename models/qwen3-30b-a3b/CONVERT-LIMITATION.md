# The v4 writer does not scale to this model (found 2026-10-17)

`format/v4_writer.py` accumulates **every tensor's raw bytes in memory** (`raw=raw` on the tensor
record) and only writes at `write()`. That is fine for a 184M-parameter model and fatal for this
one: converting a single complete layer (128 experts × 3 matrices × 768 × 2048, in fp32) needs
~2.4 GB of expert bytes per layer, plus an equal amount again in the writer, so a two-layer test
file already OOMs on a 16 GB machine.

Measured: `convert.py --shard1` was killed by SIGTERM with 10 GB free. A 30B model at Q8 is ~32 GB
of payload — there is no amount of RAM that makes the current design work.

## What has to change

The writer must **stream**: assign offsets first, then write each tensor's payload as it is
produced, never holding more than one tensor. That is a contained change to `v4_writer.py`
(the offset assignment already happens in `_place()` before the payload is written, so the ordering
is in place; what is needed is to accept payloads from an iterator/callback rather than storing
`raw`).

Two consequences worth planning for:

1. **Region digests** are computed over payload bytes at write time. Streaming means digesting as
   each tensor is written, which is straightforward (CRC-32 is incremental) but changes where the
   region table is computed — it currently precedes the payload in the file, so either the region
   table is written twice (seek back and patch) or digests are computed in a first pass.
2. **The converter must not load the whole checkpoint.** `safe_open(...).get_tensor()` per expert
   is already lazy per tensor, so the converter can stay inside the streaming design if it writes
   each expert as it is read.

## Status

Not fixed. The shard-1 wiring test is blocked on it, not on the C side: the C layer already passes
against real weights through the flat blob (`tools/verify_qwen3_blocks.py`, rel ~2e-06), so what
remains unproven is the *format* path for this model, and the blocker is the writer's memory model
rather than anything about Qwen3.

## Update — half fixed (2026-10-17)

**Output streaming: DONE.** `write()` no longer builds the file in a `bytearray`; it writes tables,
then each payload in offset order, checking every offset against the computed layout. Peak output
memory is now one tensor instead of the whole file. `--layers 1` of the real model now converts and
validates:

    convert: scratch/qwen3_l1.v4cp (3737 MB) | tensors 394 blocks 3 regions 138
    V4 FORMAT: PASS (structure, regions, digests)

and an existing port still verifies byte-identically after the change (`v4_values: PASS`), so the
streaming path did not alter the format it emits.

**Still blocking the full 48-layer checkpoint: `add_tensor` MATERIALISES every payload.**
`raw=raw` keeps each tensor's bytes from `add_tensor()` until `write()`, so peak memory is still the
sum of all tensors — ~61 GB fp32, ~32 GB at Q8, ~16 GB at 4.5 bpw. For a model whose whole point is
being larger than RAM, the writer must accept a *source* (path + offset, or a callable) and fetch
bytes at write time. That is the remaining change, and it is the same shape of fix as the output
streaming: never hold more than one tensor.

Interaction to remember when it is done: region digests cover payload bytes and the region table
precedes the payload, so a lazy writer either reads payloads twice or writes placeholder digests and
patches them by seeking back. The second is one pass and the file is already seekable.

## Update 2 — lazy payloads land, and the blocker becomes TIME (2026-10-17)

`add_tensor_lazy` + a `_payload()` accessor: peak memory is now ONE tensor, and every tensor in this
converter is registered lazily (not just the experts). Q8 happens inside the source callable, so a
tensor is quantized as it is written.

    one layer, fp32 : 3737 MB
    one layer, Q8   :  993 MB   (3.76x, matching the 1.0625/4 bytes-per-param ratio)
    V4 FORMAT: PASS (structure, regions, digests)

**New blocker: conversion TIME, and it is two separate inefficiencies.**

1. `format/q8.py::encode_q8_0` loops over rows *in Python* (`for row in flat:`). For the embedding
   (151 936 rows) and for 18 867 expert tensors that is minutes of interpreter overhead, not
   arithmetic. It vectorises straightforwardly — the group math is already numpy.
2. Each payload is produced **twice**: once when the region digest is computed (the region table
   precedes the payload) and once when it is written. Two options, both cheap: cache the payload
   when the tensors are visited in the same order, or keep it lazy and accept reading twice —
   but reading twice means paying the encoder twice.

Measured: ~10 minutes for ONE layer. At 48 layers that is ~8 hours, which is not a conversion, it is
a workday. Both fixes are small and independent; neither changes the format.
