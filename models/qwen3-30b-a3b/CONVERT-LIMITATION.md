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
