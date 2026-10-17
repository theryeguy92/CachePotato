# CPW1 v4 — CachePotato weight-file format

**Status:** DRAFT for review. No loader implements this yet. CPW1 v2/v3 remain the live
format and are untouched.

**Scope.** This document specifies a self-describing weight file: a block list, a tensor
table, a region table, and an encoding table. It exists so the engine can run models it did
not train, with no model-specific constants compiled into the engine.

**How to read the fact markers.** Every architectural statement below is either:

- **VERIFIED** — read out of a published config or the engine's own source, with the source named.
- **ASSUMED** — a design decision or an inference not yet confirmed. Ports must resolve these.

---

## 1. Axioms

1. **The file describes itself.** No engine-side constant may encode a model's shape: not
   expert width, not expert count, not which layers are sparse, not the routing activation.
   The engine walks tables; it never assumes a predicate like "layer ≥ moe_from is sparse".
   *This is a direct response to a measured failure: CPW1 v3's `moe_from` predicate silently
   disagreed with the frozen architecture spec while every gate passed.*
2. **Everything is addressed by byte offset.** Tensors carry explicit `(offset, length)`.
   A loader must be able to build the whole model from the tables with no arithmetic that
   the writer did not also do.
3. **Loud failure.** Any inconsistency — unknown block kind, unknown encoding, offset that
   leaves the file, overlapping tensors, region that is not inside its tensor — is a load
   error. Never a warning, never a fallback.
4. **One canonical in-memory representation.** Encodings are a storage detail. The engine
   decodes to fp32 and its kernels are encoding-agnostic. A block never learns its dtype.
5. **Which bytes are pageable is declared, not inferred.** The file says what a region is.
6. **Region integrity is checkable without the exporter.** Every region carries a content
   digest, so a misalignment is caught at load rather than producing plausible garbage.

## 2. Conventions

| | |
|---|---|
| Endianness | little-endian |
| Integers | unsigned `u16/u32/u64`, signed `i32` |
| Floats | IEEE-754 `f32` |
| Default alignment | 64 bytes for tensor offsets/starts. Lengths need not be 64-aligned; instead each encoding declares its own required alignment (§4), and a length must be a multiple of the encoding's group block size |
| Strings | `u16` length + bytes, **no** NUL terminator |
| Counts | every table's count precedes its entries |

## 3. File layout

| offset | field | type |
|---:|---|---|
| 0 | `magic` = `"CPW4"` | u32 |
| 4 | `version` = 4 | u32 |
| 8 | `header_bytes` | u32 |
| 12 | `flags` | u32 |
| 16 | `n_tensors` | u32 |
| 20 | `n_blocks` | u32 |
| 24 | `n_regions` | u32 |
| 28 | `n_encodings` | u32 |
| 32 | `off_encodings` | u64 |
| 40 | `off_tensors` | u64 |
| 48 | `off_blocks` | u64 |
| 56 | `off_regions` | u64 |
| 64 | `off_data` — payload start; all tensors live at or after this | u64 |
| 72 | `reserved[4]` = 0 — nonzero is a load error | u32 ×4 |
| | **`header_bytes` = 88** | |

Then the four tables in the order listed, then zero padding to `off_data`, then the raw
tensor bytes. `header_bytes` is the size of the fixed header so a future version can extend
it; a loader that does not know `version` refuses the file.

**This table is asserted, not hand-computed.** Earlier drafts of this document said 64 and
then 68 — both simple miscounts of the same 16 + 16 + 40 + 16 bytes. The reference writer
now builds the header from one explicit `struct.pack` and asserts its length against
`HDR_BYTES`, so a miscount cannot reach a file. A number that is computed by hand twice and
wrong twice should be computed by the machine instead.

## 4. Encoding table

Storage encodings are declared once and referenced by index, so a port can add one without
touching tensor records.

| field | type | meaning |
|---|---|---|
| `id` | u32 | encoding id, unique within the file |
| `kind` | u8 | `0` raw float, `1` block-quantized |
| `scalar` | u8 | `0` f32, `1` bf16, `2` f16, `3` q8_0, `4` iq2_xxs, `5` nvfp4 |
| `group_size` | u16 | elements per quantization group (0 for raw) |
| `block_bytes` | u16 | **bytes per full group block, including scales** |
| `sub_blocks` | u8 | number of scale sub-blocks per group (1 unless a format has more) |
| `flags` | u8 | bit 0: has per-group scale; bit 1: has per-group min; bit 2: sub-byte packed |
| `align` | u8 | required byte alignment for tensors in this encoding (1 = none) |
| `name` | string | human name, e.g. `"IQ2_XXS"` (diagnostics only) |

**VERIFIED encoding facts** (Nemotron 3.5 Bolt quantized `config.json`
`quantization_config.config_groups`, 2026-10-16):

| name | bpw | group_size | block_bytes | used for |
|---|---:|---:|---:|---|
| `Q8_0` | 8.5 | 32 | 34 | Latent-MoE `fc{1,2}_latent_proj`, attention `{q,k,v,o}_proj`, `lm_head`, embeddings |
| `IQ2_XXS` | 2.0625 | 256 | 66 | routed-expert `up_proj` / `down_proj` |
| `NVFP4` | 4 | — | — | Mamba `in_proj` / `out_proj` (the config's `group_1` records **no** `quant_algo`; the model page says NVFP4 — **ASSUMED**, confirm from the checkpoint or llama.cpp's enum) |

`Q8_0` and `IQ2_XXS` are llama.cpp/ggml `packing` schemes; `block_bytes` above is the
published size. The exact bit layout of `IQ2_XXS` is a hierarchical codebook and is **not**
specified here — see §6.

## 5. Tensor table

| field | type | meaning |
|---|---|---|
| `name` | string | e.g. `language_model.backbone.layers.3.mixer.experts.17.up_proj` |
| `dtype_id` | u32 | index into the encoding table |
| `rank` | u8 | number of dimensions, 1–4 |
| `shape` | u32[rank] | **logical** element shape, in this file's own order |
| `offset` | u64 | byte offset from the start of the file |
| `length` | u64 | byte length as stored (may differ from logical size when quantized) |
| `block_id` | u32 | owning block, or `0xFFFFFFFF` for shared/global tensors |
| `tensor_role` | u16 | semantic role flag (below), for diagnostics and region naming |

A loader validates, before mapping anything:

- `offset + length ≤ file_size`, and `offset ≥ off_data` unless the tensor is zero-length;
- every `offset` is a multiple of 64 **and** of its encoding's `align`;
- `length` is a multiple of its encoding's group block size (so a group is never split
  across the end of a tensor);
- no two tensors overlap, and no two share a byte;
- for quantized dtypes, `length == product_over_dims_rounded_to_group / group_size * block_bytes`
  when the tensor is contiguous — and if it is **not** contiguous, the tensor is refused
  (see §7.4).

**Ordering is explicit.** The table is the order. CPW1 v3's payload order is *a* valid
ordering that a v4 writer for that model may emit; it is not a rule.

**Why lengths are not required to be 64-aligned** (measured 2026-10-16): expressing the
existing CPW1 v3 file as a v4 tensor table gives 27 tensors whose payload end lands exactly
on the file size — but every v3 offset is `60 mod 64`, because v3's header is 60 bytes. An
"every length is 64-aligned" rule would have rejected all 27. Offset alignment is what
`madvise` and `mmap` actually need; length alignment is not, and demanding it would make the
format unable to express the one model the project already has — which is the acceptance test
in §9.2. A length must still be a multiple of its encoding's group block size, or a group
would be split and the tensor would not decode.

`tensor_role` values: `0` other, `1` embedding, `2` norm, `3` attention projection,
`4` router, `5` expert (routed), `6` expert (shared), `7` dense FFN, `8` output head,
`9` SSM/conv state-producing weight, `10` inference state (never pageable — see §8.3).

## 6. Encodings, and the dependency problem

**Raw float (kinds 0).** Element `i` at `offset + i * sizeof(scalar)`. f32/bf16/f16.

**Block-quantized (kind 1).** The tensor is a sequence of groups. Group `g` occupies
`block_bytes` and decodes to `group_size` fp32 values. The internal layout of a group
(where the scale sits, how low bits are packed) is **format-specific** and must be:
(a) named by `scalar`, (b) implemented by the port's decoder, and (c) reproduced exactly by
the port's Python decoder against the C decoder.

**This is where the project must make a decision, and it should be made deliberately.**
`IQ2_XXS`'s full definition is a hierarchical codebook (a 256-entry grid, an 8-bit sign
pattern, and a per-8-element scale). Implementing it exactly, in dependency-free C11, is the
single highest-risk item in this whole format. Three options:

1. **Engine implements ggml's schemes directly** — matches published checkpoints byte for
   byte, but couples the engine to someone else's bit layout and is a large, subtle kernel.
2. **The port's `convert.py` decodes to a simpler encoding the engine implements** (for
   example a 4-bit affine group-32 scheme) — keeps the engine's vocabulary small and its
   decoders testable, at the cost of re-encoding (and a possible quality change that must be
   measured, not assumed).
3. **Both**: format supports ggml schemes, but no port is required to use them.

The spec is written to allow all three, and **recommends 2 for the first port**, because the
engine's rule is "no runtime dependencies" and because a decoder that is wrong by one bit in
a codebook index produces confident nonsense rather than a crash.

## 7. Block table

A block is one logical layer. Its record is a **discriminated union on `kind`**: the common
fields, then kind-specific fields. Blocks appear in execution order.

**Common header**

| field | type | meaning |
|---|---|---|
| `kind` | u8 | see below |
| `layer_index` | u32 | position in the model's layer stack |
| `n_inputs` | u8 | number of tensors this block reads by name |
| `input_refs` | u32[n_inputs] | tensor ids, semantic order defined per kind |
| `n_params` | u8 | number of kind-specific `u32` parameters |
| `params` | u32[n_params] | meaning defined per kind |
| `n_outputs` | u8 | tensors this block produces (usually 1: the residual stream) |

**Kinds and their meanings.** Verified against Nemotron 3.5 Bolt's `layers_block_type` +
`block_configs` (88 entries) — the spec's stress case.

| kind | name | input_refs (in order) | params |
|---|---|---|---|
| `0x01` | `embedding` | token table | — |
| `0x02` | `norm` | weight, bias | `eps_bits` (f32 bit pattern), `kind` (0 layernorm, 1 rmsnorm) |
| `0x10` | `attention` | Wq,bq,Wk,bk,Wv,bv,Wo,bo | `n_heads`, `n_kv_heads`, `head_dim`, `rope_theta_bits`, `qk_norm`, `sliding_window` |
| `0x11` | `dense_ffn` | W1,b1,W2,b2 | `act` (0 gelu_tanh, 1 silu, 2 relu2), `d_ff`, `dense_index`, `dense_packed` |
| `0x20` | `moe` | router W,b; shared W1,b1,W2,b2; routed expert blocks | `n_experts`, `top_k`, `scoring` (0 softmax, 1 sigmoid), `norm_topk_prob`, `routed_scaling_bits`, `latent_size`, `n_shared`, `shared_overlap`, `group_limited` (`n_group`, `topk_group`) |
| `0x21` | `moe_latent` | as `moe`, plus down/up latent projections | `latent_size` ≠ 0 and the latent projections present |
| `0x30` | `ssm` (mamba) | `in_proj`, `out_proj`, conv, `A_log`, `D`, `dt_bias` | `n_heads`, `head_dim`, `state_size`, `conv_kernel`, `chunk_size`, `dt_min_bits`, `dt_max_bits` |

`dense_packed` exists because a model can legitimately hold *packed dense FFN weights on a
layer at a MoE position* — weights that exist in the checkpoint and are never executed, only
to keep tensor shapes uniform. The project's own served model does exactly this on its last
layer (an accepted, documented deviation in the decision log). The flag lets a file say
"this layer's FFN is dense even though the engine's layout rule would call the slot MoE",
so the format can express the model honestly instead of either rejecting it or loading
untrained weights. A block with `dense_packed != 0` does not contribute a MoE slot.

**Layer indices are execution positions, not a count.** A trailing `norm` block carries
`layer = n_layer` (it runs after the last layer), so a loader must derive the layer count
from the highest layer that has an FFN block, not from the maximum layer index in the table.

**Why a union rather than a predicate.** Bolt's own config is the proof: `layers_block_type`
is an explicit 88-entry list interleaving `mamba`/`moe`/`attention`, and every MoE entry
carries *its own* `n_routed_experts` and `moe_intermediate_size` — varying per layer across
{240,304,368,432,496} and {1536,2048,2304,2560}. There is no uniform expert stride in that
model, so "expert width" cannot be a file-level constant, let alone an engine constant.

**How expert weights are addressed — a decision the port must declare.** Bolt stores each
expert as **two tensors** (`experts.N.up_proj`, `experts.N.down_proj`) — VERIFIED from the
config. CPW1 v3 stores each expert as **one contiguous block** (W1e,b1e,W2e,b2e). At least
three arrangements are possible, and the port's `MODEL.md` must state which it emits:

1. **interleaved-pair** (recommended default): one tensor per expert, containing
   `[up | down]` (plus biases if any), so one expert is one contiguous, pageable unit;
2. **split**: one tensor per projection, so a region spans `2 * n_experts` tensors — smaller
   regions but an expert is no longer one sequential read;
3. **per-layer packed**: one tensor holding all experts of a layer, with regions cutting it.

The format supports all three. The pager's behaviour differs materially between them, which
is exactly the kind of thing Tier B is meant to measure rather than assume.

## 8. Region table

The pager's unit. A region is a byte range the engine may evict and re-fault; the file
declares them so that granularity is a property of the data, not of engine code.

| field | type | meaning |
|---|---|---|
| `offset` | u64 | byte offset of the region |
| `length` | u64 | byte length |
| `tensor_lo` | u32 | first tensor covered (a region may cover several) |
| `tensor_hi` | u32 | one past the last tensor covered |
| `kind` | u8 | `0` tensor, `1` expert, `2` group, `3` pinned |
| `slot` | u16 | for `expert`/`group`: the block's layer index; else `0xFFFF` |
| `member_lo` | u32 | for `expert`: first expert id in the region; for `group`: group id |
| `member_hi` | u32 | one past the last member |
| `digest` | u64 | FNV-1a over the region's bytes, verified at load |

**Rules the loader enforces**

1. Regions do not overlap, and every region lies entirely inside exactly one tensor or
   spans a contiguous run of tensors.
2. A region must be re-faultable: it maps to a file range that can be `mmap`ed and whose
   pages can be dropped. Inference state (KV cache, SSM state, conv state) is `kind: 3`
   `pinned` and is **never** evicted — it is not weight data at all.
3. At least one region must cover every tensor the block table references, or the file is
   refused (a referenced tensor outside all regions could not be paged and would silently
   read through the page cache).
4. **Non-contiguity is a load error.** A pageable tensor whose rows are packed with a
   different group size than its encoding declares must be refused.

**Expert regions and grouping.** `kind: 1` with a `slot` and a `[member_lo, member_hi)`
expert range is how a file expresses "these N experts form one pageable unit". CPW1 v3 is
exactly `member_lo = e, member_hi = e + 1` for every expert. A port may emit a coarser
grouping, and Tier B measures it.

## 9. Gates

A v4 file is accepted only if all of these hold. Gates 1–2 are format conformance; 3–5 are
the project's existing numerical discipline.

1. **Structure.** Every table parses; offsets, lengths and alignment close over the file
   size exactly; no overlapping tensors; every region digest matches; every referenced
   tensor is covered by a region.
2. **v3 round-trip.** `models/gpt2-moe/convert.py` must be able to emit a v4 file for the
   existing model that the v4 loader loads, and then reproduce the existing recorded
   artifacts: routing **exactly** and logits **to a stated tolerance** (see §10).
   *This is the acceptance test for the format itself: a format that cannot express the one
   model we have completely is not ready.*
3. **Reference equality.** C output equals the port's Python `reference.py`, exactly at the
   routing level.
4. **Permutation symmetry.** Reordering expert regions with a matching router permutation
   must leave routing exactly invariant (already measured for v3: 0/10240 mismatches).
5. **Unknown ⇒ refuse.** An unknown block kind, unknown encoding, or an encoding the binary
   lacks must fail to load. Tested by a deliberately corrupted fixture.

## 10. Exactness, and one numerics change to approve

Measured on v3 (see `bench/results/layout/2026-10-16-permutation-exactness.md`): permuting
expert weights with a matching router permutation leaves **routing exactly invariant** but
moves the logits by ~5.7e-06, because the router's columns are re-summed in a different
order.

Two ways to specify this:

- **Current behaviour.** Routing exact, logits fp32-equivalent. Cross-layout logits
  comparisons need a tolerance.
- **Recommended: canonical accumulation.** Require the routed output to be summed in a
  canonical order (expert id ascending) *after* selection. Then a permutation is a
  **bit-exact** symmetry and every gate stays `max|Δ| = 0`.

The second is cleaner and makes the project's strongest property — bit-exact gates — survive
layout compilation. **It is a numerics change to the existing engine and therefore requires
explicit approval before implementation.** Until then, v4 specifies the first behaviour with
a stated tolerance.

## 11. What a port must ship

```
models/<port>/
  MODEL.md        architecture facts, each marked VERIFIED (with source) or ASSUMED
  convert.py      source checkpoint -> CPW1 v4; must be resumable/streaming, since a
                  BF16 Bolt checkpoint is ~148 GB and the target disk is 177 GB
  reference.py    op-for-op Python forward; the numerical oracle
  reshape/        optional edits, each declaring BIT_IDENTICAL or ROUTING_CHANGED
  verify/         this port's gates, wired into tools/verify_*
```

A port **declares its tier** (A as-is / B layout / C co-design — see
`docs/2026-10-16-universal-ports.md`). A `BIT_IDENTICAL` reshape must pass the base model's
reference; a `ROUTING_CHANGED` reshape needs its own recorded trace.

## 12. Open questions

1. **`NVFP4`'s exact block layout** for Bolt's Mamba projections (ASSUMED above).
2. **Which expert arrangement** (§7.3) the first open-weight port emits — decision is a
   measurement (Tier B), not a preference.
3. **Canonical accumulation** (§10) — needs approval; decides whether permutation stays
   bit-exact.
4. **Where the encoding *table* ends and the port begins.** If two ports use different
   encodings for the same tensor role, is that a file property (yes, by design) or should
   the engine normalise on load (a cost)?
5. **Vision tensors.** Bolt is a VLM. Since the decision was text-LLM-first, this spec does
   not yet define an image-block kind; if the full VLM is ever in scope it needs one.
