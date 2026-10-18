# Handoff — starting the pager for Qwen3-30B-A3B

Written 2026-10-18 at the end of the port work. **Read this first, then start at §5.** The point of
this document is that it contains the *state*, not the history.

## 1. Where things stand

Qwen3-30B-A3B (48 layers, 128 experts, top-8, 61.1 GB bf16) has been converted to CPW1 v4 at Q8 and
**the C engine reproduces it**: argmax 5/5 vs HuggingFace, cosine 0.99938, routing mass median
99.977%. That is the first time this project has computed a model larger than its RAM correctly.

**But it has never run on the T480.** 3.3 s for 48 layers was warm page cache on a 61 GB server. The
file is 32.5 GB and the potato has 16 GB, so the model cannot even be loaded there without paging.
**None of the paging thesis has been tested.** That is the whole of what remains.

## 2. The artifacts that exist (and where)

| artifact | location | notes |
|---|---|---|
| Q8 model file | server: `weights/qwen3_30b_a3b_q8.v4cp` | 32.5 GB, 18,867 tensors, 6,579 regions |
| its manifest | server: `weights/qwen3_30b_a3b_q8.json` | name → source tensor, offset, size |
| HF baseline | server: `data/traces/qwen3_30b_a3b_full.npz` | 586 arrays: logits, 48-layer intermediates, **routing trace** |
| unpaged eval output | server: `data/traces/eval_p0.{logits,route,router}` | the B5 control |
| checkpoint | server: `weights/qwen3-30b-a3b/` | 57 GB, 16 shards |
| 1-layer files (local) | T480: `scratch/qwen3_l1f.v4cp` (f32), `qwen3_a3.l4cp` (Q8) | for fast local iteration |

**The 32.5 GB file must come to the T480** for the paging work. Everything else can stay.

## 3. What is proven, and by which gate

- **QK-norm, top-8 routing, gated expert SwiGLU** in C: rel ~2e-06 against real weights
  (`tools/verify_qwen3_blocks.py`).
- **The v4 format path equals the blob path bit-for-bit** on fp32 (`tools/verify_qwen3_v4.py`, A5).
- **The file contains the checkpoint's weights**, all 18,867 payloads byte-identical
  (`models/qwen3-30b-a3b/verify/v4_values.py`).
- **The engine reproduces the model**: B5 (`tools/compare_qwen3_eval.py`).
- Gate suite: `./tools/run_gates.sh` — 17 gates, all passing.

## 4. Constraints that must not be broken

1. **Verification first.** A port's C output is gated against an independent reference. No tolerance
   is added to make something pass.
2. **Never regenerate a recorded baseline to make a test pass.** If B5 or Tier B fails, fix the code.
3. **Refactors must be bit-identical.** The pager is a refactor of *where bytes come from*, not of
   what is computed. At an unlimited budget it must reproduce the unpaged output exactly.
4. **Engine stays C11 + libm + pthread.** No new dependencies.
5. **Tier discipline.** Tier A/B claim exactness; Tier C claims quality. The Q8 artifact is Tier C.

## 5. The task: C1 — wire the pager into the Qwen3 path

**The seam already exists.** `qwen3_bind_layer` (`engine/blocks/qwen3_moe.c`) resolves one layer's
tensors from the mapping, and experts are stored as an **index** (`m->exp[e*3+kind]`) rather than a
pointer array — deliberately, so that an individual expert can be re-resolved between tokens. That
is where the pager attaches.

1. **`pager_open(file, budget_bytes)`** next to `qwen3_map`. The file stays mapped; expert **regions**
   are admitted through a budgeted cache instead of used directly.
2. **`pager_touch(region_id)`** for each of the 384 expert regions a layer needs (48 layers × 8
   experts). The GPT-2 path's `instr_touch` is the pattern.
3. **`pager_enforce()`** between layers, evicting LRU under the budget.
4. **Counters per token, not cumulative**: regions fetched, bytes fetched, hits, evictions. A
   cumulative counter divided by the wrong token count already cost a debugging detour once.

**Acceptance for C1 is uncompromising:** at an unlimited budget, output must be **bit-identical** to
the unpaged run. A pager that changes the answer is not a pager.

Then C2: measure tok/s, **regions/token**, bytes/token, hit rate — cold and warm, at **2, 4 and 8 GB**
budgets, with the unpaged run as control.

## 6. Decisions to make before starting

- **RAM budget.** The active working set is ~3.56 GB/token (≈818 regions). At 8 GB it almost fits
  (eviction pressure); at 2 GB it certainly thrashes. Run all three rather than choosing.
- **Headline metric** is **regions fetched per token**, with and without a Tier-B layout, not tok/s
  alone. tok/s is the outcome; read count is the mechanism the thesis actually claims to move.

## 7. Traps already found the hard way — do not rediscover these

- **Ties are the norm.** bf16 router logits tie the 8th/9th expert at *every one* of 240
  (layer, token) positions. The C block's tie-break ("strict `>`, lower expert id wins") decides
  routing at every token and must be documented where it is implemented.
- **C-vs-HF routing agreement is unachievable at Q8 and must never be a gate.** Judge routing by
  probability mass relative to the baseline's top-8 (95% floor; a wrong index carries ~12.5%), with
  a 99.9% median. **Tier B compares the engine to itself**, which is where exactness lives.
- **Units and scope.** Seven of my own errors in one session were unit/scope mistakes: MemFree vs
  MemAvailable, one GPU's VRAM as all VRAM, u16 into `unsigned`, `max(dims)` for an output dim, the
  fp32→Q8 ratio applied to a bf16 checkpoint. Each was caught by comparing against an independent
  expectation, never by re-reading code.
- **A threshold chosen by feel fails a correct model.** Derive the bar from what a real error would
  look like.
- **`build/` is gitignored**; a fresh clone has no such directory.
- **`weight_map` in the index, not shard 1.** The converter once read everything from shard 1 and
  silently omitted `model.norm` and the untied `lm_head`, which live in shard 16.

## 8. First commands

```sh
git checkout feat/qwen3-30b-a3b && git pull
./tools/run_gates.sh                    # expect 17 gates, fail=0
# bring the file over
scp <server>:~/projects/CachePotato/weights/qwen3_30b_a3b_q8.v4cp weights/
python3 tools/verify_v4_file.py --file weights/qwen3_30b_a3b_q8.v4cp   # structure + region digests
```

Verify the file on arrival — a 32.5 GB transfer deserves a structural check, and the tool exists.
