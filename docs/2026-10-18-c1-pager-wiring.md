# C1 — the pager is attached to the Qwen3 path, and the answer did not move

2026-10-18, branch `feat/qwen3-30b-a3b`. Follows `docs/2026-10-18-pager-handoff.md` §5.
This is the wiring and its acceptance, not the measurement (C2).

## What was attached, and where

The seam is `qwen3_bind_layer`, and the unit is the file's own region (SPEC §8):

| piece | file | what it does |
|---|---|---|
| `pager_open_v4(fd, base, off_data, budget)` | `engine/pager/pager.c` | v4 front-end: the pager takes regions from the file's region table instead of deriving them from v3 shape constants |
| `pager_add_region(off, len, is_expert)` | `engine/pager/pager.c` | called once per region while `qwen3_map` walks the table; the id returned **is** the table index |
| `tensor_reg[n_tens]` | `qwen3_map` | tensor → region lookup, so `qwen3_bind_layer` can record ids rather than re-deriving anything |
| `reg_exp[e]`, `reg_ln1`, … | `qwen3_bind_layer` | the region ids of the layer that was just bound |
| `pager_touch_region(id)` | `qwen3_layer` | 9 core regions + 1 per selected expert, per token, BEFORE the read |
| `pager_enforce()` | `qwen3_eval.c` | between layers, after `qwen3_layer` returns |
| `pager_step(t)` / `<prefix>.pager` | `qwen3_eval.c` | per-token rows: regions fetched, bytes fetched, hits, evictions |

Three choices worth stating:

1. **Region ids are the file's table indices.** The loader, the pager and the report
   mean the same thing by "region 1234", and a touch cannot describe bytes the file did
   not declare pageable. The alternative — engine-side region construction, as the v3
   front-end must do — would be a second description of where the bytes are.
2. **The 435 non-expert regions are counted.** The budget covers the whole working set,
   so at 2 GB the ~1.6 GB of embed/head/attention/router competes with the experts. That
   is honest and it is the regime the model is in; `pager_report()` splits core from
   experts so anchor thrash cannot be mistaken for a routing effect.
3. **Off unless `CP_PAGER=1`.** The unpaged run is the B5 control; a pager that attached
   itself would change the memory behaviour of every existing gate. `CP_BUDGET=<MB>`
   alone also enables it (the engine's existing convention).

## Acceptance: bit-identity (the gate that cannot be traded)

The control is **the same binary's unpaged run on this machine**, not `eval_p0`:

```
  .logits      3038720 bytes unpaged vs unlimited-budget pager: BIT-IDENTICAL
  .route          7680 bytes unpaged vs unlimited-budget pager: BIT-IDENTICAL
  .router       122880 bytes unpaged vs unlimited-budget pager: BIT-IDENTICAL
```

and with a budget tight enough to evict:

```
  .logits   at CP_BUDGET=2048 MB: BIT-IDENTICAL
  .route    at CP_BUDGET=2048 MB: BIT-IDENTICAL
  .router   at CP_BUDGET=2048 MB: BIT-IDENTICAL
  CP_BUDGET=2048 MB: tok 0 819 regions/0 evict, tok 1 356/0, tok 2 200/0, tok 3 179/0, tok 4 136/1196
```

At 8192 MB the same run evicts 0 times, so 2048 MB is the budget that exercises the
policy: 1196 evictions, output unchanged. Eviction is `MADV_DONTNEED` on a file mapping,
which drops pages without altering them — as it must, or the pager is a numerics change.

### Why `eval_p0` cannot be the bit-identity control

`data/traces/eval_p0.*` was recorded on the AI server. Comparing this machine's unpaged
run to it:

| artifact | result |
|---|---|
| `.route` | **bit-identical** — routing does not depend on summation order |
| `.logits` | max abs diff 1.28e-04, rel 4.62e-06; 98.4% of float32 values differ in the last bits; argmax equal |
| `.router` | differs in the last bits, same reason |

`-march=native` picks different vector code on the two CPUs, so the summation order
differs and the low bits follow. `eval_p0` therefore gates **tolerance and argmax** (B5),
not byte equality; byte equality is a same-machine, same-binary claim. Both were run:
B5 `PASS` (argmax 5/5, cosine 0.99938345, mass median 99.977%), C1 `PASS`.

## Accounting checked against the file, not against itself

819 regions / 3600.2 MB for token 0 was predicted from the region table plus the emitted
routing trace — 435 core = 48×9 (ln1, ln2, q_norm, k_norm, Wq, Wk, Wv, Wo, Wgate) + 3
globals (embed, final_norm, lm_head), 384 experts = 48×8 chosen — and only then compared
with the pager's row:

```
  file says token 0 touches 435 core + 384 expert regions = 819 regions, 3600.2 MB
  pager says token 0 fetched 819 regions, 3600.2 MB
  over 5 tokens: 1690 region fetches, 0 evictions (unlimited budget: evictions must be 0)
    token 0:  819 regions   3600.2 MB    0 hits    0 evictions
    token 1:  356 regions   1784.8 MB  463 hits    0 evictions
    token 2:  200 regions   1002.7 MB  619 hits    0 evictions
    token 3:  179 regions    897.4 MB  640 hits    0 evictions
    token 4:  136 regions    681.8 MB  683 hits    0 evictions
```

3600.2 MB/token agrees with the independent arithmetic in `models/qwen3-30b-a3b/MODEL.md`
§2 (3.35 B active params × 1.0625 B/param ≈ 3.56 GB) to ~1%, the difference being
region granularity vs parameter count. Hits + fetches = 5 × 819 = 4095, which is the
number of touches the layers make — a per-token table that did not add up to the access
count would be the cumulative-vs-per-token error this project has already paid for.

## The pager refactor did not disturb the GPT-2 path

`pager.c` gained a `pager_mode`, an `is_expert` flag and step counters, and its report's
core/expert split moved from `i >= expert_base` to the flag. Compared against a binary
built from `HEAD`'s `pager.c`, same input, `CP_BUDGET=200` on the v3 file:

```
404 regions, 950 accesses, 163 hits, 1684.3 MB fetched, 992 evictions, 1744 MB evicted
```
identical in both, as was the generated text; only tok/s differed.

## Artifacts carry their machine

A baseline that cannot name its machine cannot say which claim it supports. Every
`qwen3_eval` run now writes `<prefix>.meta` next to its traces:

```
host: T480
cpu: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz
isa: avx2 fma sse4.2
compiler: 16.2.1 20260810
git: af22acf
model_bytes: 32479225664
pager: CP_BUDGET=512 MB
```

`isa` is the line that matters: it reports what the compiler actually selected
(`__AVX2__`, `__FMA__`, `__AVX512F__`), not what `-march` requested. The AI server reports
`avx512f`; the T480 does not. That is the mechanism behind the logits difference in the
table above, and it is now visible in the artifact rather than left to be inferred from
`-march=native`.

## What C1 does not decide

- **C2 measurement** (tok/s, regions/token, bytes/token, hit rate; cold and warm at 2, 4
  and 8 GB, with the unpaged control) — next, and the counters are per token precisely
  so that table can be read without dividing anything.
- **Tier B** — there is no permuted Qwen3 layout in the repo yet, so the
  "with and without a Tier-B layout" column cannot exist until one does. C1 ships the
  counter that will read it: `regions_fetched / token` at a fixed budget.
- **Prefetch.** `CP_PREFETCH_W` is v3 policy and is not read by the v4 front-end; a silent
  W=4 would make a warm run look like a policy win. C1 pages on demand only.
