# What this laptop can actually run: the corrected tok/s envelope, and the 2026 model landscape

**Date:** 2026-10-19 · **Branch:** `feat/qwen3-30b-a3b` · **Status: research note; two concrete ports proposed.**

The question: the measured 0.8 tok/s (13 GB, Q8, 30B-A3B) and my earlier estimate of 0.9–1.6 tok/s
for smaller models both felt too low. They were: the 0.9–1.6 figure was derived from the *pre-fix*
compute number, before the thread fix, the bind fix and the kernel work. This note re-derives the
envelope from today's measured cost model and checks it against the current open-weight landscape.

## The cost model, measured

`t = 374 ms compute + physical_MB / 0.389 GB/s` (fits the 8/11/13 GB arms and the unpaged control
to <1%). Compute scales with **active** parameters: **111.6 ms per 1B active** at 4 threads.

For a model that **fits**, the disk term vanishes (weights come from RAM, already inside the 374 ms).

*(Correction: the table below does not follow from 111.6 ms per 1B. Its values imply ~190 ms per 1B
active. At 111.6 ms the 0.5B would predict ~17.9 tok/s, and the measured figure is 15.5 tok/s, so
the table is conservative by roughly 1.5×. The ordering of the rows, which is what the argument
uses, is unaffected.)*

| model class | active/token | predicted tok/s | fits 16 GB? |
|---|---:|---:|---|
| 0.5B dense (calibration) | 0.5 B | 10.4 | yes |
| **1.5B dense** | 1.5 B | **3.45** | yes |
| **1.3B-active MoE** | 1.3 B | **3.98** | yes (7.3 GB at Q8) |
| **3B dense** | 3 B | **1.73** | yes |
| 2.7B-active MoE (14B total) | 2.7 B | 1.92 | yes at Q4 |
| 7B dense | 7 B | 0.74 | yes |
| 30B-A3B, Q8, 13 GB (measured) | 3.35 B | **0.80** | no — pages |

**So the honest envelope has two regimes, and my earlier single number blurred them:**

- **A model that fits** with ≤1.5B active runs at **3.4–4 tok/s**. 3B dense is 1.7. 7B dense is 0.74 —
  *slower than the 30B MoE*, which is the point of MoE on this machine.
- **A model that pages** is limited by compulsory expert reads: 348 MB/token at 13 GB → 895 ms of
  the 1252 ms token. That is where 0.8 comes from, and it is a **Q8 artifact**, not an engine limit
  (see below).

## What the 2026 landscape offers

| model | total / active | arch | portable here? | predicted on this laptop |
|---|---|---|---|---|
| **Qwen3-Coder-Next** (Feb 2026, Apache 2.0) | 80B / **3B** | hybrid **Gated DeltaNet + Gated Attention**, 48 L, **512 experts (10+1 shared)** | **no** — three new blocks | ~1.5 if ported (13 GB pages it) |
| Qwen3-Coder-480B-A35B | 480B / 35B | standard MoE | no (size) | n/a |
| **Qwen3-Coder-30B-A3B** (Jul 2025, Apache 2.0) | 30.5B / 3.3B | **identical to what we run** | **yes, drop-in** (only `rope_theta` 1e7 vs 1e6 differs) | 0.8 at Q8; **~1.3–1.5 at Q4** |
| Qwen2.5-Coder-3B | 3B dense | Qwen2 blocks | **yes, drop-in** (`qwen2.c`) | **1.73** |
| Qwen2.5-Coder-1.5B | 1.5B dense | Qwen2 blocks | **yes, drop-in** | **3.45** |
| OLMoE-7B-A1B | 6.9B / 1.3B | Llama-shaped MoE | yes (top-8, no shared) | **~4.0**, but not coding-tuned |
| Granite 4.0 | hybrid Mamba/attn | — | **no** | n/a |

Benchmark claims in the sources are vendor or community reported and harness-dependent — Qwen3-Coder-
Next's 70.6% SWE-bench Verified is with the SWE-Agent harness, and the same page notes GLM-4.7 scores
higher (74.2%). Treat them as pointers, not rankings.

## The finding that answers the question

**0.8 tok/s was never the engine's ceiling — it was Q8 paging.** The same model at Q4:

- expert regions roughly halve (~2.65 MB instead of ~5.0 MB), so the compulsory traffic drops from ~348 MB to
  ~100–150 MB/token → 260–380 ms of disk;
- compute rises (nibble unpacking is more expensive than Q8's byte→float), call it 1.5× → ~560 ms;
- total ≈ 0.8–0.95 s/token → **~1.1–1.4 tok/s**.

That is a **1.4–1.8× on the exact model we already run**, with no new architecture, no new model, and
the engine already has Q4_0 plus `verify_q4_decode`/`verify_q8_kernels` gates. The cost is Tier-C
quality (the vault records Q4_0 as rejected for the 0.5B; at 30B the same 4.5 bpw is a different
proposition and needs its own quality check).

And **3.4–4 tok/s is real**, but only for a model that fits with ≤1.5B active — i.e. Qwen2.5-Coder-
1.5B, which is a much weaker coder than a 30B.

## Two candidate next steps (both need a decision, not just work)

*(Update: option A was taken. Q4_0 on the experts measured 1.14 tok/s, and Q4_0 on the experts and
attention measured **1.22 tok/s**, passing a 120-token quality gate. See
[Where tok/s goes](2026-10-19-tok-s-levers.md), Rounds 5–7. Options B and C have not been
started.)*

**A. Q4 conversion of the current model (~1 h).** Add `q4_0` to the Qwen3 converter, convert
(≈25 min), verify with `verify_v4_file` (digests), measure at 13 GB. Predicted **1.1–1.4 tok/s**,
i.e. the best *quality-per-second* available on this machine; needs a Tier-C quality gate (B5-style
argmax/tolerance against the Q8 baseline) because 4.5 bpw is the low end.

**B. Port Qwen3-Coder-30B-A3B and run it at Q4 (~half a day).** Drop-in geometry (48 L, 128 experts,
top-8, d 2048, 32/4 heads, QK-norm, θ = 1e7), so it is a converter config + a `rope_theta` fix that
the loader needs anyway, then the same Q4 conversion and measurement. Predicted **~1.3–1.5 tok/s**
with a genuinely coding-trained model (vendor: ~50.3% SWE-bench Verified) — the best *usable* coding
model this laptop can serve, at the same speed as (A).

**C. Port Qwen3-Coder-Next — not now.** It is the model the industry uses for exactly this job
(70.6% SWE-bench Verified at 3B active, Apache 2.0), but it needs Gated DeltaNet (linear attention),
gated attention, a 512-expert MoE and shared-expert support: a month of engine work, and on 16 GB it
would page ~48 GB of Q4 weights, landing near ~1.5 tok/s anyway. The honest comparison is that it
would match (B)'s speed with much better coding — a decision to take when the engine is ready, not a
quick win.

## What is *not* available on this machine

- **60–70 tok/s** is what this model class does on a **64 GB** machine (community GGUF numbers for
  Coder-Next at Q5) because it *fits in RAM*. On 16 GB nothing fits that should page, and the SATA
  SSD's 0.389 GB/s is the wall.
- **More threads, better kernels, int8 activations**: compute is 374 ms of 1252 ms at 13 GB, so the
  whole compute programme is a 1.4× ceiling, and at Q4 the disk term shrinks further.
- **A bigger budget**: the curve saturates at 13 GB (38.3 regions/token = the compulsory demand).

## How to actually use it

```
make build/qwen3_chat
CP_PAGER=1 CP_BUDGET=13312 CP_THREADS=4 CP_PIN_CORES=1 \
  ./build/qwen3_chat scratch/qwen3_q4attn.v4cp 48 200
```

- `48` is the layer count (the v4 header has it; the driver is told), `200` is the per-answer
  token cap. Type a message, "exit" or ctrl-D to leave.
- **Budget:** 13312 MB (13 GiB) is the fast setting and leaves ~2.4 GiB of this machine's 15.4 GiB
  for everything else. Close heavy applications, and drop to 8192 if it starts swapping. A
  swapping run looks exactly like a broken model; for comparison, 8 GB at Q8_0 measured 0.36 tok/s.
- **Files:** `scratch/qwen3_q4attn.v4cp` is the fast one (4-bit experts + attention, 1.2 tok/s,
  93.3% / 97.5% next-token agreement with Q8). `scratch/qwen3_q8_chunked.v4cp` is the exact one
  (0.8 tok/s).
- Greedy only — there is no sampler, by design (see the header comment in `tools/qwen3_chat.c`).

**What it feels like, measured:** first prefill ~20 s cold (the pager has to read the layers it
needs), then ~0.8-0.9 s per token. A 60-token answer is about 50 s. Follow-up questions are
cheaper *and* faster (24-token prefill, 12.8 s) because the conversation stays in the KV cache and
the expert set is already resident — a two-turn test measured **2.22 tok/s** on the second answer
against 1.15 on the first.

**The bug worth remembering** (it is in the commit message): the first version of this driver
produced fluent-looking nonsense because it added only the expert output to the residual stream and
not the attention output. Nothing fails when you drop a residual — token 1 is just wrong, and every
token after it is confidently wrong. The eval driver, which was already gated, had the right line.

**Not a conversation partner at 0.8-1.2 tok/s.** It writes a correct in-place `strrev` in ~50
seconds, and that is the realistic unit of work: one self-contained question, one file, walk away.
Agentic loops (write → run → fix → repeat) need 10-30x this speed and are out of reach on this
machine with this model.
