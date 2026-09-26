#!/usr/bin/env python3
"""Record the Part 1 plan (goals, risk map, gates, acceptance criteria) in the vault."""
import sys
from pathlib import Path

V = Path.home() / "vaults" / "CachePotato-Vault"
P7 = V / "04-Phases/Phase-7-Integration.md"
DECLOG = V / "03-Project-Management/Decision-Log.md"

PLAN = """## Part 1 — Incremental MoE generation (planned 2026-09-25, BEFORE implementation)

The REPL's generation loop is the KV-cached `forward_step`, which has no MoE branch — the engine refuses CPW1 v3 files rather than read the wrong weights. Part 1 closes that. Its real point is that the new path must be **proven equivalent** to the full-sequence `forward_moe`, which already reproduces PyTorch's routing at 100.0000%. A second MoE implementation that nobody verified against the first is exactly how the layer-split, EOS-reset and missing-λ bugs got in.

### Goals
- **G1 Functional** — `forward_step` runs a v3 model one token at a time: router, λ locality bias, EOS reset, shared expert, top-2 routed dispatch, with the KV cache.
- **G2 Equivalence (the real goal)** — for identical tokens and weights it produces the same logits *and identical top-2 routing* as `forward_moe`. Measured, not assumed.
- **G3 Cross-call state correctness** — the locality set is the engine's first sequence state beyond the KV cache. It must persist across calls, reset at EOS, and start empty at sequence start.
- **G4 Usable** — `-i` accepts a v3 file and generates with the existing sampling flags.
- **G5 Instrument** — a first honest incremental tok/s figure, so Part 3 has a working measuring stick.

**Non-goals:** paging (Part 2), the benchmark protocol (Part 3), prefetch, speed optimization. **Tolerance discipline:** routing-identical + logits within 1e-5 — never bit-exactness (existing risk-7 rule).

### Risk map
| # | Risk | Countermeasure | Gate that catches it |
|---|---|---|---|
| 1 | **The two attention implementations diverge** — `forward` is full-sequence, `forward_step` is KV-cached; each was verified against HF but never *against each other* at MoE scale, and a small numeric difference can flip a top-2 | Measure rather than assume. Both do the same ops in the same order, so expect bit-identical; treat any difference as a finding to investigate, not to tolerate away | Gate 1 |
| 2 | **Locality state reset per call** — λ silently becomes a no-op | `moe_prev` lives in `RunState`, not the stack; only `runstate_alloc`/`runstate_reset_sequence` and the EOS branch write it | Gate 1 + Gate 2 |
| 3 | **EOS reset missing in the new path** — the exact bug fixed in `src/train.c` on 2026-09-25 | Copy the reset semantics from the now-verified `moe_ffn`, not from memory | Gate 1 (EOS fixture) + Gate 2 (across an EOS) |
| 4 | **Dense-FFN slot bug** — `forward_step` still computes `W1 + l*D*F`, but v3 indexes by **dense slot**; latent today only because the REPL refuses v3 | Fix in the same change as the branch; make the dense branch the only place those pointers are computed | Gate 1 (layers 0 and 11 read wrong weights otherwise) + v2 regressions |
| 5 | **Scratch aliasing** — `forward_step` reuses one `h`/`proj`/`x`/`xn`, and `att` is reused per head | Dedicated `moe_prev` / `moe_p` / `moe_idx` / `moe_wt` in `RunState` | Gate 1 |
| 6 | **Top-k scan drift** — ties and slot ordering must match (strict `>`, lower id wins, slot 0 = larger prob) | Reuse the identical scan from `moe_ffn`; do not rewrite it | Gate 1 with near-ties |
| 7 | **A weak gate** — comparing only final logits passes while routing diverges, which is precisely how the EOS bug escaped | The gate compares **routing**, per layer, per position | gate design itself |
| 8 | **Compounding drift over a long generation** — each token feeds back | Test generation, not just a single window | Gate 2 |
| 9 | **Breaking the dense path** while refactoring the FFN half | Run the existing v2 regressions unchanged | Gate 3 |
| 10 | **Slower than expected** (2 expert matmuls + shared per layer per token, each expert a 1.18MB mmap region) | Measure; do not optimize in Part 1 | Gate 4 (instrument only) |

### Gates
- **Gate 1 — incremental ≡ full-sequence.** Both paths over identical tokens; require **routing identical** and logits within 1e-5. Run on the 24-token EOS fixture, a 512-token held-out window, and the recorded trace windows.
- **Gate 2 — generation equivalence.** Greedily generate 64 tokens via `forward_step`, and independently re-run `forward_moe` over the growing prefix each step; require the same token ids and identical routing at every step, including across an EOS. This is what catches state-carry bugs a single window cannot.
- **Gate 3 — v2 regressions.** Toy 4.27e-07, HF GPT-2 logits 4.00e-04 with argmax match, tokenizer exact, dense REPL sampling.
- **Gate 4 — instrument.** Report incremental tok/s; no bar (that is Q3).

### Acceptance criteria
1. Routing identical to `forward_moe` on the EOS fixture, a 512-token window, and the recorded trace windows.
2. Logits within 1e-5 on all of the above.
3. 64-token greedy generation matches full-sequence generation step for step, including across an EOS.
4. All v2 regressions green.
5. `-i` runs a v3 file and produces sampled text.
6. A tok/s figure is reported.

### Build order
1. `RunState` state + allocation; the dense-slot fix; the FFN branch with MoE.
2. **Gate 1 before anything else** — no REPL work until equivalence passes.
3. Gate 2 (generation).
4. Remove the REPL guard; Gates 3 and 4.

**The risky part is not the MoE maths — it is state that must persist across calls**, which is new for this engine, and that is why Gate 2 exists. If Gate 1 returns non-identical, that is a finding about the two attention paths, not an obstacle to paper over.

"""

DECLOG_ENTRY = """## 2026-09-25 — Pre-register Part 1's acceptance criteria (incremental MoE generation)

**Decision:** Before writing the incremental MoE path, commit its gates in [[Phase-7-Integration]] § Part 1. The load-bearing ones: **Gate 1** requires `forward_step` to produce *identical top-2 routing* (not just similar logits) to the already-verified `forward_moe`, and **Gate 2** requires a 64-token greedy generation to match full-sequence generation step for step across an EOS.

**Reasoning:** Three separate silent divergences were found today — the layer split, the EOS reset, and a missing λ in the export — and every one of them passed the gate that existed at the time because that gate compared the wrong thing. The incremental path is the highest-risk code yet written here: it introduces the engine's first *sequence state* beyond the KV cache (the locality set), which must persist across calls. A gate that checked only final logits would pass while routing silently diverged.

**Alternatives considered:** Gating on logits alone (rejected — exactly how the EOS bug escaped; routing can diverge while logits look fine). Chasing bit-exactness between the two attention paths (rejected — the standing tolerance rule is routing-identical plus logits within 1e-5; the aim is to notice divergence, not to eliminate float noise). Deferring the criteria until results exist (rejected — the same pre-registration discipline that made Q1/Q2/Q3 decidable)."""

t = P7.read_text(encoding="utf-8")
anchor = "## Links\n- Previous: [[Phase-6-Scale-Training-Data]]"
if t.count(anchor) != 1:
    print("Phase-7 anchor: %d matches" % t.count(anchor)); sys.exit(1)
if "Part 1 — Incremental MoE generation" in t:
    print("SKIP Phase-7 (already present)")
else:
    P7.write_text(t.replace(anchor, PLAN + anchor), encoding="utf-8")
    print("OK  Phase-7-Integration.md (Part 1 plan recorded)")

t = DECLOG.read_text(encoding="utf-8")
if "Pre-register Part 1's acceptance criteria" in t:
    print("SKIP Decision-Log.md")
else:
    DECLOG.write_text(t.rstrip() + "\n\n" + DECLOG_ENTRY + "\n", encoding="utf-8")
    print("OK  Decision-Log.md (appended)")
print("Part 1 plan recorded.")
