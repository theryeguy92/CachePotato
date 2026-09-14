# CachePotato

Co-design of a sparse (MoE) language model and a memory-hierarchy-aware
inference/training engine in C — treating SSD, RAM, and CPU cache as one
addressable memory space so large models run on consumer hardware.

**Design docs, phase plans, and decision log live in the Obsidian vault:**
`~/vaults/CachePotato-Vault` (start at `00-MOC/MOC-Project`).
The vault is the source of truth for *why*; this repo is *what*.

## Build & run

```sh
make        # builds build/cachepotato
make run    # builds and runs the REPL (stub until Phase 1)
```

## Layout

| Path | Purpose |
|---|---|
| `src/` | The engine, in C. `model` (forward pass), `weights` (mmap loader → paging), `tokenizer`, `instr` (instrumentation) |
| `experiments/` | Phase 0 scratch programs (mmap/madvise tests, math checks). Learning artifacts, not engine code |
| `tools/` | Python sidecar — verification vs. HF reference, weight conversion. Never a runtime dependency |
| `bench/` | Benchmark harness + committed results (Phase 3+) |
| `tests/` | C unit tests for numerical kernels |
| `weights/` | Model files (gitignored) |
| `data/` | Training corpora (gitignored, Phase 6) |

## Status

Phase 0 (Foundations) — see `04-Phases/` in the vault.
