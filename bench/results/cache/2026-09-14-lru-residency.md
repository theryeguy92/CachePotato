# Experiment 2 — LRU residency under RAM budget (2026-09-14)

GPT-2 small (497MB CPW1, mmap), prompt "Alan Turing theorized that computers
would one day become", 40 greedy tokens, T480 (16GB RAM, SanDisk SATA SSD, btrfs).
Pager: tensor×layer regions, LRU eviction via MADV_DONTNEED; `CP_PAGER_SYS=1`
adds per-range POSIX_FADV_DONTNEED (simulates model > RAM).

| CP_BUDGET | CP_PAGER_SYS | tok/s | RSS | disk read | evictions (bytes) |
|---|---|---|---|---|---|
| — (hot) | — | 7.79 | 545 MB | 0 | 0 |
| 384 | off | 6.99 | 457 MB | 0 | 1,100 (4.5 GB) |
| 256 | off | 6.61 | 329 MB | 0 | 2,450 (10.9 GB) |
| 128 | off | 6.40 | 74 MB | 0 | 3,700 (23.7 GB) |
| 384 | 1 | 1.89 | 457 MB | 4.4 GB | 1,100 (4.5 GB) |
| 128 | 1 | 0.64 | 74 MB | 22.7 GB | 3,700 (23.7 GB) |
| (Exp 1 forced-cold) | system | 0.70 | 73 MB | 23.3 GB | — |

Findings:
- RSS tracks budget + ~73MB overhead exactly; 128MB < token_emb anchor (154MB)
  → anchor thrash → RSS ~74MB.
- Process-only eviction: page cache masks misses (model ≪ RAM) → disk 0, ≤18% cost.
- +system: read_bytes ≈ evicted bytes; tok/s cost ≈ evicted ÷ ~450MB/s (predictive).
- 128MB+sys ≈ Exp 1 forced-cold: LRU thrash on dense = worst case. No locality knee.

Reproduce: `printf 'Alan Turing theorized that computers would one day become\n' | CP_BUDGET=128 CP_PAGER_SYS=1 ./build/cachepotato weights/gpt2.bin -i`
