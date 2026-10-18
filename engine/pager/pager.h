#ifndef PAGER_H
#define PAGER_H

#include "core/weights.h"

/* Experiment 2 / Phase 7: region residency tracking + LRU eviction under a RAM
   budget (CP_BUDGET=<MB>). Regions are the granularity the POLICY reasons about,
   not OS pages.

   For CPW1 v2 the regions are (tensor, layer). For v3 they are (tensor, layer)
   for the dense tensors PLUS one region per routed expert — (moe slot, expert),
   1,183,488 bytes each — because the whole point of the layout is that one
   expert is one contiguous, sequentially-readable block.

   LayerNorm gains and projection biases are untracked: negligible, always hot.
   Eviction rounds ranges to page boundaries — adjacent regions may share a
   boundary page (over-evicts by <=4K per region; regions are MBs, so noise). */
void pager_init(Model *m, long budget_bytes);
void pager_touch(const char *tensor, int layer); /* no-op when disabled */

/* MoE access hook: demand-touches this token's k experts at `slot`, records them
   in the recency history for position `pos`, then prefetches the union of the
   previous W tokens' sets at the same slot (CP_PREFETCH_W, default 4; W=1 is the
   spec's stated contract, 0 disables). Prefetching marks a region resident, so
   it consumes budget and can evict something useful — which is the honest model. */
void pager_moe_access(int slot, const int *idx, int k, int pos);

void pager_enforce(void);                        /* evict LRU until under budget */
void pager_report(void);                         /* totals to stderr */

/* -- CPW1 v4 front-end (the Qwen3 path) -----------------------------------------
   A v4 file DECLARES its pageable regions (SPEC §8), so this front-end takes them
   from the file's own region table instead of deriving them from the v3 shape
   constants pager_init() must work with. A region id is that table's index, which
   lets a loader that resolved a tensor ask the pager about the region owning it.
   Both front-ends feed the SAME region array and the same LRU/budget policy, so
   accounting is comparable across models; only one may be opened per process.

   budget_bytes counts resident REGION bytes, core (non-expert) regions included —
   the same semantics as pager_init, and the reason pager_report() splits the two:
   at a tight budget the pager can evict weights every token needs, and that anchor
   thrash must not be mistaken for a routing effect.

   LONG_MAX means UNLIMITED: the pager is fully active (it counts every access and
   would re-fault on demand) but never evicts, so every byte still comes from the
   same mapping. That is the C1 refactor gate — a pager changes where bytes come
   from, not what is computed, so unlimited-budget output must be bit-identical to
   the unpaged run. (Eviction cannot change values either: MADV_DONTNEED drops
   pages of a file mapping, it does not alter them. Expect bit-identity at every
   budget; unlimited is where the attachment point itself is on trial.) */
int  pager_open_v4(int fd, const void *map_base, size_t data_off, long budget_bytes);
int  pager_add_region(size_t off, size_t len, int is_expert); /* -> region id, -1 */
void pager_touch_region(int id);                             /* demand access */

/* Per-STEP (per-token) accounting, not cumulative-only. A cumulative counter
   divided by the wrong token count has already cost this project a debugging
   detour, so each touch is charged to the step the caller declared and the dump is
   one row per step: regions fetched, bytes fetched, hits, evictions.
   Evictions happen in pager_enforce() between layers; they are charged to the step
   that was current, which is exact when one step is in flight (decode) and a rate
   when a whole prompt is batched through one forward. */
void pager_step(int step);                  /* charge subsequent touches to `step` */
int  pager_dump_steps(const char *path);    /* 0 ok; one row per step */

#endif
