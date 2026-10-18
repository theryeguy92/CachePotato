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

#endif
