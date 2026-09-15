#ifndef PAGER_H
#define PAGER_H

#include "weights.h"

/* Experiment 2 (Phase 3): tensor-region residency tracking + LRU eviction
   under a RAM budget (CP_BUDGET=<MB>). Regions = (tensor, layer) — the
   granularity the POLICY reasons about, not OS pages. LayerNorm gains and
   projection biases (~0.5MB total) are untracked: negligible, always hot.
   Eviction rounds ranges to page boundaries — adjacent regions may share a
   boundary page (over-evicts by ≤4K per region; regions are MBs, so noise). */
void pager_init(Model *m, long budget_bytes);
void pager_touch(const char *tensor, int layer); /* no-op when disabled */
void pager_enforce(void);                        /* evict LRU until under budget */
void pager_report(void);                         /* totals to stderr */

#endif
