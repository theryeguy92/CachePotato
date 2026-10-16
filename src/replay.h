#ifndef REPLAY_H
#define REPLAY_H

#include <stdint.h>
#include <stddef.h>

/* Phase 8 T2: pure-residency trace replay.
 *
 * WHY A SEPARATE SIMULATOR, AND WHY IN C
 *   The Python traffic model (analysis/traffic.py) under-predicts the engine's own
 *   measured expert hit rate by 5-26 points, so it cannot be trusted to answer
 *   "which port, at what budget". Rather than add more fidelity to a model of the
 *   engine, replay the trace and let a C implementation be fast enough to sweep
 *   configurations (a per-touch Python LRU costs ~30 s each; this costs milliseconds).
 *
 * WHAT IT MODELS
 *   Exactly the residency policy `pager.c` implements, and nothing else:
 *     - one GLOBAL LRU over regions shared by all slots (pager.c evicts from one
 *       flat region array), capacity = (budget - core_bytes) / region_bytes;
 *     - regions are (slot, expert_group), where an expert group is
 *       floor(region_bytes / expert_bytes) consecutive experts, matching what
 *       pager.c builds today at group=1;
 *     - recency resets at each window boundary and at EOS, matching pager.c's
 *       reset when pos == 0 and at end-of-sequence.
 *
 * TWO COUNTERS, DELIBERATELY (both are reported)
 *   `loads`    = distinct cold REGION loads, i.e. physical reads. This is the metric
 *                that matters on a latency-bound device, because a cold random read
 *                costs ~1.7 ms almost independently of size on the target SSD.
 *   `acc/hits` = pager.c's own counter semantics: one access per expert TOUCH, a hit
 *                if that expert's region is resident. When a region holds >1 expert
 *                this counts a cold region twice, so `misses` over-counts physical
 *                reads. Reporting both makes the over-count measurable rather than
 *                implicit.
 *
 * NO FORWARD PASS, NO WEIGHTS, NO mmap. Nothing here touches the model; it consumes a
 * recording of what the model chose.
 */

#define RPY_MAGIC 0x31505452u /* "RTP1" little-endian */

typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t  n_slots;    /* MoE layers                        */
    int32_t  n_windows;  /* sequences                         */
    int32_t  window_len; /* tokens per sequence               */
    int32_t  topk;       /* experts selected per token        */
    int32_t  n_exp;      /* expert id space                   */
    int32_t  eos;        /* eos token id; resets the LRU      */
    int32_t  lam_bits;   /* lambda, bit-cast f32 (provenance) */
} ReplayHeader;

typedef struct {
    ReplayHeader h;
    int32_t *tokens;   /* (n_windows, window_len)      */
    int16_t *sel;      /* (n_slots, n_windows, window_len, topk) */
} ReplayTrace;

typedef struct {
    long acc;             /* expert touches                        */
    long hits;            /* touches whose region was resident     */
    long loads;           /* distinct cold region loads (reads)    */
    long evictions;       /* regions evicted                       */
    long tokens;          /* tokens replayed                       */
    long resets;          /* LRU resets performed                  */
    int  n_exp;           /* expert id space (for reporting)       */
    int  cap;             /* region capacity used                  */
    int  group;           /* experts per region                    */
    int  n_regions_slot;  /* regions per slot                      */
    int  max_resident;
} ReplayStats;

typedef struct {
    long budget_bytes;
    long core_bytes;      /* non-expert regions charged against the budget */
    long region_bytes;    /* bytes per pageable region                     */
    long expert_bytes;    /* bytes per expert block                        */
    int  limit_windows;   /* 0 = all                                       */
    int  reset_windows;   /* 1 = reset LRU at each window boundary         */
} ReplayConfig;

/* Loads the RTP1 trace. Returns 0 on success, -1 with a message on stderr. */
int replay_load(ReplayTrace *t, const char *path);
void replay_free(ReplayTrace *t);

/* Replays under the given layout/budget. Returns 0 on success. */
int replay_run(const ReplayTrace *t, const ReplayConfig *cfg, ReplayStats *st);

/* One-line machine-readable summary for sweeping scripts. */
void replay_report(const ReplayStats *st, const ReplayConfig *cfg);

#endif
