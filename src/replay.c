#include "replay.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ trace I/O */

int replay_load(ReplayTrace *t, const char *path) {
    memset(t, 0, sizeof(*t));
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    if (fread(&t->h, sizeof(ReplayHeader), 1, f) != 1) {
        fprintf(stderr, "%s: short read on header\n", path);
        fclose(f); return -1;
    }
    if (t->h.magic != RPY_MAGIC) {
        fprintf(stderr, "%s: bad magic 0x%08x — not an RTP1 trace\n", path, t->h.magic);
        fclose(f); return -1;
    }
    if (t->h.version != 1) {
        fprintf(stderr, "%s: unsupported trace version %u\n", path, t->h.version);
        fclose(f); return -1;
    }
    if (t->h.n_slots <= 0 || t->h.n_windows <= 0 || t->h.window_len <= 0 ||
        t->h.topk <= 0 || t->h.n_exp <= 0) {
        fprintf(stderr, "%s: nonsensical dims (slots %d windows %d T %d topk %d N %d)\n",
                path, t->h.n_slots, t->h.n_windows, t->h.window_len,
                t->h.topk, t->h.n_exp);
        fclose(f); return -1;
    }

    size_t ntok = (size_t)t->h.n_windows * t->h.window_len;
    size_t nsel = (size_t)t->h.n_slots * ntok * t->h.topk;
    t->tokens = malloc(ntok * sizeof(int32_t));
    t->sel = malloc(nsel * sizeof(int16_t));
    if (!t->tokens || !t->sel) {
        fprintf(stderr, "%s: out of memory for the trace\n", path);
        fclose(f); free(t->tokens); free(t->sel); return -1;
    }
    if (fread(t->tokens, sizeof(int32_t), ntok, f) != ntok ||
        fread(t->sel, sizeof(int16_t), nsel, f) != nsel) {
        fprintf(stderr, "%s: short read on payload (truncated trace?)\n", path);
        fclose(f); free(t->tokens); free(t->sel); return -1;
    }
    fclose(f);
    return 0;
}

void replay_free(ReplayTrace *t) {
    free(t->tokens);
    free(t->sel);
    t->tokens = NULL;
    t->sel = NULL;
}

/* ------------------------------------------------------------------ residency */

/* One entry per tracked region. A linear LRU scan is deliberate: region counts here
   are ~10^2-10^4, where a scan of the resident set is already far faster than the
   per-touch cost of a heap, and it keeps the policy obvious to read. */
typedef struct {
    long *region;    /* region id, -1 = free          */
    long *stamp;     /* last touch (monotonic)        */
    long tick;
    int   n;
    int   cap;
    int   used;
} Lru;

static void lru_reset(Lru *l) {
    for (int i = 0; i < l->used; i++) l->region[i] = -1;
    l->used = 0;
}

static int lru_touch(Lru *l, long region) {
    l->tick++;
    for (int i = 0; i < l->used; i++) {
        if (l->region[i] == region) {
            l->stamp[i] = l->tick;
            return 1;                      /* hit */
        }
    }
    if (l->used < l->cap) {
        int i = l->used++;
        l->region[i] = region; l->stamp[i] = l->tick;
        return 0;                          /* miss (filled a free slot) */
    }
    /* evict the least recently used and reuse its slot */
    int victim = 0;
    for (int i = 1; i < l->used; i++)
        if (l->stamp[i] < l->stamp[victim]) victim = i;
    l->region[victim] = region;
    l->stamp[victim] = l->tick;
    return 0;                              /* miss (evicted something) */
}

int replay_run(const ReplayTrace *t, const ReplayConfig *cfg, ReplayStats *st) {
    memset(st, 0, sizeof(*st));

    long group = cfg->expert_bytes > 0 ? cfg->region_bytes / cfg->expert_bytes : 1;
    if (group < 1) group = 1;
    int n_regions_slot = (int)((t->h.n_exp + group - 1) / group);
    long avail = cfg->budget_bytes - cfg->core_bytes;
    long cap = avail > 0 ? avail / (group * cfg->expert_bytes) : 0;
    if (cap < 1) cap = 1;                  /* degenerate; visible via cap/report */

    Lru lru;
    lru.n = (int)cap;
    lru.cap = (int)cap;
    lru.region = malloc((size_t)cap * sizeof(long));
    lru.stamp = malloc((size_t)cap * sizeof(long));
    if (!lru.region || !lru.stamp) {
        fprintf(stderr, "replay: out of memory for %ld regions\n", cap);
        free(lru.region); free(lru.stamp);
        return -1;
    }
    lru.tick = 0; lru.used = 0;

    int W = cfg->limit_windows > 0 && cfg->limit_windows < t->h.n_windows
              ? cfg->limit_windows : t->h.n_windows;

    for (int w = 0; w < W; w++) {
        if (cfg->reset_windows) { lru_reset(&lru); st->resets++; }
        for (int p = 0; p < t->h.window_len; p++) {
            st->tokens++;
            for (int s = 0; s < t->h.n_slots; s++) {
                const int16_t *row = t->sel
                    + (((size_t)s * t->h.n_windows + w) * t->h.window_len + p) * t->h.topk;
                long base = (long)s * n_regions_slot;
                for (int k = 0; k < t->h.topk; k++) {
                    long region = base + row[k] / group;
                    st->acc++;
                    if (lru_touch(&lru, region)) st->hits++;
                    else st->loads++;
                }
            }
            if (lru.used > st->max_resident) st->max_resident = lru.used;
            if (t->tokens[(size_t)w * t->h.window_len + p] == t->h.eos) {
                lru_reset(&lru); st->resets++;
            }
        }
    }
    /* eviction count: every miss beyond the first `cap` fills is one eviction */
    st->evictions = st->loads > cap ? st->loads - cap : 0;
    st->cap = (int)cap;
    st->group = (int)group;
    st->n_exp = t->h.n_exp;
    st->n_regions_slot = n_regions_slot;

    free(lru.region); free(lru.stamp);
    return 0;
}

void replay_report(const ReplayStats *st, const ReplayConfig *cfg) {
    double hit = st->acc ? 100.0 * (double)st->hits / (double)st->acc : 0.0;
    double per_tok = st->tokens ? (double)st->loads / (double)st->tokens : 0.0;
    double mt = st->tokens ? (double)st->loads * st->group * cfg->expert_bytes
                             / (double)st->tokens / 1e6 : 0.0;
    printf("pager-sim: budget %ld MB, core %ld MB, region %ld B (%d expert(s)), "
           "expert %ld B\n", cfg->budget_bytes >> 20, cfg->core_bytes >> 20,
           st->group * cfg->expert_bytes, st->group, cfg->expert_bytes);
    printf("  capacity   : %d regions total (%d/slot of %d; pool %d experts/slot)\n",
           st->cap, st->cap, st->n_regions_slot, st->n_exp);
    printf("  expert hit : %.2f%%  (%ld/%ld touches)\n", hit, st->hits, st->acc);
    printf("  reads      : %.3f distinct region loads/token (%.2f MB/token)\n",
           per_tok, mt);
    printf("  engine counter would say %.3f misses/token%s\n",
           st->tokens ? (double)(st->acc - st->hits) / (double)st->tokens : 0.0,
           (st->loads && st->tokens &&
            (double)(st->acc - st->hits) > (double)st->loads)
             ? "  (this OVER-COUNTS physical reads; regions hold >1 expert)"
             : "");
    printf("  resets %ld, tokens %ld, peak resident %d regions, %ld evictions\n",
           st->resets, st->tokens, st->max_resident, st->evictions);
    printf("PAGER_SIM_RESULT hit_pct=%.4f loads_per_token=%.4f mb_per_token=%.4f "
           "cap=%d group=%d resets=%ld tokens=%ld\n",
           hit, per_tok, mt, st->cap, st->group, st->resets, st->tokens);
}
