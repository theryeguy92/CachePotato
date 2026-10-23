#define _GNU_SOURCE /* madvise/posix_fadvise with -std=c11 */
#include "pager.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define PAGE 4096UL

typedef struct {
    const char *name;       /* v2/v3: tensor name; v4: NULL (ids address regions) */
    int layer;              /* dense: engine layer; expert: MoE slot           */
    float *ptr;
    size_t bytes;
    unsigned long last_used; /* tick; 0 = never touched */
    int resident;
    int is_expert;          /* for the report's core/expert split. Not derivable
                               from an index: the v4 front-end gets regions in the
                               file's order, where they are interleaved. */
    int pinned;             /* never chosen as an LRU victim (CP_PIN_CORE). Still
                               counts against the budget: pinning is a policy about
                               WHO gets evicted, not a way to get memory for free. */
    void *abuf;             /* CP_ARENA: this region's copy in the arena, or NULL */
} Region;

/* Which front-end opened the pager. 0 = none, 1 = pager_init (v2/v3 layout),
   2 = pager_open_v4 (file-declared regions). The policy is shared; only region
   construction and the address of a region differ. */
static int pager_mode;

/* Dynamic: v2 needs ~74 regions, v3 needs ~414 (320 of them routed experts).
   A fixed cap used to drop regions SILENTLY, which corrupts the accounting. */
static Region *regions;
static int n_regions, cap_regions;
static int expert_base = -1;    /* regions[expert_base + slot*n_exp + e] */
static int n_slots, n_exp, topk;

static int  *hist;              /* (n_slots, pf_w, topk) recency set per slot */
static int   pf_w = 4;          /* CP_PREFETCH_W; 1 = the spec's contract   */

static long budget = -1;        /* -1 = disabled */
static unsigned long tick;
static long n_evictions, bytes_evicted;
static long n_hits, n_misses, bytes_missed, n_prefetch, n_pf_resident;
static long n_ehits, n_emisses, bytes_emissed;   /* routed experts only */
static int sys_evict;           /* CP_PAGER_SYS: also evict the system cache */
static int pin_core;            /* CP_PIN_CORE: core (non-expert) regions are never victims */
static int use_arena;           /* CP_ARENA: the pager owns the bytes (pread per region) */
static long n_arena_fallback;   /* regions too big for the budget, read from the mapping */
static long arena_bytes;        /* bytes currently held in arena buffers */
static long arena_pread_bytes;  /* bytes actually requested from the kernel */
static int warned_pinned_over;  /* one message, not one per enforce call */
static int model_fd;
static const float *model_data;

/* Per-step counters. Kept beside the cumulative ones rather than replacing them:
   the cumulative total is what pager_report() prints and what a paged run is
   judged by; the per-step table is what "regions fetched per token" means, and
   computing it from a total and a token count is exactly the unit/scope mistake
   this struct exists to prevent. */
typedef struct {
    long regions, bytes, hits, evictions;
} StepCount;
static StepCount *steps;
static long n_steps, cap_steps;
static long cur_step = -1;

static StepCount *step_at(long s) {
    if (s >= cap_steps) {
        long nc = cap_steps ? cap_steps : 256;
        while (nc <= s) nc *= 2;
        StepCount *ns = realloc(steps, (size_t)nc * sizeof *ns);
        if (!ns) { fprintf(stderr, "pager: out of memory\n"); exit(1); }
        memset(ns + cap_steps, 0, (size_t)(nc - cap_steps) * sizeof *ns);
        steps = ns;
        cap_steps = nc;
    }
    if (s >= n_steps) n_steps = s + 1;
    return &steps[s];
}

static void add_region_raw(const char *name, int layer, float *ptr, size_t bytes,
                           int is_expert) {
    if (n_regions == cap_regions) {
        int nc = cap_regions ? cap_regions * 2 : 128;
        Region *nr = realloc(regions, (size_t)nc * sizeof(Region));
        if (!nr) { fprintf(stderr, "pager: out of memory\n"); exit(1); }
        regions = nr;
        cap_regions = nc;
    }
    regions[n_regions++] = (Region){name, layer, ptr, bytes, 0, 0, is_expert,
                                    pin_core && !is_expert, NULL};
}

static void add_region(const char *name, int layer, float *ptr, size_t bytes) {
    add_region_raw(name, layer, ptr, bytes, 0);
}

void pager_init(Model *m, long budget_bytes) {
    if (pager_mode) { fprintf(stderr, "pager: already open (mode %d)\n", pager_mode); return; }
    pager_mode = 1;
    budget = budget_bytes;
    model_fd = m->fd;
    model_data = m->data;
    const char *s = getenv("CP_PAGER_SYS");
    sys_evict = s && s[0] == '1';
    const char *pc = getenv("CP_PIN_CORE");
    pin_core = pc && pc[0] == '1';
    const char *ar = getenv("CP_ARENA");
    use_arena = ar && ar[0] == '1';
    const char *pw = getenv("CP_PREFETCH_W");
    if (pw) pf_w = atoi(pw);
    if (pf_w < 0) pf_w = 0;
    if (pf_w > 64) pf_w = 64;

    int L = m->hdr.n_layer, D = m->hdr.d_model, F = m->hdr.d_ff;
    long V = m->hdr.vocab, C = m->hdr.n_ctx;
    Weights *w = &m->w;

    add_region("token_emb", -1, w->token_emb, (size_t)V * D * 4);
    add_region("pos_emb", -1, w->pos_emb, (size_t)C * D * 4);

    for (int l = 0; l < L; l++) {
        add_region("Wq", l, w->Wq + (size_t)l * D * D, (size_t)D * D * 4);
        add_region("Wk", l, w->Wk + (size_t)l * D * D, (size_t)D * D * 4);
        add_region("Wv", l, w->Wv + (size_t)l * D * D, (size_t)D * D * 4);
        add_region("Wo", l, w->Wo + (size_t)l * D * D, (size_t)D * D * 4);
        if (moe_layer_at(m, l)) {
            int slot = w->moe_slot[l];
            add_region("Wr", l, w->Wr + (size_t)slot * D * w->n_exp,
                       (size_t)D * w->n_exp * 4);
            add_region("W1s", l, w->W1s + (size_t)slot * D * D,
                       (size_t)D * D * 4);
            add_region("W2s", l, w->W2s + (size_t)slot * D * D,
                       (size_t)D * D * 4);
        } else {
            /* v3 compacts the dense FFNs into DENSE SLOTS */
            int ds = m->is_v3 ? w->dense_slot[l] : l;
            add_region("W1", l, w->W1 + (size_t)ds * D * F, (size_t)D * F * 4);
            add_region("W2", l, w->W2 + (size_t)ds * F * D, (size_t)F * D * 4);
        }
    }

    if (!(m->is_v3 && w->is_moe))
        return;

    n_slots = w->n_moe;
    n_exp = w->n_exp;
    topk = w->topk;
    expert_base = n_regions;
    long ef = expert_floats(D, w->exp_hid);
    for (int slot = 0; slot < n_slots; slot++)
        for (int e = 0; e < n_exp; e++)
            add_region_raw("expert", slot, expert_block(w, slot, e), (size_t)ef * 4, 1);

    if (pf_w > 0) {
        hist = malloc((size_t)n_slots * pf_w * topk * sizeof(int));
        if (hist) memset(hist, 0xFF, (size_t)n_slots * pf_w * topk * sizeof(int));
    }
    fprintf(stderr, "pager: %d regions (%d experts x %ld bytes), prefetch W=%d\n",
            n_regions, n_slots * n_exp, ef * 4, pf_w);
}

void pager_touch(const char *tensor, int layer) {
    if (pager_mode != 1 || budget < 0)
        return;
    /* unemb is TIED: its pages are token_emb's pages */
    if (strcmp(tensor, "unemb") == 0)
        tensor = "token_emb";
    for (int i = 0; i < n_regions; i++)
        if (regions[i].name && regions[i].layer == layer && strcmp(regions[i].name, tensor) == 0) {
            if (regions[i].resident) {
                n_hits++;
            } else {
                regions[i].resident = 1;
                n_misses++;
                bytes_missed += regions[i].bytes;
            }
            regions[i].last_used = ++tick;
            return;
        }
}

/* ---- CPW1 v4 front-end -------------------------------------------------------
   Region ids are the file region table's own indices, so the loader can record
   "this tensor came from region 1234" while it walks the tensor table and later
   charge a touch by index. No search, no name matching, and no second description
   of where the bytes are. */
static size_t map_off;          /* file's off_data: region offsets are relative */

int pager_open_v4(int fd, const void *map_base, size_t data_off, long budget_bytes) {
    if (pager_mode) { fprintf(stderr, "pager: already open (mode %d)\n", pager_mode); return -1; }
    pager_mode = 2;
    budget = budget_bytes;
    map_off = data_off;
    model_data = (const float *)map_base;  /* fadvise arithmetic is base-relative */
    model_fd = dup(fd);
    if (model_fd < 0) { perror("pager: dup"); pager_mode = 0; return -1; }
    const char *s = getenv("CP_PAGER_SYS");
    sys_evict = s && s[0] == '1';
    const char *pc = getenv("CP_PIN_CORE");
    pin_core = pc && pc[0] == '1';
    const char *ar = getenv("CP_ARENA");
    use_arena = ar && ar[0] == '1';
    /* Expert prefetch (CP_PREFETCH_W) is v3 policy: it predicts token t+1's experts
       from the history pager_moe_access() keeps. C1 pages on demand only, so the
       knob is not read here — a silent W=4 would make a warm run look like a win. */
    pf_w = 0;
    return 0;
}

int pager_add_region(size_t off, size_t len, int is_expert) {
    if (pager_mode != 2) return -1;
    float *ptr = (float *)(void *)((const unsigned char *)model_data + map_off + off);
    add_region_raw(NULL, -1, ptr, len, is_expert);
    return n_regions - 1;
}

/* Evict down to `need` free bytes (arena mode) by freeing the LRU buffer. Same policy as
   the madvise path, different mechanics: the pager allocated the memory, so it frees it. */
static void arena_make_room(size_t need) {
    while (arena_bytes + (long)need > budget) {
        int victim = -1;
        for (int i = 0; i < n_regions; i++)
            if (regions[i].abuf && !regions[i].pinned &&
                (victim < 0 || regions[i].last_used < regions[victim].last_used))
                victim = i;
        if (victim < 0) {
            if (!warned_pinned_over) {
                fprintf(stderr, "pager: arena cannot make room for %zu bytes (%.1f MB held, "
                                "%ld MB budget) — pinned or oversized region\n",
                        need, arena_bytes / 1048576.0, budget / (1024 * 1024));
                warned_pinned_over = 1;
            }
            break;
        }
        Region *v = &regions[victim];
        free(v->abuf);
        v->abuf = NULL;
        v->resident = 0;
        arena_bytes -= (long)v->bytes;
        n_evictions++;
        bytes_evicted += v->bytes;
        if (cur_step >= 0) steps[cur_step].evictions++;
    }
}

const void *pager_region_get(int id) {
    if (pager_mode != 2 || id < 0 || id >= n_regions)
        return NULL;
    Region *r = &regions[id];
    /* A region larger than the whole budget can never be cached. Reading it through the
       mapping keeps the budget honest and is the only sane answer for a 330 MB embedding
       against an 8 GB budget — it is touched once per token, not per expert. */
    const int oversized = use_arena && (long)r->bytes > budget;

    if (use_arena && !oversized) {
        if (!r->abuf) {
            arena_make_room(r->bytes);
            void *buf = malloc(r->bytes);
            if (!buf) { fprintf(stderr, "pager: arena malloc failed\n"); exit(1); }
            arena_pread_bytes += (long)r->bytes;
            if (pread(model_fd, buf, r->bytes, (off_t)((const unsigned char *)r->ptr
                                                       - (const unsigned char *)model_data)) !=
                (ssize_t)r->bytes) {
                fprintf(stderr, "pager: pread failed for region %d\n", id);
                exit(1);
            }
            /* The bytes now live in OUR buffer, so drop them from the page cache. Without
               this the page cache becomes a shadow cache the budget does not count, and a
               "miss" would be a memcpy rather than a disk read — which would flatter the
               arena by exactly the thing it is supposed to remove. Only under CP_PAGER_SYS,
               like every other honest-disk measurement here. */
            if (sys_evict)
                posix_fadvise(model_fd,
                              (off_t)((const unsigned char *)r->ptr - (const unsigned char *)model_data),
                              (off_t)r->bytes, POSIX_FADV_DONTNEED);
            r->abuf = buf;
            r->resident = 1;
            arena_bytes += (long)r->bytes;
            n_misses++;
            bytes_missed += r->bytes;
            if (r->is_expert) { n_emisses++; bytes_emissed += r->bytes; }
            if (cur_step >= 0) {
                steps[cur_step].regions++;
                steps[cur_step].bytes += (long)r->bytes;
            }
        } else {
            n_hits++;
            if (r->is_expert) n_ehits++;
            if (cur_step >= 0) steps[cur_step].hits++;
        }
        r->last_used = ++tick;
        return r->abuf;
    }

    /* no arena (or oversized): track residency, read through the mapping */
    if (r->resident) {
        n_hits++;
        if (r->is_expert) n_ehits++;
        if (cur_step >= 0) steps[cur_step].hits++;
    } else {
        r->resident = 1;
        n_misses++;
        bytes_missed += r->bytes;
        if (r->is_expert) { n_emisses++; bytes_emissed += r->bytes; }
        if (cur_step >= 0) {
            steps[cur_step].regions++;
            steps[cur_step].bytes += (long)r->bytes;
        }
    }
    if (oversized) n_arena_fallback++;
    r->last_used = ++tick;
    return r->ptr;
}

void pager_step(int step) {
    if (pager_mode != 2)
        return;
    if (step < 0) { cur_step = -1; return; }
    step_at(step);
    cur_step = step;
}

int pager_dump_steps(const char *path) {
    if (pager_mode != 2)
        return -1;
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return -1; }
    fprintf(f, "# CPW1 v4 pager per-step counters (one row per token)\n");
    fprintf(f, "# step regions_fetched bytes_fetched hits evictions\n");
    fprintf(f, "# budget %ld bytes%s\n", budget,
            budget == LONG_MAX ? " (unlimited: counts, never evicts)" : "");
    for (long s = 0; s < n_steps; s++)
        fprintf(f, "%ld %ld %ld %ld %ld\n", s, steps[s].regions, steps[s].bytes,
                steps[s].hits, steps[s].evictions);
    fclose(f);
    return 0;
}

static void touch_expert(int slot, int e, int demand) {
    if (slot < 0 || slot >= n_slots || e < 0 || e >= n_exp)
        return;
    Region *r = &regions[expert_base + slot * n_exp + e];
    if (r->resident) {
        if (demand) { n_hits++; n_ehits++; if (cur_step >= 0) steps[cur_step].hits++; }
        else n_pf_resident++;
    } else {
        r->resident = 1;
        if (demand) {
            n_misses++;
            bytes_missed += r->bytes;
            n_emisses++;
            bytes_emissed += r->bytes;
            if (cur_step >= 0) {
                steps[cur_step].regions++;
                steps[cur_step].bytes += (long)r->bytes;
            }
        } else {
            /* a real prefetch: ask the kernel to start reading it in now */
            n_prefetch++;
            unsigned long start = (unsigned long)r->ptr & ~(PAGE - 1);
            size_t len = r->bytes + ((unsigned long)r->ptr - start);
            madvise((void *)start, len, MADV_WILLNEED);
        }
    }
    r->last_used = ++tick;
}

void pager_moe_access(int slot, const int *idx, int k, int pos) {
    if (budget < 0 || expert_base < 0)
        return;

    /* demand first, always — W=0 means "no prefetch", NOT "no accounting" */
    for (int i = 0; i < k; i++)
        touch_expert(slot, idx[i], 1);

    if (pf_w == 0 || !hist)
        return;
    if (pos == 0)   /* new sequence: the history is not a predictor across it */
        memset(hist, 0xFF, (size_t)n_slots * pf_w * topk * sizeof(int));

    /* Record the current token first, then prefetch the union of the last W sets
       INCLUDING it. That is the engine's real contract: while layer l is being
       computed we know this token's selection, and the next token is what we are
       predicting for. (Issuing it before the record would read pos-1..pos-W —
       a prediction for the token already being processed, i.e. too late.) */
    int *row = hist + ((size_t)slot * pf_w + (pos % pf_w)) * topk;
    for (int i = 0; i < k && i < topk; i++)
        row[i] = idx[i];

    /* This deliberately keeps its own targets resident, so they stop being LRU
       victims: that IS the policy, and its cost is the budget they occupy. */
    for (int w = 0; w < pf_w; w++) {
        int p = pos - w;
        if (p < 0)
            break;
        const int *h = hist + ((size_t)slot * pf_w + (p % pf_w)) * topk;
        for (int i = 0; i < k && i < topk; i++)
            touch_expert(slot, h[i], 0);
    }
}

void pager_enforce(void) {
    if (budget < 0)
        return;
    if (use_arena) {  /* fetches already made room; nothing to do between layers */
        return;
    }
    long resident_bytes = 0;
    for (int i = 0; i < n_regions; i++)
        if (regions[i].resident)
            resident_bytes += regions[i].bytes;

    while (resident_bytes > budget) {
        int victim = -1;
        for (int i = 0; i < n_regions; i++)
            if (regions[i].resident && !regions[i].pinned &&
                (victim < 0 || regions[i].last_used < regions[victim].last_used))
                victim = i;
        if (victim < 0) {
            /* Everything left is pinned: the budget is below the pinned floor, so the
               policy cannot honour it. Say so once — silently running over budget would
               make every number after this one a fiction. */
            if (!warned_pinned_over) {
                fprintf(stderr, "pager: budget too small for the pinned core (%ld MB "
                                "resident, %ld MB budget) — running over budget\n",
                        resident_bytes / (1024 * 1024), budget / (1024 * 1024));
                warned_pinned_over = 1;
            }
            break;
        }
        Region *r = &regions[victim];
        unsigned long start = (unsigned long)r->ptr & ~(PAGE - 1);
        size_t len = r->bytes + ((unsigned long)r->ptr - start);
        madvise((void *)start, len, MADV_DONTNEED);
        if (sys_evict) /* also drop the SYSTEM page cache for this range —
                          simulates the model > RAM regime where the OS
                          can't mask policy misses */
            posix_fadvise(model_fd, (off_t)(start - (unsigned long)model_data),
                          (off_t)len, POSIX_FADV_DONTNEED);
        r->resident = 0;
        resident_bytes -= r->bytes;
        n_evictions++;
        bytes_evicted += r->bytes;
        if (cur_step >= 0) steps[cur_step].evictions++;
    }
}

/* Resident-byte breakdown. Exists because the trace replay needs to know the
   pager's REAL expert capacity, and the byte arithmetic (budget - core) did not
   reproduce the measured hit rates at tight budgets. Prints the resident split so
   the effective capacity is observed rather than inferred. */
/* A 13-digit budget in a report is a unit/scope trap waiting to happen: read it as
   bytes and it is wrong, as MB and it is wrong. LONG_MAX has a name. */
static void budget_str(char *buf, size_t n) {
    if (budget == LONG_MAX) snprintf(buf, n, "unlimited");
    else snprintf(buf, n, "%ld MB", budget / (1024 * 1024));
}

static void pager_report_resident(void) {
    long core_b = 0, exp_b = 0, core_n = 0, exp_n = 0, exp_total = 0, pin_b = 0;
    for (int i = 0; i < n_regions; i++) {
        if (regions[i].is_expert) exp_total++;
        if (!regions[i].resident) continue;
        if (regions[i].pinned) pin_b += (long)regions[i].bytes;
        if (regions[i].is_expert) { exp_b += (long)regions[i].bytes; exp_n++; }
        else { core_b += (long)regions[i].bytes; core_n++; }
    }
    char bs[32]; budget_str(bs, sizeof bs);
    fprintf(stderr, "  resident  : core %ld regions/%.1f MB | experts %ld/%ld regions/%.1f MB"
                    " | total %.1f MB of budget %s\n",
            core_n, core_b / 1048576.0, exp_n, exp_total,
            exp_b / 1048576.0, (core_b + exp_b) / 1048576.0, bs);
    if (pin_core)
        fprintf(stderr, "  pinned    : %.1f MB never evicted (CP_PIN_CORE) — evictable "
                        "capacity %.1f MB for experts\n",
                pin_b / 1048576.0, ((long)budget - pin_b) / 1048576.0);
    if (use_arena)
        fprintf(stderr, "  arena     : %.1f MB held in pager buffers (%ld oversized regions "
                        "read through the mapping); %.1f MB requested via pread\n",
                arena_bytes / 1048576.0, n_arena_fallback, arena_pread_bytes / 1048576.0);
}

void pager_totals(PagerTotals *out) {
    long res = 0;
    for (int i = 0; i < n_regions; i++)
        if (regions[i].resident) res += (long)regions[i].bytes;
    out->hits = n_hits; out->misses = n_misses; out->bytes_fetched = bytes_missed;
    out->evictions = n_evictions; out->resident_bytes = res; out->budget_bytes = budget;
    out->regions = n_regions;
}

void pager_report(void) {
    if (budget < 0)
        return;
    pager_report_resident();
    long access = n_hits + n_misses;
    long eaccess = n_ehits + n_emisses;
    /* Core = the hot tier (token_emb, pos_emb, attention, shared experts, dense
       FFN, router). touch_expert() increments the all-counters as well as the
       expert ones, so core is the difference and needs no separate accounting.
       Reported because at a tight CP_BUDGET the pager can begin evicting
       weights that every token needs; that "anchor thrash" would otherwise be
       indistinguishable from a routing effect. */
    long caccess = access - eaccess;
    long chits = n_hits - n_ehits;
    double cbytes = (double)(bytes_missed - bytes_emissed);
    char bs[32]; budget_str(bs, sizeof bs);
    fprintf(stderr,
            "pager: %d regions | budget %s | prefetch W=%d\n"
            "  all     : %ld accesses, %ld hits (%.2f%%), %.1f MB fetched\n"
            "  experts : %ld accesses, %ld hits (%.2f%%), %.1f MB fetched | "
            "prefetch: %ld fetches, %ld already resident\n"
            "  core    : %ld accesses, %ld hits (%.2f%%), %.1f MB fetched\n"
            "  %ld evictions, %ld MB evicted\n",
            n_regions, bs, pf_w,
            access, n_hits,
            access ? 100.0 * (double)n_hits / (double)access : 0.0,
            (double)bytes_missed / (1024 * 1024),
            eaccess, n_ehits,
            eaccess ? 100.0 * (double)n_ehits / (double)eaccess : 0.0,
            (double)bytes_emissed / (1024 * 1024), n_prefetch, n_pf_resident,
            caccess, chits,
            caccess ? 100.0 * (double)chits / (double)caccess : 0.0,
            cbytes / (1024 * 1024),
            n_evictions, bytes_evicted / (1024 * 1024));
    if (pager_mode == 2 && n_steps) {
        long tr = 0, tb = 0, th = 0;
        for (long s = 0; s < n_steps; s++) {
            tr += steps[s].regions; tb += steps[s].bytes; th += steps[s].hits;
        }
        fprintf(stderr,
                "  per token : %.2f regions fetched, %.1f MB fetched, %.2f%% hit rate "
                "(%ld steps, same events as the totals above)\n",
                (double)tr / (double)n_steps, (double)tb / 1048576.0 / (double)n_steps,
                (double)th / (double)(tr + th) * 100.0, n_steps);
    }
}
