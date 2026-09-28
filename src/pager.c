#define _GNU_SOURCE /* madvise/posix_fadvise with -std=c11 */
#include "pager.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>

#define PAGE 4096UL

typedef struct {
    const char *name;
    int layer;              /* dense: engine layer; expert: MoE slot           */
    float *ptr;
    size_t bytes;
    unsigned long last_used; /* tick; 0 = never touched */
    int resident;
} Region;

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
static int model_fd;
static const float *model_data;

static void add_region(const char *name, int layer, float *ptr, size_t bytes) {
    if (n_regions == cap_regions) {
        int nc = cap_regions ? cap_regions * 2 : 128;
        Region *nr = realloc(regions, (size_t)nc * sizeof(Region));
        if (!nr) { fprintf(stderr, "pager: out of memory\n"); exit(1); }
        regions = nr;
        cap_regions = nc;
    }
    regions[n_regions++] = (Region){name, layer, ptr, bytes, 0, 0};
}

void pager_init(Model *m, long budget_bytes) {
    budget = budget_bytes;
    model_fd = m->fd;
    model_data = m->data;
    const char *s = getenv("CP_PAGER_SYS");
    sys_evict = s && s[0] == '1';
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
            add_region("expert", slot, expert_block(w, slot, e),
                       (size_t)ef * 4);

    if (pf_w > 0) {
        hist = malloc((size_t)n_slots * pf_w * topk * sizeof(int));
        if (hist) memset(hist, 0xFF, (size_t)n_slots * pf_w * topk * sizeof(int));
    }
    fprintf(stderr, "pager: %d regions (%d experts x %ld bytes), prefetch W=%d\n",
            n_regions, n_slots * n_exp, ef * 4, pf_w);
}

void pager_touch(const char *tensor, int layer) {
    if (budget < 0)
        return;
    /* unemb is TIED: its pages are token_emb's pages */
    if (strcmp(tensor, "unemb") == 0)
        tensor = "token_emb";
    for (int i = 0; i < n_regions; i++)
        if (regions[i].layer == layer && strcmp(regions[i].name, tensor) == 0) {
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

static void touch_expert(int slot, int e, int demand) {
    if (slot < 0 || slot >= n_slots || e < 0 || e >= n_exp)
        return;
    Region *r = &regions[expert_base + slot * n_exp + e];
    if (r->resident) {
        if (demand) { n_hits++; n_ehits++; }
        else n_pf_resident++;
    } else {
        r->resident = 1;
        if (demand) {
            n_misses++;
            bytes_missed += r->bytes;
            n_emisses++;
            bytes_emissed += r->bytes;
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
    long resident_bytes = 0;
    for (int i = 0; i < n_regions; i++)
        if (regions[i].resident)
            resident_bytes += regions[i].bytes;

    while (resident_bytes > budget) {
        int victim = -1;
        for (int i = 0; i < n_regions; i++)
            if (regions[i].resident &&
                (victim < 0 || regions[i].last_used < regions[victim].last_used))
                victim = i;
        if (victim < 0)
            break;
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
    }
}

void pager_report(void) {
    if (budget < 0)
        return;
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
    fprintf(stderr,
            "pager: %d regions | budget %ld MB | prefetch W=%d\n"
            "  all     : %ld accesses, %ld hits (%.2f%%), %.1f MB fetched\n"
            "  experts : %ld accesses, %ld hits (%.2f%%), %.1f MB fetched | "
            "prefetch: %ld fetches, %ld already resident\n"
            "  core    : %ld accesses, %ld hits (%.2f%%), %.1f MB fetched\n"
            "  %ld evictions, %ld MB evicted\n",
            n_regions, budget / (1024 * 1024), pf_w,
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
}
