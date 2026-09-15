#define _GNU_SOURCE /* madvise with -std=c11 */
#include "pager.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>

#define MAX_REGIONS 128
#define PAGE 4096UL

typedef struct {
    const char *name;
    int layer;
    float *ptr;
    size_t bytes;
    unsigned long last_used; /* tick; 0 = never touched */
    int resident;
} Region;

static Region regions[MAX_REGIONS];
static int n_regions;
static long budget = -1; /* -1 = disabled */
static unsigned long tick;
static long n_evictions, bytes_evicted;
static int sys_evict;    /* CP_PAGER_SYS: also evict from system page cache */
static int model_fd;     /* for posix_fadvise offsets */
static const float *model_data;

static void add_region(const char *name, int layer, float *ptr, size_t bytes) {
    if (n_regions < MAX_REGIONS)
        regions[n_regions++] = (Region){name, layer, ptr, bytes, 0, 0};
}

void pager_init(Model *m, long budget_bytes) {
    budget = budget_bytes;
    model_fd = m->fd;
    model_data = m->data;
    const char *s = getenv("CP_PAGER_SYS");
    sys_evict = s && s[0] == '1';
    int L = m->hdr.n_layer, D = m->hdr.d_model, F = m->hdr.d_ff;
    long V = m->hdr.vocab, C = m->hdr.n_ctx;
    Weights *w = &m->w;

    add_region("token_emb", -1, w->token_emb, (size_t)V * D * 4);
    add_region("pos_emb", -1, w->pos_emb, (size_t)C * D * 4);
    for (int l = 0; l < L; l++) {
        add_region("Wq", l, w->Wq + l * D * D, (size_t)D * D * 4);
        add_region("Wk", l, w->Wk + l * D * D, (size_t)D * D * 4);
        add_region("Wv", l, w->Wv + l * D * D, (size_t)D * D * 4);
        add_region("Wo", l, w->Wo + l * D * D, (size_t)D * D * 4);
        add_region("W1", l, w->W1 + (size_t)l * D * F, (size_t)D * F * 4);
        add_region("W2", l, w->W2 + (size_t)l * F * D, (size_t)F * D * 4);
    }
}

void pager_touch(const char *tensor, int layer) {
    if (budget < 0)
        return;
    /* unemb is TIED: its pages are token_emb's pages */
    if (strcmp(tensor, "unemb") == 0)
        tensor = "token_emb";
    for (int i = 0; i < n_regions; i++)
        if (regions[i].layer == layer && strcmp(regions[i].name, tensor) == 0) {
            regions[i].last_used = ++tick;
            regions[i].resident = 1;
            return;
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
    if (budget >= 0)
        fprintf(stderr, "pager: %ld evictions, %ld MB evicted (budget %ld MB)\n",
                n_evictions, bytes_evicted / (1024 * 1024), budget / (1024 * 1024));
}
