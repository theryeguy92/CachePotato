#define _GNU_SOURCE /* madvise/posix_fadvise with -std=c11 */
#include "weights.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Floats the payload must contain, derived from the config alone.
   Independent of map_weights() ON PURPOSE — it's the cross-check. */
static long expected_floats(const Header *h) {
    long v = h->vocab, d = h->d_model, f = h->d_ff;
    long c = h->n_ctx, L = h->n_layer;
    long per_layer =
        2 * d +              /* ln1 g,b   */
        4 * (d * d + d) +    /* Wq..Wo + biases */
        2 * d +              /* ln2 g,b   */
        (d * f + f) +        /* W1 + b1   */
        (f * d + d);         /* W2 + b2   */
    return v * d + c * d + L * per_layer + 2 * d; /* + lnf g,b; unemb tied */
}

/* Assign every field of m->w by advancing ptr past each tensor, in the
   SAME order the exporter wrote them. Per-tensor-type across layers:
   layer l of Wq lives at w->Wq + l*d*d, etc. */
static void map_weights(Model *m) {
    float *ptr = m->data + sizeof(Header) / sizeof(float); /* skip header */
    long v = m->hdr.vocab, d = m->hdr.d_model, f = m->hdr.d_ff;
    long c = m->hdr.n_ctx, L = m->hdr.n_layer;
    Weights *w = &m->w;

    w->token_emb = ptr;  ptr += v * d;
    w->pos_emb = ptr;    ptr += c * d;
    w->ln1_g = ptr;      ptr += L * d;
    w->ln1_b = ptr;      ptr += L * d;
    w->Wq = ptr;         ptr += L * d * d;
    w->bq = ptr;         ptr += L * d;
    w->Wk = ptr;         ptr += L * d * d;
    w->bk = ptr;         ptr += L * d;
    w->Wv = ptr;         ptr += L * d * d;
    w->bv = ptr;         ptr += L * d;
    w->Wo = ptr;         ptr += L * d * d;
    w->bo = ptr;         ptr += L * d;
    w->ln2_g = ptr;      ptr += L * d;
    w->ln2_b = ptr;      ptr += L * d;
    w->W1 = ptr;         ptr += L * d * f;
    w->b1 = ptr;         ptr += L * f;
    w->W2 = ptr;         ptr += L * f * d;
    w->b2 = ptr;         ptr += L * d;
    w->lnf_g = ptr;      ptr += d;
    w->lnf_b = ptr;
}

int model_load(Model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) { perror("open"); return -1; }

    struct stat st;
    if (fstat(m->fd, &st) < 0) { perror("fstat"); return -1; }
    m->file_size = (size_t)st.st_size;

    if (read(m->fd, &m->hdr, sizeof(Header)) != (ssize_t)sizeof(Header)) {
        fprintf(stderr, "couldn't read header\n");
        return -1;
    }
    /* loud failure #1: wrong file / wrong format / stale version */
    if (m->hdr.magic != CP_MAGIC || m->hdr.version != CP_VERSION) {
        fprintf(stderr, "bad magic/version 0x%08x/%u — not a CPW1 v%u weight file\n",
                m->hdr.magic, m->hdr.version, CP_VERSION);
        return -1;
    }
    /* loud failure #2: truncated file or exporter/loader layout drift */
    size_t want = sizeof(Header) + (size_t)expected_floats(&m->hdr) * sizeof(float);
    if (m->file_size != want) {
        fprintf(stderr, "size mismatch: file is %zu bytes, config implies %zu\n",
                m->file_size, want);
        return -1;
    }

    m->data = mmap(NULL, m->file_size, PROT_READ, MAP_PRIVATE, m->fd, 0);
    if (m->data == MAP_FAILED) { perror("mmap"); return -1; }

    map_weights(m);
    return 0;
}

void model_free(Model *m) {
    if (m->data) munmap(m->data, m->file_size);
    if (m->fd >= 0) close(m->fd);
}

void model_evict_all(Model *m) {
    madvise(m->data, m->file_size, MADV_DONTNEED);
    posix_fadvise(m->fd, 0, 0, POSIX_FADV_DONTNEED);
}
