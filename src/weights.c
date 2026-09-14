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
    /* token_emb + unemb: 2*v*d | Wq,Wk,Wv,Wo: 4*d*d | W1+W2: 2*d*f | gains: 3*d */
    return 2 * v * d + 4 * d * d + 2 * d * f + 3 * d;
}

/* Assign every field of m->w by advancing ptr past each tensor, in the
 * SAME order the exporter wrote them:
 *   token_emb, Wq, Wk, Wv, Wo, W1, W2, ln1_g, ln2_g, lnf_g, unemb
 * This is llama2.c's memory_map_weights(), minus the multi-layer loop. */
static void map_weights(Model *m) {
    float *ptr = m->data + sizeof(Header) / sizeof(float); /* skip header */
    int v = m->hdr.vocab, d = m->hdr.d_model, f = m->hdr.d_ff;
    Weights *w = &m->w;

    w->token_emb = ptr;  ptr += v * d;
    w->Wq = ptr;         ptr += d * d;
    w->Wk = ptr;         ptr += d * d;
    w->Wv = ptr;         ptr += d * d;
    w->Wo = ptr;         ptr += d * d;
    w->W1 = ptr;         ptr += d * f;
    w->W2 = ptr;         ptr += f * d;
    w->ln1_g = ptr;      ptr += d;
    w->ln2_g = ptr;      ptr += d;
    w->lnf_g = ptr;      ptr += d;
    w->unemb = ptr;
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
    /* loud failure #1: wrong file / wrong format */
    if (m->hdr.magic != CP_MAGIC) {
        fprintf(stderr, "bad magic 0x%08x — not a CPW1 weight file\n", m->hdr.magic);
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
