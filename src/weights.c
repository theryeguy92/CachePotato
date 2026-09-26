#define _GNU_SOURCE /* madvise/posix_fadvise with -std=c11 */
#include "weights.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Floats the payload must contain, derived from the config alone.
   Independent of map_weights() ON PURPOSE — it's the cross-check. */
long expected_floats(const Header *h) {
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

/* Assign every field of w by advancing ptr past each tensor, in the
   CPW1 order. Per-tensor-type across layers: layer l of Wq lives at
   w->Wq + l*d*d, etc. Works over ANY float buffer (mmap'd file, or the
   trainer's params/grads buffers). */
void layout_weights(Weights *w, const Header *h, float *ptr) {
    long v = h->vocab, d = h->d_model, f = h->d_ff;
    long c = h->n_ctx, L = h->n_layer;

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

    /* v2 has no MoE: clear the v3 fields so nothing downstream reads garbage
       (the trainer lays this struct over its own buffers). */
    w->Wr = w->br = NULL;
    w->W1s = w->b1s = w->W2s = w->b2s = NULL;
    w->experts = NULL;
    w->is_moe = 0;
    w->d_model = (int)d;
    w->n_moe = w->n_exp = w->exp_hid = w->topk = w->moe_from = 0;
    w->moe_slot = w->dense_slot = NULL;
}

/* ------------------------------------------------------------------ CPW1 v3 */

long expert_floats(int d_model, int exp_hid) {
    /* W1e(d,EH) b1e(EH) W2e(EH,d) b2e(d) — one contiguous pager region */
    return (long)d_model * exp_hid + exp_hid + (long)exp_hid * d_model + d_model;
}

long expected_floats_v3(const HeaderV3 *h) {
    long v = h->vocab, d = h->d_model, f = h->d_ff, c = h->n_ctx, L = h->n_layer;
    long nm = h->n_moe_layers, nd = h->n_dense_layers;
    long N = h->n_exp, EH = h->exp_hid;

    long emb      = v * d + c * d;
    long norms    = 2 * L * d            /* ln1 g,b */
                  + 2 * L * d;           /* ln2 g,b */
    long attn     = L * (4 * (d * d + d));           /* Wq..Wo + biases */
    long dense    = nd * ((d * f + f) + (f * d + d));
    long router   = nm * (d * N + N);
    long shared   = nm * (2 * (d * d + d));
    long experts  = nm * N * expert_floats((int)d, (int)EH);
    long lnf      = 2 * d;
    return emb + norms + attn + dense + router + shared + experts + lnf;
}

/* Assign the v3 pointers and build the layer -> slot maps. The dense-FFN arrays
   are indexed by DENSE SLOT, not by layer, because the dead dense-FFN weights on
   MoE layers are not in the file. */
void layout_weights_v3(Weights *w, const HeaderV3 *h, float *ptr) {
    long v = h->vocab, d = h->d_model, f = h->d_ff;
    long c = h->n_ctx, L = h->n_layer;
    long N = h->n_exp, EH = h->exp_hid;
    long nm = 0, nd = 0;

    w->moe_slot = malloc((size_t)L * sizeof(int));
    w->dense_slot = malloc((size_t)L * sizeof(int));
    if (!w->moe_slot || !w->dense_slot) {
        fprintf(stderr, "out of memory for layer maps\n");
        w->is_moe = 0;
        return;
    }
    for (long l = 0; l < L; l++) {
        if (l >= h->moe_from && l < L - 1) {
            w->moe_slot[l] = (int)nm++;
            w->dense_slot[l] = -1;
        } else {
            w->moe_slot[l] = -1;
            w->dense_slot[l] = (int)nd++;
        }
    }
    if (nm != h->n_moe_layers || nd != h->n_dense_layers) {
        fprintf(stderr, "layout mismatch: moe_from=%d L=%ld implies %ld MoE / %ld "
                        "dense, header claims %d / %d\n",
                h->moe_from, L, nm, nd, h->n_moe_layers, h->n_dense_layers);
        w->is_moe = 0;
        return;
    }
    if (h->topk != 2) {
        /* the router scan in moe_ffn/moe_ffn_step is written for top-2; refuse
           loudly rather than silently dispatch the wrong number of experts */
        fprintf(stderr, "v3 file has topk=%d; the engine implements top-2 only\n",
                h->topk);
        w->is_moe = 0;
        return;
    }

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
    w->W1 = ptr;         ptr += nd * d * f;
    w->b1 = ptr;         ptr += nd * f;
    w->W2 = ptr;         ptr += nd * f * d;
    w->b2 = ptr;         ptr += nd * d;
    w->Wr = ptr;         ptr += nm * d * N;
    w->br = ptr;         ptr += nm * N;
    w->W1s = ptr;        ptr += nm * d * d;
    w->b1s = ptr;        ptr += nm * d;
    w->W2s = ptr;        ptr += nm * d * d;
    w->b2s = ptr;        ptr += nm * d;
    w->experts = ptr;    ptr += nm * N * expert_floats((int)d, (int)EH);
    w->lnf_g = ptr;      ptr += d;
    w->lnf_b = ptr;

    w->is_moe = 1;
    w->d_model = (int)d;
    w->n_moe = (int)nm;
    w->n_exp = (int)N;
    w->exp_hid = (int)EH;
    w->topk = h->topk;
    w->moe_from = h->moe_from;
}

float *expert_block(const Weights *w, int slot, int e) {
    return w->experts
         + ((size_t)slot * w->n_exp + e) * expert_floats(w->d_model, w->exp_hid);
}

int moe_layer_at(const Model *m, int l) {
    return (m->is_v3 && m->w.is_moe && l >= m->w.moe_from && l < m->hdr.n_layer - 1);
}

static void map_weights(Model *m) {
    if (m->is_v3)
        layout_weights_v3(&m->w, &m->h3, m->data + CP_V3_HDR_BYTES / sizeof(float));
    else
        layout_weights(&m->w, &m->hdr, m->data + sizeof(Header) / sizeof(float));
}

int model_load(Model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) { perror("open"); return -1; }

    struct stat st;
    if (fstat(m->fd, &st) < 0) { perror("fstat"); return -1; }
    m->file_size = (size_t)st.st_size;

    /* loud failure #1: wrong file / unknown format */
    uint32_t magic = 0, version = 0;
    if (read(m->fd, &magic, 4) != 4 || read(m->fd, &version, 4) != 4) {
        fprintf(stderr, "couldn't read magic/version\n");
        return -1;
    }
    if (magic != CP_MAGIC) {
        fprintf(stderr, "bad magic 0x%08x — not a CPW1 weight file\n", magic);
        return -1;
    }

    size_t want;
    if (version == CP_VERSION_V2) {
        m->hdr.magic = magic;
        m->hdr.version = version;
        if (read(m->fd, (char *)&m->hdr + 8, sizeof(Header) - 8)
                != (ssize_t)(sizeof(Header) - 8)) {
            fprintf(stderr, "couldn't read the v2 header\n");
            return -1;
        }
        m->is_v3 = 0;
        want = CP_V2_HDR_BYTES + (size_t)expected_floats(&m->hdr) * sizeof(float);
    } else if (version == CP_VERSION_V3) {
        m->h3.magic = magic;
        m->h3.version = version;
        if (read(m->fd, (char *)&m->h3 + 8, sizeof(HeaderV3) - 8)
                != (ssize_t)(sizeof(HeaderV3) - 8)) {
            fprintf(stderr, "couldn't read the v3 header\n");
            return -1;
        }
        /* mirror the common fields so the rest of the engine is version-agnostic */
        m->hdr.magic = magic;
        m->hdr.version = version;
        m->hdr.vocab = m->h3.vocab;   m->hdr.d_model = m->h3.d_model;
        m->hdr.n_heads = m->h3.n_heads; m->hdr.d_ff = m->h3.d_ff;
        m->hdr.n_layer = m->h3.n_layer; m->hdr.n_ctx = m->h3.n_ctx;
        m->is_v3 = 1;
        want = CP_V3_HDR_BYTES + (size_t)expected_floats_v3(&m->h3) * sizeof(float);
    } else {
        fprintf(stderr, "unsupported CPW1 version %u (loader knows v%u and v%u)\n",
                version, CP_VERSION_V2, CP_VERSION_V3);
        return -1;
    }

    /* loud failure #2: truncated file or exporter/loader layout drift */
    if (m->file_size != want) {
        fprintf(stderr, "size mismatch: file is %zu bytes, config implies %zu "
                        "(CPW1 v%u) — exporter/loader layout drift\n",
                m->file_size, want, version);
        return -1;
    }

    m->data = mmap(NULL, m->file_size, PROT_READ, MAP_PRIVATE, m->fd, 0);
    if (m->data == MAP_FAILED) { perror("mmap"); return -1; }

    map_weights(m);
    if (m->is_v3 && !m->w.is_moe) return -1;   /* layout_weights_v3 complained */
    return 0;
}

void model_free(Model *m) {
    free(m->w.moe_slot);
    free(m->w.dense_slot);
    m->w.moe_slot = m->w.dense_slot = NULL;
    if (m->data) munmap(m->data, m->file_size);
    if (m->fd >= 0) close(m->fd);
}

void model_evict_all(Model *m) {
    madvise(m->data, m->file_size, MADV_DONTNEED);
    posix_fadvise(m->fd, 0, 0, POSIX_FADV_DONTNEED);
}
