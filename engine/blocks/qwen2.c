/* qwen2.c — a complete Qwen2 forward pass in C, as a PORT PROOF.
 *
 * Why a separate file rather than bending the GPT-2 `Weights` struct: this model needs RMSNorm,
 * RoPE, GQA and SwiGLU, none of which the existing blocks express, and the point right now is to
 * prove the C arithmetic reproduces the recorded HuggingFace logits. Integration into the
 * CPW1 v4 loader comes after, so that a failure here is unambiguously arithmetic and not a
 * format/loader problem.
 *
 * Every projection in Qwen2 is stored (out, in) row-major, which is exactly `matmul_wx`.
 * That is the only kernel needed: no transposes, no repacking.
 */
#include "qwen2.h"
#include "kernels/kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_T 4096

const unsigned char *q8_slice_at(const unsigned char *base, int idx, long elems) {
    return base ? base + (size_t)idx * (size_t)((elems / 32) * 34) : (const unsigned char *)0;
}

int qwen2_forward(const QW2 *m, const int *tokens, int T, float *logits) {
    const int D = m->d_model, H = m->n_heads, KV = m->n_kv, HD = m->head_dim;
    const int F = m->d_ff, V = m->vocab, L = m->n_layers;
    const int rep = H / KV;
    if (T <= 0 || T > MAX_T) return -1;

    float *x  = malloc(sizeof(float) * (size_t)T * D);
    float *xn = malloc(sizeof(float) * (size_t)T * D);
    float *q  = malloc(sizeof(float) * (size_t)T * H * HD);
    float *k  = malloc(sizeof(float) * (size_t)T * KV * HD);
    float *vv = malloc(sizeof(float) * (size_t)T * KV * HD);
    float *ao = malloc(sizeof(float) * (size_t)T * D);
    float *sc = malloc(sizeof(float) * (size_t)MAX_T);
    float *hh = malloc(sizeof(float) * (size_t)T * (F > D ? F : D));
    float *gg = malloc(sizeof(float) * (size_t)T * (F > D ? F : D));
    if (!x || !xn || !q || !k || !vv || !ao || !sc || !hh || !gg) return -2;

    for (int t = 0; t < T; t++) {
        if (m->q_embed) q8_dequant_row(q8_slice_at(m->q_embed, tokens[t], (long)D),
                                       x + (size_t)t * D, D);
        else memcpy(x + (size_t)t * D, m->embed + (size_t)tokens[t] * D, sizeof(float) * D);
    }

    for (int l = 0; l < L; l++) {
        /* ---- attention ---- */
        for (int t = 0; t < T; t++)
            rmsnorm(xn + (size_t)t * D, x + (size_t)t * D, m->ln1 + (size_t)l * D, D, m->rms_eps);
        for (int t = 0; t < T; t++) {
            matmul_wx_auto(q + (size_t)t * H * HD,  xn + (size_t)t * D, m->Wq + (size_t)l * D * D,
                           q8_slice_at(m->q_Wq, l, (long)D * D), D, H * HD);
            matmul_wx_auto(k + (size_t)t * KV * HD, xn + (size_t)t * D, m->Wk + (size_t)l * KV * HD * D,
                           q8_slice_at(m->q_Wk, l, (long)KV * HD * D), D, KV * HD);
            matmul_wx_auto(vv + (size_t)t * KV * HD, xn + (size_t)t * D, m->Wv + (size_t)l * KV * HD * D,
                           q8_slice_at(m->q_Wv, l, (long)KV * HD * D), D, KV * HD);
            for (int j = 0; j < H * HD; j++)  q[(size_t)t * H * HD + j]  += m->bq[(size_t)l * H * HD + j];
            for (int j = 0; j < KV * HD; j++) {
                k[(size_t)t * KV * HD + j]  += m->bk[(size_t)l * KV * HD + j];
                vv[(size_t)t * KV * HD + j] += m->bv[(size_t)l * KV * HD + j];
            }
        }
        rope_apply(q, k, T, H, KV, HD, m->rope_theta);

        for (int h = 0; h < H; h++) {
            const int kh = h / rep;
            for (int i = 0; i < T; i++) {
                const float *qi = q + ((size_t)i * H + h) * HD;
                for (int j = 0; j <= i; j++) {
                    const float *kj = k + ((size_t)j * KV + kh) * HD;
                    float s = 0.0f;
                    for (int e = 0; e < HD; e++) s += qi[e] * kj[e];
                    sc[j] = s / sqrtf((float)HD);
                }
                softmax(sc, i + 1);
                for (int e = 0; e < HD; e++) {
                    float acc = 0.0f;
                    for (int j = 0; j <= i; j++)
                        acc += sc[j] * vv[((size_t)j * KV + kh) * HD + e];
                    ao[(size_t)i * D + (size_t)h * HD + e] = acc;
                }
            }
        }
        for (int t = 0; t < T; t++) {
            matmul_wx_auto(hh, ao + (size_t)t * D, m->Wo + (size_t)l * D * D,
                           q8_slice_at(m->q_Wo, l, (long)D * D), H * HD, D);
            for (int j = 0; j < D; j++) x[(size_t)t * D + j] += hh[j];
        }

        /* ---- SwiGLU FFN ---- */
        for (int t = 0; t < T; t++) {
            float *xt = x + (size_t)t * D;
            rmsnorm(xn + (size_t)t * D, xt, m->ln2 + (size_t)l * D, D, m->rms_eps);
            matmul_wx_auto(gg + (size_t)t * F, xn + (size_t)t * D, m->Wg + (size_t)l * F * D,
                           q8_slice_at(m->q_Wg, l, (long)F * D), D, F);
            matmul_wx_auto(hh + (size_t)t * F, xn + (size_t)t * D, m->Wu + (size_t)l * F * D,
                           q8_slice_at(m->q_Wu, l, (long)F * D), D, F);
            for (int j = 0; j < F; j++)
                gg[(size_t)t * F + j] = silu(gg[(size_t)t * F + j]) * hh[(size_t)t * F + j];
            matmul_wx_auto(hh, gg + (size_t)t * F, m->Wd + (size_t)l * D * F,
                           q8_slice_at(m->q_Wd, l, (long)D * F), F, D);
            for (int j = 0; j < D; j++) xt[j] += hh[j];
        }
    }

    /* final norm + tied unembedding */
    for (int t = 0; t < T; t++) {
        rmsnorm(xn + (size_t)t * D, x + (size_t)t * D, m->norm, D, m->rms_eps);
        matmul_wx_auto(logits + (size_t)t * V, xn + (size_t)t * D, m->embed, m->q_embed, D, V);
    }
    free(x); free(xn); free(q); free(k); free(vv); free(ao); free(sc); free(hh); free(gg);
    return 0;
}

/* Flat blob loader: header + tensors in a fixed order, written by
   models/qwen2.5-0.5b/convert.py. A real port uses CPW1 v4; this exists so the arithmetic can
   be proven before the format work, and a mismatch is never a loader bug. */
int qwen2_load_blob(QW2 *m, const char *path, float **owned) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    int hdr[7];
    if (fread(hdr, sizeof(int), 7, f) != 7) { fclose(f); return -1; }
    m->n_layers = hdr[0]; m->d_model = hdr[1]; m->n_heads = hdr[2];
    m->n_kv = hdr[3]; m->head_dim = hdr[4]; m->d_ff = hdr[5]; m->vocab = hdr[6];
    float fl[2];
    if (fread(fl, sizeof(float), 2, f) != 2) { fclose(f); return -1; }
    m->rope_theta = fl[0]; m->rms_eps = fl[1];

    const int L = m->n_layers, D = m->d_model, KVH = m->n_kv * m->head_dim;
    const int F = m->d_ff, V = m->vocab;
    long total = (long)V * D + D
               + (long)L * (2L * D + (long)D * D + D + 2L * KVH * D + 2L * KVH
                            + (long)D * D + 2L * F * D + (long)D * F);
    float *buf = malloc(sizeof(float) * total);
    if (!buf) { fclose(f); return -2; }
    if (fread(buf, sizeof(float), total, f) != (size_t)total) { free(buf); fclose(f); return -1; }
    fclose(f);
    *owned = buf;
    float *p = buf;
    m->embed = p; p += (long)V * D;
    m->norm  = p; p += D;
    m->ln1 = p; p += (long)L * D;
    m->ln2 = p; p += (long)L * D;
    m->Wq = p; p += (long)L * D * D;
    m->bq = p; p += (long)L * D;
    m->Wk = p; p += (long)L * KVH * D;
    m->bk = p; p += (long)L * KVH;
    m->Wv = p; p += (long)L * KVH * D;
    m->bv = p; p += (long)L * KVH;
    m->Wo = p; p += (long)L * D * D;
    m->Wg = p; p += (long)L * F * D;
    m->Wu = p; p += (long)L * F * D;
    m->Wd = p; p += (long)L * D * F;
    return 0;
}

/* ---------------------------------------------------------------- v4 loader
 *
 * Reads a CPW1 v4 file and fills QW2 from its TABLES — the model's geometry (heads, KV heads,
 * head_dim, rope theta, eps) comes from the block params, not from constants in this file. That
 * is the difference between an engine that runs a port and an engine that runs a model it was
 * compiled for.
 *
 * Only raw f32 tensors are accepted here; a quantized tensor is refused rather than decoded into
 * a layout, because QW2 indexes fp32 pointers. Q8 support for this path is a separate step.
 */
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>

#define KIND_ATTN 0x10
#define KIND_DENSE 0x11
#define KIND_NORM 0x02

static uint32_t rd32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const unsigned char *p) {
    uint64_t lo = rd32(p), hi = rd32(p + 4);
    return lo | (hi << 32);
}

typedef struct { char name[64]; uint64_t off, len; uint32_t dims[4]; int rank, dtype; } QT;

/* `dtype` in a tensor record is an INDEX INTO THE FILE'S ENCODING TABLE, not a convention.
 * Assuming 0 == f32 held only while f32 happened to be the first encoding written; once the
 * embedding was quantized first, the table became [q8_0, f32] and every f32 tensor read as
 * quantized. The meaning is per-file, so it must be looked up. */
#define MAX_ENC 8
static int g_enc_isq8[MAX_ENC];

static void parse_encodings(const unsigned char *img, uint64_t off, uint32_t n) {
    size_t o = (size_t)off;
    for (uint32_t i = 0; i < n && i < MAX_ENC; i++) {
        o += 4;                                  /* id */
        int kind = img[o++], scalar = img[o++];
        o += 4 + 3;                              /* group, block, sub, flags, align */
        uint16_t nl = (uint16_t)(img[o] | (img[o + 1] << 8)); o += 2 + nl;
        g_enc_isq8[i] = (kind == 1 && scalar == 3);
    }
}
static int dtype_is_q8(int dtype) { return (dtype >= 0 && dtype < MAX_ENC) ? g_enc_isq8[dtype] : -1; }

static const unsigned char *find_any(const QT *t, int n, const unsigned char *base, uint64_t od,
                                    const char *nm, int *dtype) {
    for (int i = 0; i < n; i++)
        if (!strcmp(t[i].name, nm)) {
            if (dtype) *dtype = t[i].dtype;
            return base + od + t[i].off;
        }
    return NULL;
}

static const float *find_q(const QT *t, int n, const unsigned char *base, uint64_t od, const char *nm) {
    for (int i = 0; i < n; i++)
        if (!strcmp(t[i].name, nm)) {
            if (dtype_is_q8(t[i].dtype)) return NULL;   /* not f32 in this file */
            return (const float *)(base + od + t[i].off);
        }
    return NULL;
}

int qwen2_load_v4(QW2 *m, const char *path, void **map, size_t *maplen) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return -1; }
    struct stat st;
    if (fstat(fd, &st)) { close(fd); return -1; }
    size_t fsz = (size_t)st.st_size;
    unsigned char *img = mmap(NULL, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (img == MAP_FAILED) { perror("mmap"); return -1; }
    if (fsz < 88 || rd32(img) != 0x34575043u) { munmap(img, fsz); return -1; }

    const uint32_t nt = rd32(img + 16), nb = rd32(img + 20);
    const uint64_t ot = rd64(img + 40), ob = rd64(img + 48), od = rd64(img + 64);
    parse_encodings(img, rd64(img + 32), rd32(img + 28));

    /* ---- tensor table (records follow a u32 count: SPEC §5) ---- */
    QT *t = calloc(nt ? nt : 1, sizeof *t);
    if (!t) { munmap(img, fsz); return -2; }
    size_t o = (size_t)ot;
    uint32_t cnt = rd32(img + o); o += 4;
    if (cnt != nt) { fprintf(stderr, "v4: tensor count %u != header %u\n", cnt, nt); goto bad; }
    for (uint32_t i = 0; i < nt; i++) {
        uint16_t nl = (uint16_t)(img[o] | (img[o + 1] << 8)); o += 2;
        if (nl >= sizeof t[i].name) goto bad;
        memcpy(t[i].name, img + o, nl); t[i].name[nl] = 0; o += nl;
        t[i].dtype = (int)rd32(img + o); o += 4;
        t[i].rank = img[o]; o += 1;
        for (int d = 0; d < t[i].rank && d < 4; d++) { t[i].dims[d] = rd32(img + o); o += 4; }
        o += 4 * (size_t)(t[i].rank > 4 ? 0 : 0);      /* dims already consumed above */
        t[i].off = rd64(img + o); o += 8;
        t[i].len = rd64(img + o); o += 8;
        o += 4 + 2;
    }

    /* ---- block table: geometry from the file, not from constants ---- */
    int n_heads = 0, n_kv = 0, hd = 0, seen_attn = 0, gated_seen = 0;
    float theta = 0.0f, eps = 0.0f;
    o = (size_t)ob;
    uint32_t cntb = rd32(img + o); o += 4;
    if (cntb != nb) { fprintf(stderr, "v4: block count %u != header %u\n", cntb, nb); goto bad; }
    for (uint32_t i = 0; i < nb; i++) {
        int kind = img[o]; o += 1;
        o += 4;                                        /* layer index */
        int nrefs = img[o]; o += 1 + 4 * (size_t)nrefs;
        int npar = img[o]; o += 1;
        uint32_t par[8] = {0};
        for (int k = 0; k < npar && k < 8; k++) { par[k] = rd32(img + o); o += 4; }
        if (kind == KIND_ATTN && !seen_attn) {
            n_heads = (int)par[0]; n_kv = (int)par[1]; hd = (int)par[2];
            memcpy(&theta, &par[3], 4);
            seen_attn = 1;
        } else if (kind == KIND_DENSE && npar >= 5 && par[4] == 1) {
            gated_seen = 1;
        } else if (kind == KIND_NORM && !eps) {
            memcpy(&eps, &par[0], 4);
        }
    }
    if (!seen_attn || !gated_seen) {
        fprintf(stderr, "v4: not a Qwen2-shaped file (attention=%d gated_ffn=%d)\n",
                seen_attn, gated_seen);
        goto bad;
    }

    /* ---- geometry cross-check against the tensors themselves ---- */
    /* embed may be f32 or Q8: the tied head is the biggest single tensor in the file. */
    int embed_dt = 0;
    const unsigned char *embed_p = find_any(t, nt, img, od, "embed", &embed_dt);
    const float *embed = (embed_p && dtype_is_q8(embed_dt) == 0) ? (const float *)embed_p : NULL;
    if (embed_p && dtype_is_q8(embed_dt) != 0) m->q_embed = embed_p;
    m->embed = embed;
    if (!embed_p || !find_any(t, nt, img, od, "Wq", NULL) || !find_any(t, nt, img, od, "Wk", NULL)) {
        fprintf(stderr, "v4: missing core tensors\n"); goto bad;
    }
    m->vocab = (int)t[0].dims[0];
    for (uint32_t i = 0; i < nt; i++) if (!strcmp(t[i].name, "embed")) { m->vocab = (int)t[i].dims[0]; m->d_model = (int)t[i].dims[1]; }
    for (uint32_t i = 0; i < nt; i++) if (!strcmp(t[i].name, "Wk")) m->n_layers = (int)t[i].dims[0];
    m->n_heads = n_heads; m->n_kv = n_kv; m->head_dim = hd;
    m->rope_theta = theta; m->rms_eps = eps;
    for (uint32_t i = 0; i < nt; i++) if (!strcmp(t[i].name, "Wg")) m->d_ff = (int)t[i].dims[1];
    if (n_heads * hd != m->d_model) {
        fprintf(stderr, "v4: attention params say %d heads x %d = %d, but d_model is %d\n",
                n_heads, hd, n_heads * hd, m->d_model);
        goto bad;
    }

    m->norm = find_q(t, nt, img, od, "norm");
    m->ln1  = find_q(t, nt, img, od, "ln1");
    m->ln2  = find_q(t, nt, img, od, "ln2");
    m->bq = find_q(t, nt, img, od, "bq");
    m->bk = find_q(t, nt, img, od, "bk");
    m->bv = find_q(t, nt, img, od, "bv");
    /* The seven large matrices may be f32 or Q8_0, independently. Each tensor carries its own
       encoding, so a file can mix them (e.g. quantized projections, exact embeddings). */
    {
        struct { const char *nm; const float **f; const unsigned char **q; } tab[] = {
            {"Wq", &m->Wq, &m->q_Wq}, {"Wk", &m->Wk, &m->q_Wk},
            {"Wv", &m->Wv, &m->q_Wv}, {"Wo", &m->Wo, &m->q_Wo},
            {"Wg", &m->Wg, &m->q_Wg}, {"Wu", &m->Wu, &m->q_Wu},
            {"Wd", &m->Wd, &m->q_Wd},
        };
        for (size_t k = 0; k < sizeof tab / sizeof tab[0]; k++) {
            int dt = 0;
            const unsigned char *p = find_any(t, nt, img, od, tab[k].nm, &dt);
            if (!p) { fprintf(stderr, "v4: missing %s\n", tab[k].nm); goto bad; }
            if (dtype_is_q8(dt) == 0) { *tab[k].f = (const float *)p; *tab[k].q = NULL; }
            else {
                *tab[k].f = NULL; *tab[k].q = p;     /* unused when quantized */
                m->quantized = 1;
                /* length must match the logical size: bytes-vs-elements has bitten twice */
                for (uint32_t i = 0; i < nt; i++)
                    if (!strcmp(t[i].name, tab[k].nm)) {
                        long elems = 1;
                        for (int d = 0; d < t[i].rank; d++) elems *= (long)t[i].dims[d];
                        if ((unsigned long long)t[i].len != (unsigned long long)((elems / 32) * 34)) {
                            fprintf(stderr, "v4: %s has %llu bytes, expected %lld for %lld "
                                            "elements\n", tab[k].nm,
                                    (unsigned long long)t[i].len, (long long)((elems / 32) * 34), (long long)elems);
                            goto bad;
                        }
                        break;
                    }
            }
        }
    }
    {
        const char *miss[12]; int nm = 0;
        if (!m->norm) miss[nm++] = "norm";
        if (!m->ln1) miss[nm++] = "ln1";
        if (!m->ln2) miss[nm++] = "ln2";
        if (!m->bq) miss[nm++] = "bq";
        if (!m->bk) miss[nm++] = "bk";
        if (!m->bv) miss[nm++] = "bv";
        if (!(m->Wo || m->q_Wo)) miss[nm++] = "Wo";
        if (!(m->Wg || m->q_Wg)) miss[nm++] = "Wg";
        if (!(m->Wu || m->q_Wu)) miss[nm++] = "Wu";
        if (!(m->Wd || m->q_Wd)) miss[nm++] = "Wd";
        if (!(m->embed || m->q_embed)) miss[nm++] = "embed";
        if (nm) {
            /* name them: "missing tensor(s)" sent me looking at the wrong field once already */
            fprintf(stderr, "v4: missing tensor(s):");
            for (int i = 0; i < nm; i++) fprintf(stderr, " %s", miss[i]);
            fprintf(stderr, "\n");
            goto bad;
        }
    }

    free(t);
    *map = img; *maplen = fsz;
    return 0;
bad:
    free(t); munmap(img, fsz); return -1;
}

/* ---------------------------------------------------------------- KV cache (GQA)
 *
 * The full-sequence forward is O(T^2) in prefill, which is fine once and wrong per token. Chat
 * needs one token at a time against a cache. The cache is sized (maxT, n_kv*head_dim) per layer:
 * GQA means only n_kv heads are stored, so it is 7x smaller than an MHA cache for this model —
 * which matters more on the T480 than the arithmetic does.
 */
typedef struct { float *k, *v; } QW2LayerCache;

struct QW2Cache { int maxT, n_layers, kv_dim; QW2LayerCache *l; };

QW2Cache *qwen2_cache_alloc(const QW2 *m, int maxT) {
    QW2Cache *c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->maxT = maxT; c->n_layers = m->n_layers; c->kv_dim = m->n_kv * m->head_dim;
    c->l = calloc((size_t)m->n_layers, sizeof *c->l);
    if (!c->l) { free(c); return NULL; }
    for (int i = 0; i < m->n_layers; i++) {
        c->l[i].k = calloc((size_t)maxT * c->kv_dim, sizeof(float));
        c->l[i].v = calloc((size_t)maxT * c->kv_dim, sizeof(float));
        if (!c->l[i].k || !c->l[i].v) return NULL;
    }
    return c;
}

void qwen2_cache_free(QW2Cache *c) {
    if (!c) return;
    for (int i = 0; i < c->n_layers; i++) { free(c->l[i].k); free(c->l[i].v); }
    free(c->l); free(c);
}

int qwen2_step(const QW2 *m, QW2Cache *c, int token, int pos, float *logits) {
    const int D = m->d_model, H = m->n_heads, KV = m->n_kv, HD = m->head_dim;
    const int F = m->d_ff, V = m->vocab, L = m->n_layers, rep = H / KV;
    if (pos < 0 || pos >= c->maxT) return -1;

    float *x = malloc(sizeof(float) * D), *xn = malloc(sizeof(float) * D);
    float *q = malloc(sizeof(float) * H * HD), *h = malloc(sizeof(float) * (F > D ? F : D));
    float *g = malloc(sizeof(float) * (F > D ? F : D));
    float *ao = malloc(sizeof(float) * D), *sc = malloc(sizeof(float) * (size_t)c->maxT);
    if (!x || !xn || !q || !h || !g || !ao || !sc) return -2;
    if (m->q_embed) q8_dequant_row(q8_slice_at(m->q_embed, token, (long)D), x, D);
    else memcpy(x, m->embed + (size_t)token * D, sizeof(float) * D);

    for (int l = 0; l < L; l++) {
        float *kc = c->l[l].k + (size_t)pos * c->kv_dim;
        float *vc = c->l[l].v + (size_t)pos * c->kv_dim;
        rmsnorm(xn, x, m->ln1 + (size_t)l * D, D, m->rms_eps);
        matmul_wx_auto(q, xn, m->Wq + (size_t)l * D * D, q8_slice_at(m->q_Wq, l, (long)D * D), D, H * HD);
        for (int j = 0; j < H * HD; j++) q[j] += m->bq[(size_t)l * H * HD + j];
        matmul_wx_auto(kc, xn, m->Wk + (size_t)l * c->kv_dim * D,
                       q8_slice_at(m->q_Wk, l, (long)c->kv_dim * D), D, c->kv_dim);
        matmul_wx_auto(vc, xn, m->Wv + (size_t)l * c->kv_dim * D,
                       q8_slice_at(m->q_Wv, l, (long)c->kv_dim * D), D, c->kv_dim);
        for (int j = 0; j < c->kv_dim; j++) {
            kc[j] += m->bk[(size_t)l * c->kv_dim + j];
            vc[j] += m->bv[(size_t)l * c->kv_dim + j];
        }
        /* RoPE for ONE position: same formula as rope_apply, applied to q and the new k */
        for (int hh = 0; hh < 2; hh++) {
            float *base = hh ? kc : q;
            int nh = hh ? KV : H;
            for (int i = 0; i < HD / 2; i++) {
                float inv = powf(m->rope_theta, -2.0f * (float)i / (float)HD);
                float ang = (float)pos * inv, cc = cosf(ang), ss = sinf(ang);
                for (int hd = 0; hd < nh; hd++) {
                    float *v = base + (size_t)hd * HD;
                    float a = v[i], b = v[i + HD / 2];
                    v[i] = a * cc - b * ss;
                    v[i + HD / 2] = b * cc + a * ss;
                }
            }
        }
        for (int hd = 0; hd < H; hd++) {
            const int kh = hd / rep;
            const float *qh = q + (size_t)hd * HD;
            for (int j = 0; j <= pos; j++) {
                const float *kj = c->l[l].k + (size_t)j * c->kv_dim + (size_t)kh * HD;
                float s = 0.0f;
                for (int e = 0; e < HD; e++) s += qh[e] * kj[e];
                sc[j] = s / sqrtf((float)HD);
            }
            softmax(sc, pos + 1);
            for (int e = 0; e < HD; e++) {
                float acc = 0.0f;
                for (int j = 0; j <= pos; j++)
                    acc += sc[j] * c->l[l].v[(size_t)j * c->kv_dim + (size_t)kh * HD + e];
                ao[(size_t)hd * HD + e] = acc;
            }
        }
        matmul_wx_auto(h, ao, m->Wo + (size_t)l * D * D, q8_slice_at(m->q_Wo, l, (long)D * D), H * HD, D);
        for (int j = 0; j < D; j++) x[j] += h[j];

        rmsnorm(xn, x, m->ln2 + (size_t)l * D, D, m->rms_eps);
        matmul_wx_auto(g, xn, m->Wg + (size_t)l * F * D, q8_slice_at(m->q_Wg, l, (long)F * D), D, F);
        matmul_wx_auto(h, xn, m->Wu + (size_t)l * F * D, q8_slice_at(m->q_Wu, l, (long)F * D), D, F);
        for (int j = 0; j < F; j++) g[j] = silu(g[j]) * h[j];
        matmul_wx_auto(h, g, m->Wd + (size_t)l * D * F, q8_slice_at(m->q_Wd, l, (long)D * F), F, D);
        for (int j = 0; j < D; j++) x[j] += h[j];
    }
    rmsnorm(xn, x, m->norm, D, m->rms_eps);
    matmul_wx_auto(logits, xn, m->embed, m->q_embed, D, V);
    free(x); free(xn); free(q); free(h); free(g); free(ao); free(sc);
    return 0;
}
