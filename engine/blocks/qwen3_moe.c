/* qwen3_moe.c — Qwen3-30B-A3B decoder layer in C.
 *
 * Ported against models/qwen3-30b-a3b/reference.py, which is itself gated to ~1e-7 against the
 * published weights. So a disagreement here is this file's fault, not the model's.
 */
#include "qwen3_moe.h"
#include "kernels/kernels.h"

#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int qwen3_layer_alloc(QW3Out *o, const QW3Layer *m, int T) {
    o->attn = calloc((size_t)T * m->d_model, sizeof(float));
    o->mlp = calloc((size_t)T * m->d_model, sizeof(float));
    o->router = calloc((size_t)T * m->n_experts, sizeof(float));
    o->topk_idx = calloc((size_t)T * m->topk, sizeof(int));
    return (o->attn && o->mlp && o->router && o->topk_idx) ? 0 : -1;
}
void qwen3_layer_free(QW3Out *o) {
    free(o->attn); free(o->mlp); free(o->router); free(o->topk_idx);
    memset(o, 0, sizeof *o);
}

/* One matmul, encoding-agnostic. Same idea as qwen2.c's mm_wx: the block size and the kernel must
   be chosen together, because a Q8 payload read as f32 is not an error you can see. */
static void mm(const QW3T *t, float *out, const float *x, int n_in, int n_out) {
    if (t->enc == QW3_ENC_Q8) matmul_wx_q8(out, x, (const uint8_t *)t->p, n_in, n_out);
    else                      matmul_wx(out, x, (const float *)t->p, n_in, n_out);
}

/* per-head RMSNorm: rows of `head_dim`, gain is (head_dim) and shared by every head.
   This is the block the config does not document; the reference proves it sits BEFORE RoPE. */
static void head_rmsnorm(float *v, int rows, int hd, const float *g, float eps) {
    for (int r = 0; r < rows; r++) {
        float *row = v + (size_t)r * hd;
        rmsnorm(row, row, g, hd, eps);
    }
}

int qwen3_layer(const QW3Layer *m, const float *x, int T, QW3Out *o) {
    const int D = m->d_model, H = m->n_heads, KV = m->n_kv, HD = m->head_dim;
    const int E = m->n_experts, K = m->topk, F = m->d_exp, rep = H / KV;
    float *xn = malloc(sizeof(float) * (size_t)T * D);
    float *q = malloc(sizeof(float) * (size_t)T * H * HD);
    float *k = malloc(sizeof(float) * (size_t)T * KV * HD);
    float *v = malloc(sizeof(float) * (size_t)T * KV * HD);
    float *ao = malloc(sizeof(float) * (size_t)T * H * HD);  /* NOT xn: H*HD is 2x D */
    float *sc = malloc(sizeof(float) * (size_t)T);
    float *hh = malloc(sizeof(float) * (size_t)(F > D ? F : D));
    float *gu = malloc(sizeof(float) * (size_t)(F > D ? F : D));
    float *probs = malloc(sizeof(float) * E);
    int *idx = malloc(sizeof(int) * (size_t)K);
    float *wt = malloc(sizeof(float) * (size_t)T * K);
    if (!xn || !q || !k || !v || !ao || !sc || !hh || !gu || !probs || !idx || !wt) return -1;

    /* ---- attention: norm -> qkv -> QK-norm -> RoPE -> GQA -> o_proj ---- */
    for (int t = 0; t < T; t++) {
        rmsnorm(xn + (size_t)t * D, x + (size_t)t * D, m->ln1, D, m->rms_eps);
        mm(&m->Wq, q + (size_t)t * H * HD,  xn + (size_t)t * D, D, H * HD);
        mm(&m->Wk, k + (size_t)t * KV * HD, xn + (size_t)t * D, D, KV * HD);
        mm(&m->Wv, v + (size_t)t * KV * HD, xn + (size_t)t * D, D, KV * HD);
        head_rmsnorm(q + (size_t)t * H * HD,  H,  HD, m->q_norm, m->rms_eps);
        head_rmsnorm(k + (size_t)t * KV * HD, KV, HD, m->k_norm, m->rms_eps);
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
                for (int j = 0; j <= i; j++) acc += sc[j] * v[((size_t)j * KV + kh) * HD + e];
                ao[(size_t)i * H * HD + (size_t)h * HD + e] = acc;
            }
        }
    }
    for (int t = 0; t < T; t++)
        mm(&m->Wo, o->attn + (size_t)t * D, ao + (size_t)t * H * HD, H * HD, D);

    /* ---- MoE: router -> softmax -> top-k -> renormalise -> gated experts ---- */
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < D; j++) hh[j] = x[(size_t)t * D + j] + o->attn[(size_t)t * D + j];
        rmsnorm(xn + (size_t)t * D, hh, m->ln2, D, m->rms_eps);
        mm(&m->Wgate, o->router + (size_t)t * E, xn + (size_t)t * D, D, E);
        memcpy(probs, o->router + (size_t)t * E, sizeof(float) * E);
        softmax(probs, E);

        /* strict `>` scan with lower id winning ties, the same rule as the gpt2-moe block */
        for (int j = 0; j < K; j++) {
            int best = -1;
            for (int e = 0; e < E; e++) {
                int taken = 0;
                for (int p = 0; p < j; p++) if (idx[p] == e) { taken = 1; break; }
                if (taken) continue;
                if (best < 0 || probs[e] > probs[best]) best = e;
            }
            idx[j] = best;
        }
        float sum = 0.0f;
        for (int j = 0; j < K; j++) sum += probs[idx[j]];      /* norm_topk_prob */
        for (int j = 0; j < K; j++) wt[(size_t)t * K + j] = probs[idx[j]] / sum;

        for (int j = 0; j < K; j++) {
            const int e = idx[j];
            o->topk_idx[(size_t)t * K + j] = e;
            const QW3T *wg = &m->exp[(size_t)e * 3 + 0];
            const QW3T *wu = &m->exp[(size_t)e * 3 + 1];
            const QW3T *wd = &m->exp[(size_t)e * 3 + 2];
            mm(wg, gu, xn + (size_t)t * D, D, F);              /* gate */
            mm(wu, hh, xn + (size_t)t * D, D, F);              /* up   */
            for (int i = 0; i < F; i++) gu[i] = silu(gu[i]) * hh[i];
            mm(wd, hh, gu, F, D);                              /* down */
            for (int i = 0; i < D; i++) o->mlp[(size_t)t * D + i] += wt[(size_t)t * K + j] * hh[i];
        }
    }
    free(xn); free(q); free(k); free(v); free(ao); free(sc); free(hh); free(gu); free(probs); free(idx); free(wt);
    return 0;
}

/* Flat blob for the shard-1 gate: header, then tensors in a fixed order written by
   tools/verify_qwen3_blocks.py. The real port reads CPW1 v4; this exists so a mismatch here is
   unambiguously arithmetic. */
int qwen3_load_blob(QW3Layer *m, const char *path, float **owned) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    int h[8];
    if (fread(h, sizeof(int), 8, f) != 8) { fclose(f); return -1; }
    m->d_model = h[0]; m->n_heads = h[1]; m->n_kv = h[2]; m->head_dim = h[3];
    m->d_exp = h[4]; m->n_experts = h[5]; m->topk = h[6]; m->vocab = h[7];
    float fl[2];
    if (fread(fl, sizeof(float), 2, f) != 2) { fclose(f); return -1; }
    m->rope_theta = fl[0]; m->rms_eps = fl[1];
    const int D = m->d_model, H = m->n_heads, KV = m->n_kv, HD = m->head_dim;
    const int E = m->n_experts, F = m->d_exp;
    long total = 2L*D + 2L*HD + (long)H*HD*D + 2L*KV*HD*D + (long)D*H*HD + (long)E*D
               + 3L*E*F*D;
    float *buf = malloc(sizeof(float) * total);
    if (!buf) { fclose(f); return -2; }
    if (fread(buf, sizeof(float), total, f) != (size_t)total) { free(buf); fclose(f); return -1; }
    fclose(f);
    *owned = buf;
    float *p = buf;
    m->ln1 = p; p += D;
    m->ln2 = p; p += D;
    m->q_norm = p; p += HD;
    m->k_norm = p; p += HD;
    m->Wq.p = p; p += (long)H*HD*D;
    m->Wk.p = p; p += (long)KV*HD*D;
    m->Wv.p = p; p += (long)KV*HD*D;
    m->Wo.p = p; p += (long)D*H*HD;
    m->Wgate.p = p; p += (long)E*D;
    m->Wq.enc = m->Wk.enc = m->Wv.enc = m->Wo.enc = m->Wgate.enc = QW3_ENC_F32;
    /* experts: contiguous in the blob, but indexed the same way the v4 path is, so both loaders
       hand qwen3_layer an identical shape and A5 compares like with like. */
    m->exp = calloc((size_t)E * 3, sizeof(QW3T));
    if (!m->exp) { free(buf); return -2; }
    m->exp_owned = 1;
    const float *wg = p; p += (long)E*F*D;
    const float *wu = p; p += (long)E*F*D;
    const float *wd = p; p += (long)E*D*F;
    for (int e = 0; e < E; e++) {
        m->exp[(size_t)e*3+0].p = wg + (size_t)e*F*D;
        m->exp[(size_t)e*3+1].p = wu + (size_t)e*F*D;
        m->exp[(size_t)e*3+2].p = wd + (size_t)e*D*F;
        m->exp[(size_t)e*3+0].enc = m->exp[(size_t)e*3+1].enc = m->exp[(size_t)e*3+2].enc
                                  = QW3_ENC_F32;
    }
    return 0;
}

void qwen3_layer_free_weights(QW3Layer *m) {
    if (m->exp_owned) free(m->exp);
    m->exp = NULL; m->exp_owned = 0;
}


/* ------------------------------------------------------------------ CPW1 v4
 * Table layout, read from format/v4_writer.py rather than guessed:
 *   header 0: magic, version, hdr_bytes, rsvd
 *          16: n_tensors, n_blocks, n_regions, n_encodings
 *          32: off_enc, off_tens, off_blk, off_reg, off_data       (all u64)
 *   encoding entry: u32 id, u8 kind, u8 scalar, u16 group, u16 block, u8 sub, u8 flags,
 *                   u8 align, string(name)      [string = u16 length + bytes]
 *   tensor:  string(name), u32 enc, u8 ndim, u32 dims[ndim], u64 off, u64 nbytes, u32 block,
 *            u16 role
 * A wrong offset here reads every weight one place off and still produces numbers, so the
 * parser is written against the writer's source, not against memory.
 */
static uint32_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t rd64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }

static int name_is(const char *s, unsigned len, const char *want) {
    size_t w = strlen(want);
    return len == w && memcmp(s, want, w) == 0;
}

int qwen3_load_v4(QW3Layer *m, const char *path, int layer) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    const unsigned char *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) { perror("mmap"); return -1; }
    m->map = base; m->map_len = (size_t)st.st_size;

    const uint32_t n_tens = rd32(base + 16), n_enc = rd32(base + 28);
    const uint64_t off_enc = rd64(base + 32), off_tens = rd64(base + 40), off_data = rd64(base + 64);

    /* encoding index -> engine encoding code */
    int enc_code[64];
    {
        const unsigned char *cur = base + off_enc;
        for (uint32_t i = 0; i < n_enc && i < 64; i++) {
            uint32_t id = rd32(cur); cur += 4;
            uint8_t kind = *cur++; uint8_t scalar = *cur++; cur += 2 /*group*/; cur += 2 /*block*/;
            cur += 3 /*sub, flags, align*/;
            uint16_t len; memcpy(&len, cur, 2); cur += 2 + len;
            (void)id;
            enc_code[i] = (kind == 0) ? QW3_ENC_F32
                        : (scalar == 3) ? QW3_ENC_Q8
                        : (scalar == 4) ? 4 : -1;
        }
    }

    /* Metadata comes from the model's own tensors, not from a config: shape is what the file has. */
    m->exp = calloc((size_t)128 * 3, sizeof(QW3T));
    if (!m->exp) return -2;
    m->exp_owned = 1;
    int n_experts = 0;
    
    const unsigned char *cur = base + off_tens;
    uint32_t cnt = rd32(cur); cur += 4;
    if (cnt != n_tens) { fprintf(stderr, "v4: tensor count %u != header %u\n", cnt, n_tens); return -3; }
    char want[64];
    for (uint32_t i = 0; i < cnt; i++) {
        uint16_t nlen; memcpy(&nlen, cur, 2); cur += 2;
        const char *nm = (const char *)cur; cur += nlen;
        uint32_t eidx = rd32(cur); cur += 4;
        uint8_t nd = *cur++; uint32_t dims[4] = {0,0,0,0};
        for (uint8_t d = 0; d < nd; d++) { dims[d] = rd32(cur); cur += 4; }
        uint64_t off = rd64(cur); cur += 8;
        uint64_t nb  = rd64(cur); cur += 8;
        cur += 4 /*block*/; cur += 2 /*role*/;
        if (eidx >= n_enc) continue;
        int enc = enc_code[eidx];
        const void *ptr = base + off_data + off;
        (void)nb;

        int l = -1, e = -1, kind = -1;
        if (nlen > 7 && memcmp(nm, "expert.", 7) == 0) {
            char buf[64];
            if (nlen < sizeof buf) {
                memcpy(buf, nm, nlen); buf[nlen] = 0;
                char ks[8];
                if (sscanf(buf, "expert.%d.%d.%3s", &l, &e, ks) == 3 && l == layer) {
                    if (!strcmp(ks, "Wg")) kind = 0;
                    else if (!strcmp(ks, "Wu")) kind = 1;
                    else if (!strcmp(ks, "Wd")) kind = 2;
                    if (kind >= 0) {
                        if (e + 1 > n_experts) n_experts = e + 1;
                        m->exp[(size_t)e * 3 + kind].p = ptr;
                        m->exp[(size_t)e * 3 + kind].enc = enc;
                    }
                }
            }
            continue;
        }
        /* per-layer tensors are named "<stem>.<layer>" */
        snprintf(want, sizeof want, ".%d", layer);
        size_t wl = strlen(want);
        if (nlen <= wl || memcmp(nm + nlen - wl, want, wl) != 0) continue;
        unsigned stem = nlen - (unsigned)wl;
        if      (name_is(nm, stem, "ln1"))    { m->ln1 = ptr; m->d_model = (int)dims[0]; }
        else if (name_is(nm, stem, "ln2"))    m->ln2 = ptr;
        else if (name_is(nm, stem, "q_norm")) { m->q_norm = ptr; m->head_dim = (int)dims[0]; }
        else if (name_is(nm, stem, "k_norm")) m->k_norm = ptr;
        else if (name_is(nm, stem, "Wgate"))  { m->Wgate.p = ptr; m->Wgate.enc = enc;
                                                m->n_experts = (int)dims[0]; }
        else if (name_is(nm, stem, "Wq"))     { m->Wq.p = ptr; m->Wq.enc = enc;
                                                m->n_heads = (int)(dims[0] / (m->head_dim ? m->head_dim : 1)); }
        else if (name_is(nm, stem, "Wk"))     { m->Wk.p = ptr; m->Wk.enc = enc;
                                                m->n_kv = (int)(dims[0] / (m->head_dim ? m->head_dim : 1)); }
        else if (name_is(nm, stem, "Wv"))     { m->Wv.p = ptr; m->Wv.enc = enc; }
        else if (name_is(nm, stem, "Wo"))     { m->Wo.p = ptr; m->Wo.enc = enc; }
    }
    if (!m->ln1 || !m->Wq.p || !m->Wgate.p || !m->exp[0].p) {
        fprintf(stderr, "v4: layer %d is incomplete\n", layer);
        return -4;
    }
    /* Geometry the file states rather than implies. */
    m->d_exp = 0;
    snprintf(want, sizeof want, "expert.%d.0.Wg", layer);
    /* d_exp comes from the expert shape; take it from the gate matrix's output dim */
    {
        const unsigned char *c2 = base + off_tens + 4;
        for (uint32_t i = 0; i < cnt; i++) {
            uint16_t nlen; memcpy(&nlen, c2, 2); c2 += 2;
            const char *nm = (const char *)c2; c2 += nlen;
            uint32_t eidx = rd32(c2); c2 += 4;
            uint8_t nd = *c2++; uint32_t dims[4] = {0,0,0,0};
            for (uint8_t d = 0; d < nd; d++) { dims[d] = rd32(c2); c2 += 4; }
            c2 += 8 + 8 + 4 + 2;
            if (eidx >= n_enc) continue;
            if (nlen == strlen(want) && memcmp(nm, want, nlen) == 0) {
                m->d_exp = (int)dims[0];
                break;
            }
        }
    }
    m->rope_theta = 1000000.0f;
    m->rms_eps = 1e-6f;
    m->topk = 8;
    m->vocab = 151936;
    if (n_experts) m->n_experts = n_experts;
    if (!m->d_exp) { fprintf(stderr, "v4: cannot determine d_exp\n"); return -5; }
        return 0;
}
