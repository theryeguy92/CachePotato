/* qwen3_moe.c — Qwen3-30B-A3B decoder layer in C.
 *
 * Ported against models/qwen3-30b-a3b/reference.py, which is itself gated to ~1e-7 against the
 * published weights. So a disagreement here is this file's fault, not the model's.
 */
#include "qwen3_moe.h"
#include "kernels/kernels.h"
#include "pager/pager.h"

#include <math.h>
#include <fcntl.h>
#include <limits.h>
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

int qwen3_kv_alloc(QW3KV *kv, const QW3Layer *m, int cap) {
    kv->k = malloc(sizeof(float) * (size_t)cap * m->n_kv * m->head_dim);
    kv->v = malloc(sizeof(float) * (size_t)cap * m->n_kv * m->head_dim);
    kv->cap = cap;
    if (!kv->k || !kv->v) { free(kv->k); free(kv->v); kv->k = kv->v = NULL; kv->cap = 0; return -1; }
    return 0;
}

void qwen3_kv_free(QW3KV *kv) {
    free(kv->k); free(kv->v);
    memset(kv, 0, sizeof *kv);
}

int qwen3_layer(const QW3Layer *m, const float *x, int T, QW3Out *o) {
    QW3KV tmp;
    if (qwen3_kv_alloc(&tmp, m, T)) return -1;
    int rc = qwen3_layer_kv(m, x, T, 0, &tmp, o);
    qwen3_kv_free(&tmp);
    return rc;
}

int qwen3_layer_kv(const QW3Layer *m, const float *x, int T, int pos0, QW3KV *cache, QW3Out *o) {
    const int D = m->d_model, H = m->n_heads, KV = m->n_kv, HD = m->head_dim;
    const int E = m->n_experts, K = m->topk, F = m->d_exp, rep = H / KV;
    if (!cache || pos0 < 0 || T <= 0 || pos0 + T > cache->cap) {
        fprintf(stderr, "qwen3: kv range %d+%d outside cache cap %d\n", pos0, T,
                cache ? cache->cap : 0);
        return -1;
    }
    float *xn = malloc(sizeof(float) * (size_t)T * D);
    float *q = malloc(sizeof(float) * (size_t)T * H * HD);
    float *ao = malloc(sizeof(float) * (size_t)T * H * HD);  /* NOT xn: H*HD is 2x D */
    /* attention can run over pos0+T positions, not just the new ones */
    float *sc = malloc(sizeof(float) * (size_t)(pos0 + T));
    float *hh = malloc(sizeof(float) * (size_t)(F > D ? F : D));
    float *gu = malloc(sizeof(float) * (size_t)(F > D ? F : D));
    float *probs = malloc(sizeof(float) * E);
    int *idx = malloc(sizeof(int) * (size_t)K);
    float *wt = malloc(sizeof(float) * (size_t)T * K);
    if (!xn || !q || !ao || !sc || !hh || !gu || !probs || !idx || !wt) return -1;

    /* ---- attention: norm -> qkv -> QK-norm -> RoPE -> GQA -> o_proj ---- */
    for (int t = 0; t < T; t++) {
        /* The pager's unit is the region, and this token is about to read six of them.
           Demand-touch BEFORE the read: charging after it would describe the previous
           state of the cache, which is the whole distinction between a hit and a miss.
           Nine core regions per layer (six here, Wo, ln2, the router) plus one per
           selected expert is 17 per layer, 816 per token across 48 layers — the
           working set the budget is measured against. */
        if (m->pager) {
            pager_step(pos0 + t);
            pager_touch_region(m->reg_ln1);
            pager_touch_region(m->reg_Wq);
            pager_touch_region(m->reg_Wk);
            pager_touch_region(m->reg_Wv);
            pager_touch_region(m->reg_qnorm);
            pager_touch_region(m->reg_knorm);
        }
        rmsnorm(xn + (size_t)t * D, x + (size_t)t * D, m->ln1, D, m->rms_eps);
        mm(&m->Wq, q + (size_t)t * H * HD,  xn + (size_t)t * D, D, H * HD);
        mm(&m->Wk, cache->k + (size_t)(pos0 + t) * KV * HD, xn + (size_t)t * D, D, KV * HD);
        mm(&m->Wv, cache->v + (size_t)(pos0 + t) * KV * HD, xn + (size_t)t * D, D, KV * HD);
        head_rmsnorm(q + (size_t)t * H * HD,  H,  HD, m->q_norm, m->rms_eps);
        head_rmsnorm(cache->k + (size_t)(pos0 + t) * KV * HD, KV, HD, m->k_norm, m->rms_eps);
    }
    /* the new rows start at pos0, so token n is rotated at position n */
    rope_apply_from(q, cache->k + (size_t)pos0 * KV * HD, T, pos0, H, KV, HD, m->rope_theta);

    for (int h = 0; h < H; h++) {
        const int kh = h / rep;
        for (int i = 0; i < T; i++) {
            const float *qi = q + ((size_t)i * H + h) * HD;
            const int last = pos0 + i;          /* causal: every position up to here */
            for (int j = 0; j <= last; j++) {
                const float *kj = cache->k + ((size_t)j * KV + kh) * HD;
                float s = 0.0f;
                for (int e = 0; e < HD; e++) s += qi[e] * kj[e];
                sc[j] = s / sqrtf((float)HD);
            }
            softmax(sc, last + 1);
            for (int e = 0; e < HD; e++) {
                float acc = 0.0f;
                for (int j = 0; j <= last; j++)
                    acc += sc[j] * cache->v[((size_t)j * KV + kh) * HD + e];
                ao[(size_t)i * H * HD + (size_t)h * HD + e] = acc;
            }
        }
    }
    for (int t = 0; t < T; t++) {
        if (m->pager) { pager_step(pos0 + t); pager_touch_region(m->reg_Wo); }
        mm(&m->Wo, o->attn + (size_t)t * D, ao + (size_t)t * H * HD, H * HD, D);
    }

    /* ---- MoE: router -> softmax -> top-k -> renormalise -> gated experts ---- */
    for (int t = 0; t < T; t++) {
        if (m->pager) {
            pager_step(pos0 + t);
            pager_touch_region(m->reg_ln2);
            pager_touch_region(m->reg_Wgate);
        }
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
            /* One region per (layer, expert), declared by the file. This is the touch
               that makes the experiment: 8 of 128 per layer, chosen by the router, and
               nothing else in the layer depends on which ones. */
            if (m->pager) pager_touch_region(m->reg_exp[e]);
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
    free(xn); free(q); free(ao); free(sc); free(hh); free(gu); free(probs); free(idx); free(wt);
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

int qwen3_map(QW3Layer *m, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    const unsigned char *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); close(fd); return -1; }
    m->map = base; m->map_len = (size_t)st.st_size;

    m->n_tens = rd32(base + 16); m->n_regions = (int)rd32(base + 24);
    m->n_enc = rd32(base + 28);
    const uint64_t off_enc = rd64(base + 32);
    m->off_tens = rd64(base + 40);
    const uint64_t off_reg = rd64(base + 56);
    m->off_data = rd64(base + 64);

    /* encoding index -> engine encoding code */
    const unsigned char *cur = base + off_enc;
    for (uint32_t i = 0; i < m->n_enc && i < 64; i++) {
        cur += 4;                                  /* id */
        uint8_t kind = *cur++; uint8_t scalar = *cur++;
        cur += 2; cur += 2;                        /* group, block bytes */
        cur += 3;                                  /* sub, flags, align */
        uint16_t len; memcpy(&len, cur, 2); cur += 2 + len;
        m->enc_code[i] = (kind == 0) ? QW3_ENC_F32
                       : (scalar == 3) ? QW3_ENC_Q8
                       : (scalar == 4) ? 4 : -1;
    }
    m->exp = calloc((size_t)128 * 3, sizeof(QW3T));
    m->reg_exp = calloc((size_t)128, sizeof(int));
    m->tensor_reg = calloc(m->n_tens ? m->n_tens : 1, sizeof(int));
    if (!m->exp || !m->reg_exp || !m->tensor_reg) { close(fd); return -2; }
    m->exp_owned = 1;
    for (uint32_t i = 0; i < m->n_tens; i++) m->tensor_reg[i] = -1;
    m->rope_theta = 1000000.0f;
    m->rms_eps = 1e-6f;
    m->topk = 8;
    m->vocab = 151936;

    /* ---- region table: the file's own declaration of what is pageable (SPEC §8).
       Region ids handed to the pager ARE this table's indices, so "region 1234" means
       the same thing to the file, the loader and the report — and a touch can never
       describe bytes the file did not declare pageable.
       Digests are NOT checked here. tools/verify_v4_file.py owns that claim, and doing
       it at load would read all 32.5 GB on every run, which would both destroy the
       measurement (every region starts resident) and make a model that does not fit in
       RAM impossible to load on the machine the pager exists for. */
    {
        const char *pg = getenv("CP_PAGER"), *pb = getenv("CP_BUDGET");
        /* Off unless asked for: the unpaged run IS the control (B5), and a pager that
           attached itself would change the memory behaviour of every existing gate. */
        if ((pg && pg[0] == '1') || pb) {
            long budget = LONG_MAX;                 /* active, counts, never evicts */
            if (pb) budget = atol(pb) * 1024L * 1024L;
            if (pager_open_v4(fd, base, (size_t)m->off_data, budget)) {
                fprintf(stderr, "v4: pager open failed\n");
                close(fd);
                return -6;
            }
            m->pager = 1;
        }
    }
    {
        const unsigned char *reg = base + off_reg;
        uint32_t nreg = rd32(reg); reg += 4;
        if (nreg != (uint32_t)m->n_regions) {
            fprintf(stderr, "v4: region table count %u != header %d\n", nreg, m->n_regions);
            close(fd);
            return -3;
        }
        for (uint32_t i = 0; i < nreg; i++) {
            uint64_t off = rd64(reg); reg += 8;
            uint64_t len = rd64(reg); reg += 8;
            uint32_t lo = rd32(reg); reg += 4;
            uint32_t hi = rd32(reg); reg += 4;
            const uint8_t kind = *reg++;
            reg += 2 + 4 + 4 + 8;               /* slot, member_lo/hi, digest */
            if (m->pager && pager_add_region((size_t)off, (size_t)len, kind == 1) != (int)i) {
                fprintf(stderr, "v4: pager region id != table index %u\n", i);
                close(fd);
                return -6;
            }
            for (uint32_t t = lo; t < hi && t < m->n_tens; t++)
                m->tensor_reg[t] = (int)i;
        }
    }
    close(fd);
    return 0;
}

/* Walk the tensor table once, resolving what this layer needs and the model-wide tensors. */
int qwen3_bind_layer(QW3Layer *m, int layer) {
    const unsigned char *base = m->map;
    const unsigned char *cur = base + m->off_tens;
    uint32_t cnt = rd32(cur); cur += 4;
    if (cnt != m->n_tens) { fprintf(stderr, "v4: tensor count %u != header %u\n", cnt, m->n_tens); return -3; }
    char want[64];
    m->ln1 = m->ln2 = NULL; m->q_norm = m->k_norm = NULL; m->final_norm = NULL;
    m->Wq.p = m->Wk.p = m->Wv.p = m->Wo.p = m->Wgate.p = NULL;
    m->embed.p = m->lm_head.p = NULL;
    /* region ids are per layer, so they are cleared with the pointers */
    m->reg_ln1 = m->reg_ln2 = m->reg_qnorm = m->reg_knorm = -1;
    m->reg_Wq = m->reg_Wk = m->reg_Wv = m->reg_Wo = m->reg_Wgate = -1;
    m->reg_embed = m->reg_lm_head = m->reg_final_norm = -1;
    if (m->reg_exp) for (int e = 0; e < 128; e++) m->reg_exp[e] = -1;
    for (uint32_t i = 0; i < cnt; i++) {
        uint16_t nlen; memcpy(&nlen, cur, 2); cur += 2;
        const char *nm = (const char *)cur; cur += nlen;
        uint32_t eidx = rd32(cur); cur += 4;
        uint8_t nd = *cur++; uint32_t dims[4] = {0,0,0,0};
        for (uint8_t d = 0; d < nd; d++) { dims[d] = rd32(cur); cur += 4; }
        uint64_t off = rd64(cur); cur += 8;
        cur += 8;                                  /* nbytes */
        cur += 4; cur += 2;                        /* block, role */
        if (eidx >= m->n_enc) continue;
        const int enc = m->enc_code[eidx];
        const void *ptr = base + m->off_data + off;
        const int reg = m->tensor_reg ? m->tensor_reg[i] : -1;

        if (name_is(nm, nlen, "embed"))     { m->embed.p = ptr; m->embed.enc = enc;
                                              m->reg_embed = reg; continue; }
        if (name_is(nm, nlen, "lm_head"))   { m->lm_head.p = ptr; m->lm_head.enc = enc;
                                              m->reg_lm_head = reg; continue; }
        if (name_is(nm, nlen, "final_norm")){ m->final_norm = (const float *)ptr;
                                              m->d_model = (int)dims[0];
                                              m->reg_final_norm = reg; continue; }

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
                        m->exp[(size_t)e * 3 + kind].p = ptr;
                        m->exp[(size_t)e * 3 + kind].enc = enc;
                        /* An expert's three matrices share ONE region (the file
                           declares a region per expert, SPEC §8), so the id is the
                           same for Wg/Wu/Wd and only the expert index matters. */
                        if (e >= 0 && e < 128) m->reg_exp[e] = reg;
                    }
                }
            }
            continue;
        }
        snprintf(want, sizeof want, ".%d", layer);
        size_t wl = strlen(want);
        if (nlen <= wl || memcmp(nm + nlen - wl, want, wl) != 0) continue;
        const unsigned stem = nlen - (unsigned)wl;
        if      (name_is(nm, stem, "ln1"))    { m->ln1 = (const float *)ptr; m->d_model = (int)dims[0];
                                                m->reg_ln1 = reg; }
        else if (name_is(nm, stem, "ln2"))    { m->ln2 = (const float *)ptr; m->reg_ln2 = reg; }
        else if (name_is(nm, stem, "q_norm")) { m->q_norm = (const float *)ptr; m->head_dim = (int)dims[0];
                                                m->reg_qnorm = reg; }
        else if (name_is(nm, stem, "k_norm")) { m->k_norm = (const float *)ptr; m->reg_knorm = reg; }
        else if (name_is(nm, stem, "Wgate"))  { m->Wgate.p = ptr; m->Wgate.enc = enc;
                                                m->n_experts = (int)dims[0]; m->reg_Wgate = reg; }
        else if (name_is(nm, stem, "Wq"))     { m->Wq.p = ptr; m->Wq.enc = enc;
                                                if (m->head_dim) m->n_heads = (int)(dims[0] / m->head_dim);
                                                m->reg_Wq = reg; }
        else if (name_is(nm, stem, "Wk"))     { m->Wk.p = ptr; m->Wk.enc = enc;
                                                if (m->head_dim) m->n_kv = (int)(dims[0] / m->head_dim);
                                                m->reg_Wk = reg; }
        else if (name_is(nm, stem, "Wv"))     { m->Wv.p = ptr; m->Wv.enc = enc; m->reg_Wv = reg; }
        else if (name_is(nm, stem, "Wo"))     { m->Wo.p = ptr; m->Wo.enc = enc; m->reg_Wo = reg; }
        else if (m->d_exp == 0 && nlen > 12 && memcmp(nm, "expert.", 7) == 0) { /* handled above */ }
    }
    if (!m->ln1 || !m->Wq.p || !m->Wgate.p || !m->exp[0].p) {
        fprintf(stderr, "v4: layer %d is incomplete\n", layer);
        return -4;
    }
    /* d_exp from an expert's gate matrix: (d_exp, d_model) — the OUTPUT dim is d_exp. */
    {
        const unsigned char *c2 = base + m->off_tens + 4;
        snprintf(want, sizeof want, "expert.%d.0.Wg", layer);
        for (uint32_t i = 0; i < cnt; i++) {
            uint16_t nlen; memcpy(&nlen, c2, 2); c2 += 2;
            const char *nm = (const char *)c2; c2 += nlen;
            uint32_t eidx = rd32(c2); c2 += 4;
            uint8_t nd = *c2++; uint32_t dims[4] = {0,0,0,0};
            for (uint8_t d = 0; d < nd; d++) { dims[d] = rd32(c2); c2 += 4; }
            c2 += 8 + 8 + 4 + 2;
            if (eidx >= m->n_enc) continue;
            if (nlen == strlen(want) && memcmp(nm, want, nlen) == 0) { m->d_exp = (int)dims[0]; break; }
        }
    }
    if (!m->d_exp) { fprintf(stderr, "v4: cannot determine d_exp for layer %d\n", layer); return -5; }
    return 0;
}

int qwen3_load_v4(QW3Layer *m, const char *path, int layer) {
    int rc = qwen3_map(m, path);
    return rc ? rc : qwen3_bind_layer(m, layer);
}
