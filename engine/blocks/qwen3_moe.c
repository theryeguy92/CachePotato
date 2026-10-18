/* qwen3_moe.c — Qwen3-30B-A3B decoder layer in C.
 *
 * Ported against models/qwen3-30b-a3b/reference.py, which is itself gated to ~1e-7 against the
 * published weights. So a disagreement here is this file's fault, not the model's.
 */
#include "qwen3_moe.h"
#include "kernels/kernels.h"

#include <math.h>
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
        matmul_wx(q + (size_t)t * H * HD,  xn + (size_t)t * D, m->Wq, D, H * HD);
        matmul_wx(k + (size_t)t * KV * HD, xn + (size_t)t * D, m->Wk, D, KV * HD);
        matmul_wx(v + (size_t)t * KV * HD, xn + (size_t)t * D, m->Wv, D, KV * HD);
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
        matmul_wx(o->attn + (size_t)t * D, ao + (size_t)t * H * HD, m->Wo, H * HD, D);

    /* ---- MoE: router -> softmax -> top-k -> renormalise -> gated experts ---- */
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < D; j++) hh[j] = x[(size_t)t * D + j] + o->attn[(size_t)t * D + j];
        rmsnorm(xn + (size_t)t * D, hh, m->ln2, D, m->rms_eps);
        matmul_wx(o->router + (size_t)t * E, xn + (size_t)t * D, m->Wgate, D, E);
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
            const float *wg = m->Wg + (size_t)e * F * D;
            const float *wu = m->Wu + (size_t)e * F * D;
            const float *wd = m->Wd + (size_t)e * D * F;
            matmul_wx(gu, xn + (size_t)t * D, wg, D, F);       /* gate */
            matmul_wx(hh, xn + (size_t)t * D, wu, D, F);       /* up   */
            for (int i = 0; i < F; i++) gu[i] = silu(gu[i]) * hh[i];
            matmul_wx(hh, gu, wd, F, D);                       /* down */
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
    /* ln1, ln2 (2*D) and q_norm, k_norm (2*HD) — the first version counted 3*D and silently
       mis-sized the buffer, which would have read one tensor shifted into the next. */
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
    m->Wq = p; p += (long)H*HD*D;
    m->Wk = p; p += (long)KV*HD*D;
    m->Wv = p; p += (long)KV*HD*D;
    m->Wo = p; p += (long)D*H*HD;
    m->Wgate = p; p += (long)E*D;
    m->Wg = p; p += (long)E*F*D;
    m->Wu = p; p += (long)E*F*D;
    m->Wd = p; p += (long)E*D*F;
    return 0;
}
