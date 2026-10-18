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

    for (int t = 0; t < T; t++)
        memcpy(x + (size_t)t * D, m->embed + (size_t)tokens[t] * D, sizeof(float) * D);

    for (int l = 0; l < L; l++) {
        /* ---- attention ---- */
        for (int t = 0; t < T; t++)
            rmsnorm(xn + (size_t)t * D, x + (size_t)t * D, m->ln1 + (size_t)l * D, D, m->rms_eps);
        for (int t = 0; t < T; t++) {
            matmul_wx(q + (size_t)t * H * HD,  xn + (size_t)t * D, m->Wq + (size_t)l * D * D, D, H * HD);
            matmul_wx(k + (size_t)t * KV * HD, xn + (size_t)t * D, m->Wk + (size_t)l * KV * HD * D, D, KV * HD);
            matmul_wx(vv + (size_t)t * KV * HD, xn + (size_t)t * D, m->Wv + (size_t)l * KV * HD * D, D, KV * HD);
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
            matmul_wx(hh, ao + (size_t)t * D, m->Wo + (size_t)l * D * D, H * HD, D);
            for (int j = 0; j < D; j++) x[(size_t)t * D + j] += hh[j];
        }

        /* ---- SwiGLU FFN ---- */
        for (int t = 0; t < T; t++) {
            float *xt = x + (size_t)t * D;
            rmsnorm(xn + (size_t)t * D, xt, m->ln2 + (size_t)l * D, D, m->rms_eps);
            matmul_wx(gg + (size_t)t * F, xn + (size_t)t * D, m->Wg + (size_t)l * F * D, D, F);
            matmul_wx(hh + (size_t)t * F, xn + (size_t)t * D, m->Wu + (size_t)l * F * D, D, F);
            for (int j = 0; j < F; j++)
                gg[(size_t)t * F + j] = silu(gg[(size_t)t * F + j]) * hh[(size_t)t * F + j];
            matmul_wx(hh, gg + (size_t)t * F, m->Wd + (size_t)l * D * F, F, D);
            for (int j = 0; j < D; j++) xt[j] += hh[j];
        }
    }

    /* final norm + tied unembedding */
    for (int t = 0; t < T; t++) {
        rmsnorm(xn + (size_t)t * D, x + (size_t)t * D, m->norm, D, m->rms_eps);
        matmul_wx(logits + (size_t)t * V, xn + (size_t)t * D, m->embed, D, V);
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
