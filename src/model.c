#include "model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* out = (x - mean) / sqrt(var + eps) * gain + bias, over n floats (one row) */
static void layernorm(float *out, const float *x, const float *gain,
                      const float *bias, int n) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * gain[i] + bias[i];
}

/* softmax in place over n floats, max-subtracted (same as run.c) */
static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* GPT-2's GELU (tanh approximation) */
static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

/* out[j] = bias[j] + sum_k x[k] * w[k*n_out + j]  — "x @ W + b",
   W stored (n_in, n_out). All GPT-2 projections use this layout
   (HF Conv1D stores (in, out) — the exporter writes it as-is). */
static void matmul_xw(float *out, const float *x, const float *w,
                      const float *bias, int n_in, int n_out) {
    for (int j = 0; j < n_out; j++) {
        float acc = bias[j];
        for (int k = 0; k < n_in; k++) acc += x[k] * w[k * n_out + j];
        out[j] = acc;
    }
}

/* out[i] = sum_k w[i*n_in + k] * x[k]  — "W @ x", W stored (n_out, n_in),
   contiguous rows. Used for logits = xf @ token_emb^T (tied unemb). */
static void matmul_wx(float *out, const float *x, const float *w, int n_in, int n_out) {
    for (int i = 0; i < n_out; i++) {
        float acc = 0.0f;
        for (int k = 0; k < n_in; k++) acc += w[i * n_in + k] * x[k];
        out[i] = acc;
    }
}

void forward(const Model *m, const int *tokens, int T, float *logits_out) {
    const int D = m->hdr.d_model, H = m->hdr.n_heads, L = m->hdr.n_layer;
    const int F = m->hdr.d_ff, V = m->hdr.vocab;
    const int HD = D / H;
    const Weights *w = &m->w;

    float *x   = malloc(T * D * sizeof(float)); /* residual stream */
    float *xn  = malloc(T * D * sizeof(float)); /* normed copy */
    float *q   = malloc(T * D * sizeof(float));
    float *k   = malloc(T * D * sizeof(float));
    float *vv  = malloc(T * D * sizeof(float));
    float *ao  = malloc(T * D * sizeof(float)); /* attention out, heads concat'd */
    float *att = malloc(T * sizeof(float));     /* one query's scores at a time */
    float *h   = malloc(T * F * sizeof(float)); /* FFN hidden */
    float *proj = malloc(D * sizeof(float));    /* residual scratch */

    /* step 0: token embedding + learned position embedding */
    for (int t = 0; t < T; t++)
        for (int e = 0; e < D; e++)
            x[t * D + e] = w->token_emb[tokens[t] * D + e] + w->pos_emb[t * D + e];

    for (int l = 0; l < L; l++) {
        const float *ln1_g = w->ln1_g + l * D, *ln1_b = w->ln1_b + l * D;
        const float *Wq = w->Wq + l * D * D, *bq = w->bq + l * D;
        const float *Wk = w->Wk + l * D * D, *bk = w->bk + l * D;
        const float *Wv = w->Wv + l * D * D, *bv = w->bv + l * D;
        const float *Wo = w->Wo + l * D * D, *bo = w->bo + l * D;
        const float *ln2_g = w->ln2_g + l * D, *ln2_b = w->ln2_b + l * D;
        const float *W1 = w->W1 + l * D * F, *b1 = w->b1 + l * F;
        const float *W2 = w->W2 + l * F * D, *b2 = w->b2 + l * D;

        /* attention half: norm -> qkv -> per-head masked attention -> Wo -> +res */
        for (int t = 0; t < T; t++) layernorm(xn + t * D, x + t * D, ln1_g, ln1_b, D);
        for (int t = 0; t < T; t++) {
            matmul_xw(q  + t * D, xn + t * D, Wq, bq, D, D);
            matmul_xw(k  + t * D, xn + t * D, Wk, bk, D, D);
            matmul_xw(vv + t * D, xn + t * D, Wv, bv, D, D);
        }
        for (int hh = 0; hh < H; hh++) {
            for (int i = 0; i < T; i++) {
                float scale = 1.0f / sqrtf((float)HD);
                for (int j = 0; j <= i; j++) {  /* causal: keys up to i only */
                    float s = 0.0f;
                    for (int e = 0; e < HD; e++)
                        s += q[i * D + hh * HD + e] * k[j * D + hh * HD + e];
                    att[j] = s * scale;
                }
                softmax(att, i + 1);
                for (int e = 0; e < HD; e++) {
                    float acc = 0.0f;
                    for (int j = 0; j <= i; j++)
                        acc += att[j] * vv[j * D + hh * HD + e];
                    ao[i * D + hh * HD + e] = acc;
                }
            }
        }
        for (int t = 0; t < T; t++) {
            matmul_xw(proj, ao + t * D, Wo, bo, D, D);
            for (int e = 0; e < D; e++) x[t * D + e] += proj[e];
        }

        /* FFN half: norm -> W1 -> gelu -> W2 -> +res */
        for (int t = 0; t < T; t++) layernorm(xn + t * D, x + t * D, ln2_g, ln2_b, D);
        for (int t = 0; t < T; t++) {
            matmul_xw(h + t * F, xn + t * D, W1, b1, D, F);
            for (int e = 0; e < F; e++) h[t * F + e] = gelu(h[t * F + e]);
            matmul_xw(proj, h + t * F, W2, b2, F, D);
            for (int e = 0; e < D; e++) x[t * D + e] += proj[e];
        }
    }

    /* head: final norm -> logits = xf @ token_emb^T (tied unembedding) */
    for (int t = 0; t < T; t++) {
        layernorm(xn + t * D, x + t * D, w->lnf_g, w->lnf_b, D);
        matmul_wx(logits_out + t * V, xn + t * D, w->token_emb, D, V);
    }

    free(x); free(xn); free(q); free(k); free(vv);
    free(ao); free(att); free(h); free(proj);
}
