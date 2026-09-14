#include "model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* out = (x - mean) / sqrt(var + eps) * gain, over n floats (one row) */
static void layernorm(float *out, const float *x, const float *gain, int n) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * gain[i];
}

/* softmax in place over n floats, max-subtracted (same as run.c) */
static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* out[j] = sum_k x[k] * w[k*n_out + j]  — "x @ W", W stored (n_in, n_out).
   Our exercise convention (Wq, Wk, Wv, Wo, W1, W2 all use this). */
static void matmul_xw(float *out, const float *x, const float *w, int n_in, int n_out) {
    for (int j = 0; j < n_out; j++) {
        float acc = 0.0f;
        for (int k = 0; k < n_in; k++) acc += x[k] * w[k * n_out + j];
        out[j] = acc;
    }
}

/* out[i] = sum_k w[i*n_in + k] * x[k]  — "W @ x", W stored (n_out, n_in).
   llama2.c convention, contiguous rows — used for logits = xf @ unemb^T. */
static void matmul_wx(float *out, const float *x, const float *w, int n_in, int n_out) {
    for (int i = 0; i < n_out; i++) {
        float acc = 0.0f;
        for (int k = 0; k < n_in; k++) acc += w[i * n_in + k] * x[k];
        out[i] = acc;
    }
}

void forward(const Model *m, const int *tokens, int T, float *logits_out) {
    const int D = m->hdr.d_model, H = m->hdr.n_heads;
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

    /* step 0: embedding lookup */
    for (int t = 0; t < T; t++)
        memcpy(x + t * D, w->token_emb + tokens[t] * D, D * sizeof(float));

    /* attention half: norm -> qkv -> per-head masked attention -> Wo -> +res */
    for (int t = 0; t < T; t++) layernorm(xn + t * D, x + t * D, w->ln1_g, D);
    for (int t = 0; t < T; t++) {
        matmul_xw(q  + t * D, xn + t * D, w->Wq, D, D);
        matmul_xw(k  + t * D, xn + t * D, w->Wk, D, D);
        matmul_xw(vv + t * D, xn + t * D, w->Wv, D, D);
    }
    for (int hh = 0; hh < H; hh++) {
        for (int i = 0; i < T; i++) {
            float scale = 1.0f / sqrtf((float)HD);
            for (int j = 0; j <= i; j++) {      /* causal: keys up to i only */
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
        matmul_xw(proj, ao + t * D, w->Wo, D, D);
        for (int e = 0; e < D; e++) x[t * D + e] += proj[e];
    }

    /* FFN half: norm -> W1 -> relu -> W2 -> +res */
    for (int t = 0; t < T; t++) layernorm(xn + t * D, x + t * D, w->ln2_g, D);
    for (int t = 0; t < T; t++) {
        matmul_xw(h + t * F, xn + t * D, w->W1, D, F);
        for (int e = 0; e < F; e++)
            if (h[t * F + e] < 0.0f) h[t * F + e] = 0.0f;
        matmul_xw(proj, h + t * F, w->W2, F, D);
        for (int e = 0; e < D; e++) x[t * D + e] += proj[e];
    }

    /* head: final norm -> logits = xf @ unemb^T */
    for (int t = 0; t < T; t++) {
        layernorm(xn + t * D, x + t * D, w->lnf_g, D);
        matmul_wx(logits_out + t * V, xn + t * D, w->unemb, D, V);
    }

    free(x); free(xn); free(q); free(k); free(vv);
    free(ao); free(att); free(h); free(proj);
}
