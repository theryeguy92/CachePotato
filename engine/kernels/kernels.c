#include "kernels.h"
#include "parallel.h"

#include <math.h>
#include <stddef.h>

/* out = (x - mean) / sqrt(var + eps) * gain + bias, over n floats (one row) */
void layernorm(float *out, const float *x, const float *gain,
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
void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* GPT-2's GELU (tanh approximation) */
float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

/* out[j] = bias[j] + sum_k x[k] * w[k*n_out + j]  — "x @ W + b",
   W stored (n_in, n_out). Row-streaming: init with bias, then scatter
   x[k]*w_row into out — inner loop walks w CONTIGUOUSLY (vectorizes). */
/* Split over OUTPUT COLUMNS: each out[j] still accumulates over k in exactly the order
   0..n_in-1, so threading changes nothing. Below CP_MATMUL_MIN the dispatch costs more than
   the work, so small matmuls (the router: 768x32) stay serial. */
#define CP_MATMUL_MIN 60000L

typedef struct { float *out; const float *x, *w, *bias; int n_in, n_out; } XWCtx;

static void xw_slice(void *v, int jlo, int jhi) {
    XWCtx *c = v;
    for (int j = jlo; j < jhi; j++) c->out[j] = c->bias[j];
    for (int k = 0; k < c->n_in; k++) {
        float xk = c->x[k];
        const float *wrow = c->w + (size_t)k * c->n_out;
        for (int j = jlo; j < jhi; j++) c->out[j] += xk * wrow[j];
    }
}

void matmul_xw(float *out, const float *x, const float *w,
               const float *bias, int n_in, int n_out) {
    if (cp_would_parallelise((long)n_in * n_out, CP_MATMUL_MIN)) {
        XWCtx c = { out, x, w, bias, n_in, n_out };
        cp_parallel_for(n_out, xw_slice, &c);
        return;
    }
    for (int j = 0; j < n_out; j++) out[j] = bias[j];
    for (int k = 0; k < n_in; k++) {
        float xk = x[k];
        const float *wrow = w + (size_t)k * n_out;
        for (int j = 0; j < n_out; j++) out[j] += xk * wrow[j];
    }
}

/* out[i] = sum_k w[i*n_in + k] * x[k]  — "W @ x", W stored (n_out, n_in),
   contiguous rows. Used for logits = xf @ token_emb^T (tied unemb). */
/* Split over OUTPUT ROWS: each out[i] is a serial dot product, unchanged. This one matters
   a lot — the tied unembedding is n_out = vocab (50257 for gpt2-moe), i.e. the widest
   single matmul in the engine. */
typedef struct { float *out; const float *x, *w; int n_in, n_out; } WXCtx;

static void wx_slice(void *v, int ilo, int ihi) {
    WXCtx *c = v;
    for (int i = ilo; i < ihi; i++) {
        float acc = 0.0f;
        for (int k = 0; k < c->n_in; k++) acc += c->w[(size_t)i * c->n_in + k] * c->x[k];
        c->out[i] = acc;
    }
}

void matmul_wx(float *out, const float *x, const float *w, int n_in, int n_out) {
    if (cp_would_parallelise((long)n_in * n_out, CP_MATMUL_MIN)) {
        WXCtx c = { out, x, w, n_in, n_out };
        cp_parallel_for(n_out, wx_slice, &c);
        return;
    }
    for (int i = 0; i < n_out; i++) {
        float acc = 0.0f;
        for (int k = 0; k < n_in; k++) acc += w[(size_t)i * n_in + k] * x[k];
        out[i] = acc;
    }
}
