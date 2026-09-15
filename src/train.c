/* train.c — C trainer for the dense model (Phase 5).
 *
 *   train --gradcheck   finite-difference gradient gate on the toy config.
 *                       NOTHING trains unless this passes.
 *   train [steps]       dense TinyStories sanity run (AdamW), then save a
 *                       CPW1 checkpoint the REPL can load.
 *
 * Backward mirrors the engine's forward exactly (same kernels).
 * Layer structure: attention couples positions, so backward runs per-layer
 * over ALL positions at once; FFN/norms are per-position.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "kernels.h"
#include "weights.h"

/* ---------------- config ---------------- */

static Header HDR;
static long N_PARAMS;

#define T_STEPS   300
#define BATCH     4
#define SEQ_LEN   64
#define LR        1e-3f
#define GRADCHECK_EPS 1e-3f

static void set_toy_config(void)   { HDR = (Header){0, 0, 8, 4, 2, 8, 1, 8}; }
static void set_dense_config(void) { HDR = (Header){0, 0, 50257, 128, 4, 512, 4, 256}; }

/* ---------------- params & grads ---------------- */

static float *params, *grads, *opt_m, *opt_v;
static Weights W, G;

static void model_alloc(void) {
    N_PARAMS = expected_floats(&HDR);
    params = malloc(N_PARAMS * sizeof(float));
    grads = calloc(N_PARAMS, sizeof(float));
    opt_m = calloc(N_PARAMS, sizeof(float));
    opt_v = calloc(N_PARAMS, sizeof(float));
    layout_weights(&W, &HDR, params);
    layout_weights(&G, &HDR, grads);
}

static float frand(void) { return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f; }

static void model_init(void) {
    int D = HDR.d_model, L = HDR.n_layer;
    long F = HDR.d_ff, V = HDR.vocab, C = HDR.n_ctx;
    for (long i = 0; i < V * D; i++) W.token_emb[i] = frand() * 0.02f;
    for (long i = 0; i < C * D; i++) W.pos_emb[i] = frand() * 0.02f;
    for (long i = 0; i < L * D * D; i++) {
        W.Wq[i] = frand() * 0.02f; W.Wk[i] = frand() * 0.02f;
        W.Wv[i] = frand() * 0.02f; W.Wo[i] = frand() * 0.02f;
    }
    for (long i = 0; i < L * D * F; i++) W.W1[i] = frand() * 0.02f;
    for (long i = 0; i < L * F * D; i++) W.W2[i] = frand() * 0.02f;
    for (long i = 0; i < L * D; i++) {
        W.ln1_g[i] = 1.0f; W.ln2_g[i] = 1.0f;
        W.ln1_b[i] = W.ln2_b[i] = W.bq[i] = W.bk[i] = W.bv[i] = W.bo[i] = W.b2[i] = 0.0f;
    }
    for (long i = 0; i < L * F; i++) W.b1[i] = 0.0f;
    for (int i = 0; i < D; i++) { W.lnf_g[i] = 1.0f; W.lnf_b[i] = 0.0f; }
}

/* ---------------- activations & grad workspaces ---------------- */

static float *x_in, *xmid, *xn1, *q, *kk, *vv, *ao, *xn2, *hpre, *hact, *xf, *logits, *att, *scratchF, *zeros;
static float *dx_next, *dx_cur, *dq, *dk, *dv, *dao, *dxn, *dh, *dlog, *rowbuf;

static inline float *arow(float *base, int l, int t, int dim) {
    return base + ((size_t)l * SEQ_LEN + t) * dim;
}
static inline float *trow(float *base, int t, int dim) {
    return base + (size_t)t * dim;
}

static void buffers_alloc(void) {
    int D = HDR.d_model, L = HDR.n_layer, H = HDR.n_heads;
    long F = HDR.d_ff, V = HDR.vocab;
    size_t td = (size_t)L * SEQ_LEN * D, tf = (size_t)L * SEQ_LEN * F;
    x_in = malloc((size_t)(L + 1) * SEQ_LEN * D * sizeof(float));
    xmid = malloc(td * sizeof(float));
    xn1 = malloc(td * sizeof(float));
    q = malloc(td * sizeof(float));
    kk = malloc(td * sizeof(float));
    vv = malloc(td * sizeof(float));
    ao = malloc(td * sizeof(float));
    xn2 = malloc(td * sizeof(float));
    hpre = malloc(tf * sizeof(float));
    hact = malloc(tf * sizeof(float));
    xf = malloc(SEQ_LEN * D * sizeof(float));
    logits = malloc((size_t)SEQ_LEN * V * sizeof(float));
    att = malloc((size_t)L * H * SEQ_LEN * SEQ_LEN * sizeof(float));
    scratchF = malloc(F * sizeof(float));
    zeros = calloc(V > F ? V : F, sizeof(float));

    dx_next = malloc(SEQ_LEN * D * sizeof(float));
    dx_cur = malloc(SEQ_LEN * D * sizeof(float));
    dq = malloc(td * sizeof(float));
    dk = malloc(td * sizeof(float));
    dv = malloc(td * sizeof(float));
    dao = malloc(td * sizeof(float));
    dxn = malloc(td * sizeof(float));
    dh = malloc(tf * sizeof(float));
    dlog = malloc(V * sizeof(float));
    rowbuf = malloc(SEQ_LEN * D * sizeof(float));
}

/* ---------------- forward (stores activations) ---------------- */

static float forward_loss(const int *tokens, const int *targets, int T) {
    const int D = HDR.d_model, H = HDR.n_heads, L = HDR.n_layer;
    const long F = HDR.d_ff, V = HDR.vocab;
    const int HD = D / H;

    for (int t = 0; t < T; t++)
        for (int e = 0; e < D; e++)
            arow(x_in, 0, t, D)[e] = W.token_emb[tokens[t] * D + e] + W.pos_emb[t * D + e];

    for (int l = 0; l < L; l++) {
        const float *ln1_g = W.ln1_g + l * D, *ln1_b = W.ln1_b + l * D;
        const float *Wq = W.Wq + l * D * D, *bq = W.bq + l * D;
        const float *Wk = W.Wk + l * D * D, *bk = W.bk + l * D;
        const float *Wv = W.Wv + l * D * D, *bv = W.bv + l * D;
        const float *Wo = W.Wo + l * D * D, *bo = W.bo + l * D;
        const float *ln2_g = W.ln2_g + l * D, *ln2_b = W.ln2_b + l * D;
        const float *W1 = W.W1 + (size_t)l * D * F, *b1 = W.b1 + l * F;
        const float *W2 = W.W2 + (size_t)l * F * D, *b2 = W.b2 + l * D;
        float *attl = att + (size_t)l * H * SEQ_LEN * SEQ_LEN;

        for (int t = 0; t < T; t++) {
            layernorm(arow(xn1, l, t, D), arow(x_in, l, t, D), ln1_g, ln1_b, D);
            matmul_xw(arow(q, l, t, D), arow(xn1, l, t, D), Wq, bq, D, D);
            matmul_xw(arow(kk, l, t, D), arow(xn1, l, t, D), Wk, bk, D, D);
            matmul_xw(arow(vv, l, t, D), arow(xn1, l, t, D), Wv, bv, D, D);
        }
        for (int hh = 0; hh < H; hh++) {
            float *atth = attl + (size_t)hh * SEQ_LEN * SEQ_LEN;
            float scale = 1.0f / sqrtf((float)HD);
            for (int i = 0; i < T; i++) {
                const float *qh = arow(q, l, i, D) + hh * HD;
                for (int j = 0; j <= i; j++) {
                    float s = 0.0f;
                    const float *kj = arow(kk, l, j, D) + hh * HD;
                    for (int e = 0; e < HD; e++) s += qh[e] * kj[e];
                    atth[i * SEQ_LEN + j] = s * scale;
                }
                softmax(atth + i * SEQ_LEN, i + 1);
                float *aoi = arow(ao, l, i, D) + hh * HD;
                for (int e = 0; e < HD; e++) {
                    float acc = 0.0f;
                    for (int j = 0; j <= i; j++)
                        acc += atth[i * SEQ_LEN + j] * arow(vv, l, j, D)[hh * HD + e];
                    aoi[e] = acc;
                }
            }
        }
        for (int t = 0; t < T; t++) {
            matmul_xw(scratchF, arow(ao, l, t, D), Wo, bo, D, D);
            for (int e = 0; e < D; e++)
                arow(xmid, l, t, D)[e] = arow(x_in, l, t, D)[e] + scratchF[e];
            layernorm(arow(xn2, l, t, D), arow(xmid, l, t, D), ln2_g, ln2_b, D);
            matmul_xw(arow(hpre, l, t, F), arow(xn2, l, t, D), W1, b1, D, F);
            for (int e = 0; e < F; e++)
                arow(hact, l, t, F)[e] = gelu(arow(hpre, l, t, F)[e]);
            matmul_xw(scratchF, arow(hact, l, t, F), W2, b2, F, D);
            for (int e = 0; e < D; e++)
                arow(x_in, l + 1, t, D)[e] = arow(xmid, l, t, D)[e] + scratchF[e];
        }
    }

    float loss = 0.0f;
    for (int t = 0; t < T; t++) {
        layernorm(trow(xf, t, D), arow(x_in, L, t, D), W.lnf_g, W.lnf_b, D);
        matmul_wx(logits + (size_t)t * V, trow(xf, t, D), W.token_emb, D, V);
        softmax(logits + (size_t)t * V, (int)V);
        float p = logits[(size_t)t * V + targets[t]];
        loss += -logf(p > 1e-30f ? p : 1e-30f);
    }
    return loss / T;
}

/* ---------------- backward helpers ---------------- */

/* dx = LN'(dout); accumulates dgain/dbias. Needs the ORIGINAL pre-norm x. */
static void layernorm_bwd(float *dx, const float *dout, const float *x,
                          const float *gain, float *dgain, float *dbias, int n) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    float s1 = 0.0f, s2 = 0.0f;
    for (int i = 0; i < n; i++) {
        float xhat = (x[i] - mean) * inv;
        s1 += dout[i] * gain[i];
        s2 += dout[i] * gain[i] * xhat;
        dgain[i] += dout[i] * xhat;
        dbias[i] += dout[i];
    }
    for (int i = 0; i < n; i++) {
        float xhat = (x[i] - mean) * inv;
        dx[i] = inv / n * (n * dout[i] * gain[i] - s1 - xhat * s2);
    }
}

static float gelu_grad(float x) {
    float c = 0.7978845608028654f;
    float u = c * (x + 0.044715f * x * x * x);
    float t = tanhf(u);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * 0.044715f * x * x);
}

/* softmax backward over a causal row (j <= i): ds_j = att_j*(d_j - sum_k d_k att_k) */
static void softmax_bwd(float *ds, const float *dout, const float *s, int n) {
    float dot = 0.0f;
    for (int j = 0; j < n; j++) dot += dout[j] * s[j];
    for (int j = 0; j < n; j++) ds[j] = s[j] * (dout[j] - dot);
}

/* ---------------- backward (per-layer, all positions) ---------------- */

static void backward(const int *tokens, const int *targets, int T) {
    const int D = HDR.d_model, H = HDR.n_heads, L = HDR.n_layer;
    const long F = HDR.d_ff, V = HDR.vocab;
    const int HD = D / H;
    const float scale = 1.0f / sqrtf((float)HD);

    /* ---- output stage: dlogits, unemb (tied) grad, final-LN grad ---- */
    for (int t = 0; t < T; t++) {
        const float *probs = logits + (size_t)t * V;
        const float *xf_t = trow(xf, t, D);
        for (long j = 0; j < V; j++) dlog[j] = probs[j] / T;
        dlog[targets[t]] -= 1.0f / T;
        /* dtoken_emb (unemb role) += dlog outer xf */
        for (long j = 0; j < V; j++) {
            float dl = dlog[j];
            if (dl != 0.0f) {
                float *ge = G.token_emb + j * D;
                for (int e = 0; e < D; e++) ge[e] += dl * xf_t[e];
            }
        }
        /* dxf = dlog @ token_emb */
        matmul_xw(rowbuf + (size_t)t * D, dlog, W.token_emb, zeros, (int)V, D);
        /* grad on stream input to final norm */
        layernorm_bwd(trow(dx_next, t, D), rowbuf + (size_t)t * D,
                      arow(x_in, L, t, D), W.lnf_g, G.lnf_g, G.lnf_b, D);
    }

    /* ---- layers, top to bottom ---- */
    for (int l = L - 1; l >= 0; l--) {
        const float *Wq = W.Wq + l * D * D, *Wk = W.Wk + l * D * D;
        const float *Wv = W.Wv + l * D * D, *Wo = W.Wo + l * D * D;
        const float *ln1_g = W.ln1_g + l * D, *ln2_g = W.ln2_g + l * D;
        const float *W1 = W.W1 + (size_t)l * D * F, *W2 = W.W2 + (size_t)l * F * D;
        float *gWq = G.Wq + l * D * D, *gWk = G.Wk + l * D * D;
        float *gWv = G.Wv + l * D * D, *gWo = G.Wo + l * D * D;
        float *gbq = G.bq + l * D, *gbk = G.bk + l * D, *gbv = G.bv + l * D, *gbo = G.bo + l * D;
        float *gln1_g = G.ln1_g + l * D, *gln1_b = G.ln1_b + l * D;
        float *gln2_g = G.ln2_g + l * D, *gln2_b = G.ln2_b + l * D;
        float *gW1 = G.W1 + (size_t)l * D * F, *gW2 = G.W2 + (size_t)l * F * D;
        float *gb1 = G.b1 + l * F, *gb2 = G.b2 + l * D;
        float *attl = att + (size_t)l * H * SEQ_LEN * SEQ_LEN;

        /* ===== FFN backward (per position; x_in[l+1] = xmid + proj2) ===== */
        float *dxmid = dx_cur; /* alias: we build d(xmid) into dx_cur first */
        for (int t = 0; t < T; t++) {
            const float *dproj2 = trow(dx_next, t, D);
            const float *hact_t = arow(hact, l, t, F);
            const float *hpre_t = arow(hpre, l, t, F);
            const float *xn2_t = arow(xn2, l, t, D);
            float *dh_t = trow(dh, t, F);
            /* dW2 += hact outer dproj2 ; db2 += dproj2 */
            for (long f = 0; f < F; f++) {
                float a = hact_t[f];
                if (a != 0.0f) {
                    float *gw = gW2 + f * D;
                    for (int e = 0; e < D; e++) gw[e] += a * dproj2[e];
                }
                /* dhact = dproj2 @ W2^T */
                float acc = 0.0f;
                for (int e = 0; e < D; e++) acc += W2[f * D + e] * dproj2[e];
                dh_t[f] = acc * gelu_grad(hpre_t[f]); /* -> dhpre */
            }
            for (int e = 0; e < D; e++) gb2[e] += dproj2[e];
            /* dW1 += xn2 outer dhpre ; db1 += dhpre ; dxn2 = dhpre @ W1^T */
            float *dxn_t = trow(dxn, t, D);
            for (int d = 0; d < D; d++) {
                float xv = xn2_t[d];
                if (xv != 0.0f) {
                    float *gw = gW1 + (size_t)d * F;
                    for (long f = 0; f < F; f++) gw[f] += xv * dh_t[f];
                }
                float acc = 0.0f;
                for (long f = 0; f < F; f++) acc += W1[(size_t)d * F + f] * dh_t[f];
                dxn_t[d] = acc;
            }
            for (long f = 0; f < F; f++) gb1[f] += dh_t[f];
            /* d(xmid) = dx_in[l+1] + ln2_bwd(dxn2) */
            layernorm_bwd(rowbuf + (size_t)t * D, dxn_t, arow(xmid, l, t, D),
                          ln2_g, gln2_g, gln2_b, D);
            for (int e = 0; e < D; e++)
                trow(dxmid, t, D)[e] = dproj2[e] + rowbuf[(size_t)t * D + e];
        }

        /* ===== attention backward (whole layer; xmid = x_in + proj1) ===== */
        for (int t = 0; t < T; t++) {
            const float *dproj1 = trow(dxmid, t, D);
            const float *ao_t = arow(ao, l, t, D);
            /* dWo += ao outer dproj1 ; dbo += dproj1 ; dao = dproj1 @ Wo^T */
            for (int d = 0; d < D; d++) {
                float a = ao_t[d];
                if (a != 0.0f) {
                    float *gw = gWo + (size_t)d * D;
                    for (int e = 0; e < D; e++) gw[e] += a * dproj1[e];
                }
                float acc = 0.0f;
                for (int e = 0; e < D; e++) acc += Wo[(size_t)d * D + e] * dproj1[e];
                trow(dao, t, D)[d] = acc;
                gbo[d] += dproj1[d];
            }
        }
        /* per head: datt, dv, softmax-bwd, dq, dk (zero dq/dk/dv first) */
        memset(arow(dq, l, 0, D), 0, (size_t)SEQ_LEN * D * sizeof(float));
        memset(arow(dk, l, 0, D), 0, (size_t)SEQ_LEN * D * sizeof(float));
        memset(arow(dv, l, 0, D), 0, (size_t)SEQ_LEN * D * sizeof(float));
        for (int hh = 0; hh < H; hh++) {
            const float *atth = attl + (size_t)hh * SEQ_LEN * SEQ_LEN;
            float datt[SEQ_LEN], ds[SEQ_LEN];
            for (int i = 0; i < T; i++) {
                const float *dao_i = trow(dao, i, D) + hh * HD;
                for (int j = 0; j <= i; j++) {
                    const float *v_j = arow(vv, l, j, D) + hh * HD;
                    float acc = 0.0f;
                    for (int e = 0; e < HD; e++) acc += dao_i[e] * v_j[e];
                    datt[j] = acc;
                }
                softmax_bwd(ds, datt, atth + i * SEQ_LEN, i + 1);
                const float *q_i = arow(q, l, i, D) + hh * HD;
                float *dq_i = arow(dq, l, i, D) + hh * HD;
                for (int j = 0; j <= i; j++) {
                    const float *k_j = arow(kk, l, j, D) + hh * HD;
                    float *dk_j = arow(dk, l, j, D) + hh * HD;
                    float *dv_j = arow(dv, l, j, D) + hh * HD;
                    float s = ds[j] * scale;
                    float aij = atth[i * SEQ_LEN + j];
                    for (int e = 0; e < HD; e++) {
                        dq_i[e] += s * k_j[e];
                        dk_j[e] += s * q_i[e];
                        dv_j[e] += aij * dao_i[e];
                    }
                }
            }
        }
        /* dWq/dWk/dWv += xn1 outer dq/dk/dv ; biases ; dxn1 = sum of three backprops */
        for (int t = 0; t < T; t++) {
            const float *xn1_t = arow(xn1, l, t, D);
            const float *dq_t = arow(dq, l, t, D);
            const float *dk_t = arow(dk, l, t, D);
            const float *dv_t = arow(dv, l, t, D);
            float *dxn_t = trow(dxn, t, D);
            for (int d1 = 0; d1 < D; d1++) {
                float xv = xn1_t[d1];
                float *gwq = gWq + (size_t)d1 * D;
                float *gwk = gWk + (size_t)d1 * D;
                float *gwv = gWv + (size_t)d1 * D;
                float acc = 0.0f;
                for (int d2 = 0; d2 < D; d2++) {
                    gwq[d2] += xv * dq_t[d2];
                    gwk[d2] += xv * dk_t[d2];
                    gwv[d2] += xv * dv_t[d2];
                    acc += Wq[(size_t)d1 * D + d2] * dq_t[d2]
                         + Wk[(size_t)d1 * D + d2] * dk_t[d2]
                         + Wv[(size_t)d1 * D + d2] * dv_t[d2];
                }
                dxn_t[d1] = acc;
                gbq[d1] += dq_t[d1];
                gbk[d1] += dk_t[d1];
                gbv[d1] += dv_t[d1];
            }
            /* d(x_in[l]) = d(xmid) + ln1_bwd(dxn1) */
            layernorm_bwd(rowbuf + (size_t)t * D, dxn_t, arow(x_in, l, t, D),
                          ln1_g, gln1_g, gln1_b, D);
            for (int e = 0; e < D; e++)
                trow(dx_cur, t, D)[e] = trow(dxmid, t, D)[e] + rowbuf[(size_t)t * D + e];
        }
        /* layer done: dx_cur = d(x_in[l]); swap for next iteration */
        float *tmp = dx_next; dx_next = dx_cur; dx_cur = tmp;
    }

    /* ---- embedding backward: dx_next now holds d(x_in[0]) ---- */
    for (int t = 0; t < T; t++) {
        const float *dx0 = trow(dx_next, t, D);
        float *gtok = G.token_emb + (size_t)tokens[t] * D;
        float *gpos = G.pos_emb + (size_t)t * D;
        for (int e = 0; e < D; e++) {
            gtok[e] += dx0[e];
            gpos[e] += dx0[e];
        }
    }
}

/* ---------------- gradient check ---------------- */

static int gradcheck(void) {
    set_toy_config();
    model_alloc();
    buffers_alloc();
    srand(42);
    model_init();

    int tokens[] = {5, 1, 7};
    int targets[] = {1, 7, 2};

    float loss0 = forward_loss(tokens, targets, 3);
    memset(grads, 0, N_PARAMS * sizeof(float));
    backward(tokens, targets, 3);

    double max_abs = 0.0, max_rel = 0.0;
    long worst = -1;
    for (long i = 0; i < N_PARAMS; i++) {
        float save = params[i];
        params[i] = save + GRADCHECK_EPS;
        float lp = forward_loss(tokens, targets, 3);
        params[i] = save - GRADCHECK_EPS;
        float lm = forward_loss(tokens, targets, 3);
        params[i] = save;
        float num = (lp - lm) / (2.0f * GRADCHECK_EPS);
        float ana = grads[i];
        double abs_err = fabs((double)ana - num);
        /* rel err with clamped denominator — near-zero gradients would
           otherwise blow up the ratio on finite-difference noise alone */
        double rel_err = abs_err / fmax(fabs((double)ana) + fabs((double)num), 1e-2);
        if (abs_err > max_abs) { max_abs = abs_err; worst = i; }
        if (rel_err > max_rel) max_rel = rel_err;
    }
    printf("gradcheck: loss=%.6f params=%ld\n", loss0, N_PARAMS);
    printf("  max abs err: %.3e (param %ld) | max rel err: %.3e\n", max_abs, worst, max_rel);
    int pass = max_abs < 5e-3 && max_rel < 5e-2;
    printf("  %s\n", pass ? "PASS — analytic gradients match finite differences"
                          : "FAIL — backprop bug somewhere");
    return pass ? 0 : 1;
}

/* ---------------- data loader ---------------- */

static FILE *train_bin;
static long train_tokens;

static void data_open(const char *path) {
    train_bin = fopen(path, "rb");
    if (!train_bin) { perror("open train bin"); exit(1); }
    fseek(train_bin, 0, SEEK_END);
    train_tokens = ftell(train_bin) / 2;
}

static void data_window(int *inputs, int *targets, int T) {
    long pos = rand() % (train_tokens - T - 2);
    uint16_t buf[SEQ_LEN + 1];
    fseek(train_bin, pos * 2, SEEK_SET);
    if (fread(buf, 2, T + 1, train_bin) != (size_t)(T + 1)) { perror("read window"); exit(1); }
    for (int i = 0; i < T; i++) { inputs[i] = buf[i]; targets[i] = buf[i + 1]; }
}

/* ---------------- checkpoint (CPW1 v2) ---------------- */

static int save_checkpoint(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("save checkpoint"); return -1; }
    Header out = HDR;
    out.magic = CP_MAGIC;
    out.version = CP_VERSION;
    fwrite(&out, sizeof(out), 1, f);
    fwrite(params, sizeof(float), N_PARAMS, f);
    fclose(f);
    fprintf(stderr, "saved %s (%ld params, %zu bytes)\n",
            path, N_PARAMS, sizeof(Header) + N_PARAMS * 4);
    return 0;
}

/* ---------------- training run ---------------- */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int train_run(int steps) {
    set_dense_config();
    model_alloc();
    buffers_alloc();
    srand(1337);
    model_init();
    data_open("data/tinystories_train.bin");
    fprintf(stderr, "dense sanity: %ld params, %ld train tokens, %d steps (B=%d T=%d lr=%g)\n",
            N_PARAMS, train_tokens, steps, BATCH, SEQ_LEN, LR);

    int inputs[SEQ_LEN], targets[SEQ_LEN];
    for (int step = 1; step <= steps; step++) {
        double t0 = now_s();
        memset(grads, 0, N_PARAMS * sizeof(float));
        float loss = 0.0f;
        for (int b = 0; b < BATCH; b++) {
            data_window(inputs, targets, SEQ_LEN);
            loss += forward_loss(inputs, targets, SEQ_LEN);
            backward(inputs, targets, SEQ_LEN);
        }
        loss /= BATCH;
        for (long i = 0; i < N_PARAMS; i++) grads[i] /= BATCH;

        /* AdamW */
        for (long i = 0; i < N_PARAMS; i++) {
            float g = grads[i];
            opt_m[i] = 0.9f * opt_m[i] + 0.1f * g;
            opt_v[i] = 0.999f * opt_v[i] + 0.001f * g * g;
            float mhat = opt_m[i] / (1.0f - powf(0.9f, step));
            float vhat = opt_v[i] / (1.0f - powf(0.999f, step));
            params[i] -= LR * mhat / (sqrtf(vhat) + 1e-8f);
        }

        if (step == 1 || step % 20 == 0)
            fprintf(stderr, "step %4d | loss %.4f | %.2fs/step\n",
                    step, loss, now_s() - t0);
    }
    fclose(train_bin);
    return save_checkpoint("weights/tinystories_dense.bin");
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--gradcheck") == 0)
        return gradcheck();
    return train_run(argc > 1 ? atoi(argv[1]) : T_STEPS);
}
