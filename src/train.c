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

#include "kernels/kernels.h"
#include "weights.h"

/* ---------------- config ---------------- */

static Header HDR;
static long N_PARAMS;

#define T_STEPS   300
#define BATCH     4
#define SEQ_LEN   64
#define EOS_TOKEN 50256   /* GPT-2 <|endoftext|>; also resets the locality set */
#define LR        1e-3f
#define GRADCHECK_EPS 1e-3f

static void set_toy_config(void)   { HDR = (Header){0, 0, 8, 4, 2, 8, 3, 8}; }
static void set_dense_config(void) { HDR = (Header){0, 0, 50257, 128, 4, 512, 4, 256}; }

/* ---------------- MoE config (Model-Architecture-Spec, scaled for CPU sanity) ----------------
   Dense early/late, MoE middle: moe_layer(l) = 0 < l < L-1.
   NOTE (2026-09-25): this is the AS-BUILT layout -- MoE on layers 1..L-2, dense
   on 0 and L-1 -- which does NOT match Model-Architecture-Spec, which asks for
   MoE on 2..L-2 with dense on 0-1 and L-1. Both trainers share the deviation,
   which is exactly why the G1 bridge and G2 smoke gates passed and nothing
   caught it: they test implementation-vs-implementation, not spec conformance.
   train_moe.py now carries moe_from (default 1 = as-built, 2 = spec) and
   tools/check_spec_conformance.py checks against the spec. If you change the
   layout here, change it there too, or the bridge compares nothing meaningful.
   See vault Decision-Log 2026-09-25.
   Per MoE layer: 1 shared expert (hidden = d_model) + N_EXP routed (top-2).
   Router: linear D->N_EXP + softmax(biased logits), top-2, renormalize.
   Locality bias: score += LAMBDA if expert was selected by the previous
   token at this layer (spec mechanism; LAMBDA=0 = ablation baseline).
   Aux loss: Switch-style alpha * N * sum_e f_e * P_e per MoE layer. */
static int MOE = 0;
static int MOE_FROM = 1;       /* first sparse-FFN layer; 1 = as-built, 2 = spec */
static int N_EXP = 8;          /* routed experts per MoE layer (sanity scale) */
static int EXP_HID = 32;       /* routed expert hidden dim (sanity scale) */
static float LAMBDA = 0.5f;    /* locality bias strength; 0 = ablation */
static float AUX_ALPHA = 0.01f;
static int moe_layer(int l) {
    return MOE && l >= MOE_FROM && l < HDR.n_layer - 1;
}

/* --- spec-conformance check (2026-09-25) -----------------------------------
   G1 (bridge) and G2 (smoke) compare the two implementations to EACH OTHER;
   neither compares either one to Model-Architecture-Spec. That is how the
   Phase 6 checkpoints ended up with MoE on layers 1..L-2 while the spec asks for
   2..L-2, with every gate green. This makes the layout explicit and loud.
   Mirrors tools/check_spec_conformance.py.
     moe_from 1 = AS-BUILT Phase 6 layout (dense on 0 and L-1)
     moe_from 2 = spec layout (dense on 0-1 and L-1)
   Returns 0 if conformant, 1 if divergent. */
static int check_spec(void) {
    printf("layout: moe_from=%d  n_layer=%d  (sanity-scale rule check)\n",
           MOE_FROM, HDR.n_layer);
    printf("  MoE layers  :");
    for (int l = 0; l < HDR.n_layer; l++) if (moe_layer(l)) printf(" %d", l);
    printf("\n  dense layers:");
    for (int l = 0; l < HDR.n_layer; l++) if (!moe_layer(l)) printf(" %d", l);
    printf("\n  spec rule   : MoE on 2..L-2, dense on 0-1 and L-1\n");
    int conformant = (MOE_FROM == 2);
    printf("\nSPEC CONFORMANCE: %s\n", conformant ? "PASS" : "DIVERGENT");
    if (!conformant)
        fprintf(stderr,
                "  -> moe_from=%d is the AS-BUILT Phase 6 deviation, NOT the spec.\n"
                "     Use --moe-from 2 for a conformant run. See Decision-Log 2026-09-25.\n",
                MOE_FROM);
    return conformant ? 0 : 1;
}

static float frand(void);

typedef struct {
    float *Wr, *br;      /* router: (L,D,N), (L,N) */
    float *W1s, *b1s;    /* shared expert: (L,D,D), (L,D) */
    float *W2s, *b2s;    /* shared expert: (L,D,D), (L,D) */
    float *W1e, *b1e;    /* routed: (L,N,D,EH), (L,N,EH) */
    float *W2e, *b2e;    /* routed: (L,N,EH,D), (L,N,D) */
} MoeW;

static MoeW MW, MG;
static float *moe_params, *moe_grads;
static long N_MOE;

static void moe_layout(MoeW *m, float *ptr) {
    int L = HDR.n_layer, D = HDR.d_model, N = N_EXP, EH = EXP_HID;
    m->Wr = ptr;  ptr += (long)L * D * N;
    m->br = ptr;  ptr += (long)L * N;
    m->W1s = ptr; ptr += (long)L * D * D;
    m->b1s = ptr; ptr += (long)L * D;
    m->W2s = ptr; ptr += (long)L * D * D;
    m->b2s = ptr; ptr += (long)L * D;
    m->W1e = ptr; ptr += (long)L * N * D * EH;
    m->b1e = ptr; ptr += (long)L * N * EH;
    m->W2e = ptr; ptr += (long)L * N * EH * D;
    m->b2e = ptr;
}

static long moe_count(void) {
    int L = HDR.n_layer, D = HDR.d_model, N = N_EXP, EH = EXP_HID;
    return (long)L * (D * N + N + 2 * (D * D + D) + N * (D * EH + EH + EH * D + D));
}

static void moe_alloc(void) {
    if (!MOE) return;
    N_MOE = moe_count();
    moe_params = malloc(N_MOE * sizeof(float));
    moe_grads = calloc(N_MOE, sizeof(float));
    moe_layout(&MW, moe_params);
    moe_layout(&MG, moe_grads);
}

static void moe_init(void) {
    int L = HDR.n_layer, D = HDR.d_model, N = N_EXP, EH = EXP_HID;
    if (!MOE) return;
    for (long i = 0; i < (long)L * D * N; i++) MW.Wr[i] = frand() * 0.02f;
    for (long i = 0; i < (long)L * D * D; i++) { MW.W1s[i] = frand() * 0.02f; MW.W2s[i] = frand() * 0.02f; }
    for (long i = 0; i < (long)L * N * D * EH; i++) MW.W1e[i] = frand() * 0.02f;
    for (long i = 0; i < (long)L * N * EH * D; i++) MW.W2e[i] = frand() * 0.02f;
    for (long i = 0; i < (long)L * N; i++) MW.br[i] = 0.0f;
    for (long i = 0; i < (long)L * D; i++) { MW.b1s[i] = 0.0f; MW.b2s[i] = 0.0f; }
    for (long i = 0; i < (long)L * N * EH; i++) MW.b1e[i] = 0.0f;
    for (long i = 0; i < (long)L * N * D; i++) MW.b2e[i] = 0.0f;
}

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

/* MoE activation buffers (per MoE layer l) */
static float *rprob;     /* (L,T,N) biased softmax router probs */
static int *topk_id;     /* (L,T,2) selected expert ids */
static float *topk_w;    /* (L,T,2) renormalized selection weights */
static float *slot_hpre, *slot_hact; /* (L,T,2,EH) per selected slot */
static float *slot_out;  /* (L,T,2,D) expert outputs (for dw_k grad) */
static float *sh_hpre, *sh_hact;     /* (L,T,D) shared expert hidden */
static float *aux_fe;    /* (L,N) fraction of tokens routed to e (window) */
static long *util;       /* (L,N) selection counts across the run */
static int *prev_sel;    /* (L,N) previous-token selection set (locality) */

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

    if (MOE) {
        int N = N_EXP, EH = EXP_HID;
        rprob = malloc((size_t)L * SEQ_LEN * N * sizeof(float));
        topk_id = malloc((size_t)L * SEQ_LEN * 2 * sizeof(int));
        topk_w = malloc((size_t)L * SEQ_LEN * 2 * sizeof(float));
        slot_hpre = malloc((size_t)L * SEQ_LEN * 2 * EH * sizeof(float));
        slot_hact = malloc((size_t)L * SEQ_LEN * 2 * EH * sizeof(float));
        slot_out = malloc((size_t)L * SEQ_LEN * 2 * D * sizeof(float));
        sh_hpre = malloc((size_t)L * SEQ_LEN * D * sizeof(float));
        sh_hact = malloc((size_t)L * SEQ_LEN * D * sizeof(float));
        aux_fe = malloc((size_t)L * N * sizeof(float));
        util = calloc((size_t)L * N, sizeof(long));
        prev_sel = malloc((size_t)L * N * sizeof(int));
    }
}

/* ---------------- MoE forward (per layer, all tokens) ---------------- */

static float aux_loss_total; /* accumulated per forward_loss call */

static void moe_ffn_forward(int l, int T, const int *tokens) {
    const int D = HDR.d_model, N = N_EXP, EH = EXP_HID;
    float *Wr = MW.Wr + (size_t)l * D * N, *br = MW.br + (size_t)l * N;
    float *W1s = MW.W1s + (size_t)l * D * D, *b1s = MW.b1s + (size_t)l * D;
    float *W2s = MW.W2s + (size_t)l * D * D, *b2s = MW.b2s + (size_t)l * D;
    float *W1e = MW.W1e + (size_t)l * N * D * EH;
    float *b1e = MW.b1e + (size_t)l * N * EH;
    float *W2e = MW.W2e + (size_t)l * N * EH * D;
    float *b2e = MW.b2e + (size_t)l * N * D;
    float *fe = aux_fe + (size_t)l * N;
    int *prev = prev_sel + (size_t)l * N;
    float rlog[64], p[64];
    if (N > 64) { fprintf(stderr, "N_EXP>64 unsupported\n"); exit(1); }

    memset(fe, 0, N * sizeof(float));
    memset(prev, 0, N * sizeof(int)); /* window start: no previous token */

    for (int t = 0; t < T; t++) {
        const float *xn2_t = arow(xn2, l, t, D);
        float *rp = rprob + ((size_t)l * SEQ_LEN + t) * N;
        int *ids = topk_id + ((size_t)l * SEQ_LEN + t) * 2;
        float *ws = topk_w + ((size_t)l * SEQ_LEN + t) * 2;

        /* router: logits -> + locality bias -> softmax */
        matmul_xw(rlog, xn2_t, Wr, br, D, N);
        for (int e = 0; e < N; e++) p[e] = rlog[e] + (prev[e] ? LAMBDA : 0.0f);
        softmax(p, N);
        memcpy(rp, p, N * sizeof(float));

        /* top-2 by prob, renormalize */
        int a = 0, b = 1;
        if (p[b] > p[a]) { a = 1; b = 0; }
        for (int e = 2; e < N; e++)
            if (p[e] > p[a]) { b = a; a = e; }
            else if (p[e] > p[b]) b = e;
        float S = p[a] + p[b];
        ids[0] = a; ids[1] = b;
        ws[0] = p[a] / S; ws[1] = p[b] / S;
        fe[a] += 1.0f / (T * 2); fe[b] += 1.0f / (T * 2);
        util[(size_t)l * N + a]++; util[(size_t)l * N + b]++;
        memset(prev, 0, N * sizeof(int));
        /* Locality set for the NEXT token. Spec (2026-09-15): it resets at
           window start AND at every EOS token — cross-document stickiness is
           meaningless. This check was MISSING until 2026-09-25, so the C
           trainer carried routing across document boundaries; the extended G1
           bridge caught it because the old 5-token fixture contained no EOS. */
        if (tokens[t] != EOS_TOKEN) {
            prev[a] = 1;
            prev[b] = 1;
        }

        /* shared expert (always active) */
        float *shp = sh_hpre + ((size_t)l * SEQ_LEN + t) * D;
        float *sha = sh_hact + ((size_t)l * SEQ_LEN + t) * D;
        matmul_xw(shp, xn2_t, W1s, b1s, D, D);
        for (int e = 0; e < D; e++) sha[e] = gelu(shp[e]);
        float *out = arow(x_in, l + 1, t, D);
        matmul_xw(out, sha, W2s, b2s, D, D);

        /* routed experts: dispatch top-2 */
        for (int s = 0; s < 2; s++) {
            int e = ids[s];
            const float *w1 = W1e + (size_t)e * D * EH;
            const float *b1 = b1e + (size_t)e * EH;
            const float *w2 = W2e + (size_t)e * EH * D;
            const float *b2 = b2e + (size_t)e * D;
            float *hp = slot_hpre + (((size_t)l * SEQ_LEN + t) * 2 + s) * EH;
            float *ha = slot_hact + (((size_t)l * SEQ_LEN + t) * 2 + s) * EH;
            float *so = slot_out + (((size_t)l * SEQ_LEN + t) * 2 + s) * D;
            matmul_xw(hp, xn2_t, w1, b1, D, EH);
            for (int i = 0; i < EH; i++) ha[i] = gelu(hp[i]);
            matmul_xw(so, ha, w2, b2, EH, D);
            for (int d = 0; d < D; d++) out[d] += ws[s] * so[d];
        }
        for (int d = 0; d < D; d++) out[d] += arow(xmid, l, t, D)[d];
    }

    /* aux loss (Switch): alpha * N * sum_e f_e * P_e, P_e = mean prob */
    float aux = 0.0f;
    for (int e = 0; e < N; e++) {
        float P = 0.0f;
        for (int t = 0; t < T; t++) P += rprob[((size_t)l * SEQ_LEN + t) * N + e];
        P /= T;
        aux += fe[e] * P;
    }
    aux_loss_total += AUX_ALPHA * N * aux;
}

/* ---------------- forward (stores activations) ---------------- */

static float forward_loss(const int *tokens, const int *targets, int T) {
    const int D = HDR.d_model, H = HDR.n_heads, L = HDR.n_layer;
    const long F = HDR.d_ff, V = HDR.vocab;
    const int HD = D / H;
    aux_loss_total = 0.0f;

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
        }
        if (moe_layer(l)) {
            for (int t = 0; t < T; t++)
                layernorm(arow(xn2, l, t, D), arow(xmid, l, t, D), ln2_g, ln2_b, D);
            moe_ffn_forward(l, T, tokens);
        } else {
            for (int t = 0; t < T; t++) {
                layernorm(arow(xn2, l, t, D), arow(xmid, l, t, D), ln2_g, ln2_b, D);
                matmul_xw(arow(hpre, l, t, F), arow(xn2, l, t, D), W1, b1, D, F);
                for (int e = 0; e < F; e++)
                    arow(hact, l, t, F)[e] = gelu(arow(hpre, l, t, F)[e]);
                matmul_xw(scratchF, arow(hact, l, t, F), W2, b2, F, D);
                for (int e = 0; e < D; e++)
                    arow(x_in, l + 1, t, D)[e] = arow(xmid, l, t, D)[e] + scratchF[e];
            }
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
    return loss / T + aux_loss_total;
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

/* ---------------- MoE backward (per layer, all tokens) ----------------
   Fills dxmid (= grad on xmid rows) from dx_next; accumulates all MoE grads. */
static void moe_ffn_backward(int l, int T, float *dxmid, float *dx_next_g) {
    const int D = HDR.d_model, N = N_EXP, EH = EXP_HID;
    const float *ln2_g = W.ln2_g + l * D;
    float *gln2_g = G.ln2_g + l * D, *gln2_b = G.ln2_b + l * D;
    float *Wr = MW.Wr + (size_t)l * D * N;
    float *W1s = MW.W1s + (size_t)l * D * D;
    float *W2s = MW.W2s + (size_t)l * D * D;
    const float *W1e = MW.W1e + (size_t)l * N * D * EH;
    const float *W2e = MW.W2e + (size_t)l * N * EH * D;
    float *gWr = MG.Wr + (size_t)l * D * N, *gbr = MG.br + (size_t)l * N;
    float *gW1s = MG.W1s + (size_t)l * D * D, *gb1s = MG.b1s + (size_t)l * D;
    float *gW2s = MG.W2s + (size_t)l * D * D, *gb2s = MG.b2s + (size_t)l * D;
    float *gW1e = MG.W1e + (size_t)l * N * D * EH, *gb1e = MG.b1e + (size_t)l * N * EH;
    float *gW2e = MG.W2e + (size_t)l * N * EH * D, *gb2e = MG.b2e + (size_t)l * N * D;
    const float *fe = aux_fe + (size_t)l * N;
    float dprobs[64], drlog[64];

    for (int t = 0; t < T; t++) {
        const float *dout = trow(dx_next_g, t, D);
        const float *xn2_t = arow(xn2, l, t, D);
        const float *rp = rprob + ((size_t)l * SEQ_LEN + t) * N;
        const int *ids = topk_id + ((size_t)l * SEQ_LEN + t) * 2;
        const float *ws = topk_w + ((size_t)l * SEQ_LEN + t) * 2;
        float *dxn_t = trow(dxn, t, D);
        memset(dxn_t, 0, D * sizeof(float));

        /* ---- router path ----
           grad on selection weights: gw_s = dout . slot_out[s] ;
           through renormalize w = p/S and aux loss -> dprobs -> softmax_bwd */
        int a = ids[0], b = ids[1];
        const float *so0 = slot_out + (((size_t)l * SEQ_LEN + t) * 2 + 0) * D;
        const float *so1 = slot_out + (((size_t)l * SEQ_LEN + t) * 2 + 1) * D;
        float gw0 = 0.0f, gw1 = 0.0f;
        for (int d = 0; d < D; d++) { gw0 += dout[d] * so0[d]; gw1 += dout[d] * so1[d]; }
        float S = rp[a] + rp[b];
        memset(dprobs, 0, N * sizeof(float));
        dprobs[a] = (gw0 - gw1) * rp[b] / (S * S);
        dprobs[b] = (gw1 - gw0) * rp[a] / (S * S);
        for (int e = 0; e < N; e++) dprobs[e] += AUX_ALPHA * N * fe[e] / T;
        softmax_bwd(drlog, dprobs, rp, N);
        /* dWr += xn2 outer drlog ; dbr += drlog ; dxn2 += drlog @ Wr^T */
        for (int d = 0; d < D; d++) {
            float xv = xn2_t[d];
            float *gw = gWr + (size_t)d * N;
            for (int e = 0; e < N; e++) {
                gw[e] += xv * drlog[e];
                dxn_t[d] += Wr[(size_t)d * N + e] * drlog[e];
            }
        }
        for (int e = 0; e < N; e++) gbr[e] += drlog[e];

        /* ---- routed experts backward (per slot) ---- */
        for (int s = 0; s < 2; s++) {
            int e = ids[s];
            const float *hp = slot_hpre + (((size_t)l * SEQ_LEN + t) * 2 + s) * EH;
            const float *w1 = W1e + (size_t)e * D * EH;
            const float *w2 = W2e + (size_t)e * EH * D;
            float *gw1 = gW1e + (size_t)e * D * EH, *gb1 = gb1e + (size_t)e * EH;
            float *gw2 = gW2e + (size_t)e * EH * D, *gb2 = gb2e + (size_t)e * D;
            /* dout_e = ws[s] * dout */
            for (int i = 0; i < EH; i++) {
                float acc = 0.0f;
                for (int d = 0; d < D; d++) acc += w2[(size_t)i * D + d] * (ws[s] * dout[d]);
                dh[t * EXP_HID + i] = acc * gelu_grad(hp[i]); /* dhpre */
            }
            for (int i = 0; i < EH; i++) {
                float ha = slot_hact[(((size_t)l * SEQ_LEN + t) * 2 + s) * EH + i];
                if (ha != 0.0f) {
                    float *gw = gw2 + (size_t)i * D;
                    for (int d = 0; d < D; d++) gw[d] += ha * ws[s] * dout[d];
                }
                gb1[i] += dh[t * EXP_HID + i];
            }
            for (int d = 0; d < D; d++) {
                float xv = xn2_t[d];
                float *gw = gw1 + (size_t)d * EH;
                float acc = 0.0f;
                for (int i = 0; i < EH; i++) {
                    gw[i] += xv * dh[t * EXP_HID + i];
                    acc += w1[(size_t)d * EH + i] * dh[t * EXP_HID + i];
                }
                dxn_t[d] += acc;
                gb2[d] += ws[s] * dout[d];
            }
        }

        /* ---- shared expert backward (full dout) ---- */
        const float *shp = sh_hpre + ((size_t)l * SEQ_LEN + t) * D;
        const float *sha = sh_hact + ((size_t)l * SEQ_LEN + t) * D;
        /* dh_s reuses rowbuf's slot for this t — consumed before the
           layernorm_bwd call below overwrites it */
        float *dh_s = rowbuf + (size_t)t * D;
        for (int d = 0; d < D; d++) {
            float acc = 0.0f;
            for (int e = 0; e < D; e++) acc += W2s[(size_t)d * D + e] * dout[e];
            dh_s[d] = acc * gelu_grad(shp[d]);
        }
        for (int d = 0; d < D; d++) {
            if (sha[d] != 0.0f) {
                float *gw = gW2s + (size_t)d * D;
                for (int e = 0; e < D; e++) gw[e] += sha[d] * dout[e];
            }
            gb1s[d] += dh_s[d];
        }
        for (int d1 = 0; d1 < D; d1++) {
            float xv = xn2_t[d1];
            float *gw = gW1s + (size_t)d1 * D;
            float acc = 0.0f;
            for (int d2 = 0; d2 < D; d2++) {
                gw[d2] += xv * dh_s[d2];
                acc += W1s[(size_t)d1 * D + d2] * dh_s[d2];
            }
            dxn_t[d1] += acc;
            gb2s[d1] += dout[d1];
        }

        /* d(xmid) = dout + ln2_bwd(dxn2) */
        layernorm_bwd(rowbuf + (size_t)t * D, dxn_t, arow(xmid, l, t, D),
                      ln2_g, gln2_g, gln2_b, D);
        for (int e = 0; e < D; e++)
            trow(dxmid, t, D)[e] = dout[e] + rowbuf[(size_t)t * D + e];
    }
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

        /* ===== FFN/MoE backward (x_in[l+1] = xmid + ffn_out) ===== */
        float *dxmid = dx_cur; /* alias: we build d(xmid) into dx_cur first */
        if (moe_layer(l)) {
            moe_ffn_backward(l, T, dxmid, dx_next);
        } else for (int t = 0; t < T; t++) {
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

static void check_buffer(float *p, const float *g, long n,
                         const int *tokens, const int *targets,
                         double *max_abs, double *max_rel, long *worst, long base) {
    for (long i = 0; i < n; i++) {
        float save = p[i];
        p[i] = save + GRADCHECK_EPS;
        float lp = forward_loss(tokens, targets, 3);
        p[i] = save - GRADCHECK_EPS;
        float lm = forward_loss(tokens, targets, 3);
        p[i] = save;
        float num = (lp - lm) / (2.0f * GRADCHECK_EPS);
        float ana = g[i];
        double abs_err = fabs((double)ana - num);
        /* rel err with clamped denominator — near-zero gradients would
           otherwise blow up the ratio on finite-difference noise alone */
        double rel_err = abs_err / fmax(fabs((double)ana) + fabs((double)num), 1e-2);
        if (abs_err > *max_abs) { *max_abs = abs_err; *worst = base + i; }
        if (rel_err > *max_rel) *max_rel = rel_err;
    }
}

static int gradcheck(int with_moe) {
    MOE = with_moe;
    if (MOE) { N_EXP = 4; EXP_HID = 2; }
    set_toy_config();
    model_alloc();
    moe_alloc();
    buffers_alloc();
    srand(42);
    model_init();
    moe_init();

    int tokens[] = {5, 1, 7};
    int targets[] = {1, 7, 2};

    float loss0 = forward_loss(tokens, targets, 3);
    memset(grads, 0, N_PARAMS * sizeof(float));
    if (MOE) memset(moe_grads, 0, N_MOE * sizeof(float));
    backward(tokens, targets, 3);

    double max_abs = 0.0, max_rel = 0.0;
    long worst = -1;
    check_buffer(params, grads, N_PARAMS, tokens, targets, &max_abs, &max_rel, &worst, 0);
    if (MOE)
        check_buffer(moe_params, moe_grads, N_MOE, tokens, targets, &max_abs, &max_rel, &worst, N_PARAMS);

    printf("gradcheck%s: loss=%.6f params=%ld+%ld\n", MOE ? " (MoE)" : "",
           loss0, N_PARAMS, MOE ? N_MOE : 0);
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
    out.version = CP_VERSION_V2;
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

static float *opt_m2, *opt_v2; /* MoE optimizer state */

static void adamw_update(float *p, float *g, float *m, float *v, long n, int step) {
    for (long i = 0; i < n; i++) {
        float gi = g[i];
        m[i] = 0.9f * m[i] + 0.1f * gi;
        v[i] = 0.999f * v[i] + 0.001f * gi * gi;
        float mhat = m[i] / (1.0f - powf(0.9f, step));
        float vhat = v[i] / (1.0f - powf(0.999f, step));
        p[i] -= LR * mhat / (sqrtf(vhat) + 1e-8f);
    }
}

static int train_run(int steps) {
    set_dense_config();
    model_alloc();
    moe_alloc();
    if (MOE) { opt_m2 = calloc(N_MOE, sizeof(float)); opt_v2 = calloc(N_MOE, sizeof(float)); }
    buffers_alloc();
    srand(1337);
    model_init();
    moe_init();
    data_open("data/tinystories_train.bin");
    fprintf(stderr, "%s: %ld+%ld params, %ld train tokens, %d steps (B=%d T=%d lr=%g lambda=%g)\n",
            MOE ? "moe sanity" : "dense sanity", N_PARAMS, MOE ? N_MOE : 0,
            train_tokens, steps, BATCH, SEQ_LEN, LR, MOE ? LAMBDA : 0.0f);

    int inputs[SEQ_LEN], targets[SEQ_LEN];
    for (int step = 1; step <= steps; step++) {
        double t0 = now_s();
        memset(grads, 0, N_PARAMS * sizeof(float));
        if (MOE) memset(moe_grads, 0, N_MOE * sizeof(float));
        float loss = 0.0f;
        for (int b = 0; b < BATCH; b++) {
            data_window(inputs, targets, SEQ_LEN);
            loss += forward_loss(inputs, targets, SEQ_LEN);
            backward(inputs, targets, SEQ_LEN);
        }
        loss /= BATCH;
        for (long i = 0; i < N_PARAMS; i++) grads[i] /= BATCH;
        if (MOE) for (long i = 0; i < N_MOE; i++) moe_grads[i] /= BATCH;

        adamw_update(params, grads, opt_m, opt_v, N_PARAMS, step);
        if (MOE) adamw_update(moe_params, moe_grads, opt_m2, opt_v2, N_MOE, step);

        if (step == 1 || step % 20 == 0) {
            fprintf(stderr, "step %4d | loss %.4f", step, loss);
            if (MOE) {
                /* collapse watch: max expert share per MoE layer (spec: >25% sustained = collapse) */
                for (int l = 0; l < HDR.n_layer; l++) {
                    if (!moe_layer(l)) continue;
                    long mx = 0, tot = 0;
                    for (int e = 0; e < N_EXP; e++) {
                        if (util[(size_t)l * N_EXP + e] > mx) mx = util[(size_t)l * N_EXP + e];
                        tot += util[(size_t)l * N_EXP + e];
                    }
                    fprintf(stderr, " | L%d maxshare %.2f", l, tot ? (double)mx / tot : 0.0);
                }
            }
            fprintf(stderr, " | %.2fs/step\n", now_s() - t0);
        }
    }
    fclose(train_bin);
    if (MOE) {
        fprintf(stderr, "MoE checkpoint saving deferred (CPW1 v3 is Phase 7) — run was for the loss-curve milestone\n");
        return 0;
    }
    return save_checkpoint("weights/tinystories_dense.bin");
}

/* --export-moe <path>: dump the sanity-scale MoE model (fixed seed) + a
   fixed-window forward's logits, for the PyTorch bridge test (G1).
   Layout: 9 ints {vocab,d_model,n_heads,d_ff,n_layer,n_ctx,N_EXP,EXP_HID,T}
   then longs {n_params,n_moe}, float loss, params, moe_params, logits (T*V). */
static int export_moe(const char *path) {
    MOE = 1; N_EXP = 8; EXP_HID = 32; LAMBDA = 0.5f;
    set_dense_config();
    model_alloc();
    moe_alloc();
    buffers_alloc();
    srand(99);
    model_init();
    moe_init();

    /* Bridge fixture (format CPBRIDG2, 2026-09-25). Real corpus text with an
       EOS at index 8, so the locality reset is actually exercised — the old
       5-token fixture had no EOS and could not test it. The export carries the
       tokens, so the Python side cannot drift from what C actually ran. */
    int toks[] = {262, 2071, 318, 17764, 351, 6868, 18366, 737, 50256, 14906,
                  314, 25, 1041, 3145, 896, 2422, 3842, 11, 355, 564, 250,
                  13217, 283, 28914};
    int ntok = (int)(sizeof(toks) / sizeof(toks[0]));
    int T = ntok - 1;                      /* next-token setup */
    int targets[SEQ_LEN];
    for (int t = 0; t < T; t++) targets[t] = toks[t + 1];
    float loss = forward_loss(toks, targets, T);

    int n_moe_layers = 0;
    for (int l = 0; l < HDR.n_layer; l++) if (moe_layer(l)) n_moe_layers++;

    FILE *f = fopen(path, "wb");
    if (!f) { perror("export"); return 1; }
    fwrite("CPBRIDG2", 1, 8, f);
    int hd[11] = {HDR.vocab, HDR.d_model, HDR.n_heads, HDR.d_ff,
                  HDR.n_layer, HDR.n_ctx, N_EXP, EXP_HID, T,
                  n_moe_layers, MOE_FROM};
    long ld[2] = {N_PARAMS, N_MOE};
    int eos = 50256;
    fwrite(hd, sizeof(int), 11, f);
    fwrite(ld, sizeof(long), 2, f);
    fwrite(&loss, sizeof(float), 1, f);
    fwrite(&eos, sizeof(int), 1, f);
    fwrite(toks, sizeof(int), ntok, f);
    fwrite(params, sizeof(float), N_PARAMS, f);
    fwrite(moe_params, sizeof(float), N_MOE, f);
    fwrite(logits, sizeof(float), (size_t)T * HDR.vocab, f);
    for (int l = 0; l < HDR.n_layer; l++) {
        if (!moe_layer(l)) continue;
        for (int t = 0; t < T; t++)
            fwrite(topk_id + ((size_t)l * SEQ_LEN + t) * 2, sizeof(int), 2, f);
    }
    fclose(f);
    fprintf(stderr, "exported %s: loss=%.6f params=%ld+%ld T=%d tokens=%d "
                    "moe_layers=%d moe_from=%d (CPBRIDG2)\n",
            path, loss, N_PARAMS, N_MOE, T, ntok, n_moe_layers, MOE_FROM);
    return 0;
}

int main(int argc, char **argv) {
    int steps = T_STEPS, do_gradcheck = 0, do_check_spec = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gradcheck") == 0) do_gradcheck = 1;
        else if (strcmp(argv[i], "--moe") == 0) MOE = 1;
        else if (strcmp(argv[i], "--moe-from") == 0 && i + 1 < argc)
            MOE_FROM = atoi(argv[++i]);
        else if (strcmp(argv[i], "--check-spec") == 0) do_check_spec = 1;
        else if (strcmp(argv[i], "--export-moe") == 0 && i + 1 < argc)
            return export_moe(argv[++i]);
        else if (strcmp(argv[i], "--lambda") == 0 && i + 1 < argc) LAMBDA = atof(argv[++i]);
        else steps = atoi(argv[i]);
    }
    if (do_check_spec) {
        if (!MOE) {
            fprintf(stderr, "check-spec: pass --moe (the spec's sparse layout is "
                            "the only conformant configuration)\n");
            return 1;
        }
        set_dense_config();
        return check_spec();
    }
    if (do_gradcheck) {
        int rc = gradcheck(0);   /* dense gate must pass first */
        if (rc) return rc;
        return gradcheck(1);     /* then the MoE gate */
    }
    return train_run(steps);
}
