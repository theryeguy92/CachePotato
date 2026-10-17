#include "model.h"
#include "blocks/blocks.h"
#include "instr.h"
#include "kernels/kernels.h"
#include "pager/pager.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EOS_TOKEN CP_EOS_TOKEN /* also resets the locality set */

/* --- MoE FFN (CPW1 v3) -----------------------------------------------------
   router -> +lambda locality bias -> softmax -> top-k, then one shared expert
   (always active, hot by architecture) plus the k routed experts. The locality
   set is the previous token's selection at THIS layer and is CLEARED at every
   EOS: cross-document stickiness is meaningless (spec decision 2026-09-15).
   Mirrors train_moe.py's moe_ffn and the bridge-verified src/train.c: the top-k
   scan below is the same strict-> form, so lower id wins ties. */

void forward_moe(const Model *m, const int *tokens, int T, float *logits_out,
                 int *sel_out, float lambda) {
    const int D = m->hdr.d_model, L = m->hdr.n_layer;
    const int F = m->hdr.d_ff, V = m->hdr.vocab;
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
    for (int t = 0; t < T; t++) {
        instr_touch("token_emb", -1, t, D * sizeof(float));
        instr_touch("pos_emb", -1, t, D * sizeof(float));
        for (int e = 0; e < D; e++)
            x[t * D + e] = w->token_emb[tokens[t] * D + e] + w->pos_emb[t * D + e];
    }

    for (int l = 0; l < L; l++) {
        const float *ln2_g = w->ln2_g + l * D, *ln2_b = w->ln2_b + l * D;

        /* attention half — the block owns norm -> qkv -> masked attention -> Wo -> +res */
        blk_attention_mha(m, l, T, x, xn, q, k, vv, ao, att, proj, x);

        /* FFN half: dense on the outer layers, sparse (MoE) in the middle */
        for (int t = 0; t < T; t++) {
            instr_touch("ln2", l, t, 2 * D * sizeof(float));
            layernorm(xn + t * D, x + t * D, ln2_g, ln2_b, D);
        }
        if (moe_layer_at(m, l)) {
            blk_moe_ffn(m, m->w.moe_slot[l], l, xn, x, tokens, T, h, proj, sel_out, lambda);
        } else {
            blk_ffn_dense(m, l, T, xn, x, h, proj);
        }
    }

    /* head: final norm -> logits = xf @ token_emb^T (tied unembedding) */
    for (int t = 0; t < T; t++) {
        instr_touch("lnf", -1, t, 2 * D * sizeof(float));
        instr_touch("unemb", -1, t, (size_t)V * D * sizeof(float));
        layernorm(xn + t * D, x + t * D, w->lnf_g, w->lnf_b, D);
        matmul_wx(logits_out + t * V, xn + t * D, w->token_emb, D, V);
    }

    free(x); free(xn); free(q); free(k); free(vv);
    free(ao); free(att); free(h); free(proj);
}

/* v2-compatible entry point (dense): no routing trace, no locality bias. */
void forward(const Model *m, const int *tokens, int T, float *logits_out) {
    forward_moe(m, tokens, T, logits_out, NULL, 0.0f);
}

/* ---- incremental generation mode (KV-cached, one token per call) ------- */

void runstate_alloc(RunState *s, const Model *m) {
    s->lam = m->h3.lam;   /* callers may override (v4 files carry no lambda) */
    int D = m->hdr.d_model, F = m->hdr.d_ff, V = m->hdr.vocab;
    int L = m->hdr.n_layer, N = m->w.n_exp, K = m->w.topk;
    size_t cache = (size_t)m->hdr.n_layer * m->hdr.n_ctx * D;
    s->x = malloc(D * sizeof(float));
    s->xn = malloc(D * sizeof(float));
    s->q = malloc(D * sizeof(float));
    s->ao = malloc(D * sizeof(float));
    s->att = malloc(m->hdr.n_ctx * sizeof(float));
    s->h = malloc(F * sizeof(float));
    s->proj = malloc(D * sizeof(float));
    s->logits = malloc(V * sizeof(float));
    s->key_cache = calloc(cache, sizeof(float));
    s->value_cache = calloc(cache, sizeof(float));
    /* MoE state (NULL/0 for dense v2 files) */
    s->moe_prev = (L > 0 && N > 0) ? calloc((size_t)L * N, sizeof(int)) : NULL;
    s->moe_p = N > 0 ? malloc((size_t)N * sizeof(float)) : NULL;
    s->moe_idx = K > 0 ? malloc((size_t)K * sizeof(int)) : NULL;
    s->moe_wt = K > 0 ? malloc((size_t)K * sizeof(float)) : NULL;
    /* MUST be initialised: forward_step dereferences this if non-NULL. Gates 1
       and 2 set it explicitly, so neither could catch it — the REPL did. */
    s->moe_sel_out = NULL;
    s->moe_sel_stride = 0;
}

void runstate_free(RunState *s, const Model *m) {
    (void)m;
    free(s->x); free(s->xn); free(s->q); free(s->ao); free(s->att);
    free(s->h); free(s->proj); free(s->logits);
    free(s->key_cache); free(s->value_cache);
    free(s->moe_prev); free(s->moe_p); free(s->moe_idx); free(s->moe_wt);
}

void runstate_reset_sequence(RunState *s, const Model *m) {
    if (s->moe_prev && m->w.n_exp > 0)
        memset(s->moe_prev, 0, (size_t)m->hdr.n_layer * m->w.n_exp * sizeof(int));
}

/* Single-token MoE FFN for the KV-cached path. Identical maths and the same
   top-k scan as moe_ffn() — the only difference is that the locality set lives
   in RunState because it must persist across calls. Cleared at EOS. */

void forward_step(const Model *m, RunState *s, int token, int pos) {
    const int D = m->hdr.d_model, L = m->hdr.n_layer;
    const int F = m->hdr.d_ff, V = m->hdr.vocab;
    const Weights *w = &m->w;

    instr_touch("token_emb", -1, pos, D * sizeof(float));
    instr_touch("pos_emb", -1, pos, D * sizeof(float));
    for (int e = 0; e < D; e++)
        s->x[e] = w->token_emb[token * D + e] + w->pos_emb[pos * D + e];

    for (int l = 0; l < L; l++) {
        const float *ln2_g = w->ln2_g + l * D, *ln2_b = w->ln2_b + l * D;

        blk_attention_mha_step(m, s, l, pos, token);

        instr_touch("ln2", l, pos, 2 * D * sizeof(float));
        layernorm(s->xn, s->x, ln2_g, ln2_b, D);
        if (moe_layer_at(m, l)) {
            blk_moe_ffn_step(m, s, m->w.moe_slot[l], l, s->xn, s->x, token, pos);
        } else {
            /* instr_touch stays here: the step path knows `pos`, the block does not */
            instr_touch("W1", l, pos, (D * F + F) * sizeof(float));
            instr_touch("W2", l, pos, (F * D + D) * sizeof(float));
            blk_ffn_dense_step(m, s, l);
        }
    }

    instr_touch("lnf", -1, pos, 2 * D * sizeof(float));
    instr_touch("unemb", -1, pos, (size_t)V * D * sizeof(float));
    layernorm(s->xn, s->x, w->lnf_g, w->lnf_b, D);
    matmul_wx(s->logits, s->xn, w->token_emb, D, V); /* tied unemb */

    pager_enforce(); /* evict LRU regions if over CP_BUDGET (no-op otherwise) */
}
