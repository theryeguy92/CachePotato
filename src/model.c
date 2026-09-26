#include "model.h"
#include "instr.h"
#include "kernels.h"
#include "pager.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EOS_TOKEN 50256 /* GPT-2 <|endoftext|>; also resets the locality set */

/* --- MoE FFN (CPW1 v3) -----------------------------------------------------
   router -> +lambda locality bias -> softmax -> top-k, then one shared expert
   (always active, hot by architecture) plus the k routed experts. The locality
   set is the previous token's selection at THIS layer and is CLEARED at every
   EOS: cross-document stickiness is meaningless (spec decision 2026-09-15).
   Mirrors train_moe.py's moe_ffn and the bridge-verified src/train.c: the top-k
   scan below is the same strict-> form, so lower id wins ties. */
static void moe_ffn(const Model *m, int slot, int layer, const float *xn, float *x,
                    const int *tokens, int T, float *h, float *proj,
                    int *sel_out, float lambda) {
    const int D = m->hdr.d_model, N = m->w.n_exp, EH = m->w.exp_hid, K = m->w.topk;
    const Weights *w = &m->w;
    const float *Wr = w->Wr + (size_t)slot * D * N;
    const float *br = w->br + (size_t)slot * N;
    const float *W1s = w->W1s + (size_t)slot * D * D;
    const float *b1s = w->b1s + (size_t)slot * D;
    const float *W2s = w->W2s + (size_t)slot * D * D;
    const float *b2s = w->b2s + (size_t)slot * D;

    float *rlog = malloc(N * sizeof(float));
    float *p = malloc(N * sizeof(float));
    int *prev = calloc(N, sizeof(int));
    int *idx = malloc(K * sizeof(int));
    float *wt = malloc(K * sizeof(float));
    if (!rlog || !p || !prev || !idx || !wt) { fprintf(stderr, "moe_ffn: oom\n"); exit(1); }

    for (int t = 0; t < T; t++) {
        const float *xt = xn + (size_t)t * D;
        float *out = x + (size_t)t * D;

        matmul_xw(rlog, xt, Wr, br, D, N);
        for (int e = 0; e < N; e++) p[e] = rlog[e] + (prev[e] ? lambda : 0.0f);
        softmax(p, N);

        int a = 0, b = 1;
        if (p[b] > p[a]) { a = 1; b = 0; }
        for (int e = 2; e < N; e++)
            if (p[e] > p[a]) { b = a; a = e; }
            else if (p[e] > p[b]) b = e;
        idx[0] = a; idx[1] = b;
        float S = p[a] + p[b];
        wt[0] = p[a] / S; wt[1] = p[b] / S;

        /* pager: demand-touch these experts, record them, and prefetch the
           previous W tokens' sets (issued before the expert matmuls so the
           WILLNEED hint precedes the use) */
        instr_touch("Wr", layer, t, ((size_t)D * N + N) * sizeof(float));
        instr_touch("W1s", layer, t, 2 * ((size_t)D * D) * sizeof(float));
        pager_moe_access(slot, idx, K, t);

        /* shared expert */
        matmul_xw(h, xt, W1s, b1s, D, D);
        for (int e = 0; e < D; e++) h[e] = gelu(h[e]);
        matmul_xw(proj, h, W2s, b2s, D, D);
        for (int e = 0; e < D; e++) out[e] += proj[e];

        /* routed experts: one contiguous block per expert = one pager region */
        for (int s = 0; s < K; s++) {
            float *blk = expert_block(w, slot, idx[s]);
            const float *b1e = blk + (size_t)D * EH;
            const float *W2e = b1e + EH;
            const float *b2e = W2e + (size_t)EH * D;
            matmul_xw(h, xt, blk, b1e, D, EH);
            for (int e = 0; e < EH; e++) h[e] = gelu(h[e]);
            matmul_xw(proj, h, W2e, b2e, EH, D);
            for (int e = 0; e < D; e++) out[e] += wt[s] * proj[e];
        }

        if (sel_out)
            for (int s = 0; s < K; s++)
                sel_out[((size_t)slot * T + t) * K + s] = idx[s];

        /* locality set for the NEXT token; cleared at EOS */
        memset(prev, 0, N * sizeof(int));
        if (tokens[t] != EOS_TOKEN)
            for (int s = 0; s < K; s++) prev[idx[s]] = 1;
    }
    free(rlog); free(p); free(prev); free(idx); free(wt);
}

void forward_moe(const Model *m, const int *tokens, int T, float *logits_out,
                 int *sel_out, float lambda) {
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
    for (int t = 0; t < T; t++) {
        instr_touch("token_emb", -1, t, D * sizeof(float));
        instr_touch("pos_emb", -1, t, D * sizeof(float));
        for (int e = 0; e < D; e++)
            x[t * D + e] = w->token_emb[tokens[t] * D + e] + w->pos_emb[t * D + e];
    }

    for (int l = 0; l < L; l++) {
        const float *ln1_g = w->ln1_g + l * D, *ln1_b = w->ln1_b + l * D;
        const float *Wq = w->Wq + l * D * D, *bq = w->bq + l * D;
        const float *Wk = w->Wk + l * D * D, *bk = w->bk + l * D;
        const float *Wv = w->Wv + l * D * D, *bv = w->bv + l * D;
        const float *Wo = w->Wo + l * D * D, *bo = w->bo + l * D;
        const float *ln2_g = w->ln2_g + l * D, *ln2_b = w->ln2_b + l * D;

        /* attention half: norm -> qkv -> per-head masked attention -> Wo -> +res */
        for (int t = 0; t < T; t++) {
            instr_touch("ln1", l, t, 2 * D * sizeof(float));
            layernorm(xn + t * D, x + t * D, ln1_g, ln1_b, D);
        }
        for (int t = 0; t < T; t++) {
            instr_touch("Wq", l, t, (D * D + D) * sizeof(float));
            instr_touch("Wk", l, t, (D * D + D) * sizeof(float));
            instr_touch("Wv", l, t, (D * D + D) * sizeof(float));
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
            instr_touch("Wo", l, t, (D * D + D) * sizeof(float));
            matmul_xw(proj, ao + t * D, Wo, bo, D, D);
            for (int e = 0; e < D; e++) x[t * D + e] += proj[e];
        }

        /* FFN half: dense on the outer layers, sparse (MoE) in the middle */
        for (int t = 0; t < T; t++) {
            instr_touch("ln2", l, t, 2 * D * sizeof(float));
            layernorm(xn + t * D, x + t * D, ln2_g, ln2_b, D);
        }
        if (moe_layer_at(m, l)) {
            moe_ffn(m, m->w.moe_slot[l], l, xn, x, tokens, T, h, proj, sel_out, lambda);
        } else {
            /* dense slots are compacted in v3 (dead dense FFNs are not in the
               file), so layer -> slot; v2 has one dense FFN per layer */
            const int ds = m->is_v3 ? m->w.dense_slot[l] : l;
            const float *W1 = w->W1 + (size_t)ds * D * F, *b1 = w->b1 + (size_t)ds * F;
            const float *W2 = w->W2 + (size_t)ds * F * D, *b2 = w->b2 + (size_t)ds * D;
            for (int t = 0; t < T; t++) {
                instr_touch("W1", l, t, (D * F + F) * sizeof(float));
                instr_touch("W2", l, t, (F * D + D) * sizeof(float));
                matmul_xw(h + t * F, xn + t * D, W1, b1, D, F);
                for (int e = 0; e < F; e++) h[t * F + e] = gelu(h[t * F + e]);
                matmul_xw(proj, h + t * F, W2, b2, F, D);
                for (int e = 0; e < D; e++) x[t * D + e] += proj[e];
            }
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
static void moe_ffn_step(const Model *m, RunState *s, int slot, int layer,
                         const float *xn, float *x, int token, int pos) {
    const int D = m->hdr.d_model, N = m->w.n_exp, EH = m->w.exp_hid, K = m->w.topk;
    const Weights *w = &m->w;
    const float *Wr = w->Wr + (size_t)slot * D * N;
    const float *br = w->br + (size_t)slot * N;
    const float *W1s = w->W1s + (size_t)slot * D * D;
    const float *b1s = w->b1s + (size_t)slot * D;
    const float *W2s = w->W2s + (size_t)slot * D * D;
    const float *b2s = w->b2s + (size_t)slot * D;
    int *prev = s->moe_prev + (size_t)layer * N;
    float *p = s->moe_p;

    matmul_xw(p, xn, Wr, br, D, N);
    const float lam = m->h3.lam;
    for (int e = 0; e < N; e++) p[e] += prev[e] ? lam : 0.0f;
    softmax(p, N);

    /* same strict-> scan as moe_ffn(): lower id wins ties, slot 0 = larger */
    int a = 0, b = 1;
    if (p[b] > p[a]) { a = 1; b = 0; }
    for (int e = 2; e < N; e++)
        if (p[e] > p[a]) { b = a; a = e; }
        else if (p[e] > p[b]) b = e;
    s->moe_idx[0] = a; s->moe_idx[1] = b;
    float S = p[a] + p[b];
    s->moe_wt[0] = p[a] / S; s->moe_wt[1] = p[b] / S;

    instr_touch("Wr", layer, pos, ((size_t)D * N + N) * sizeof(float));
    instr_touch("W1s", layer, pos, 2 * ((size_t)D * D) * sizeof(float));
    pager_moe_access(slot, s->moe_idx, K, pos);

    /* shared expert */
    matmul_xw(s->h, xn, W1s, b1s, D, D);
    for (int e = 0; e < D; e++) s->h[e] = gelu(s->h[e]);
    matmul_xw(s->proj, s->h, W2s, b2s, D, D);
    for (int e = 0; e < D; e++) x[e] += s->proj[e];

    /* routed experts */
    for (int k = 0; k < K; k++) {
        float *blk = expert_block(w, slot, s->moe_idx[k]);
        const float *b1e = blk + (size_t)D * EH;
        const float *W2e = b1e + EH;
        const float *b2e = W2e + (size_t)EH * D;
        matmul_xw(s->h, xn, blk, b1e, D, EH);
        for (int e = 0; e < EH; e++) s->h[e] = gelu(s->h[e]);
        matmul_xw(s->proj, s->h, W2e, b2e, EH, D);
        for (int e = 0; e < D; e++) x[e] += s->moe_wt[k] * s->proj[e];
    }

    /* locality set for the NEXT token; cleared at EOS */
    memset(prev, 0, N * sizeof(int));
    if (token != EOS_TOKEN)
        for (int k = 0; k < K; k++) prev[s->moe_idx[k]] = 1;

    if (s->moe_sel_out)
        for (int k = 0; k < K; k++)
            s->moe_sel_out[((size_t)slot * s->moe_sel_stride + pos) * K + k] =
                s->moe_idx[k];
}

void forward_step(const Model *m, RunState *s, int token, int pos) {
    const int D = m->hdr.d_model, H = m->hdr.n_heads, L = m->hdr.n_layer;
    const int F = m->hdr.d_ff, V = m->hdr.vocab, C = m->hdr.n_ctx;
    const int HD = D / H;
    const Weights *w = &m->w;

    instr_touch("token_emb", -1, pos, D * sizeof(float));
    instr_touch("pos_emb", -1, pos, D * sizeof(float));
    for (int e = 0; e < D; e++)
        s->x[e] = w->token_emb[token * D + e] + w->pos_emb[pos * D + e];

    for (int l = 0; l < L; l++) {
        const float *ln1_g = w->ln1_g + l * D, *ln1_b = w->ln1_b + l * D;
        const float *Wq = w->Wq + l * D * D, *bq = w->bq + l * D;
        const float *Wk = w->Wk + l * D * D, *bk = w->bk + l * D;
        const float *Wv = w->Wv + l * D * D, *bv = w->bv + l * D;
        const float *Wo = w->Wo + l * D * D, *bo = w->bo + l * D;
        const float *ln2_g = w->ln2_g + l * D, *ln2_b = w->ln2_b + l * D;
        float *kc = s->key_cache + (size_t)l * C * D;
        float *vc = s->value_cache + (size_t)l * C * D;

        instr_touch("ln1", l, pos, 2 * D * sizeof(float));
        layernorm(s->xn, s->x, ln1_g, ln1_b, D);

        /* q into s->q; k,v straight into this layer's cache at pos */
        instr_touch("Wq", l, pos, (D * D + D) * sizeof(float));
        instr_touch("Wk", l, pos, (D * D + D) * sizeof(float));
        instr_touch("Wv", l, pos, (D * D + D) * sizeof(float));
        matmul_xw(s->q, s->xn, Wq, bq, D, D);
        matmul_xw(kc + pos * D, s->xn, Wk, bk, D, D);
        matmul_xw(vc + pos * D, s->xn, Wv, bv, D, D);

        for (int hh = 0; hh < H; hh++) {
            float scale = 1.0f / sqrtf((float)HD);
            const float *qh = s->q + hh * HD;
            for (int j = 0; j <= pos; j++) {
                float sc = 0.0f;
                const float *kj = kc + j * D + hh * HD;
                for (int e = 0; e < HD; e++) sc += qh[e] * kj[e];
                s->att[j] = sc * scale;
            }
            softmax(s->att, pos + 1);
            float *ao = s->ao + hh * HD;
            for (int e = 0; e < HD; e++) {
                float acc = 0.0f;
                for (int j = 0; j <= pos; j++)
                    acc += s->att[j] * vc[j * D + hh * HD + e];
                ao[e] = acc;
            }
        }
        instr_touch("Wo", l, pos, (D * D + D) * sizeof(float));
        matmul_xw(s->proj, s->ao, Wo, bo, D, D);
        for (int e = 0; e < D; e++) s->x[e] += s->proj[e];

        instr_touch("ln2", l, pos, 2 * D * sizeof(float));
        layernorm(s->xn, s->x, ln2_g, ln2_b, D);
        if (moe_layer_at(m, l)) {
            moe_ffn_step(m, s, m->w.moe_slot[l], l, s->xn, s->x, token, pos);
        } else {
            /* dense slots are compacted in v3 (see forward_moe) */
            const int ds = m->is_v3 ? m->w.dense_slot[l] : l;
            const float *W1 = w->W1 + (size_t)ds * D * F, *b1 = w->b1 + (size_t)ds * F;
            const float *W2 = w->W2 + (size_t)ds * F * D, *b2 = w->b2 + (size_t)ds * D;
            instr_touch("W1", l, pos, (D * F + F) * sizeof(float));
            instr_touch("W2", l, pos, (F * D + D) * sizeof(float));
            matmul_xw(s->h, s->xn, W1, b1, D, F);
            for (int e = 0; e < F; e++) s->h[e] = gelu(s->h[e]);
            matmul_xw(s->proj, s->h, W2, b2, F, D);
            for (int e = 0; e < D; e++) s->x[e] += s->proj[e];
        }
    }

    instr_touch("lnf", -1, pos, 2 * D * sizeof(float));
    instr_touch("unemb", -1, pos, (size_t)V * D * sizeof(float));
    layernorm(s->xn, s->x, w->lnf_g, w->lnf_b, D);
    matmul_wx(s->logits, s->xn, w->token_emb, D, V); /* tied unemb */

    pager_enforce(); /* evict LRU regions if over CP_BUDGET (no-op otherwise) */
}
