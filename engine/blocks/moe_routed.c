#include "blocks/blocks.h"
#include "core/instr.h"
#include "core/model.h"
#include "pager/pager.h"
#include "kernels/kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EOS_TOKEN CP_EOS_TOKEN

/* Routed MoE: router -> lambda locality bias -> top-k -> shared + k routed experts.
 *
 * Moved verbatim from engine/core/model.c: same strict `>` top-k scan (lower id wins ties),
 * same order of shared-then-routed accumulation, same pager hook position, same routing
 * trace write. Bit-identical output is what the gate checks.
 *
 * This is the block a port is most likely to replace, and the one that carries the engine's
 * memory behaviour: pager_moe_access() is called here because the block is what knows which
 * experts this token selected, and the pager's prefetch prediction is built from that.
 */

void blk_moe_ffn(const Model *m, int slot, int layer, const float *xn, float *x,
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

        matmul_xw_auto(rlog, xt, Wr, q8_slice(w->q_Wr, slot, (long)D*N), br, D, N);
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
        matmul_xw_auto(h, xt, W1s, q8_slice(w->q_W1s, slot, (long)D*D), b1s, D, D);
        for (int e = 0; e < D; e++) h[e] = gelu(h[e]);
        matmul_xw_auto(proj, h, W2s, q8_slice(w->q_W2s, slot, (long)D*D), b2s, D, D);
        for (int e = 0; e < D; e++) out[e] += proj[e];

        /* routed experts: one contiguous block per expert = one pager region */
        for (int s = 0; s < K; s++) {
            if (w->experts_q8) {
                /* quantized: stream the blocks, never materialise fp32 weights */
                const unsigned char *W1q = expert_block_q8(w, slot, idx[s]);
                const float *b1e = expert_bias_ptr(w, slot, idx[s]);
                const float *b2e = b1e + EH;
                const unsigned char *W2q = W1q + q8_bytes_for((long)D * EH);
                matmul_xw_q8(h, xt, W1q, b1e, D, EH);
                for (int e = 0; e < EH; e++) h[e] = gelu(h[e]);
                matmul_xw_q8(proj, h, W2q, b2e, EH, D);
            } else {
                float *blk = expert_block(w, slot, idx[s]);
                const float *b1e = blk + (size_t)D * EH;
                const float *W2e = b1e + EH;
                const float *b2e = W2e + (size_t)EH * D;
                matmul_xw(h, xt, blk, b1e, D, EH);
                for (int e = 0; e < EH; e++) h[e] = gelu(h[e]);
                matmul_xw(proj, h, W2e, b2e, EH, D);
            }
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

void blk_moe_ffn_step(const Model *m, RunState *s, int slot, int layer,
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

    matmul_xw_auto(p, xn, Wr, q8_slice(w->q_Wr, slot, (long)D*N), br, D, N);
    const float lam = s->lam;
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
    matmul_xw_auto(s->h, xn, W1s, q8_slice(w->q_W1s, slot, (long)D*D), b1s, D, D);
    for (int e = 0; e < D; e++) s->h[e] = gelu(s->h[e]);
    matmul_xw_auto(s->proj, s->h, W2s, q8_slice(w->q_W2s, slot, (long)D*D), b2s, D, D);
    for (int e = 0; e < D; e++) x[e] += s->proj[e];

    /* routed experts */
    for (int k = 0; k < K; k++) {
        if (w->experts_q8) {
            const unsigned char *W1q = expert_block_q8(w, slot, s->moe_idx[k]);
            const float *b1e = expert_bias_ptr(w, slot, s->moe_idx[k]);
            const float *b2e = b1e + EH;
            const unsigned char *W2q = W1q + q8_bytes_for((long)D * EH);
            matmul_xw_q8(s->h, xn, W1q, b1e, D, EH);
            for (int e = 0; e < EH; e++) s->h[e] = gelu(s->h[e]);
            matmul_xw_q8(s->proj, s->h, W2q, b2e, EH, D);
        } else {
            float *blk = expert_block(w, slot, s->moe_idx[k]);
            const float *b1e = blk + (size_t)D * EH;
            const float *W2e = b1e + EH;
            const float *b2e = W2e + (size_t)EH * D;
            matmul_xw(s->h, xn, blk, b1e, D, EH);
            for (int e = 0; e < EH; e++) s->h[e] = gelu(s->h[e]);
            matmul_xw(s->proj, s->h, W2e, b2e, EH, D);
        }
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
