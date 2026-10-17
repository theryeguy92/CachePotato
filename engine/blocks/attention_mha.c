#include "blocks/blocks.h"
#include "core/instr.h"
#include "core/model.h"
#include "kernels/kernels.h"

#include <math.h>

/* Attention (multi-head, causal). The maths and operation ORDER are exactly what
 * engine/core/model.c did inline, so this is a move, not a redesign — bit-identical output
 * is what the gate checks.
 *
 * Two arities exist because the two forward paths differ: the prompt path computes q/k/v for
 * all T positions and then does a per-head masked pass over the block, while the incremental
 * path computes one position and attends against the layer's KV cache. They share the maths
 * but not the memory, so they are separate blocks rather than one with a flag.
 */

void blk_attention_mha(const Model *m, int layer, int T,
                       const float *x, float *xn, float *q, float *k, float *vv,
                       float *ao, float *att, float *proj, float *out_x) {
    const int D = m->hdr.d_model, H = m->hdr.n_heads;
    const int HD = D / H;
    const Weights *w = &m->w;

    const float *ln1_g = w->ln1_g + layer * D, *ln1_b = w->ln1_b + layer * D;
    const float *Wq = w->Wq + (size_t)layer * D * D, *bq = w->bq + layer * D;
    const float *Wk = w->Wk + (size_t)layer * D * D, *bk = w->bk + layer * D;
    const float *Wv = w->Wv + (size_t)layer * D * D, *bv = w->bv + layer * D;
    const float *Wo = w->Wo + (size_t)layer * D * D, *bo = w->bo + layer * D;

    for (int t = 0; t < T; t++) {
        instr_touch("ln1", layer, t, 2 * D * sizeof(float));
        layernorm(xn + t * D, x + t * D, ln1_g, ln1_b, D);
    }
    for (int t = 0; t < T; t++) {
        instr_touch("Wq", layer, t, (D * D + D) * sizeof(float));
        instr_touch("Wk", layer, t, (D * D + D) * sizeof(float));
        instr_touch("Wv", layer, t, (D * D + D) * sizeof(float));
        matmul_xw_auto(q  + t * D, xn + t * D, Wq, q8_slice(w->q_Wq, layer, (long)D*D), bq, D, D);
        matmul_xw_auto(k  + t * D, xn + t * D, Wk, q8_slice(w->q_Wk, layer, (long)D*D), bk, D, D);
        matmul_xw_auto(vv + t * D, xn + t * D, Wv, q8_slice(w->q_Wv, layer, (long)D*D), bv, D, D);
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
        instr_touch("Wo", layer, t, (D * D + D) * sizeof(float));
        matmul_xw_auto(proj, ao + t * D, Wo, q8_slice(w->q_Wo, layer, (long)D*D), bo, D, D);
        for (int e = 0; e < D; e++) out_x[t * D + e] += proj[e];
    }
}

void blk_attention_mha_step(const Model *m, RunState *s, int layer, int pos, int token) {
    (void)token;
    const int D = m->hdr.d_model, H = m->hdr.n_heads, C = m->hdr.n_ctx;
    const int HD = D / H;
    const Weights *w = &m->w;
    const float *ln1_g = w->ln1_g + layer * D, *ln1_b = w->ln1_b + layer * D;
    const float *Wq = w->Wq + (size_t)layer * D * D, *bq = w->bq + layer * D;
    const float *Wk = w->Wk + (size_t)layer * D * D, *bk = w->bk + layer * D;
    const float *Wv = w->Wv + (size_t)layer * D * D, *bv = w->bv + layer * D;
    const float *Wo = w->Wo + (size_t)layer * D * D, *bo = w->bo + layer * D;
    float *kc = s->key_cache + (size_t)layer * C * D;
    float *vc = s->value_cache + (size_t)layer * C * D;

    instr_touch("ln1", layer, pos, 2 * D * sizeof(float));
    layernorm(s->xn, s->x, ln1_g, ln1_b, D);

    /* q into s->q; k,v straight into this layer's cache at pos */
    instr_touch("Wq", layer, pos, (D * D + D) * sizeof(float));
    instr_touch("Wk", layer, pos, (D * D + D) * sizeof(float));
    instr_touch("Wv", layer, pos, (D * D + D) * sizeof(float));
    matmul_xw_auto(s->q, s->xn, Wq, q8_slice(w->q_Wq, layer, (long)D*D), bq, D, D);
    matmul_xw_auto(kc + pos * D, s->xn, Wk, q8_slice(w->q_Wk, layer, (long)D*D), bk, D, D);
    matmul_xw_auto(vc + pos * D, s->xn, Wv, q8_slice(w->q_Wv, layer, (long)D*D), bv, D, D);

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
    instr_touch("Wo", layer, pos, (D * D + D) * sizeof(float));
    matmul_xw_auto(s->proj, s->ao, Wo, q8_slice(w->q_Wo, layer, (long)D*D), bo, D, D);
    for (int e = 0; e < D; e++) s->x[e] += s->proj[e];
}
