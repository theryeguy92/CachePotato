#ifndef MODEL_H
#define MODEL_H

#include "weights.h"

/* Full-sequence forward pass (prompt processing / verification mode).
   Fills logits_out: (T, vocab) row-major; row t scores what follows token t.
   Mirrors experiments/forward_by_hand.py exactly — same op order, same mask.
   This is the reference implementation everything else gets verified against. */
void forward(const Model *m, const int *tokens, int T, float *logits_out);

/* Same forward pass, but for CPW1 v3/MoE files it also applies the locality
   bias `lambda` and, when sel_out != NULL, records the per-token top-k expert
   selections as (n_moe_layers, T, topk) row-major — the routing gate's output.
   Pass m->h3.lam for lambda to reproduce what the model was trained with. */
void forward_moe(const Model *m, const int *tokens, int T, float *logits_out,
                 int *sel_out, float lambda);

/* Incremental generation mode: one token per call, K/V cached per layer.
   logits lands in s->logits (vocab floats) — the distribution over what
   follows `token` at position `pos`. */
typedef struct {
    float *x, *xn, *q, *ao, *att, *h, *proj, *logits;
    float *key_cache;   /* (n_layer, n_ctx, d_model) */
    float *value_cache; /* (n_layer, n_ctx, d_model) */
} RunState;

void runstate_alloc(RunState *s, const Model *m);
void runstate_free(RunState *s, const Model *m);
void forward_step(const Model *m, RunState *s, int token, int pos);

#endif
