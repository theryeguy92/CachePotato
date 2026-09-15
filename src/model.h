#ifndef MODEL_H
#define MODEL_H

#include "weights.h"

/* Full-sequence forward pass (prompt processing / verification mode).
   Fills logits_out: (T, vocab) row-major; row t scores what follows token t.
   Mirrors experiments/forward_by_hand.py exactly — same op order, same mask.
   This is the reference implementation everything else gets verified against. */
void forward(const Model *m, const int *tokens, int T, float *logits_out);

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
