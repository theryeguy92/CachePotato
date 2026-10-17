#ifndef MODEL_H
#define MODEL_H

#include "weights.h"

/* End-of-sequence token: resets the locality set, so the engine needs it for its own state
   and a block needs it for the same reason. One definition, in the header, so a port with a
   different EOS can change it once. */
#define CP_EOS_TOKEN 50256

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
   follows `token` at position `pos`.

   moe_prev is the engine's first piece of SEQUENCE state beyond the KV cache:
   the per-layer locality set, which must persist across forward_step calls.
   It is cleared by runstate_reset_sequence() at the start of a sequence and by
   the EOS branch inside forward_step — never per call, or λ becomes a no-op. */
typedef struct {
    float *x, *xn, *q, *ao, *att, *h, *proj, *logits;
    float *key_cache;   /* (n_layer, n_ctx, d_model) */
    float *value_cache; /* (n_layer, n_ctx, d_model) */
    int   *moe_prev;    /* (n_layer, n_exp) locality set; NULL for dense files */
    float *moe_p;       /* (n_exp) router probs scratch */
    int   *moe_idx;     /* (topk)  selected expert ids */
    float *moe_wt;      /* (topk)  selection weights */
    /* Optional routing trace for Gate 1: (n_moe, moe_sel_stride, topk) row-major,
       written by forward_step when non-NULL. NULL in normal generation. */
    int   *moe_sel_out;
    int    moe_sel_stride;
    /* The locality bias actually applied by forward_step. The full-sequence path takes it
       as an argument; the incremental path used to read m->h3.lam directly, which is 0 for
       a v4 file (v4 has no home for lambda yet). Gate 1 then compared a lambda=0.5
       full-sequence pass against a lambda=0 incremental one and reported a divergence that
       was not there. One field, so both paths are driven by the same value. */
    float  lam;
} RunState;

void runstate_alloc(RunState *s, const Model *m);
void runstate_free(RunState *s, const Model *m);
/* Clear the locality set. Call at the start of each new sequence: the KV cache
   needs no clearing because attention only ever reads j <= pos. */
void runstate_reset_sequence(RunState *s, const Model *m);
void forward_step(const Model *m, RunState *s, int token, int pos);

#endif
