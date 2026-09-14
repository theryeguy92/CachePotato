#ifndef MODEL_H
#define MODEL_H

#include "weights.h"

/* Full-sequence forward pass (prompt processing / verification mode).
   Fills logits_out: (T, vocab) row-major; row t scores what follows token t.
   Mirrors experiments/forward_by_hand.py exactly — same op order, same mask.
   KV-cached one-token-at-a-time mode comes with the REPL; this is the
   reference implementation everything else gets verified against. */
void forward(const Model *m, const int *tokens, int T, float *logits_out);

#endif
