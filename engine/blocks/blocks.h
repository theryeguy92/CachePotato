#ifndef BLOCKS_H
#define BLOCKS_H

/* Blocks depend on core (the allowed direction). The structs in weights.h/model.h are
   anonymous typedefs, so forward-declaring them here would create a DIFFERENT `struct
   Model` and conflict; including the real headers is both simpler and correct. core never
   includes blocks, so there is no cycle. */
#include "core/model.h"
#include "core/weights.h"

/* Reusable layer blocks — the architecture half of the engine.
 *
 * `engine/kernels/` is driven by how weights are ENCODED (f32 today; quantized later).
 * This directory is driven by how a model is BUILT: attention variants, dense FFN, routed
 * MoE. The distinction matters because a block must never learn its own dtype — if it does,
 * every new encoding touches every block and the engine stops being modular.
 *
 * This is a MOVE, not a redesign: the maths and the operation ORDER are exactly what
 * engine/core/model.c contained inline, so the refactor is bit-identical (verified by
 * comparing routing, logits and pager byte counts against a captured baseline). The
 * meaningful change is that a port can now add a block without touching the forward-pass
 * orchestration, and that model.c is about *composition* rather than arithmetic.
 *
 * Signatures carry what the existing blocks need and nothing more. The MoE block takes the
 * Model because expert geometry, the pager hook and the routing trace are all model-level;
 * attention and dense FFN need only the weights and the layer index. Widening a signature
 * is cheap; guessing at a block's future requirements is not.
 */

/* MHA over T positions, full-sequence (prompt processing). Writes into x (residual). */
void blk_attention_mha(const Model *m, int layer, int T,
                       const float *x, float *xn, float *q, float *k, float *vv,
                       float *ao, float *att, float *proj, float *out_x);

/* MHA, one token at a time against the KV cache in RunState. */
void blk_attention_mha_step(const Model *m, RunState *s, int layer, int pos, int token);

/* Dense FFN (gelu), full-sequence. */
void blk_ffn_dense(const Model *m, int layer, int T,
                   const float *xn, float *x, float *h, float *proj);

/* Dense FFN, one token. */
void blk_ffn_dense_step(const Model *m, RunState *s, int layer);

/* Routed MoE (router -> top-k -> shared + k routed experts). These keep their original
   names and signatures: they are the blocks a port is most likely to replace, and changing
   their call shape now would be a redesign rather than a move. */
void blk_moe_ffn(const Model *m, int slot, int layer, const float *xn, float *x,
                 const int *tokens, int T, float *h, float *proj,
                 int *sel_out, float lambda);
void blk_moe_ffn_step(const Model *m, RunState *s, int slot, int layer,
                      const float *xn, float *x, int token, int pos);

#endif
