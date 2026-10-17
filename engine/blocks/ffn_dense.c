#include "blocks/blocks.h"
#include "core/instr.h"
#include "core/model.h"
#include "kernels/kernels.h"

/* Dense FFN (gelu). Moved verbatim from engine/core/model.c — same op order, same
 * instr_touch calls, so bit-identical output.
 *
 * The slot indirection is the one subtlety: a v2/v3/v4 file stores only the dense FFNs that
 * are actually executed, so a layer maps to a compacted dense slot when the file is a MoE
 * layout. A dense v2 file has one dense FFN per layer and no slot table.
 */

void blk_ffn_dense(const Model *m, int layer, int T,
                   const float *xn, float *x, float *h, float *proj) {
    const int D = m->hdr.d_model, F = m->hdr.d_ff;
    const Weights *w = &m->w;
    const int ds = w->is_moe ? w->dense_slot[layer] : layer;
    const float *W1 = w->W1 + (size_t)ds * D * F, *b1 = w->b1 + (size_t)ds * F;
    const float *W2 = w->W2 + (size_t)ds * F * D, *b2 = w->b2 + (size_t)ds * D;
    for (int t = 0; t < T; t++) {
        instr_touch("W1", layer, t, (D * F + F) * sizeof(float));
        instr_touch("W2", layer, t, (F * D + D) * sizeof(float));
        matmul_xw(h + t * F, xn + t * D, W1, b1, D, F);
        for (int e = 0; e < F; e++) h[t * F + e] = gelu(h[t * F + e]);
        matmul_xw(proj, h + t * F, W2, b2, F, D);
        for (int e = 0; e < D; e++) x[t * D + e] += proj[e];
    }
}

void blk_ffn_dense_step(const Model *m, RunState *s, int layer) {
    const int D = m->hdr.d_model, F = m->hdr.d_ff;
    const Weights *w = &m->w;
    const int ds = w->is_moe ? w->dense_slot[layer] : layer;
    const float *W1 = w->W1 + (size_t)ds * D * F, *b1 = w->b1 + (size_t)ds * F;
    const float *W2 = w->W2 + (size_t)ds * F * D, *b2 = w->b2 + (size_t)ds * D;
    /* instr_touch happens in the caller's loop position: forward_step uses `pos`, which the
       step path knows and the block does not (it has no position argument). Kept there so
       the touch order is unchanged. */
    matmul_xw(s->h, s->xn, W1, b1, D, F);
    for (int e = 0; e < F; e++) s->h[e] = gelu(s->h[e]);
    matmul_xw(s->proj, s->h, W2, b2, F, D);
    for (int e = 0; e < D; e++) s->x[e] += s->proj[e];
}
