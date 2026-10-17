#ifndef WEIGHTS_V4_H
#define WEIGHTS_V4_H

#include "weights.h"

/* CPW1 v4 loader — see format/SPEC.md.
 *
 * This is the model-agnostic path: it walks the file's own encoding/tensor/block/region
 * tables instead of relying on compiled-in shape constants. It exists to be the other half
 * of the format's acceptance test (§9.2): the model the project already has must load from
 * v4 and produce the same routing and logits as the v3 it was written from.
 *
 * SCOPE OF THIS FIRST VERSION, stated so the limits are not mistaken for completeness:
 *   - encoding: raw f32 only. Quantized encodings are refused loudly (the spec defines
 *     them, but no decoder exists yet, and a wrong decoder produces confident nonsense).
 *   - block kinds: embedding, norm, attention (MHA, head_dim == d_model/n_heads,
 *     n_kv_heads == n_heads), dense_ffn (gelu_tanh), moe (softmax, latent_size == 0,
 *     1 shared expert, no group-limited routing). Every other kind or parameter combination
 *     is REFUSED rather than approximated.
 *   - expert arrangement: the spec's "split" form (one tensor per projection) is refused;
 *     "interleaved" per-expert tensors are accepted and copied into one contiguous
 *     layer-major buffer, because the existing forward pass indexes experts by
 *     (slot, expert) stride.
 *
 * REGION VERIFICATION is done at load: every region's CRC-32 is checked against the
 * payload. That is what makes the format's "misalignment fails loudly" axiom real.
 *
 * Returns 0 on success, -1 with a message on stderr. On success m->v4_scratch owns the
 * routed-expert copy and must be freed by model_free().
 */
int model_load_v4(Model *m, const char *path);

/* Exposed for the gate and for callers that want the tables without loading weights. */
typedef struct {
    uint32_t n_tensors, n_blocks, n_regions, n_encodings;
    uint64_t off_encodings, off_tensors, off_blocks, off_regions, off_data;
} V4Header;

int v4_read_header(int fd, V4Header *h);

#endif
