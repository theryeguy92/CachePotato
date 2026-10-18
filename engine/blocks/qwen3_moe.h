#ifndef QWEN3_MOE_H
#define QWEN3_MOE_H
#include <stddef.h>

/* Qwen3-30B-A3B: one decoder layer. Three things here are new to this engine:
 *   QK-norm   per-head RMSNorm on q (n_heads) and k (n_kv) BEFORE RoPE
 *   top-k     generalised routing (8 of 128), softmax + norm_topk_prob renormalisation
 *   experts   gated SwiGLU per expert, no shared expert — every FFN read is an expert read
 * Every weight is (out, in) row-major, which is `matmul_wx`. No transposes anywhere.
 */
typedef struct {
    int d_model, n_heads, n_kv, head_dim, d_exp, n_experts, topk, vocab;
    float rope_theta, rms_eps;
    const float *ln1, *ln2;                 /* (d_model) */
    const float *Wq, *Wk, *Wv, *Wo;         /* (n_heads*hd, d) (n_kv*hd, d) (d, n_heads*hd) */
    const float *q_norm, *k_norm;           /* (head_dim) */
    const float *Wgate;                     /* (n_experts, d_model) */
    const float *Wg, *Wu, *Wd;              /* per expert: (d_exp, d) (d_exp, d) (d, d_exp) */
} QW3Layer;

typedef struct {
    float *attn;        /* (T, d_model)  pre-residual attention output */
    float *mlp;         /* (T, d_model)  pre-residual MoE output */
    float *router;      /* (T, n_experts) router logits */
    int   *topk_idx;    /* (T, topk) selected experts, descending probability */
} QW3Out;

int qwen3_layer_alloc(QW3Out *o, const QW3Layer *m, int T);
void qwen3_layer_free(QW3Out *o);
int qwen3_layer(const QW3Layer *m, const float *x, int T, QW3Out *o);
int qwen3_load_blob(QW3Layer *m, const char *path, float **owned);
#endif
