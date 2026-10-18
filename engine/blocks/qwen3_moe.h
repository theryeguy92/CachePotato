#ifndef QWEN3_MOE_H
#define QWEN3_MOE_H
#include <stddef.h>

/* Qwen3-30B-A3B: one decoder layer. Three things here are new to this engine:
 *   QK-norm   per-head RMSNorm on q (n_heads) and k (n_kv) BEFORE RoPE
 *   top-k     generalised routing (8 of 128), softmax + norm_topk_prob renormalisation
 *   experts   gated SwiGLU per expert, no shared expert — every FFN read is an expert read
 *
 * WEIGHTS MAY BE QUANTIZED. The served file is Q8_0 (32.5 GB for 61 GB of bf16), so a tensor
 * carries its encoding with its pointer. Dequantizing at load time is not an option — it would
 * need the full 61 GB — so the encoding decides which kernel runs, every call.
 */
#define QW3_ENC_F32 0
#define QW3_ENC_Q8  8

typedef struct { const void *p; int enc; } QW3T;

typedef struct {
    int d_model, n_heads, n_kv, head_dim, d_exp, n_experts, topk, vocab;
    float rope_theta, rms_eps;
    const float *ln1, *ln2, *q_norm, *k_norm;   /* (d_model) and (head_dim) */
    QW3T Wq, Wk, Wv, Wo;                        /* (n_heads*hd, d) (n_kv*hd, d) (d, n_heads*hd) */
    QW3T Wgate;                                 /* (n_experts, d_model) */
    /* Expert index, [e * 3 + kind] with kind 0=gate 1=up 2=down.
       An INDEX rather than a contiguous block on purpose: 128 experts per layer cannot be one
       tensor, and the pager must be able to evict and re-resolve an individual expert between
       tokens. A baked-in pointer array would have to be replaced later; this is the shape the
       pager needs. */
    QW3T *exp;
    int exp_owned;
    const void *map;        /* whole-file mapping, owned by the caller-facing loader */
    size_t map_len;
} QW3Layer;

typedef struct {
    float *attn;        /* (T, d_model)  pre-residual attention output */
    float *mlp;         /* (T, d_model)  pre-residual MoE output */
    float *router;      /* (T, n_experts) router logits */
    int   *topk_idx;    /* (T, topk) selected experts, descending probability */
} QW3Out;

int qwen3_layer_alloc(QW3Out *o, const QW3Layer *m, int T);
void qwen3_layer_free(QW3Out *o);
void qwen3_layer_free_weights(QW3Layer *m);
int qwen3_layer(const QW3Layer *m, const float *x, int T, QW3Out *o);

/* Flat-blob loader: the arithmetic gate. Host-endian, no tables. */
int qwen3_load_blob(QW3Layer *m, const char *path, float **owned);

/* CPW1 v4 loader: reads the published format, one layer at a time.
   Returns 0 on success; the file is mmapped and stays mapped for the model's lifetime. */
int qwen3_load_v4(QW3Layer *m, const char *path, int layer);
#endif
