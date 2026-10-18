#ifndef QWEN3_MOE_H
#define QWEN3_MOE_H
#include <stddef.h>
#include <stdint.h>

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
    /* global tensors, bound once for the whole model */
    QW3T embed, lm_head;
    const float *final_norm;
    /* parsed table geometry, reused by every bind so a 48-layer run maps the file ONCE */
    uint64_t off_tens, off_data;
    uint32_t n_tens, n_enc;
    int enc_code[64];   /* encoding index -> QW3_ENC_* */

    /* ---- pager attachment (C1) -------------------------------------------------
       The pager evicts REGIONS, and a v4 file declares them (SPEC §8), so the unit of
       the policy is a region id, not an address. qwen3_map() adds the file's regions to
       the pager and remembers which region covers each tensor; qwen3_bind_layer() then
       records the ids of the tensors this layer bound. At run time a touch is an array
       index — no search, and no second description of where the bytes are, which is
       what lets the accounting be checked against the file instead of against itself.
       Every id is -1 when no pager is attached and pager_touch_region(-1) is a no-op,
       so the unpaged path runs the same code with the same arithmetic. */
    int pager;              /* 1 when a pager is attached to this mapping */
    int n_regions;          /* regions in the file's table */
    int *tensor_reg;        /* [n_tens]: region covering that tensor, or -1 */
    int *reg_exp;           /* [n_experts]: region for the BOUND layer's expert e */
    int reg_ln1, reg_ln2, reg_qnorm, reg_knorm;
    int reg_Wq, reg_Wk, reg_Wv, reg_Wo, reg_Wgate;
    int reg_embed, reg_lm_head, reg_final_norm;
} QW3Layer;

typedef struct {
    float *attn;        /* (T, d_model)  pre-residual attention output */
    float *mlp;         /* (T, d_model)  pre-residual MoE output */
    float *router;      /* (T, n_experts) router logits */
    int   *topk_idx;    /* (T, topk) selected experts, descending probability */
} QW3Out;

/* KV cache for incremental decode. The layout is the prefill path's (pos, n_kv*head_dim)
   array unchanged, so both entry points read the same bytes in the same order and the
   step gate can demand BIT-IDENTITY instead of a tolerance. cap is in positions. */
typedef struct { float *k, *v; int cap; } QW3KV;

int qwen3_layer_alloc(QW3Out *o, const QW3Layer *m, int T);
void qwen3_layer_free(QW3Out *o);
void qwen3_layer_free_weights(QW3Layer *m);

/* Prefill: T new tokens at positions 0..T-1, no cache kept afterwards. */
int qwen3_layer(const QW3Layer *m, const float *x, int T, QW3Out *o);

/* Incremental: T new tokens at positions pos0..pos0+T-1, attending to everything in
   `cache` at positions < pos0+T (the new rows are written before attention). pos0 = 0
   and T = the whole sequence is exactly qwen3_layer — same code, not a second copy. */
int qwen3_layer_kv(const QW3Layer *m, const float *x, int T, int pos0, QW3KV *cache, QW3Out *o);
int qwen3_kv_alloc(QW3KV *kv, const QW3Layer *m, int cap);
void qwen3_kv_free(QW3KV *kv);

/* Flat-blob loader: the arithmetic gate. Host-endian, no tables. */
int qwen3_load_blob(QW3Layer *m, const char *path, float **owned);

/* CPW1 v4 loader: reads the published format, one layer at a time.
   Returns 0 on success; the file is mmapped and stays mapped for the model's lifetime. */
int qwen3_map(QW3Layer *m, const char *path);        /* mmap + read the encoding table */
int qwen3_bind_layer(QW3Layer *m, int layer);        /* resolve one layer's tensors */
int qwen3_load_v4(QW3Layer *m, const char *path, int layer);   /* map + bind (A5 gate) */
#endif
