#ifndef QWEN2_H
#define QWEN2_H
#include <stddef.h>
/* Qwen2 forward: RMSNorm + RoPE + GQA + SwiGLU + tied unembedding. */
typedef struct {
    int n_layers, d_model, n_heads, n_kv, head_dim, d_ff, vocab;
    float rope_theta, rms_eps;
    const float *embed, *norm;
    const float *ln1, *ln2;                 /* (L, D) */
    const float *Wq, *bq;                   /* (L, D, D) / (L, D) */
    const float *Wk, *bk, *Wv, *bv;         /* (L, n_kv*hd, D) / (L, n_kv*hd) */
    const float *Wo;                        /* (L, D, D) */
    const float *Wg, *Wu, *Wd;              /* (L, d_ff, D) / (L, D, d_ff) */
} QW2;

int qwen2_forward(const QW2 *m, const int *tokens, int T, float *logits);
int qwen2_load_blob(QW2 *m, const char *path, float **owned);
/* v4: geometry (heads, KV heads, head_dim, rope theta, eps) comes from the file's BLOCK TABLE. */
int qwen2_load_v4(QW2 *m, const char *path, void **map, size_t *maplen);

/* Incremental decode with a GQA-sized KV cache: (maxT, n_kv*head_dim) per layer. */
typedef struct QW2Cache QW2Cache;
QW2Cache *qwen2_cache_alloc(const QW2 *m, int maxT);
void qwen2_cache_free(QW2Cache *c);
int qwen2_step(const QW2 *m, QW2Cache *c, int token, int pos, float *logits);
#endif
