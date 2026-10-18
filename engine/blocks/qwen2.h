#ifndef QWEN2_H
#define QWEN2_H
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
#endif
