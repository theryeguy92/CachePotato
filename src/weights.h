#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <stddef.h>
#include <stdint.h>

#define CP_MAGIC 0x31575043u /* "CPW1" little-endian */
#define CP_VERSION 2u

/* File header: magic + version + 6 config ints = 32 bytes. */
typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t vocab, d_model, n_heads, d_ff, n_layer, n_ctx;
} Header;

/* The model's parameters as pointers INTO the mmap'd file — no copying,
   the OS pages them in as we touch them. Per-tensor-type across layers
   (llama2.c style): e.g. Wq holds ALL layers, layer l at Wq + l*d*d.
   Unembedding is TIED: logits use token_emb (GPT-2 semantics). */
typedef struct {
    float *token_emb;             /* (vocab, d_model)   */
    float *pos_emb;               /* (n_ctx, d_model)   */
    float *ln1_g, *ln1_b;         /* (n_layer, d_model) */
    float *Wq, *bq;               /* (n_layer, d,d) / (n_layer, d) */
    float *Wk, *bk;
    float *Wv, *bv;
    float *Wo, *bo;
    float *ln2_g, *ln2_b;         /* (n_layer, d_model) */
    float *W1, *b1;               /* (n_layer, d,f) / (n_layer, f) */
    float *W2, *b2;               /* (n_layer, f,d) / (n_layer, d) */
    float *lnf_g, *lnf_b;         /* (d_model)          */
} Weights;

typedef struct {
    Header hdr;
    Weights w;
    int fd;
    float *data; /* mmap base */
    size_t file_size;
} Model;

/* Load + validate + mmap the weight file. Returns 0 on success,
   -1 (message on stderr) on any failure. */
int model_load(Model *m, const char *path);
void model_free(Model *m);

/* The pointer walk, exposed: lay out a Weights struct over any float
   buffer (used by the mmap loader AND by the trainer, which lays the same
   struct over its params and grads buffers). */
void layout_weights(Weights *w, const Header *h, float *ptr);
/* floats the CPW1 payload contains for this config (cross-check + sizing) */
long expected_floats(const Header *h);

/* Experiment 1 (Phase 3): evict ALL weight pages — process-level
   (MADV_DONTNEED) AND system page cache (POSIX_FADV_DONTNEED).
   Next touch re-faults everything. Used by CP_COLD mode. */
void model_evict_all(Model *m);

#endif
