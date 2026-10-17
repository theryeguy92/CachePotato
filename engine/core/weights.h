#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <stddef.h>
#include <stdint.h>

#define CP_MAGIC 0x31575043u /* "CPW1" little-endian */
#define CP_VERSION_V2 2u
#define CP_VERSION_V3 3u
#define CP_V2_HDR_BYTES 32
#define CP_V3_HDR_BYTES 60

/* v2 header: magic + version + 6 config ints = 32 bytes. */
typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t vocab, d_model, n_heads, d_ff, n_layer, n_ctx;
} Header;

/* v3 header = the v2 fields + MoE config + lambda = 60 bytes.
   lambda lives in the file because the locality bias is part of the model:
   serving at a different lambda routes differently from what was trained. */
typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t vocab, d_model, n_heads, d_ff, n_layer, n_ctx;
    int32_t n_exp, exp_hid, topk, moe_from, n_moe_layers, n_dense_layers;
    float lam;
} HeaderV3;

/* The model's parameters as pointers INTO the mmap'd file — no copying, the OS
   pages them in as we touch them. Per-tensor-type across layers (llama2.c
   style): layer l of Wq lives at Wq + l*d*d. Unembedding is TIED (GPT-2).
   For v3 the dense-FFN arrays hold DENSE SLOTS ONLY (see dense_slot below), so
   layer l's W1 is at W1 + dense_slot[l]*d*f — the dead dense-FFN weights that
   the trainer allocates on MoE layers are not in the file at all. */
typedef struct {
    float *token_emb;             /* (vocab, d_model)   */
    float *pos_emb;               /* (n_ctx, d_model)   */
    float *ln1_g, *ln1_b;         /* (n_layer, d_model) */
    float *Wq, *bq;               /* (n_layer, d,d) / (n_layer, d) */
    float *Wk, *bk;
    float *Wv, *bv;
    float *Wo, *bo;
    float *ln2_g, *ln2_b;         /* (n_layer, d_model) */
    float *W1, *b1;               /* (n_dense, d,f) / (n_dense, f) */
    float *W2, *b2;               /* (n_dense, f,d) / (n_dense, d) */
    float *lnf_g, *lnf_b;         /* (d_model)          */

    /* --- CPW1 v3 / MoE (NULL and zero for v2) --- */
    float *Wr, *br;               /* router:   (n_moe, d, N) / (n_moe, N) */
    float *W1s, *b1s, *W2s, *b2s; /* shared:   (n_moe, d, d) / (n_moe, d) */
    float *experts;               /* routed:   (n_moe, N, expert_floats)  */
    /* Quantized experts (Q8_0). When `experts_q8` is non-NULL the routed experts are read
       from the file's blocks on the fly and `experts` is unused: that is the difference
       between a quantized model that FITS and one that only has a smaller file. Layout per
       expert: q8(W1) then q8(W2); biases live separately (b1 then b2) because a bias must not
       share a group scale with weights. */
    const unsigned char **experts_q8;   /* [n_moe * N] -> q8(W1)|q8(W2) */
    const float **expert_bias;          /* [n_moe * N] -> b1[EH] then b2[D] */
    int is_moe;
    int d_model, n_moe, n_exp, exp_hid, topk, moe_from;
    int *moe_slot;                /* (n_layer) layer -> MoE slot  (-1 if dense) */
    int *dense_slot;              /* (n_layer) layer -> dense slot (-1 if MoE)  */
} Weights;

typedef struct {
    Header hdr;
    HeaderV3 h3;      /* populated only when is_v3 */
    int is_v3;
    Weights w;
    int fd;
    float *data;      /* mmap base */
    size_t file_size;
    float *v4_scratch; /* v4 only: the engine-layout copy of the tensors. A v4 file makes no
                         promise about tensor order, and the forward pass indexes per-layer
                         arrays by stride, so the loader copies each tensor into its slot
                         (~701 MB for the current model) rather than assuming layout. A
                         v4-native forward pass would remove the copy; that is a later
                         change, and this one keeps the acceptance test about the FILE. */
} Model;

/* Load + validate + mmap the weight file (CPW1 v2 or v3). Returns 0 on success,
   -1 (message on stderr) on any failure. */
int model_load(Model *m, const char *path);
void model_free(Model *m);

/* The pointer walk, exposed: lay out a Weights struct over any float
   buffer (used by the mmap loader AND by the trainer, which lays the same
   struct over its params and grads buffers). */
void layout_weights(Weights *w, const Header *h, float *ptr);
/* floats the CPW1 payload contains for this config (cross-check + sizing) */
long expected_floats(const Header *h);

/* --- CPW1 v3 --- */
void layout_weights_v3(Weights *w, const HeaderV3 *h, float *ptr);
long expected_floats_v3(const HeaderV3 *h);
/* floats in one routed-expert block: W1e(EH*d) b1e(EH) W2e(d*EH) b2e(d) */
long expert_floats(int d_model, int exp_hid);
/* pointer to one (MoE slot, expert) block — one pager region */
float *expert_block(const Weights *w, int slot, int e);
const unsigned char *expert_block_q8(const Weights *w, int slot, int e);
const float *expert_bias_ptr(const Weights *w, int slot, int e);
long q8_bytes_for(long elems);
/* true when engine layer l uses a sparse FFN */
int moe_layer_at(const Model *m, int l);

/* Experiment 1 (Phase 3): evict ALL weight pages — process-level
   (MADV_DONTNEED) AND system page cache (POSIX_FADV_DONTNEED).
   Next touch re-faults everything. Used by CP_COLD mode. */
void model_evict_all(Model *m);

#endif

/* --- CPW1 v4 (format/SPEC.md) --- */
#include "weights_v4.h"
