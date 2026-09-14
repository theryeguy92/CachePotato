#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <stddef.h>
#include <stdint.h>

#define CP_MAGIC 0x31575043u /* "CPW1" little-endian */

/* File header: magic + reserved + 5 config ints = 28 bytes. */
typedef struct {
    uint32_t magic;
    uint32_t reserved;
    int32_t vocab, d_model, n_heads, d_ff, n_layer;
} Header;

/* The model's parameters as pointers INTO the mmap'd file — no copying,
   the OS pages them in as we touch them. */
typedef struct {
    float *token_emb;           /* (vocab, d_model)   */
    float *Wq, *Wk, *Wv, *Wo;   /* (d_model, d_model) */
    float *W1;                  /* (d_model, d_ff)    */
    float *W2;                  /* (d_ff, d_model)    */
    float *ln1_g, *ln2_g, *lnf_g; /* (d_model)        */
    float *unemb;               /* (vocab, d_model)   */
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

#endif
