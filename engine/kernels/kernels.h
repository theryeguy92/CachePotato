#ifndef KERNELS_H
#define KERNELS_H
#include <stdint.h>

/* Shared math kernels — used by both the inference engine (model.c) and
   the trainer (train.c). Keep them static-inline-free and single-threaded;
   the row-streaming layout is the performance contract. */

void layernorm(float *out, const float *x, const float *gain,
               const float *bias, int n);
void softmax(float *x, int n);
float gelu(float x);
/* out[j] = bias[j] + sum_k x[k] * w[k*n_out + j]  — "x @ W + b", W (n_in, n_out) */
void matmul_xw(float *out, const float *x, const float *w,
               const float *bias, int n_in, int n_out);
/* out[i] = sum_k w[i*n_in + k] * x[k]  — "W @ x", W (n_out, n_in) */
void matmul_wx(float *out, const float *x, const float *w, int n_in, int n_out);

/* Q8_0 (format/SPEC.md §6): 34-byte group = fp16 scale + 32 int8, y = q * d.
   The decoder must equal format/q8.py bit for bit (tools/verify_q8_decode.py). */
void q8_dequant_row(const uint8_t *blocks, float *out, int n);
void matmul_xw_q8(float *out, const float *x, const uint8_t *w,
                  const float *bias, int n_in, int n_out);
/* Dispatch: use the Q8 kernel when a quantized block pointer is present, else fp32. Lets a
   block stay representation-agnostic — it never learns whether its weights are quantized. */
void matmul_xw_auto(float *out, const float *x, const float *w, const uint8_t *q8,
                    const float *bias, int n_in, int n_out);
void matmul_wx_q8(float *out, const float *x, const uint8_t *w, int n_in, int n_out);

#endif
