#ifndef KERNELS_H
#define KERNELS_H

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

#endif
