#include "kernels.h"
#include "parallel.h"

#include <math.h>
#include <stddef.h>

/* out = (x - mean) / sqrt(var + eps) * gain + bias, over n floats (one row) */
void layernorm(float *out, const float *x, const float *gain,
               const float *bias, int n) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * gain[i] + bias[i];
}

/* softmax in place over n floats, max-subtracted (same as run.c) */
void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* GPT-2's GELU (tanh approximation) */
float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

/* out[j] = bias[j] + sum_k x[k] * w[k*n_out + j]  — "x @ W + b",
   W stored (n_in, n_out). Row-streaming: init with bias, then scatter
   x[k]*w_row into out — inner loop walks w CONTIGUOUSLY (vectorizes). */
/* Split over OUTPUT COLUMNS: each out[j] still accumulates over k in exactly the order
   0..n_in-1, so threading changes nothing. Below CP_MATMUL_MIN the dispatch costs more than
   the work, so small matmuls (the router: 768x32) stay serial. */
#define CP_MATMUL_MIN 60000L

typedef struct { float *out; const float *x, *w, *bias; int n_in, n_out; } XWCtx;

static void xw_slice(void *v, int jlo, int jhi) {
    XWCtx *c = v;
    for (int j = jlo; j < jhi; j++) c->out[j] = c->bias[j];
    for (int k = 0; k < c->n_in; k++) {
        float xk = c->x[k];
        const float *wrow = c->w + (size_t)k * c->n_out;
        for (int j = jlo; j < jhi; j++) c->out[j] += xk * wrow[j];
    }
}

void matmul_xw(float *out, const float *x, const float *w,
               const float *bias, int n_in, int n_out) {
    if (cp_would_parallelise((long)n_in * n_out, CP_MATMUL_MIN)) {
        XWCtx c = { out, x, w, bias, n_in, n_out };
        cp_parallel_for(n_out, xw_slice, &c);
        return;
    }
    for (int j = 0; j < n_out; j++) out[j] = bias[j];
    for (int k = 0; k < n_in; k++) {
        float xk = x[k];
        const float *wrow = w + (size_t)k * n_out;
        for (int j = 0; j < n_out; j++) out[j] += xk * wrow[j];
    }
}

/* out[i] = sum_k w[i*n_in + k] * x[k]  — "W @ x", W stored (n_out, n_in),
   contiguous rows. Used for logits = xf @ token_emb^T (tied unemb). */
/* Split over OUTPUT ROWS: each out[i] is a serial dot product, unchanged. This one matters
   a lot — the tied unembedding is n_out = vocab (50257 for gpt2-moe), i.e. the widest
   single matmul in the engine. */
typedef struct { float *out; const float *x, *w; int n_in, n_out; } WXCtx;

static void wx_slice(void *v, int ilo, int ihi) {
    WXCtx *c = v;
    for (int i = ilo; i < ihi; i++) {
        float acc = 0.0f;
        for (int k = 0; k < c->n_in; k++) acc += c->w[(size_t)i * c->n_in + k] * c->x[k];
        c->out[i] = acc;
    }
}

void matmul_wx(float *out, const float *x, const float *w, int n_in, int n_out) {
    if (cp_would_parallelise((long)n_in * n_out, CP_MATMUL_MIN)) {
        WXCtx c = { out, x, w, n_in, n_out };
        cp_parallel_for(n_out, wx_slice, &c);
        return;
    }
    for (int i = 0; i < n_out; i++) {
        float acc = 0.0f;
        for (int k = 0; k < n_in; k++) acc += w[(size_t)i * n_in + k] * x[k];
        out[i] = acc;
    }
}

/* ---------------------------------------------------------------- Q8_0 (format/SPEC.md §6)
 *
 * Group = 34 bytes: fp16 scale, then 32 int8.  decode: y[i] = (float)q[i] * d
 *
 * The layout is llama.cpp's Q8_0 on purpose, so a published checkpoint needs no re-encoding
 * (SPEC §6 option 1). `format/q8.py` is the reference and `tools/verify_q8_decode.py` requires
 * EXACT equality between the two decoders: the scale is fp16, an exact subset of fp32, so
 * `q * d` has exactly one correct fp32 answer and there is no tolerance to hide a mistake in.
 */
#include <stdint.h>
#include <string.h>

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {                       /* subnormal: renormalise */
            uint32_t e = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            f = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31) {          /* inf / nan: preserve */
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &f, sizeof r);        /* avoid a strict-aliasing violation */
    return r;
}

static float q8_scale(const uint8_t *blk) {
    uint16_t h = (uint16_t)(blk[0] | (blk[1] << 8));
    return f16_to_f32(h);
}

void matmul_xw_auto(float *out, const float *x, const float *w, const uint8_t *q8,
                    const float *bias, int n_in, int n_out) {
    if (q8) matmul_xw_q8(out, x, q8, bias, n_in, n_out);
    else    matmul_xw(out, x, w, bias, n_in, n_out);
}

void q8_dequant_row(const uint8_t *blocks, float *out, int n) {
    int ng = n / 32;
    for (int g = 0; g < ng; g++) {
        const uint8_t *blk = blocks + (size_t)g * 34;
        float d = q8_scale(blk);
        const int8_t *q = (const int8_t *)(blk + 2);
        for (int i = 0; i < 32; i++) out[g * 32 + i] = (float)q[i] * d;
    }
}

/* out[j] += x[k] * W[k][j], W row-quantized with groups along n_out. Dequantised inline so
 * no scratch buffer (and no per-call allocation) is needed.
 *
 * Split by GROUP, not by element: a group's 32 outputs share one scale, so a group is the
 * smallest unit that can be moved to another thread without re-reading the scale. */
/* ------------------------------------------------------------ AVX2 dequant
 *
 * The scalar decoder costs ~3 ops per element (int8->float, x scale, FMA) against 1 for fp32,
 * which is why Q8 measured SLOWER than fp32 on every cache-resident matrix (0.62-0.95x). The
 * vector form decodes 32 elements with 4 sign-extends, 4 converts and 4 FMAs, so the byte
 * saving (3.76x) can finally reach the arithmetic.
 *
 * Guarded by __AVX2__ rather than a runtime check: the build already passes -march=native, so
 * a binary either has AVX2 or is compiled for a CPU without it. The scalar path is kept as the
 * fallback.
 *
 * NOT bit-identical to the scalar path, and it cannot be: FMA fuses the multiply-add (one
 * rounding instead of two) and the horizontal sum reduces in lanes. This is the SAME class of
 * difference as C-vs-numpy fp32 accumulation, and it is why the *decoder* (q8_dequant_row, the
 * definition of what the file's values ARE) is gated bit-exactly while the *matmul* is gated
 * with a stated tolerance (tools/verify_q8_kernels.py). Quantised-vs-fp32 is already lossy, so
 * demanding exactness here would mean forbidding FMA — the main reason the kernel is fast.
 */
#if defined(__AVX2__)
#include <immintrin.h>

static inline void q8_axpy_avx2(float *out, const uint8_t *blk, float xk) {
    const float d = q8_scale(blk);
    const int8_t *q = (const int8_t *)(blk + 2);
    const __m256 dv = _mm256_set1_ps(d);
    const __m256 xv = _mm256_set1_ps(xk);
    for (int h = 0; h < 4; h++) {
        __m128i b = _mm_loadl_epi64((const __m128i *)(q + h * 8));
        __m256i i32 = _mm256_cvtepi8_epi32(b);
        __m256 f = _mm256_cvtepi32_ps(i32);
        __m256 v = _mm256_mul_ps(f, dv);
        __m256 o = _mm256_loadu_ps(out + h * 8);
        o = _mm256_fmadd_ps(v, xv, o);
        _mm256_storeu_ps(out + h * 8, o);
    }
}

/* out[i] += (float)q[i] * d * xk, 32 elements */
static inline __m256 q8_dot_acc_avx2(__m256 s, const uint8_t *blk, const float *x) {
    const float d = q8_scale(blk);
    const int8_t *q = (const int8_t *)(blk + 2);
    const __m256 dv = _mm256_set1_ps(d);
    for (int h = 0; h < 4; h++) {
        __m128i b = _mm_loadl_epi64((const __m128i *)(q + h * 8));
        __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)), dv);
        s = _mm256_fmadd_ps(v, _mm256_loadu_ps(x + h * 8), s);
    }
    return s;
}

static inline float q8_hsum8(__m256 s) {
    float lane[8];
    _mm256_storeu_ps(lane, s);
    return ((lane[0] + lane[1]) + (lane[2] + lane[3])) +
           ((lane[4] + lane[5]) + (lane[6] + lane[7]));
}

static inline void q8_dot_avx2(float *acc, const uint8_t *blk, const float *x) {
    const float d = q8_scale(blk);
    const int8_t *q = (const int8_t *)(blk + 2);
    const __m256 dv = _mm256_set1_ps(d);
    __m256 s = _mm256_setzero_ps();
    for (int h = 0; h < 4; h++) {
        __m128i b = _mm_loadl_epi64((const __m128i *)(q + h * 8));
        __m256i i32 = _mm256_cvtepi8_epi32(b);
        __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(i32), dv);
        s = _mm256_fmadd_ps(v, _mm256_loadu_ps(x + h * 8), s);
    }
    /* horizontal sum in a FIXED order so the result does not depend on lane count */
    float lane[8];
    _mm256_storeu_ps(lane, s);
    float r = ((lane[0] + lane[1]) + (lane[2] + lane[3])) +
              ((lane[4] + lane[5]) + (lane[6] + lane[7]));
    *acc += r;
}
#endif

typedef struct { float *out; const float *x; const uint8_t *w; int n_in, ng;
                 size_t rowbytes; } XWQ8Ctx;

static void xw_q8_slice(void *v, int glo, int ghi) {
    XWQ8Ctx *c = v;
    for (int k = 0; k < c->n_in; k++) {
        float xk = c->x[k];
        const uint8_t *row = c->w + (size_t)k * c->rowbytes;
        for (int g = glo; g < ghi; g++) {
            const uint8_t *blk = row + (size_t)g * 34;
            float *o = c->out + g * 32;
#if defined(__AVX2__)
            q8_axpy_avx2(o, blk, xk);
#else
            float d = q8_scale(blk);
            const int8_t *q = (const int8_t *)(blk + 2);
            for (int i = 0; i < 32; i++) o[i] += xk * ((float)q[i] * d);
#endif
        }
    }
}

void matmul_xw_q8(float *out, const float *x, const uint8_t *w,
                  const float *bias, int n_in, int n_out) {
    const int ng = n_out / 32;
    const size_t rowbytes = (size_t)ng * 34;
    for (int j = 0; j < n_out; j++) out[j] = bias[j];
    if (cp_would_parallelise((long)n_in * n_out, CP_MATMUL_MIN) && ng > 1) {
        XWQ8Ctx c = { out, x, w, n_in, ng, rowbytes };
        cp_parallel_for(ng, xw_q8_slice, &c);
        return;
    }
    for (int k = 0; k < n_in; k++) {
        float xk = x[k];
        const uint8_t *row = w + (size_t)k * rowbytes;
        for (int g = 0; g < ng; g++) {
            float d = q8_scale(row + (size_t)g * 34);
            const int8_t *q = (const int8_t *)(row + (size_t)g * 34 + 2);
            float *o = out + g * 32;
            for (int i = 0; i < 32; i++) o[i] += xk * ((float)q[i] * d);
        }
    }
}

/* out[i] = sum_k W[i][k] * x[k], W row-quantized. Same accumulation order as the fp32
 * version (groups in order, elements within a group in order); only the values differ. */
void matmul_wx_auto(float *out, const float *x, const float *w, const uint8_t *q8,
                    int n_in, int n_out) {
    if (q8) matmul_wx_q8(out, x, q8, n_in, n_out);
    else    matmul_wx(out, x, w, n_in, n_out);
}

typedef struct { float *out; const float *x; const uint8_t *w; int ng; size_t rowbytes; } WXQ8Ctx;

static void wx_q8_slice(void *v, int ilo, int ihi) {
    WXQ8Ctx *c = v;
    for (int i = ilo; i < ihi; i++) {
        const uint8_t *row = c->w + (size_t)i * c->rowbytes;
#if defined(__AVX2__)
        /* Accumulate in a VECTOR across every group, then horizontal-sum ONCE per row.
           The previous version summed 8 lanes per 32-element GROUP — for a 4864-wide row that
           is 152 horizontal sums per row, each a 32-byte store plus 7 scalar adds. Profiling
           put Wd at 0.81 ms/token and 5.7 GB/s (half the ceiling); the summation, not the
           arithmetic, was the cost. */
        __m256 s = _mm256_setzero_ps();
        for (int g = 0; g < c->ng; g++)
            s = q8_dot_acc_avx2(s, row + (size_t)g * 34, c->x + g * 32);
        c->out[i] = q8_hsum8(s);
#else
        float acc = 0.0f;
        for (int g = 0; g < c->ng; g++) {
            const uint8_t *blk = row + (size_t)g * 34;
            float d = q8_scale(blk);
            const int8_t *q = (const int8_t *)(blk + 2);
            for (int j = 0; j < 32; j++) acc += ((float)q[j] * d) * c->x[g * 32 + j];
        }
        c->out[i] = acc;
#endif
    }
}

void matmul_wx_q8(float *out, const float *x, const uint8_t *w, int n_in, int n_out) {
    const int ng = n_in / 32;
    const size_t rowbytes = (size_t)ng * 34;
    if (cp_would_parallelise((long)n_in * n_out, CP_MATMUL_MIN) && n_out > 1) {
        WXQ8Ctx c = { out, x, w, ng, rowbytes };
        cp_parallel_for(n_out, wx_q8_slice, &c);
        return;
    }
    for (int i = 0; i < n_out; i++) {
        const uint8_t *row = w + (size_t)i * rowbytes;
        float acc = 0.0f;
        for (int g = 0; g < ng; g++) {
            float d = q8_scale(row + (size_t)g * 34);
            const int8_t *q = (const int8_t *)(row + (size_t)g * 34 + 2);
            for (int j = 0; j < 32; j++) acc += ((float)q[j] * d) * x[g * 32 + j];
        }
        out[i] = acc;
    }
}

/* ---------------------------------------------------------------- Qwen2.5 primitives
 *
 * Three things this engine has never had. Each is gated against the independent numpy
 * reference (models/qwen2.5-0.5b/reference.py) on random data, because a `read the config`
 * mistake here is silent: an RMSNorm implemented as LayerNorm still runs and still produces
 * fluent text.
 */

/* RMSNorm: x / sqrt(mean(x^2) + eps) * g.  NO mean subtraction, NO bias. This is the single
   easiest block to get wrong by reusing a LayerNorm, and the failure is invisible in output. */
void rmsnorm(float *out, const float *x, const float *g, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * g[i];
}

/* SiLU (swish): x * sigmoid(x). */
float silu(float x) { return x / (1.0f + expf(-x)); }

/* RoPE, applied in place to q and k.
 *
 * Layout: q is (T, n_q_heads, hd), k is (T, n_kv_heads, hd), row-major. Only the FIRST half of
 * each head's cos/sin table is needed: the pair (i, i + hd/2) rotates together, which is the
 * same thing as HF's duplicated-table + rotate_half formulation.
 *
 * theta is the RoPE base (1e6 for Qwen2.5), taken from the file rather than assumed. */
void rope_apply(float *q, float *k, int T, int n_q_heads, int n_kv_heads, int hd, float theta) {
    const int half = hd / 2;
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < half; i++) {
            /* inv_freq = theta^(-2i/hd); angle = pos * inv_freq */
            float inv = powf(theta, -2.0f * (float)i / (float)hd);
            float ang = (float)t * inv;
            float c = cosf(ang), s = sinf(ang);
            float *rows[2] = { q + (size_t)t * n_q_heads * hd, k + (size_t)t * n_kv_heads * hd };
            int nh[2] = { n_q_heads, n_kv_heads };
            for (int r = 0; r < 2; r++) {
                float *base = rows[r];
                for (int h = 0; h < nh[r]; h++) {
                    float *v = base + (size_t)h * hd;
                    float a = v[i], b = v[i + half];
                    v[i]        = a * c - b * s;
                    v[i + half] = b * c + a * s;
                }
            }
        }
    }
}
