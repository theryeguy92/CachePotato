/* Corrected + extended: what actually limits single-token GEMV on this CPU, and what
 * lifts the ceiling. Variants:
 *   A naive      : acc += x[i]*W[i]           (serial dependency on acc)
 *   B ilp4       : four partial accumulators   (breaks the latency chain)
 *   C q4         : 4-bit packed weights, dequantised on the fly (the byte lever)
 * Work accounting: each rep streams the matrix ONCE total across all threads.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

typedef struct { const float *W; const uint8_t *Q; const float *scale, *x; float *out;
                 int M, N, reps, tid, nth, variant; } Arg;

static void row_naive(const float *row, const float *x, int N, float *out) {
    float acc = 0.0f;
    for (int i = 0; i < N; i++) acc += x[i] * row[i];
    *out = acc;
}
static void row_ilp4(const float *row, const float *x, int N, float *out) {
    float a0=0,a1=0,a2=0,a3=0; int i=0;
    for (; i + 3 < N; i += 4) { a0 += x[i]*row[i]; a1 += x[i+1]*row[i+1];
                                a2 += x[i+2]*row[i+2]; a3 += x[i+3]*row[i+3]; }
    float acc = (a0+a1)+(a2+a3);
    for (; i < N; i++) acc += x[i]*row[i];
    *out = acc;
}
/* 4-bit: 32 values packed in 16 bytes + one f32 scale (symmetric per-32 group). */
static void row_q4(const uint8_t *packed, const float *scale, const float *x, int N, float *out) {
    float a0=0,a1=0,a2=0,a3=0;
    int g = 0;
    for (int i = 0; i < N; i += 32, g++) {
        const uint8_t *b = packed + (size_t)g * 16;
        const float s = scale[g];
        for (int j = 0; j < 16; j++) {
            int lo = b[j] & 0xF, hi = b[j] >> 4;
            float v0 = (float)(lo - 8) * s, v1 = (float)(hi - 8) * s;
            a0 += x[i + 2*j]     * v0;
            a1 += x[i + 2*j + 1] * v1;
        }
    }
    *out = (a0 + a1);
}

static void *worker(void *p) {
    Arg *a = p;
    int lo = (int)((long)a->M * a->tid / a->nth), hi = (int)((long)a->M * (a->tid + 1) / a->nth);
    for (int r = 0; r < a->reps; r++)
        for (int o = lo; o < hi; o++) {
            if (a->variant == 0) row_naive(a->W + (size_t)o*a->N, a->x, a->N, &a->out[o]);
            else if (a->variant == 1) row_ilp4(a->W + (size_t)o*a->N, a->x, a->N, &a->out[o]);
            else row_q4(a->Q + (size_t)o*(a->N/2), a->scale + (size_t)o*(a->N/32), a->x, a->N, &a->out[o]);
        }
    return NULL;
}

static void run(const char *name, int M, int N, int reps, int nth, int variant) {
    size_t mb = (size_t)M * N * sizeof(float);
    float *W = aligned_alloc(64, mb), *x = aligned_alloc(64, (size_t)N * 4);
    float *out = aligned_alloc(64, (size_t)M * 4);
    uint8_t *Q = NULL; float *scale = NULL;
    for (size_t i = 0; i < (size_t)M*N; i++) W[i] = (float)(i % 13) * 0.01f;
    for (int i = 0; i < N; i++) x[i] = (float)(i % 7) * 0.1f;
    if (variant == 2) {
        Q = aligned_alloc(64, (size_t)M * (N/2));
        scale = aligned_alloc(64, (size_t)M * (N/32) * 4);
        for (int o = 0; o < M; o++)
            for (int g = 0; g < N/32; g++) {
                float mx = 1e-9f;
                for (int j = 0; j < 32; j++) { float v = W[(size_t)o*N+g*32+j]; if (v<0) v=-v; if (v>mx) mx=v; }
                scale[(size_t)o*(N/32)+g] = mx/7.0f;
                for (int j = 0; j < 16; j++) {
                    int a = (int)(W[(size_t)o*N+g*32+2*j]   / (mx/7.0f)); a = a<-8?-8:(a>7?7:a);
                    int b = (int)(W[(size_t)o*N+g*32+2*j+1] / (mx/7.0f)); b = b<-8?-8:(b>7?7:b);
                    Q[(size_t)o*(N/2)+g*16+j] = (uint8_t)((a+8) | ((b+8)<<4));
                }
            }
    }
    pthread_t th[16]; Arg args[16];
    for (int i = 0; i < nth; i++) args[i] = (Arg){W,Q,scale,x,out,M,N,reps,i,nth,variant};
    double t0 = now();
    for (int i = 1; i < nth; i++) pthread_create(&th[i], NULL, worker, &args[i]);
    worker(&args[0]);
    for (int i = 1; i < nth; i++) pthread_join(th[i], NULL);
    double dt = now() - t0;
    double flops = 2.0 * M * N * (double)reps;          /* matrix streamed once per rep */
    double bytes = (variant == 2 ? (double)M * (N/2 + N/8) : (double)mb) * reps;
    printf("  %-9s %5dx%-5d %6.1f MB %2dt %8.2f GFLOP/s %7.2f GB/s\n",
           name, M, N, bytes/reps/1e6, nth, flops/dt/1e9, bytes/dt/1e9);
    free(W); free(x); free(out); free(Q); free(scale);
}

int main(void) {
    printf("single-token GEMV ceiling on this CPU (i5-8250U, DDR4-2400 dual ~25 GB/s real)\n\n");
    for (int nth = 1; nth <= 8; nth *= 8) {
        printf(" threads=%d\n", nth);
        run("naive",  768, 3072, 2000, nth, 0);
        run("ilp4",   768, 3072, 2000, nth, 1);
        run("q4",     768, 3072, 2000, nth, 2);
        printf("\n");
    }
    printf(" threads=8, larger (out of L3)\n");
    run("naive", 4096, 4096, 60, 8, 0);
    run("ilp4",  4096, 4096, 60, 8, 1);
    run("q4",    4096, 4096, 60, 8, 2);
    return 0;
}
