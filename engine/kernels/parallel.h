#ifndef PARALLEL_H
#define PARALLEL_H

/* Minimal persistent thread pool (POSIX threads only — no third-party dependency).
 *
 * WHY THIS IS SAFE TO ADD TO A "BIT-IDENTICAL" ENGINE
 *   Every kernel here computes independent outputs: `matmul_xw` produces one `out[j]` per
 *   column, `matmul_wx` one `out[i]` per row. Splitting those outputs across threads leaves
 *   each output's accumulation order untouched, so the result is not merely equivalent but
 *   bit-identical. The pool therefore cannot change any logit, any routing decision or any
 *   pager byte count — verified by comparing CP_THREADS=1 against CP_THREADS=8.
 *
 *   It is deliberately NOT used for reductions (layernorm's mean/variance, softmax's sum,
 *   the MoE locality scan): those are serial over their axis, and splitting them would
 *   change the summation order. That distinction is the whole design.
 *
 * DEFAULT IS 1 THREAD. The project's benchmarks and the recorded reference runs are
 * single-core ("the single-core reference point"), so adding cores must never silently move
 * a number. Set CP_THREADS=N to opt in.
 */

/* Initialise lazily from CP_THREADS (default 1, clamped to [1, 64]). Returns thread count. */
int  cp_threads(void);

/* Start the pool with n threads instead of reading CP_THREADS (n <= 0 falls back to it).
 * pin: 1/0 overrides CP_PIN_CORES, -1 keeps it. Only the first call that starts the pool
 * counts; later calls return the running thread count. */
int  cp_threads_init(int n, int pin);

/* Run fn(ctx, lo, hi) over contiguous slices of [0, n). Blocks until all slices finish.
 * Runs serially when the pool has one thread. */
void cp_parallel_for(int n, void (*fn)(void *ctx, int lo, int hi), void *ctx);

/* 1 if the call would actually be spread over more than one thread for `work` units. */
int  cp_would_parallelise(long work, long threshold);

void cp_pool_shutdown(void);

#endif
