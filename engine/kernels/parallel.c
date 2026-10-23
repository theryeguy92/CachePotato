#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "parallel.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAX_THREADS 64
#define SPIN_LIMIT  60000   /* ~0.5 ms of spinning. A token issues ~144 dispatches separated
                               by up to ~1 ms of serial work; with a 15 us spin the workers
                               slept between almost every one. Measured: dispatch costs 2.0 us
                               hot but 63 us after a 1 ms gap = ~9 ms/token of wake-up. */

typedef void (*task_fn)(void *ctx, int lo, int hi);

/* Dispatch cost is what decides whether threading pays. Measured on the T480 with a
   condvar-only barrier: 20.8 us per dispatch, which is larger than several of the matmuls
   being dispatched (a 768x192 expert GEMV is 23 us). So the barrier spins briefly before
   blocking: a dispatch that arrives while workers are still hot costs ~1-2 us, and a
   genuinely idle pool still sleeps instead of burning a core. Fields are published by the
   caller before `gen` is released and read by workers after `gen` is acquired. */
static struct {
    _Atomic int gen;
    _Atomic int remaining;
    _Atomic int shutdown;
    task_fn fn;
    void *ctx;
    int n, chunk;
    int nworkers, nthreads, started;
    pthread_t tid[MAX_THREADS];
    pthread_mutex_t mu;
    pthread_cond_t sleep;
} J = { .mu = PTHREAD_MUTEX_INITIALIZER, .sleep = PTHREAD_COND_INITIALIZER };

/* Optional: bind each thread to its own core. On a 4C/8T part the OS otherwise migrates
   threads between HT siblings, and a migrated thread loses its L1/L2 contents — the x vector
   and the four accumulators are exactly what a decode matmul keeps hot. Off by default so no
   recorded benchmark moves; CP_PIN_CORES=1 turns it on. */
static int pin_override = -1;   /* cp_threads_init(): -1 = use CP_PIN_CORES */
static int pin_cores(void) {
    if (pin_override >= 0) return pin_override;
    const char *pc = getenv("CP_PIN_CORES");
    return pc && *pc && atoi(pc) != 0;
}
static void pin_self(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    sched_setaffinity(0, sizeof set, &set);
}

static void *worker(void *arg) {
    long id = (long)arg;
    if (pin_cores()) pin_self((int)id + 1);   /* the child pins itself */
    int seen = 0;
    for (;;) {
        int g, spins = 0;
        while ((g = atomic_load_explicit(&J.gen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&J.shutdown, memory_order_relaxed)) return NULL;
            if (++spins < SPIN_LIMIT) { __builtin_ia32_pause(); continue; }
            /* TIMED wait on purpose. A lost wakeup would deadlock the whole engine, and
               this pool has no other backstop. The timeout makes a missed broadcast a short
               delay instead of a hang; it should never fire in practice. */
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 5 * 1000 * 1000;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            pthread_mutex_lock(&J.mu);
            while (atomic_load_explicit(&J.gen, memory_order_acquire) == seen &&
                   !atomic_load_explicit(&J.shutdown, memory_order_relaxed))
                pthread_cond_timedwait(&J.sleep, &J.mu, &ts);
            pthread_mutex_unlock(&J.mu);
            spins = 0;
        }
        seen = g;
        task_fn fn = J.fn;
        void *ctx = J.ctx;
        int n = J.n, chunk = J.chunk;

        int lo = (int)((id + 1) * (long)chunk);
        int hi = lo + chunk;
        if (hi > n) hi = n;
        if (lo < n) fn(ctx, lo, hi);
        atomic_fetch_sub_explicit(&J.remaining, 1, memory_order_release);
    }
}

static void start(int nthreads) {
    if (J.started) return;
    J.started = 1;
    J.nthreads = nthreads;
    J.nworkers = nthreads - 1;
    if (pin_cores()) pin_self(0);   /* the caller also runs chunk 0 */
    for (long i = 0; i < J.nworkers; i++)
        if (pthread_create(&J.tid[i], NULL, worker, (void *)i) != 0) {
            fprintf(stderr, "parallel: pthread_create failed at %ld; using %ld thread(s)\n",
                    i, i + 1);
            J.nworkers = (int)i;
            J.nthreads = (int)i + 1;
            break;
        }
}

static int clamp_threads(int n) {
    if (n <= 0) n = 1;
    if (n > MAX_THREADS) n = MAX_THREADS;
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpus > 0 && n > (int)cpus) n = (int)cpus;
    return n;
}

int cp_threads(void) {
    if (J.started) return J.nthreads;
    int n = 1;
    const char *s = getenv("CP_THREADS");
    if (s && *s) n = atoi(s);
    start(clamp_threads(n));
    return J.nthreads;
}

int cp_threads_init(int n, int pin) {
    if (J.started) return J.nthreads;
    if (pin >= 0) pin_override = pin != 0;
    if (n <= 0) return cp_threads();
    start(clamp_threads(n));
    return J.nthreads;
}

int cp_would_parallelise(long work, long threshold) {
    return work >= threshold && cp_threads() > 1;
}

void cp_parallel_for(int n, task_fn fn, void *ctx) {
    int nth = cp_threads();
    if (nth <= 1 || n <= 0 || J.nworkers == 0) { if (n > 0) fn(ctx, 0, n); return; }

    int chunk = (n + nth - 1) / nth;
    if (chunk < 1) chunk = 1;

    J.fn = fn;
    J.ctx = ctx;
    J.n = n;
    J.chunk = chunk;
    atomic_store_explicit(&J.remaining, J.nworkers, memory_order_relaxed);
    atomic_fetch_add_explicit(&J.gen, 1, memory_order_release);
    /* MUST take the lock: a trylock can fail while a worker is inside the critical section
       between its generation check and its wait, and skipping the broadcast then strands it.
       That bug deadlocked the 8-thread build. Uncontended this costs ~50ns. */
    pthread_mutex_lock(&J.mu);
    pthread_cond_broadcast(&J.sleep);
    pthread_mutex_unlock(&J.mu);

    int hi = chunk < n ? chunk : n;
    fn(ctx, 0, hi);

    /* spin for the workers; yield rather than block, so the fast path stays ~1us */
    int spins = 0;
    while (atomic_load_explicit(&J.remaining, memory_order_acquire) != 0) {
        if (++spins < SPIN_LIMIT) { __builtin_ia32_pause(); continue; }
        sched_yield();
    }
}

void cp_pool_shutdown(void) {
    if (!J.started) return;
    atomic_store(&J.shutdown, 1);
    atomic_fetch_add(&J.gen, 1);
    pthread_mutex_lock(&J.mu);
    pthread_cond_broadcast(&J.sleep);
    pthread_mutex_unlock(&J.mu);
    for (int i = 0; i < J.nworkers; i++) pthread_join(J.tid[i], NULL);
    J.nworkers = 0;
    J.nthreads = 1;
}
