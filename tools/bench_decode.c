/* bench_decode.c — one number per run, no chat loop, no tokenizer, no stdout.
 *
 * Measures decode only: N qwen2_step calls after a warm-up pass, so page cache is populated and
 * the first-touch cost is excluded. Prints ONE machine-readable line so a harness can parse it.
 *
 * Deliberately not a per-shape synthetic loop: that measured L3-resident bandwidth and was
 * optimistic by ~2x. This runs the real step on the real file.
 */
#define _POSIX_C_SOURCE 200809L
#include "blocks/qwen2.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
                          return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    const char *weights = argc > 1 ? argv[1] : "scratch/qwen_q8e.v4cp";
    int ntok = argc > 2 ? atoi(argv[2]) : 48;
    int warm = argc > 3 ? atoi(argv[3]) : 16;

    QW2 m; void *map = NULL; size_t ml = 0;
    if (qwen2_load_v4(&m, weights, &map, &ml)) return 1;
    QW2Cache *c = qwen2_cache_alloc(&m, 2048);
    if (!c) return 1;
    float *logits = malloc(sizeof(float) * (size_t)m.vocab);

    int tok = 1000, pos = 0;
    for (int i = 0; i < warm; i++) { if (qwen2_step(&m, c, tok, pos++, logits)) return 1; }
    double t0 = now();
    for (int i = 0; i < ntok; i++) {
        int best = 0;
        for (int j = 1; j < m.vocab; j++) if (logits[j] > logits[best]) best = j;
        if (qwen2_step(&m, c, best, pos++, logits)) return 1;
    }
    double dt = now() - t0;
    printf("BENCH ms_per_token=%.3f tok_s=%.3f\n", dt * 1e3 / ntok, ntok / dt);
    return 0;
}
