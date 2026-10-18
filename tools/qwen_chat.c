#define _POSIX_C_SOURCE 200809L
/* qwen_chat.c — a real conversation with Qwen2.5-0.5B through the engine.
 *
 * Prompt -> ChatML -> tokenize -> prefill (KV cache) -> greedy decode to <|im_end|>.
 * This is the first path where a human types something and the model answers.
 */
#include "blocks/qwen2.h"
#include "tokenizers/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int tokenizer_qwen_load_unicode(const char *path);
int tokenizer_qwen_encode(const Tokenizer *t, const char *text, int *ids, int max_ids);

#define IM_END 151645
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    const char *weights = argc > 1 ? argv[1] : "scratch/qwen.v4cp";
    int max_new = argc > 2 ? atoi(argv[2]) : 96;
    if (tokenizer_qwen_load_unicode("weights/qwen2.5-unicode.bin")) return 1;
    Tokenizer tk;
    if (tokenizer_load(&tk, "weights/qwen2.5-tokenizer.bin")) return 1;
    QW2 m; void *map = NULL; size_t ml = 0;
    if (qwen2_load_v4(&m, weights, &map, &ml)) return 1;
    QW2Cache *c = qwen2_cache_alloc(&m, 2048);
    if (!c) return 1;

    char line[4096];
    fprintf(stderr, "qwen2.5-0.5b — %d layers, d=%d, heads %d/%d. ctrl-D to quit.\n",
            m.n_layers, m.d_model, m.n_heads, m.n_kv);

    while (1) {
        fprintf(stderr, "\nuser> ");
        if (!fgets(line, sizeof line, stdin)) break;
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (!L) continue;

        char prompt[8192];
        snprintf(prompt, sizeof prompt,
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", line);
        int ids[4096];
        int n = tokenizer_qwen_encode(&tk, prompt, ids, 4096);
        if (n <= 0) continue;

        float *logits = malloc(sizeof(float) * (size_t)m.vocab);
        double t0 = now();
        for (int i = 0; i < n; i++)
            if (qwen2_step(&m, c, ids[i], i, logits)) { fprintf(stderr, "step failed\n"); return 1; }
        double t1 = now();

        fprintf(stderr, "assistant> ");
        int pos = n, produced = 0;
        double t2 = now();
        double t_arg = 0, t_emit = 0, t_step = 0;
        while (produced < max_new && pos < 2047) {
            double a0 = now();
            int best = 0;
            for (int j = 1; j < m.vocab; j++) if (logits[j] > logits[best]) best = j;
            t_arg += now() - a0;
            if (best == IM_END) { fflush(stdout); break; }
            double e0 = now();
            tokenizer_decode(&tk, &best, 1, stdout);
            fflush(stdout);
            t_emit += now() - e0;
            double s0 = now();
            if (qwen2_step(&m, c, best, pos, logits)) break;
            t_step += now() - s0;
            pos++; produced++;
        }
        free(logits);
        {
            double wall = now() - t2;
            double acc = t_step + t_arg + t_emit;
            fprintf(stderr, "\n[loop] step %.0f | argmax %.0f | emit %.0f | "
                    "UNACCOUNTED %.0f ms  (of %.0f ms)\n",
                    t_step*1e3, t_arg*1e3, t_emit*1e3, (wall-acc)*1e3, wall*1e3);
        }
        double t3 = now();
        fprintf(stderr, "\n[prefill %d tok in %.2fs (%.1f tok/s) | %d tok in %.2fs (%.2f tok/s)]\n",
                n, t1 - t0, n / (t1 - t0), produced, t3 - t2,
                produced ? produced / (t3 - t2) : 0.0);
    }
    if (getenv("CP_PROF")) {
        extern double g_t_attn, g_t_mlp, g_t_head;
        double tot = g_t_attn + g_t_mlp + g_t_head;
        fprintf(stderr, "\n[profile] attention %.1f ms  mlp %.1f ms  head %.1f ms  "
                        "= %.1f ms of compute\n", g_t_attn*1e3, g_t_mlp*1e3, g_t_head*1e3, tot*1e3);
    }
    qwen2_cache_free(c);

    return 0;
}
