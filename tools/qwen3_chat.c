/* qwen3_chat.c — talk to Qwen3-30B-A3B through the paging engine, from the terminal.
 *
 *   ./build/qwen3_chat <model.v4cp> [max_new]
 *
 * A thin client of engine/api/cachepotato.h: everything that produces tokens lives there, so
 * this driver and the Rust CLI run the same code. The old form `<model> <layers> <max_new>`
 * is still accepted; the layer count now comes from the file and the argument is ignored.
 *
 * Defaults are chosen for the machine: the pager budget from MemAvailable, one thread per
 * physical core, and the tokenizer from next to the model (or weights/). The old environment
 * variables still win when set:
 *
 *   CP_BUDGET=13312 CP_THREADS=4 CP_PIN_CORES=1 ./build/qwen3_chat scratch/qwen3_q4attn.v4cp
 *
 * Expected on this T480: the first prefill is disk-bound (~20-25 s cold, ~3 s warm), then
 * ~0.8 s per token (1.2 tok/s) at a 13 GB budget with the 4-bit file, ~1.25 s at Q8.
 *
 * Greedy only. There is no sampler on purpose: the engine's whole gate structure rests on the
 * output being a deterministic function of the weights.
 */
#include "api/cachepotato.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int print_token(const char *bytes, size_t len, void *user) {
    (void)user;
    fwrite(bytes, 1, len, stdout);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    const char *weights = argc > 1 ? argv[1] : "scratch/qwen3_q4attn.v4cp";
    const int max_new = argc > 3 ? atoi(argv[3]) : argc > 2 ? atoi(argv[2]) : 128;

    cp_options opt;
    cp_options_default(&opt);
    if (getenv("CP_BUDGET") || getenv("CP_PAGER")) opt.budget_mb = CP_BUDGET_ENV;
    if (getenv("CP_THREADS")) opt.threads = atoi(getenv("CP_THREADS"));

    char err[512];
    cp_model *m = cp_open(weights, &opt, err, sizeof err);
    if (!m) { fprintf(stderr, "qwen3_chat: %s\n", err); return 1; }

    cp_stats s;
    cp_get_stats(m, &s);
    char budget[48];
    if (s.budget_mb < 0) snprintf(budget, sizeof budget, "no pager");
    else if (s.budget_mb >= (1L << 40)) snprintf(budget, sizeof budget, "unlimited");
    else snprintf(budget, sizeof budget, "%ld MB", s.budget_mb);
    fprintf(stderr, "qwen3-30b-a3b — %d layers, d=%d, %d experts top-%d, vocab %d, budget %s, "
                    "%d threads.\ntype a message, ctrl-D to quit. greedy decoding; "
                    "\"exit\" to leave.\n",
            s.n_layers, s.d_model, s.n_experts, s.topk, s.vocab, budget, s.threads);

    char line[8192];
    for (;;) {
        fprintf(stderr, "\nuser> ");
        if (!fgets(line, sizeof line, stdin)) break;
        size_t n = strlen(line);
        while (n && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
        if (!n) continue;
        if (!strcmp(line, "exit") || !strcmp(line, "quit")) break;

        fprintf(stderr, "assistant> ");
        cp_turn t;
        int rc = cp_chat(m, line, max_new, print_token, NULL, &t);
        if (t.context_reset)
            fprintf(stderr, "(context was full — started a new conversation)\n");
        if (rc == -2) { fprintf(stderr, "tokenize failed\n"); continue; }
        if (rc < 0) { fprintf(stderr, "forward pass failed\n"); break; }
        fprintf(stderr, "\n[prefill %d new tok (context %d) in %.1fs | %d tok in %.1fs (%.2f tok/s)]\n",
                t.prompt_tokens, t.context_tokens, t.prefill_s, t.generated_tokens, t.decode_s,
                t.generated_tokens ? t.generated_tokens / t.decode_s : 0.0);
    }
    cp_close(m);
    return 0;
}
