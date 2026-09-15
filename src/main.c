#define _POSIX_C_SOURCE 199309L /* clock_gettime with -std=c11 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "instr.h"
#include "model.h"
#include "tokenizer.h"
#include "weights.h"

#define EOS_TOKEN 50256 /* GPT-2 <|endoftext|> */
#define MAX_NEW 40

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Stage-1 REPL (Interface-TUI-Design): prompt in, streamed tokens out,
   greedy decoding, tok/s on stderr. */
static int repl(const char *path) {
    Tokenizer tk;
    if (tokenizer_load(&tk, "weights/tokenizer.bin") != 0)
        return 1;
    Model m;
    if (model_load(&m, path) != 0)
        return 1;
    RunState rs;
    runstate_alloc(&rs, &m);

    fprintf(stderr, "cachepotato repl — %s (%d layers, d=%d). ctrl-d to quit.\n",
            path, m.hdr.n_layer, m.hdr.d_model);

    char line[1024];
    while (fprintf(stderr, "\nprompt> "), fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\n")] = 0;
        if (!line[0])
            continue;

        int ids[512];
        int n = tokenizer_encode(&tk, line, ids, 512);
        if (n >= m.hdr.n_ctx - MAX_NEW - 1) {
            fprintf(stderr, "(prompt too long)\n");
            continue;
        }

        double t0 = now_s();
        for (int i = 0; i < n; i++)          /* warm the KV cache on the prompt */
            forward_step(&m, &rs, ids[i], i);

        int gen = 0, pos = n;
        for (; gen < MAX_NEW && pos < m.hdr.n_ctx; gen++, pos++) {
            int next = argmax(rs.logits, m.hdr.vocab);
            if (next == EOS_TOKEN)
                break;
            tokenizer_decode(&tk, &next, 1, stdout);
            fflush(stdout);
            forward_step(&m, &rs, next, pos);
        }
        double dt = now_s() - t0;
        fprintf(stderr, "\n[%d tokens in %.1fs — %.2f tok/s]\n", gen, dt, gen / dt);
    }

    runstate_free(&rs, &m);
    model_free(&m);
    tokenizer_free(&tk);
    return 0;
}


/* usage:
   cachepotato [weights.bin] tok0 tok1 ...   forward on raw token ids
   cachepotato [weights.bin] -p "prompt"     encode -> forward -> next token
   env: CP_LOGITS=1  print the full logits block (verify scripts use this)
        CP_INSTR=<path>  tensor-touch log (Phase 3 groundwork) */
int main(int argc, char **argv) {
    const char *path = "weights/toy.bin";
    const char *prompt = NULL;
    int interactive = 0;
    int tokens[256];
    int T = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc)
            prompt = argv[++i];
        else if (strcmp(argv[i], "-i") == 0)
            interactive = 1;
        else if (argv[i][0] < '0' || argv[i][0] > '9')
            path = argv[i];
        else if (T < 256)
            tokens[T++] = atoi(argv[i]);
    }

    if (interactive)
        return repl(path);

    Tokenizer tk;
    int have_tk = 0;
    if (prompt) {
        if (tokenizer_load(&tk, "weights/tokenizer.bin") != 0)
            return 1;
        have_tk = 1;
        T = tokenizer_encode(&tk, prompt, tokens, 256);
    }
    if (T == 0) { tokens[0] = 5; tokens[1] = 1; tokens[2] = 7; T = 3; }

    Model m;
    if (model_load(&m, path) != 0)
        return 1;

    fprintf(stderr, "loaded %s: vocab=%d d_model=%d n_heads=%d d_ff=%d n_layer=%d n_ctx=%d\n",
            path, m.hdr.vocab, m.hdr.d_model, m.hdr.n_heads,
            m.hdr.d_ff, m.hdr.n_layer, m.hdr.n_ctx);

    int V = m.hdr.vocab;
    float *logits = malloc((size_t)T * V * sizeof(float));
    forward(&m, tokens, T, logits);

    if (getenv("CP_LOGITS")) {
        /* stdout carries ONLY the logits block — verify scripts parse this */
        printf("logits:\n");
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < V; j++)
                printf(" %9.6f", logits[t * V + j]);
            printf("\n");
        }
    }

    if (have_tk) {
        printf("tokens:");
        for (int i = 0; i < T; i++) printf(" %d", tokens[i]);
        printf("\n");
        printf("decoded: '");
        tokenizer_decode(&tk, tokens, T, stdout);
        printf("'\n");

        const float *row = logits + (size_t)(T - 1) * V;
        int best = 0;
        for (int j = 1; j < V; j++)
            if (row[j] > row[best]) best = j;
        printf("next: %d '", best);
        tokenizer_decode(&tk, &best, 1, stdout);
        printf("'\n");
        tokenizer_free(&tk);
    }

    free(logits);
    instr_report();
    model_free(&m);
    return 0;
}
