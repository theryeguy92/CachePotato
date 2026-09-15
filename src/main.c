#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "instr.h"
#include "model.h"
#include "tokenizer.h"
#include "weights.h"

/* usage:
   cachepotato [weights.bin] tok0 tok1 ...   forward on raw token ids
   cachepotato [weights.bin] -p "prompt"     encode -> forward -> next token
   env: CP_LOGITS=1  print the full logits block (verify scripts use this)
        CP_INSTR=<path>  tensor-touch log (Phase 3 groundwork) */
int main(int argc, char **argv) {
    const char *path = "weights/toy.bin";
    const char *prompt = NULL;
    int tokens[256];
    int T = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc)
            prompt = argv[++i];
        else if (argv[i][0] < '0' || argv[i][0] > '9')
            path = argv[i];
        else if (T < 256)
            tokens[T++] = atoi(argv[i]);
    }

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
