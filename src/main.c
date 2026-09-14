#include <stdio.h>
#include <stdlib.h>

#include "instr.h"
#include "model.h"
#include "weights.h"

/* usage: cachepotato [weights.bin] [tok0 tok1 ...]
   defaults: weights/toy.bin, tokens 5 1 7 */
int main(int argc, char **argv) {
    const char *path = "weights/toy.bin";
    int tokens[256];
    int T = 0;

    int argi = 1;
    if (argc > 1 && (argv[1][0] < '0' || argv[1][0] > '9'))
        path = argv[argi++]; /* first arg is non-numeric -> weight file */
    while (argi < argc && T < 256)
        tokens[T++] = atoi(argv[argi++]);
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

    /* stdout carries ONLY the logits block — verify scripts parse this */
    printf("logits:\n");
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < V; j++)
            printf(" %9.6f", logits[t * V + j]);
        printf("\n");
    }

    free(logits);
    instr_report();
    model_free(&m);
    return 0;
}
