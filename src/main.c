#include <stdio.h>
#include <stdlib.h>

#include "model.h"
#include "weights.h"

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "weights/toy.bin";

    Model m;
    if (model_load(&m, path) != 0)
        return 1;

    printf("loaded %s: vocab=%d d_model=%d n_heads=%d d_ff=%d n_layer=%d\n",
           path, m.hdr.vocab, m.hdr.d_model, m.hdr.n_heads,
           m.hdr.d_ff, m.hdr.n_layer);

    /* toy forward pass on the exercise's tokens [5,1,7] — verified by
       tools/verify_toy.py against the numpy ground truth */
    int tokens[] = {5, 1, 7};
    int T = 3, V = m.hdr.vocab;
    float *logits = malloc(T * V * sizeof(float));
    forward(&m, tokens, T, logits);

    printf("logits:\n");
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < V; j++)
            printf(" %9.6f", logits[t * V + j]);
        printf("\n");
    }

    free(logits);
    model_free(&m);
    return 0;
}
