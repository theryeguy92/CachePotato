#define _GNU_SOURCE
/* qwen3_chat.c — talk to Qwen3-30B-A3B through the paging engine.
 *
 * Prompt -> ChatML -> tokenize -> prefill (KV cache) -> greedy decode, streamed token by token,
 * stopping at <|im_end|>. The 30B does not fit in 16 GB, so the weights come from the pager:
 *
 *   CP_PAGER=1 CP_BUDGET=13312 CP_THREADS=4 CP_PIN_CORES=1 \
 *     ./build/qwen3_chat scratch/qwen3_q4attn.v4cp 128
 *
 * Expected on this T480: the first prefill is disk-bound (~20-25 s cold, ~3 s warm), then
 * ~0.8 s per token (1.2 tok/s) at a 13 GB budget with the 4-bit file, ~1.25 s at Q8.
 *
 * Greedy only. There is no sampler on purpose: the engine's whole gate structure rests on the
 * output being a deterministic function of the weights, and a temperature knob would be the
 * first thing to break that. If sampling is ever wanted it goes behind a flag with its own
 * recorded test, not silently in this loop.
 */
#include "blocks/qwen3_moe.h"
#include "kernels/kernels.h"
#include "kernels/parallel.h"
#include "pager/pager.h"
#include "tokenizers/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int tokenizer_qwen_load_unicode(const char *path);
int tokenizer_qwen_encode(const Tokenizer *t, const char *text, int *ids, int max_ids);

#define IM_END  151645        /* <|im_end|> */
#define IM_START 151644
#define MAX_CTX 2048

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- the three pieces qwen3_eval also has: embed a row, project to logits, run the layers.
 * The matmuls all go through qwen3_mm(), the single encoding-agnostic dispatch, so this file
 * does not need to know whether the file is f32, Q8_0 or Q4_0. ---- */

static void embed_rows(QW3Layer *m, const int *ids, int T, int pos0, float *x, int D) {
    const size_t q8_row = (size_t)((D / 32) * 34);
    const size_t q4_row = (size_t)((D / 32) * 18);
    for (int t = 0; t < T; t++) {
        int row = ids[t];
        const void *base;
        if (m->embed_chunks > 0 && m->embed_rows > 0) {
            int c = row / m->embed_rows, r = row - c * m->embed_rows;
            if (c >= m->embed_chunks) { c = m->embed_chunks - 1; r = row - c * m->embed_rows; }
            if (m->pager) { pager_step(pos0 + t); base = pager_region_get(m->reg_embed_chunk[c]); }
            else base = m->embed_chunk_ptr[c];
            row = r;
        } else {
            if (m->pager) { pager_step(pos0 + t); base = pager_region_get(m->reg_embed); }
            else base = m->embed.p;
        }
        if (m->embed.enc == QW3_ENC_Q8)
            q8_dequant_row((const unsigned char *)base + (size_t)row * q8_row,
                           x + (size_t)t * D, D);
        else if (m->embed.enc == QW3_ENC_Q4)
            q4_dequant_row((const unsigned char *)base + (size_t)row * q4_row,
                           x + (size_t)t * D, D);
        else memcpy(x + (size_t)t * D, (const float *)base + (size_t)row * D,
                    sizeof(float) * D);
    }
}

static void apply_head(QW3Layer *m, const float *x, int T, int pos0, float *logits, int V) {
    const int D = m->d_model;
    for (int t = 0; t < T; t++) {
        const float *fn = m->final_norm;
        if (m->pager) { pager_step(pos0 + t); fn = pager_region_get(m->reg_final_norm); }
        rmsnorm((float *)x + (size_t)t * D, (float *)x + (size_t)t * D, fn, D, m->rms_eps);
    }
    for (int t = 0; t < T; t++) {
        QW3T head = m->lm_head;
        if (m->pager) { pager_step(pos0 + t); head.p = pager_region_get(m->reg_lm_head); }
        qwen3_mm(&head, logits + (size_t)t * V, x + (size_t)t * D, D, V);
    }
}

static int run_layers(QW3Layer *layers, float *x, int T, int pos0, QW3KV *kv, int L) {
    QW3Out o;
    for (int l = 0; l < L; l++) {
        QW3Layer *m = &layers[l];
        if (qwen3_layer_alloc(&o, m, T)) return -1;
        if (qwen3_layer_kv(m, x, T, pos0, kv ? &kv[l] : NULL, &o)) { qwen3_layer_free(&o); return -1; }
        /* BOTH residuals: attention and expert output. Dropping o.attn here produces
           fluent-looking garbage — it is not the kind of bug that fails loudly. */
        for (size_t t = 0; t < (size_t)T; t++)
            for (int j = 0; j < m->d_model; j++)
                x[t * m->d_model + j] += o.attn[t * m->d_model + j] + o.mlp[t * m->d_model + j];
        qwen3_layer_free(&o);
        /* Between layers, not inside one: a layer's 8 experts are still in use for the next
           token, and evicting between two of their own reads is what the policy avoids. */
        pager_enforce();
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *weights = argc > 1 ? argv[1] : "scratch/qwen3_q4attn.v4cp";
    /* the layer count is not in QW3Layer: the v4 header carries it, and qwen3_map keeps
       it in the pager's geometry, so the driver is told (same as qwen3_eval). */
    const int L = argc > 2 ? atoi(argv[2]) : 48;
    const int max_new  = argc > 3 ? atoi(argv[3]) : 128;
    const char *budget = getenv("CP_BUDGET") ? getenv("CP_BUDGET") : "13312";

    if (tokenizer_qwen_load_unicode("weights/qwen2.5-unicode.bin")) return 1;
    Tokenizer tk;
    if (tokenizer_load(&tk, "weights/qwen2.5-tokenizer.bin")) return 1;

    QW3Layer m;
    if (qwen3_map(&m, weights)) { fprintf(stderr, "cannot map %s\n", weights); return 1; }
    if (qwen3_bind_layer(&m, 0)) { fprintf(stderr, "cannot bind layer 0\n"); return 1; }
    const int D = m.d_model, E = m.n_experts, V = m.vocab;

    /* bind every layer once (the 48x-per-token walk cost 11% back when it was per token) */
    QW3Layer *layers = calloc((size_t)L, sizeof(QW3Layer));
    if (!layers) return 1;
    for (int l = 0; l < L; l++) {
        layers[l] = m;
        layers[l].exp = calloc((size_t)E * 3, sizeof(QW3T));
        layers[l].reg_exp = calloc((size_t)E, sizeof(int));
        if (!layers[l].exp || !layers[l].reg_exp || qwen3_bind_layer(&layers[l], l)) {
            fprintf(stderr, "bind layer %d failed\n", l); return 1;
        }
    }
    QW3KV *kv = calloc((size_t)L, sizeof(QW3KV));
    if (!kv) return 1;
    for (int l = 0; l < L; l++)
        if (qwen3_kv_alloc(&kv[l], &layers[l], MAX_CTX)) { fprintf(stderr, "kv alloc failed\n"); return 1; }

    float *x = malloc(sizeof(float) * (size_t)MAX_CTX * D);
    float *logits = malloc(sizeof(float) * (size_t)V);
    if (!x || !logits) return 1;

    int *ids = malloc(sizeof(int) * (size_t)MAX_CTX);   /* the conversation, tokenised once */
    int *turn = malloc(sizeof(int) * (size_t)MAX_CTX);  /* just the new fragment */
    if (!ids || !turn) return 1;
    int hist = 0;                                       /* tokens already in the KV cache */

    fprintf(stderr, "qwen3-30b-a3b — %d layers, d=%d, %d experts top-%d, vocab %d, budget %s MB, "
                    "%d threads.\ntype a message, ctrl-D to quit. greedy decoding; "
                    "\"exit\" to leave.\n",
            L, D, E, m.topk, V, budget, cp_threads());

    char line[8192];
    for (;;) {
        fprintf(stderr, "\nuser> ");
        if (!fgets(line, sizeof line, stdin)) break;
        size_t n = strlen(line);
        while (n && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
        if (!n) continue;
        if (!strcmp(line, "exit") || !strcmp(line, "quit")) break;

        /* ChatML, with /no_think: Qwen3 is a thinking model, and without this the whole
           visible budget goes into a  block before the answer starts. The fragment is
           APPENDED to the conversation, so earlier turns stay in the KV cache instead of being
           re-prefilled — that is what makes follow-ups work (and what keeps the prefill cost
           proportional to the new message, not the whole history). */
        if (hist > MAX_CTX - max_new - 64) {
            fprintf(stderr, "(context full: %d tokens — starting a new conversation)\n", hist);
            hist = 0;
        }
        char prompt[16384];
        snprintf(prompt, sizeof prompt,
                 "<|im_start|>user\n%s /no_think<|im_end|>\n<|im_start|>assistant\n", line);
        int Tn = tokenizer_qwen_encode(&tk, prompt, turn, MAX_CTX - hist - 8);
        if (Tn <= 0) { fprintf(stderr, "tokenize failed\n"); continue; }
        memcpy(ids + hist, turn, sizeof(int) * (size_t)Tn);
        const int pos0 = hist;                 /* where this fragment starts in the cache */
        hist += Tn;

        double t0 = now();
        embed_rows(&layers[0], ids + pos0, Tn, pos0, x, D);
        if (run_layers(layers, x, Tn, pos0, kv, L)) { fprintf(stderr, "prefill failed\n"); break; }
        double t1 = now();

        /* the last prefill position gives the first token */
        apply_head(&layers[0], x + (size_t)(Tn - 1) * D, 1, hist - 1, logits, V);
        fprintf(stderr, "assistant> ");
        int pos = hist, produced = 0;
        double t2 = now();
        for (;;) {
            int best = 0;
            for (int j = 1; j < V; j++) if (logits[j] > logits[best]) best = j;
            if (best == IM_END || produced >= max_new || pos >= MAX_CTX - 1) break;
            tokenizer_decode(&tk, &best, 1, stdout);
            fflush(stdout);
            if (pos < MAX_CTX - 2) ids[pos] = best, hist = pos + 1;
            embed_rows(&layers[0], &best, 1, pos, x, D);
            if (run_layers(layers, x, 1, pos, kv, L)) break;
            apply_head(&layers[0], x, 1, pos, logits, V);
            pos++; produced++;
        }
        double t3 = now();
        fprintf(stderr, "\n[prefill %d new tok (context %d) in %.1fs | %d tok in %.1fs (%.2f tok/s)]\n",
                Tn, hist, t1 - t0, produced, t3 - t2, produced ? produced / (t3 - t2) : 0.0);
    }
    free(turn); free(ids); free(logits); free(x); free(kv); free(layers);
    return 0;
}
