/* qwen3_eval.c — run all 48 layers of Qwen3-30B-A3B from a CPW1 v4 file, UNPAGED.
 *
 * This is the B5 driver: it answers "does the engine compute this model", deliberately without the
 * pager, so that a disagreement is arithmetic rather than policy. It needs the whole 32.5 GB file
 * mapped, which the T480 cannot do — correctness lives on a machine with the RAM, paging lives on
 * the potato.
 *
 * Emits, for comparison against the recorded baseline:
 *   <out>.logits   T x vocab f32    (final norm + untied lm_head)
 *   <out>.route    L x T x topk i32 (the routing trace)
 *   <out>.router   L x T x experts f32 (router logits, so a mismatch can be localised to a layer)
 */
#include "blocks/qwen3_moe.h"
#include "kernels/kernels.h"
#include "pager/pager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* PAGING (C1). The pager is attached in qwen3_map() when CP_PAGER is set, and this
   driver supplies what only the model loop knows: the token boundaries, the order of
   note (embed once per token, final norm and head at the end), and where a layer ends.
   With CP_PAGER unset every pager call is a no-op and this is the same unpaged run
   that produced data/traces/eval_p0.* — which is what makes an unlimited-budget paged
   run comparable to it at all. */

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.v4cp> <n_layers> <token_id,...> <out-prefix>\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const int L = atoi(argv[2]);
    int ids[64]; int T = 0;
    for (char *s = strtok(argv[3], ","); s && T < 64; s = strtok(NULL, ",")) ids[T++] = atoi(s);
    const char *pre = argv[4];

    QW3Layer m; memset(&m, 0, sizeof m);
    if (qwen3_map(&m, path)) { fprintf(stderr, "map failed\n"); return 1; }
    if (qwen3_bind_layer(&m, 0)) return 1;          /* geometry (d_model, head_dim, ...) */
    const int D = m.d_model, V = m.vocab, K = m.topk, E = m.n_experts;
    fprintf(stderr, "model: D=%d heads=%d/%d hd=%d d_exp=%d experts=%d top%d vocab=%d layers=%d\n",
            D, m.n_heads, m.n_kv, m.head_dim, m.d_exp, E, K, V, L);
    if (!m.embed.p || !m.lm_head.p || !m.final_norm) {
        fprintf(stderr, "missing embed/lm_head/final_norm\n"); return 1;
    }

    float *x = malloc(sizeof(float) * (size_t)T * D);
    for (int t = 0; t < T; t++) {                    /* embedding rows, Q8 or f32 */
        if (m.pager) { pager_step(t); pager_touch_region(m.reg_embed); }
        const unsigned char *row = (const unsigned char *)m.embed.p + (size_t)ids[t] * (size_t)((D / 32) * 34);
        if (m.embed.enc == QW3_ENC_Q8) q8_dequant_row(row, x + (size_t)t * D, D);
        else memcpy(x + (size_t)t * D, (const float *)m.embed.p + (size_t)ids[t] * D, sizeof(float) * D);
    }

    int *route = malloc(sizeof(int) * (size_t)L * T * K);
    float *router = malloc(sizeof(float) * (size_t)L * T * E);
    QW3Out o;
    for (int l = 0; l < L; l++) {
        if (qwen3_bind_layer(&m, l)) { fprintf(stderr, "bind layer %d failed\n", l); return 1; }
        if (qwen3_layer_alloc(&o, &m, T)) return 1;
        if (qwen3_layer(&m, x, T, &o)) return 1;
        for (int t = 0; t < T; t++)
            for (int j = 0; j < D; j++)
                x[(size_t)t * D + j] += o.attn[(size_t)t * D + j] + o.mlp[(size_t)t * D + j];
        memcpy(route + (size_t)l * T * K, o.topk_idx, sizeof(int) * (size_t)T * K);
        memcpy(router + (size_t)l * T * E, o.router, sizeof(float) * (size_t)T * E);
        qwen3_layer_free(&o);
        /* Between layers, not inside one: the 8 experts a layer selected are still
           being used by the next token of this layer, and evicting one between two of
           its own reads is the kind of read amplification the policy exists to avoid. */
        pager_enforce();
        fprintf(stderr, "  layer %d/%d done\n", l + 1, L);
    }

    for (int t = 0; t < T; t++) {
        if (m.pager) { pager_step(t); pager_touch_region(m.reg_final_norm); }
        rmsnorm(x + (size_t)t * D, x + (size_t)t * D, m.final_norm, D, m.rms_eps);
    }
    float *logits = malloc(sizeof(float) * (size_t)T * V);
    for (int t = 0; t < T; t++) {
        if (m.pager) { pager_step(t); pager_touch_region(m.reg_lm_head); }
        if (m.lm_head.enc == QW3_ENC_Q8)
            matmul_wx_q8(logits + (size_t)t * V, x + (size_t)t * D, (const uint8_t *)m.lm_head.p, D, V);
        else
            matmul_wx(logits + (size_t)t * V, x + (size_t)t * D, (const float *)m.lm_head.p, D, V);
    }

    char fn[512], fn0[512];
    if (m.pager) {
        snprintf(fn0, sizeof fn0, "%s.pager", pre);
        if (pager_dump_steps(fn0)) fprintf(stderr, "pager: could not write %s\n", fn0);
        pager_report();
    }

    snprintf(fn, sizeof fn, "%s.logits", pre); FILE *f = fopen(fn, "wb");
    fwrite(logits, sizeof(float), (size_t)T * V, f); fclose(f);
    snprintf(fn, sizeof fn, "%s.route", pre); f = fopen(fn, "wb");
    fwrite(route, sizeof(int), (size_t)L * T * K, f); fclose(f);
    snprintf(fn, sizeof fn, "%s.router", pre); f = fopen(fn, "wb");
    fwrite(router, sizeof(float), (size_t)L * T * E, f); fclose(f);

    for (int t = 0; t < T; t++) {
        int best = 0;
        for (int v = 1; v < V; v++) if (logits[(size_t)t * V + v] > logits[(size_t)t * V + best]) best = v;
        printf("token %d: argmax %d\n", t, best);
    }
    return 0;
}
