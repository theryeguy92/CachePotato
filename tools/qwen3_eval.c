/* qwen3_eval.c — run all 48 layers of Qwen3-30B-A3B from a CPW1 v4 file.
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
 *   <out>.pager    one row per token when paged (regions/bytes fetched, hits, evictions)
 *   <out>.meta     the machine and build that produced the three files above
 */
#define _GNU_SOURCE             /* gethostname, gmtime_r, popen with -std=c11 */
#include "blocks/qwen3_moe.h"
#include "kernels/kernels.h"
#include "pager/pager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

/* PAGING (C1). The pager is attached in qwen3_map() when CP_PAGER is set, and this
   driver supplies what only the model loop knows: the token boundaries, the order of
   note (embed once per token, final norm and head at the end), and where a layer ends.
   With CP_PAGER unset every pager call is a no-op and this is the same unpaged run
   that produced data/traces/eval_p0.* — which is what makes an unlimited-budget paged
   run comparable to it at all. */

/* ---- machine stamp ------------------------------------------------------------
   A recorded artifact has to say WHICH claim it can support. Bit-identity is a
   property of a binary on a machine: the same source compiled for another CPU picks
   different vector code, so the summation order changes and the low bits move even
   with identical weights (measured 2026-10-18: routing bit-identical, logits rel
   4.6e-06, argmax equal — docs/2026-10-18-c1-pager-wiring.md). A baseline without its
   machine cannot be told apart from one that should have matched, so the stamp is
   written next to the outputs and travels with them.

   The instruction-set line is the part that answers the determinism question: which
   SIMD/FMA width the compiler selected for THIS binary, not what -march was asked for.
   That matters because `-march=native` is a request and __AVX2__ is the result. */
static void cpu_model(char *out, size_t n) {
    snprintf(out, n, "unknown");
#ifdef __linux__
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "model name", 10) != 0) continue;
        char *c = strchr(line, ':');
        if (!c) continue;
        for (c++; *c == ' '; c++) { }
        size_t len = strcspn(c, "\n");
        if (len >= n) len = n - 1;
        memcpy(out, c, len);
        out[len] = 0;
        break;
    }
    fclose(f);
#endif
}

static void isa_string(char *out, size_t n) {
    out[0] = 0;
#define CP_ADD(x) do { \
        if (out[0] && strlen(out) + 1 < n) strncat(out, " ", n - strlen(out) - 1); \
        strncat(out, (x), n - strlen(out) - 1); \
    } while (0)
#ifdef __AVX512F__
    CP_ADD("avx512f");
#endif
#ifdef __AVX2__
    CP_ADD("avx2");
#endif
#ifdef __FMA__
    CP_ADD("fma");
#endif
#ifdef __SSE4_2__
    CP_ADD("sse4.2");
#endif
#undef CP_ADD
    if (!out[0]) snprintf(out, n, "baseline x86-64 (no SIMD macros)");
}

static void git_rev(char *out, size_t n) {
    snprintf(out, n, "unknown");
    FILE *p = popen("git rev-parse --short HEAD 2>/dev/null", "r");
    if (!p) return;
    char buf[64];
    if (fgets(buf, sizeof buf, p)) {
        buf[strcspn(buf, "\n")] = 0;
        if (buf[0] && strcmp(buf, "HEAD") != 0) snprintf(out, n, "%s", buf);
    }
    pclose(p);
}

static void write_meta(const char *pre, const char *model_path, const QW3Layer *m,
                       const char *ids_arg, int T, int L, const char *mode, int gen) {
    char fn[512], host[256], cpu[256], isa[96], sha[64], ts[40], pager[96];
    cpu_model(cpu, sizeof cpu);
    isa_string(isa, sizeof isa);
    git_rev(sha, sizeof sha);
    if (gethostname(host, sizeof host) != 0) snprintf(host, sizeof host, "unknown");
    host[sizeof host - 1] = 0;
    time_t now = time(NULL);
    struct tm tmv;
    if (gmtime_r(&now, &tmv)) { if (!strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", &tmv)) snprintf(ts, sizeof ts, "unknown"); }
    else snprintf(ts, sizeof ts, "unknown");

    const char *pe = getenv("CP_PAGER"), *pb = getenv("CP_BUDGET"), *ps = getenv("CP_PAGER_SYS");
    const int sys = ps && ps[0] == '1';
    if (pb) snprintf(pager, sizeof pager, "CP_BUDGET=%s MB%s", pb, sys ? " + CP_PAGER_SYS=1" : "");
    else if (pe && pe[0] == '1') snprintf(pager, sizeof pager, "unlimited (counts, never evicts)");
    else snprintf(pager, sizeof pager, "off (unpaged control)");

    struct stat st;
    long long model_bytes = 0;
    if (stat(model_path, &st) == 0) model_bytes = (long long)st.st_size;

    snprintf(fn, sizeof fn, "%s.meta", pre);
    FILE *f = fopen(fn, "w");
    if (!f) { perror(fn); return; }
    fprintf(f, "# CachePotato eval artifact metadata.\n"
               "# BIT-IDENTITY IS PER-MACHINE: two runs of THIS binary on THIS cpu are byte-identical;\n"
               "# the same source on another cpu is not, because different vector code reorders the\n"
               "# summations. Compare bytes only within a machine; across machines compare the function.\n");
    fprintf(f, "utc: %s\n", ts);
    fprintf(f, "host: %s\n", host);
    fprintf(f, "cpu: %s\n", cpu);
    fprintf(f, "isa: %s\n", isa);
    fprintf(f, "compiler: %s\n", __VERSION__);
    fprintf(f, "git: %s\n", sha);
    fprintf(f, "model: %s\n", model_path);
    fprintf(f, "model_bytes: %lld\n", model_bytes);
    fprintf(f, "mode: %s\n", mode);
    fprintf(f, "layers: %d\n", L);
    fprintf(f, "tokens: %s\n", ids_arg);
    fprintf(f, "n_tokens: %d\n", T);
    fprintf(f, "n_gen: %d\n", gen);
    fprintf(f, "weight_regions: %d\n", m->n_regions);
    fprintf(f, "pager: %s\n", pager);
    fclose(f);
}

/* Run every layer once over T rows. `kv` non-NULL means incremental: the rows are
   positions pos0..pos0+T-1 in the per-layer caches. route/router are optional. */
static int run_layers(QW3Layer *m, float *x, int T, int pos0, QW3KV *kv, int L,
                      int *route, float *router) {
    const int D = m->d_model, K = m->topk, E = m->n_experts;
    QW3Out o;
    for (int l = 0; l < L; l++) {
        if (qwen3_bind_layer(m, l)) { fprintf(stderr, "bind layer %d failed\n", l); return -1; }
        if (qwen3_layer_alloc(&o, m, T)) return -1;
        int rc = kv ? qwen3_layer_kv(m, x, T, pos0, &kv[l], &o)
                    : qwen3_layer(m, x, T, &o);
        if (rc) { qwen3_layer_free(&o); return -1; }
        for (int t = 0; t < T; t++)
            for (int j = 0; j < D; j++)
                x[(size_t)t * D + j] += o.attn[(size_t)t * D + j] + o.mlp[(size_t)t * D + j];
        if (route)  memcpy(route  + (size_t)l * T * K, o.topk_idx, sizeof(int) * (size_t)T * K);
        if (router) memcpy(router + (size_t)l * T * E, o.router, sizeof(float) * (size_t)T * E);
        qwen3_layer_free(&o);
        /* Between layers, not inside one: the 8 experts a layer selected are still being
           used by the next token of this layer, and evicting one between two of its own
           reads is the read amplification the policy exists to avoid. */
        pager_enforce();
        if (!kv) fprintf(stderr, "  layer %d/%d done\n", l + 1, L);
    }
    return 0;
}

/* Final norm + untied head. Overwrites x: the caller has no further use for the
   pre-norm hidden state, and the prefill path never did. */
static void apply_head(QW3Layer *m, float *x, int T, int pos0, float *logits, int V) {
    const int D = m->d_model;
    for (int t = 0; t < T; t++) {
        if (m->pager) { pager_step(pos0 + t); pager_touch_region(m->reg_final_norm); }
        rmsnorm(x + (size_t)t * D, x + (size_t)t * D, m->final_norm, D, m->rms_eps);
    }
    for (int t = 0; t < T; t++) {
        if (m->pager) { pager_step(pos0 + t); pager_touch_region(m->reg_lm_head); }
        if (m->lm_head.enc == QW3_ENC_Q8)
            matmul_wx_q8(logits + (size_t)t * V, x + (size_t)t * D, (const uint8_t *)m->lm_head.p, D, V);
        else
            matmul_wx(logits + (size_t)t * V, x + (size_t)t * D, (const float *)m->lm_head.p, D, V);
    }
}

static void embed_rows(QW3Layer *m, const int *ids, int T, int pos0, float *x, int D) {
    for (int t = 0; t < T; t++) {
        if (m->pager) { pager_step(pos0 + t); pager_touch_region(m->reg_embed); }
        const unsigned char *row = (const unsigned char *)m->embed.p
                                 + (size_t)ids[t] * (size_t)((D / 32) * 34);
        if (m->embed.enc == QW3_ENC_Q8) q8_dequant_row(row, x + (size_t)t * D, D);
        else memcpy(x + (size_t)t * D, (const float *)m->embed.p + (size_t)ids[t] * D,
                    sizeof(float) * D);
    }
}

/* Every artifact this driver writes is checked: a run that computes for two minutes and
   then silently drops its output (or segfaults on a NULL FILE because the directory did
   not exist) is worse than one that refuses to start. */
static FILE *out_file(const char *fn) {
    FILE *f = fopen(fn, "wb");
    if (!f) { perror(fn); exit(1); }
    return f;
}

static int argmax_row(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* Physical read counter: /proc/self/io read_bytes counts block-device reads only, so it
   is the honest check on whether a miss really went to disk (the page cache can serve a
   pager "miss" when CP_PAGER_SYS is off, and that difference must be visible). */
static long io_read_kb(void) {
    FILE *f = fopen("/proc/self/io", "r");
    long val = 0;
    char key[64];
    if (!f) return 0;
    while (fscanf(f, "%63[^:]: %ld\n", key, &val) == 2)
        if (strcmp(key, "read_bytes") == 0) break;
    fclose(f);
    return val / 1024;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.v4cp> <n_layers> <token_id,...> <out-prefix>"
                        " [--decode N]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const int L = atoi(argv[2]);
    int ids[64]; int T = 0;
    /* strtok() rewrites argv[3] in place, so the literal is copied for the metadata
       BEFORE it is parsed — otherwise the .meta records only the first token id. */
    char ids_arg[512];
    snprintf(ids_arg, sizeof ids_arg, "%s", argv[3]);
    for (char *s = strtok(argv[3], ","); s && T < 64; s = strtok(NULL, ",")) ids[T++] = atoi(s);
    const char *pre = argv[4];
    /* Greedy and measurement-only on purpose: C2 measures the paging cost of producing a
       token, so the token itself must be the deterministic argmax. Sampling would change
       which experts are read and belongs with a gate of its own. */
    int gen = 0;
    for (int i = 5; i + 1 < argc; i++)
        if (!strcmp(argv[i], "--decode")) gen = atoi(argv[i + 1]);

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
    embed_rows(&m, ids, T, 0, x, D);

    char fn[512], fn0[512];
    if (gen > 0) {
        /* ---- incremental decode: prefill the prompt, then one token at a time ---- */
        const int cap = T + gen;
        QW3KV *kv = calloc((size_t)L, sizeof *kv);
        if (!kv) return 1;
        for (int l = 0; l < L; l++)
            if (qwen3_kv_alloc(&kv[l], &m, cap)) { fprintf(stderr, "kv alloc failed\n"); return 1; }
        if (run_layers(&m, x, T, 0, kv, L, NULL, NULL)) return 1;

        int   *gtok   = malloc(sizeof(int) * (size_t)gen);
        float *glog   = malloc(sizeof(float) * (size_t)gen * V);
        float *grow   = malloc(sizeof(float) * D);
        double *gms   = malloc(sizeof(double) * (size_t)gen);
        long  *gkb    = malloc(sizeof(long) * (size_t)gen);
        if (!gtok || !glog || !grow || !gms || !gkb) return 1;

        const float *last = x + (size_t)(T - 1) * D;
        for (int s = 0; s < gen; s++) {
            const int pos = T + s;              /* position of the token we are about to feed */
            long io0 = io_read_kb();
            double t0 = now_s();
            memcpy(grow, last, sizeof(float) * D);   /* apply_head normalises in place */
            apply_head(&m, grow, 1, pos - 1, glog + (size_t)s * V, V);
            gtok[s] = argmax_row(glog + (size_t)s * V, V);
            embed_rows(&m, &gtok[s], 1, pos, grow, D);
            if (run_layers(&m, grow, 1, pos, kv, L, NULL, NULL)) return 1;
            gms[s] = now_s() - t0;
            gkb[s] = io_read_kb() - io0;
            last = grow;
            fprintf(stderr, "  token %d/%d: id %d  %.0f ms  %ld KB read\n",
                    s + 1, gen, gtok[s], gms[s] * 1e3, gkb[s]);
        }

        if (m.pager) {
            snprintf(fn0, sizeof fn0, "%s.pager", pre);
            if (pager_dump_steps(fn0)) fprintf(stderr, "pager: could not write %s\n", fn0);
            pager_report();
        }
        snprintf(fn, sizeof fn, "%s.logits", pre); FILE *f = out_file(fn);
        fwrite(glog, sizeof(float), (size_t)gen * V, f); fclose(f);
        snprintf(fn, sizeof fn, "%s.tokens", pre); f = out_file(fn);
        fwrite(gtok, sizeof(int), (size_t)gen, f); fclose(f);
        snprintf(fn, sizeof fn, "%s.steps", pre); f = out_file(fn);
        fprintf(f, "# position token ms disk_kb  (one line per GENERATED token)\n");
        for (int s = 0; s < gen; s++)
            fprintf(f, "%d %d %.3f %ld\n", T + s, gtok[s], gms[s] * 1e3, gkb[s]);
        fclose(f);
        write_meta(pre, path, &m, ids_arg, T, L, "decode", gen);
        printf("generated %d tokens:", gen);
        for (int s = 0; s < gen; s++) printf(" %d", gtok[s]);
        printf("\n");
        return 0;
    }

    /* ---- prefill: all T tokens in one forward (B5 / A5 / the C1 control) ---- */
    int *route = malloc(sizeof(int) * (size_t)L * T * K);
    float *router = malloc(sizeof(float) * (size_t)L * T * E);
    if (run_layers(&m, x, T, 0, NULL, L, route, router)) return 1;

    float *logits = malloc(sizeof(float) * (size_t)T * V);
    apply_head(&m, x, T, 0, logits, V);

    if (m.pager) {
        snprintf(fn0, sizeof fn0, "%s.pager", pre);
        if (pager_dump_steps(fn0)) fprintf(stderr, "pager: could not write %s\n", fn0);
        pager_report();
    }

    snprintf(fn, sizeof fn, "%s.logits", pre); FILE *f = out_file(fn);
    fwrite(logits, sizeof(float), (size_t)T * V, f); fclose(f);
    snprintf(fn, sizeof fn, "%s.route", pre); f = out_file(fn);
    fwrite(route, sizeof(int), (size_t)L * T * K, f); fclose(f);
    snprintf(fn, sizeof fn, "%s.router", pre); f = out_file(fn);
    fwrite(router, sizeof(float), (size_t)L * T * E, f); fclose(f);
    write_meta(pre, path, &m, ids_arg, T, L, "prefill", 0);

    for (int t = 0; t < T; t++)
        printf("token %d: argmax %d\n", t, argmax_row(logits + (size_t)t * V, V));
    return 0;
}
