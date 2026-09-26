#define _POSIX_C_SOURCE 199309L /* clock_gettime with -std=c11 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "instr.h"
#include "model.h"
#include "pager.h"
#include "tokenizer.h"
#include "weights.h"

#define EOS_TOKEN 50256 /* GPT-2 <|endoftext|> */
#define MAX_NEW 40

/* --- decoding config -------------------------------------------------------
   Both Phase 6 arms degenerate into phrase loops under long GREEDY decoding
   (bench/results/quality-q2-2026-09-25.md), so the REPL defaults to nucleus
   sampling. --greedy restores the previous behaviour. */
static double g_temp = 0.8;
static double g_top_p = 0.95;
static int g_max_new = 128;

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* xorshift64* — deterministic, no shared libc RNG state. */
static unsigned long long g_rng = 88172645463325252ULL;
static void rng_seed(unsigned long long s) {
    g_rng = s ? s : 88172645463325252ULL;
}
static double rng_uniform(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) * (1.0 / 9007199254740992.0);
}

static const float *g_sort_w;
static int cmp_desc(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    if (g_sort_w[ia] != g_sort_w[ib]) return g_sort_w[ia] < g_sort_w[ib] ? 1 : -1;
    return ia - ib; /* deterministic tie-break by token id */
}

/* Temperature + nucleus (top-p) sampling. temp <= 0 falls back to argmax. */
static int sample_token(const float *logits, int n, double temp, double top_p) {
    if (temp <= 0.0) return argmax(logits, n);
    static float *w = NULL;
    static int *idx = NULL;
    if (!w) {
        w = malloc((size_t)n * sizeof(float));
        idx = malloc((size_t)n * sizeof(int));
        if (!w || !idx) return argmax(logits, n);
    }
    float mx = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double e = exp(((double)logits[i] - mx) / temp);
        w[i] = (float)e;
        sum += e;
        idx[i] = i;
    }
    g_sort_w = w;
    qsort(idx, (size_t)n, sizeof(int), cmp_desc);
    double cum = 0.0;
    int k = 0;
    for (; k < n; k++) {
        cum += (double)w[idx[k]] / sum;
        if (cum >= top_p) { k++; break; }
    }
    if (k < 1) k = 1;
    double tot = 0.0;
    for (int i = 0; i < k; i++) tot += w[idx[i]];
    double r = rng_uniform() * tot, acc = 0.0;
    for (int i = 0; i < k; i++) {
        acc += w[idx[i]];
        if (r < acc) return idx[i];
    }
    return idx[k - 1];
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long rss_kb(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    long total, resident = 0;
    if (f) { if (fscanf(f, "%ld %ld", &total, &resident) != 2) resident = 0; fclose(f); }
    return resident * 4; /* 4K pages */
}

static long disk_read_kb(void) {
    /* /proc/self/io read_bytes = actual block-device reads (NOT page-cache
       hits) — the honest measure of whether weights came off disk */
    FILE *f = fopen("/proc/self/io", "r");
    char key[64];
    long val = 0;
    if (f) {
        while (fscanf(f, "%63[^:]: %ld\n", key, &val) == 2)
            if (strcmp(key, "read_bytes") == 0) break;
        fclose(f);
    }
    return val / 1024;
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
    if (m.is_v3 && m.w.is_moe) {
        fprintf(stderr,
                "this is a CPW1 v3 MoE model, and the incremental (KV-cached) "
                "path does not implement MoE yet.\n"
                "Use --moe-sel <tokens.bin> <start> <T> <out.bin> for the routing "
                "check; sampled MoE generation is the next Phase 7 step.\n");
        return 1;
    }
    const char *b = getenv("CP_BUDGET");
    if (b)
        pager_init(&m, atol(b) * 1024L * 1024L);
    RunState rs;
    runstate_alloc(&rs, &m);

    fprintf(stderr,
            "cachepotato repl — %s (%d layers, d=%d). %s, up to %d new tokens."
            " ctrl-d to quit.\n",
            path, m.hdr.n_layer, m.hdr.d_model,
            g_temp > 0.0 ? "nucleus sampling" : "greedy", g_max_new);
    if (g_temp > 0.0)
        fprintf(stderr, "  temp %.2f | top-p %.2f  (--greedy for the old path)\n",
                g_temp, g_top_p);

    char line[1024];
    while (fprintf(stderr, "\nprompt> "), fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\n")] = 0;
        if (!line[0])
            continue;

        int ids[512];
        int n = tokenizer_encode(&tk, line, ids, 512);
        if (n >= m.hdr.n_ctx - g_max_new - 1) {
            fprintf(stderr, "(prompt too long)\n");
            continue;
        }

        int cold = getenv("CP_COLD") != NULL;
        long io0 = disk_read_kb();
        double t0 = now_s();
        for (int i = 0; i < n; i++) {        /* warm the KV cache on the prompt */
            forward_step(&m, &rs, ids[i], i);
            if (cold) model_evict_all(&m);
        }

        int gen = 0, pos = n;
        for (; gen < g_max_new && pos < m.hdr.n_ctx; gen++, pos++) {
            int next = sample_token(rs.logits, m.hdr.vocab, g_temp, g_top_p);
            if (next == EOS_TOKEN)
                break;
            tokenizer_decode(&tk, &next, 1, stdout);
            fflush(stdout);
            forward_step(&m, &rs, next, pos);
            if (cold) model_evict_all(&m);
        }
        double dt = now_s() - t0;
        fprintf(stderr, "\n[%d tokens in %.1fs — %.2f tok/s | rss %ld MB | disk %ld MB]%s\n",
                gen, dt, gen / dt, rss_kb() / 1024,
                (disk_read_kb() - io0) / 1024, cold ? " (forced-cold)" : "");
    }

    pager_report();
    runstate_free(&rs, &m);
    model_free(&m);
    tokenizer_free(&tk);
    return 0;
}


/* --- MoE routing gate (CPW1 v3) --------------------------------------------
   Run the trained model over a token window and dump the per-token top-k expert
   selections, so tools/verify_cpw1_v3.py can require the C engine to reproduce
   what was recorded from the PyTorch model. This is the check that a weight
   export is faithful AT THE ROUTING LEVEL, not merely loadable. */
static int moe_sel_mode(const char *path, const char *tokbin, long start, int T,
                        const char *out_path) {
    Model m;
    if (model_load(&m, path) != 0)
        return 1;
    if (!m.is_v3 || !m.w.is_moe) {
        fprintf(stderr, "%s is not a CPW1 v3 MoE file\n", path);
        return 1;
    }
    FILE *f = fopen(tokbin, "rb");
    if (!f) { perror("tokens"); return 1; }
    if (fseek(f, start * 2L, SEEK_SET) != 0) { perror("seek"); return 1; }
    uint16_t *raw = malloc((size_t)T * sizeof(uint16_t));
    int *toks = malloc((size_t)T * sizeof(int));
    if (fread(raw, sizeof(uint16_t), (size_t)T, f) != (size_t)T) {
        fprintf(stderr, "short read from %s\n", tokbin);
        return 1;
    }
    fclose(f);
    for (int i = 0; i < T; i++) toks[i] = raw[i];

    int nm = m.w.n_moe, K = m.w.topk;
    int *sel = malloc((size_t)nm * T * K * sizeof(int));
    float *logits = malloc((size_t)T * m.hdr.vocab * sizeof(float));
    if (!raw || !toks || !sel || !logits) { fprintf(stderr, "oom\n"); return 1; }

    double t0 = now_s();
    forward_moe(&m, toks, T, logits, sel, m.h3.lam);
    double dt = now_s() - t0;

    FILE *o = fopen(out_path, "wb");
    if (!o) { perror("out"); return 1; }
    fwrite(&nm, sizeof(int), 1, o);
    fwrite(&T, sizeof(int), 1, o);
    fwrite(&K, sizeof(int), 1, o);
    fwrite(sel, sizeof(int), (size_t)nm * T * K, o);
    fclose(o);

    int best = 0;
    const float *row = logits + (size_t)(T - 1) * m.hdr.vocab;
    for (int j = 1; j < m.hdr.vocab; j++) if (row[j] > row[best]) best = j;
    fprintf(stderr,
            "moe-sel: %d MoE layers x %d tokens x top-%d, lam=%.3f, %.1fs "
            "(%.0f tok/s) | next-argmax %d -> %s\n",
            nm, T, K, m.h3.lam, dt, dt > 0 ? T / dt : 0.0, best, out_path);

    free(raw); free(toks); free(sel); free(logits);
    model_free(&m);
    return 0;
}

/* usage:
   cachepotato [weights.bin] tok0 tok1 ...   forward on raw token ids
   cachepotato [weights.bin] -p "prompt"     encode -> forward -> next token
   cachepotato [weights.bin] -i              interactive REPL (sampled)
     --temp T --top-p P --max-new N --seed S   decoding (default 0.8 / 0.95 / 128)
     --greedy                                  restore greedy decoding
   env: CP_LOGITS=1  print the full logits block (verify scripts use this)
        CP_INSTR=<path>  tensor-touch log (Phase 3 groundwork)
        CP_BUDGET=<MB>   pager budget; CP_COLD forces eviction every token */
int main(int argc, char **argv) {
    const char *path = "weights/toy.bin";
    const char *prompt = NULL;
    int interactive = 0;
    int tokens[256];
    int T = 0;
    int moe_sel = 0;
    const char *ms_bin = NULL, *ms_out = NULL;
    long ms_start = 0;
    int ms_T = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc)
            g_temp = atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc)
            g_top_p = atof(argv[++i]);
        else if (strcmp(argv[i], "--max-new") == 0 && i + 1 < argc)
            g_max_new = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            rng_seed(strtoull(argv[++i], NULL, 10));
        else if (strcmp(argv[i], "--greedy") == 0)
            g_temp = 0.0;
        else if (strcmp(argv[i], "--moe-sel") == 0 && i + 4 < argc) {
            ms_bin = argv[++i];
            ms_start = atol(argv[++i]);
            ms_T = atoi(argv[++i]);
            ms_out = argv[++i];
            moe_sel = 1;
        }
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc)
            prompt = argv[++i];
        else if (strcmp(argv[i], "-i") == 0)
            interactive = 1;
        else if (argv[i][0] < '0' || argv[i][0] > '9')
            path = argv[i];
        else if (T < 256)
            tokens[T++] = atoi(argv[i]);
    }

    if (moe_sel)
        return moe_sel_mode(path, ms_bin, ms_start, ms_T, ms_out);

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
