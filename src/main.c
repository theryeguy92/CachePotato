#define _POSIX_C_SOURCE 199309L /* clock_gettime with -std=c11 */
#define _GNU_SOURCE         /* strdup/strtok_r with -std=c11 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "instr.h"
#include "model.h"
#include "pager.h"
#include "replay.h"
#include "tokenizer.h"
#include "weights.h"

#define EOS_TOKEN 50256 /* GPT-2 <|endoftext|> */
#define MAX_NEW 40

/* --- decoding config -------------------------------------------------------
   Both Phase 6 arms degenerate into phrase loops under long GREEDY decoding
   (bench/results/routing/2026-09-25-quality-q2.md), so the REPL defaults to nucleus
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
        /* new sequence: clear the locality set (the KV cache needs no clearing,
           attention only reads j <= pos) */
        runstate_reset_sequence(&rs, &m);
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

/* --- Phase 8 T2: pager residency replay --------------------------------------
   Why this exists: analysis/traffic.py models this policy in Python and
   under-predicts the engine's measured expert hit rate by 5-26 points, so it could
   not be trusted to answer "which port, at what budget". src/replay.c implements the
   residency policy directly and runs a full sweep in milliseconds, so the model and
   the simulator can be compared against each other AND against the engine.

   Two readings worth knowing before quoting a number:
     - `reads`  = distinct cold region loads = physical I/O (the metric that matters
                  on a latency-bound device);
     - the engine-counter line = what pager.c itself would report, which counts one
                  miss per expert TOUCH and so over-counts once a region holds >1 expert.
   Pass --reset 0 to test the hypothesis that the trace's window boundaries are not
   real sequence resets (the trace has many more windows than a bench has prompts). */
static int pager_sim_mode(const char *trace_path, long budget_bytes, long region_bytes,
                          long expert_bytes, long core_bytes, int limit_windows,
                          int reset_windows, const char *sweep) {
    ReplayTrace t;
    if (replay_load(&t, trace_path) != 0)
        return 1;
    ReplayConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.budget_bytes = budget_bytes;
    cfg.core_bytes = core_bytes;
    cfg.region_bytes = region_bytes > 0 ? region_bytes : expert_bytes;
    cfg.expert_bytes = expert_bytes;
    cfg.limit_windows = limit_windows;
    cfg.reset_windows = reset_windows;

    fprintf(stderr, "trace : %s — %d MoE slots x %d windows x %d tokens, top-%d, "
            "N=%d, eos=%d\n", trace_path, t.h.n_slots, t.h.n_windows, t.h.window_len,
            t.h.topk, t.h.n_exp, t.h.eos);
    fprintf(stderr, "sim   : budget %ld MB, core %ld MB, region %ld B, expert %ld B, "
            "reset_windows=%d%s\n", budget_bytes >> 20, core_bytes >> 20, cfg.region_bytes,
            expert_bytes, reset_windows,
            limit_windows ? "  (limited windows)" : "");

    int rc = 0;
    if (sweep && *sweep) {
        printf("  %8s %6s %8s %9s %11s %10s\n", "budget", "cap", "hit%",
               "reads/tok", "MB/token", "evictions");
        char *spec = strdup(sweep), *save = NULL;
        for (char *tok = strtok_r(spec, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
            ReplayConfig c = cfg;
            c.budget_bytes = atol(tok) * 1024L * 1024L;
            ReplayStats st;
            if (replay_run(&t, &c, &st) != 0) { rc = 1; break; }
            printf("  %8ld %6d %8.2f %9.3f %11.2f %10ld\n",
                   c.budget_bytes >> 20, st.cap,
                   st.acc ? 100.0 * (double)st.hits / (double)st.acc : 0.0,
                   st.tokens ? (double)st.loads / (double)st.tokens : 0.0,
                   st.tokens ? (double)st.loads * st.group * c.expert_bytes
                               / (double)st.tokens / 1e6 : 0.0,
                   st.evictions);
        }
        free(spec);
    } else {
        ReplayStats st;
        if (replay_run(&t, &cfg, &st) != 0) rc = 1;
        else replay_report(&st, &cfg);
    }
    replay_free(&t);
    return rc;
}

/* --- Gate 1: incremental (KV-cached) vs full-sequence equivalence -----------
   The two paths use different attention implementations. A gate comparing only
   final logits would pass while ROUTING diverged — which is exactly how the
   EOS-reset bug escaped. So this compares the per-token top-k selections too. */
static int step_check_mode(const char *path, const char *tokbin, long start, int T) {
    Model m;
    if (model_load(&m, path) != 0) return 1;
    if (!m.is_v3 || !m.w.is_moe) {
        fprintf(stderr, "%s is not a CPW1 v3 MoE file\n", path);
        return 1;
    }
    const int V = m.hdr.vocab, nm = m.w.n_moe, K = m.w.topk;

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

    size_t nsel = (size_t)nm * T * K;
    int *selA = malloc(nsel * sizeof(int));
    int *selB = calloc(nsel, sizeof(int));
    float *logA = malloc((size_t)T * V * sizeof(float));
    float *logB = malloc((size_t)T * V * sizeof(float));
    if (!raw || !toks || !selA || !selB || !logA || !logB) {
        fprintf(stderr, "oom\n"); return 1;
    }

    forward_moe(&m, toks, T, logA, selA, m.h3.lam);

    RunState rs;
    runstate_alloc(&rs, &m);
    runstate_reset_sequence(&rs, &m);
    rs.moe_sel_out = selB;
    rs.moe_sel_stride = T;
    double t0 = now_s();
    for (int i = 0; i < T; i++) {
        forward_step(&m, &rs, toks[i], i);
        memcpy(logB + (size_t)i * V, rs.logits, (size_t)V * sizeof(float));
    }
    double dt = now_s() - t0;
    rs.moe_sel_out = NULL;

    double dmax = 0.0;
    for (size_t i = 0; i < (size_t)T * V; i++) {
        double d = fabs((double)logA[i] - (double)logB[i]);
        if (d > dmax) dmax = d;
    }
    int am = 0;
    for (int t = 0; t < T; t++) {
        int a = 0, b = 0;
        for (int j = 1; j < V; j++) {
            if (logA[(size_t)t * V + j] > logA[(size_t)t * V + a]) a = j;
            if (logB[(size_t)t * V + j] > logB[(size_t)t * V + b]) b = j;
        }
        if (a != b) am++;
    }
    int sm = 0;
    for (size_t i = 0; i < nsel; i++) if (selA[i] != selB[i]) sm++;
    if (sm) {
        int shown = 0;
        for (size_t i = 0; i < nsel && shown < 6; i++)
            if (selA[i] != selB[i]) {
                size_t slot = i / ((size_t)T * K), rem = i % ((size_t)T * K);
                fprintf(stderr, "  moe slot %zu pos %zu k %zu: full=%d step=%d\n",
                        slot, rem / K, rem % K, selA[i], selB[i]);
                shown++;
            }
    }

    printf("step-check: T=%d  moe_layers=%d  top-%d  lam=%.3f\n", T, nm, K, m.h3.lam);
    printf("  logits  : max|delta| = %.3e   argmax mismatches %d/%d\n", dmax, am, T);
    printf("  routing : %d / %zu selections differ\n", sm, nsel);
    printf("  speed   : %.1f tok/s (%.1fs for %d tokens, incremental)\n",
           dt > 0 ? T / dt : 0.0, dt, T);
    int ok = (sm == 0) && (dmax < 1e-5);
    printf("GATE 1: %s\n", ok ? "PASS (incremental == full-sequence)" : "FAIL");

    runstate_free(&rs, &m);
    free(raw); free(toks); free(selA); free(selB); free(logA); free(logB);
    model_free(&m);
    return ok ? 0 : 1;
}

/* --- Gate 2: generation equivalence -----------------------------------------
   A single window cannot catch state-carry bugs: the locality set persists
   across forward_step calls and every generated token feeds back. So generate
   greedily with the incremental path and, independently, re-run the
   full-sequence forward over the growing prefix at each step. Token ids and
   last-position routing must match at every step, including across an EOS. */
static int gen_check_mode(const char *path, const char *tokbin, long start,
                          int prompt_T, int N) {
    Model m;
    if (model_load(&m, path) != 0) return 1;
    if (!m.is_v3 || !m.w.is_moe) {
        fprintf(stderr, "%s is not a CPW1 v3 MoE file\n", path);
        return 1;
    }
    const int V = m.hdr.vocab, nm = m.w.n_moe, K = m.w.topk;
    const int Lmax = prompt_T + N;

    int *seq = malloc((size_t)Lmax * sizeof(int));
    uint16_t *raw = malloc((size_t)prompt_T * sizeof(uint16_t));
    FILE *f = fopen(tokbin, "rb");
    if (!f) { perror("tokens"); return 1; }
    if (fseek(f, start * 2L, SEEK_SET) != 0) { perror("seek"); return 1; }
    if (fread(raw, sizeof(uint16_t), (size_t)prompt_T, f) != (size_t)prompt_T) {
        fprintf(stderr, "short read\n"); return 1;
    }
    fclose(f);
    for (int i = 0; i < prompt_T; i++) seq[i] = raw[i];

    /* ---- path A: incremental */
    int *selA = calloc((size_t)nm * Lmax * K, sizeof(int));
    int *idsA = malloc((size_t)N * sizeof(int));
    RunState rs;
    runstate_alloc(&rs, &m);
    runstate_reset_sequence(&rs, &m);
    rs.moe_sel_out = selA;
    rs.moe_sel_stride = Lmax;
    for (int i = 0; i < prompt_T; i++) forward_step(&m, &rs, seq[i], i);
    double t0 = now_s();
    for (int i = 0; i < N; i++) {
        int best = 0;
        for (int j = 1; j < V; j++) if (rs.logits[j] > rs.logits[best]) best = j;
        idsA[i] = best;
        seq[prompt_T + i] = best;
        forward_step(&m, &rs, best, prompt_T + i);
    }
    double dt_gen = now_s() - t0;
    rs.moe_sel_out = NULL;

    /* ---- path B: full sequence re-run over the growing prefix */
    int *selB = malloc((size_t)nm * Lmax * K * sizeof(int));
    float *logits = malloc((size_t)Lmax * V * sizeof(float));
    int id_mm = 0, sel_mm = 0, first_bad = -1;
    for (int i = 0; i < N; i++) {
        int P = prompt_T + i;
        forward_moe(&m, seq, P, logits, selB, m.h3.lam);
        const float *row = logits + (size_t)(P - 1) * V;
        int best = 0;
        for (int j = 1; j < V; j++) if (row[j] > row[best]) best = j;
        if (best != idsA[i]) { id_mm++; if (first_bad < 0) first_bad = i; }
        for (int slot = 0; slot < nm; slot++)
            for (int k = 0; k < K; k++)
                if (selA[((size_t)slot * Lmax + (P - 1)) * K + k]
                        != selB[((size_t)slot * P + (P - 1)) * K + k])
                    sel_mm++;
    }

    int eos_seen = 0;
    for (int i = 0; i < N; i++) if (idsA[i] == EOS_TOKEN) eos_seen++;
    int eos_prompt = 0;
    for (int i = 0; i < prompt_T; i++) if (seq[i] == EOS_TOKEN) eos_prompt++;

    printf("gen-check: prompt %d tokens, %d generated greedily, lam=%.3f\n",
           prompt_T, N, m.h3.lam);
    printf("  generated ids    : %d / %d differ from full-sequence%s\n",
           id_mm, N, id_mm ? "   <-- DIVERGENCE" : "");
    printf("  last-pos routing : %d / %d selections differ\n", sel_mm, nm * N * K);
    printf("  EOS              : %d in prompt, %d generated%s\n",
           eos_prompt, eos_seen,
           eos_prompt ? "  (reset covered here + by Gate 1)"
                      : "  (no reset exercised)");
    printf("  speed            : %.1f tok/s incremental\n",
           dt_gen > 0 ? N / dt_gen : 0.0);
    int ok = (id_mm == 0 && sel_mm == 0);
    printf("GATE 2: %s\n", ok ? "PASS (generation matches step for step)" : "FAIL");

    runstate_free(&rs, &m);
    free(raw); free(seq); free(selA); free(idsA); free(selB); free(logits);
    model_free(&m);
    return ok ? 0 : 1;
}

/* --- Q3 benchmark (protocol pre-registered in Phase-7-Integration § Part 3) --
   Steady-state GENERATION throughput, prompt warm-up excluded. Greedy, because
   sampling injects token-choice randomness -> different routing -> different
   I/O, on a margin of ~0.2 tok/s. Per-prompt timings are kept so the CI is taken
   across prompts (the independent unit), and prompt 1 is discarded as warm-up. */
static double t95(int df) {
    static const double tab[] = {12.706, 4.303, 3.182, 2.776, 2.571, 2.447,
        2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
        2.110, 2.101, 2.093, 2.086};
    if (df < 1) return 12.706;
    if (df <= 20) return tab[df - 1];
    if (df <= 30) return 2.042;
    if (df <= 40) return 2.021;
    if (df <= 60) return 2.000;
    if (df <= 120) return 1.980;
    return 1.960;
}

/* Machine-state telemetry, sampled once per prompt boundary — never inside the
   generation loop, so it cannot perturb the timing it reports. The 2026-09-28
   sweep showed a discrete ~13.1 vs ~11.0 tok/s regime switch across runs with
   the same binary and byte-identical pager statistics; logging this per prompt
   makes such a switch observed rather than inferred from the tok/s spread. */
static void read_sysfs(const char *path, char *out, size_t n) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { snprintf(out, n, "?"); return; }
    ssize_t r = read(fd, out, n - 1);
    close(fd);
    if (r <= 0) { snprintf(out, n, "?"); return; }
    out[r] = '\0';
    char *nl = strchr(out, '\n');
    if (nl) *nl = '\0';
}

static int bench_mode(const char *path, const char *tokbin, long start,
                      int prompt_T, int gen_N, int n_prompts) {
    Model m;
    if (model_load(&m, path) != 0) return 1;
    const int V = m.hdr.vocab;

    /* The pager must be enabled here too. Without this, CP_BUDGET has no effect
       and the benchmark silently measures a fully-resident model — which is
       exactly what happened on the first run (no pager output, flat curve). */
    const char *pb = getenv("CP_BUDGET");
    if (pb)
        pager_init(&m, atol(pb) * 1024L * 1024L);
    else
        fprintf(stderr, "WARNING: CP_BUDGET unset — pager disabled, this is a "
                        "fully-resident measurement, not a paged one\n");

    int fd = open(tokbin, O_RDONLY);
    if (fd < 0) { perror("tokens"); return 1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); return 1; }
    size_t nbytes = (size_t)st.st_size;
    long n_tok = (long)(nbytes / 2);
    uint16_t *ids = mmap(NULL, nbytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ids == MAP_FAILED) { perror("mmap"); return 1; }

    /* deterministic document-start prompts spread across the whole bin */
    long stride = n_tok / (n_prompts + 1);
    long *offs = malloc((size_t)n_prompts * sizeof(long));
    for (int i = 0; i < n_prompts; i++) {
        long p = start + (long)i * stride;
        while (p < n_tok - prompt_T - 2 && ids[p] != EOS_TOKEN) p++;
        offs[i] = p + 1;
    }

    RunState rs;
    runstate_alloc(&rs, &m);
    double *tps = malloc((size_t)n_prompts * sizeof(double));

    /* Prime the page cache before any timing. Without this, run-to-run drift —
       whatever the PREVIOUS run happened to leave cached — dwarfs the effect
       being measured (three identical runs measured 8.13 / 8.78 / 10.67 tok/s).
       After priming, every run starts from the same defined state and the only
       cold pages are the ones the pager itself evicted, which is the regime the
       simulated budget is meant to represent. CP_BENCH_NOPRIME=1 disables it. */
    if (!getenv("CP_BENCH_NOPRIME")) {
        volatile unsigned char sink = 0;
        const unsigned char *pg = (const unsigned char *)m.data;
        for (size_t i = 0; i < m.file_size; i += 4096) sink ^= pg[i];
        (void)sink;
        fprintf(stderr, "  primed %zu MB of page cache\n", m.file_size >> 20);
    }

    fprintf(stderr, "bench: %s\n  prompt_T=%d gen_N=%d n_prompts=%d lam=%.3f\n",
            path, prompt_T, gen_N, n_prompts, m.h3.lam);
    for (int i = 0; i < n_prompts; i++) {
        runstate_reset_sequence(&rs, &m);
        long o = offs[i];
        for (int t = 0; t < prompt_T; t++)
            forward_step(&m, &rs, ids[o + t], t);
        double t0 = now_s();
        int pos = prompt_T;
        for (int g = 0; g < gen_N && pos < m.hdr.n_ctx; g++, pos++) {
            int best = 0;
            for (int j = 1; j < V; j++)
                if (rs.logits[j] > rs.logits[best]) best = j;
            forward_step(&m, &rs, best, pos);
        }
        double dt = now_s() - t0;
        int ng = pos - prompt_T;
        tps[i] = dt > 0 ? ng / dt : 0.0;
        /* Sampled AFTER dt is taken, at the prompt boundary only: telemetry
           must never sit between the clock start and the clock stop. */
        char khz[24], ttemp[24];
        read_sysfs("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq",
                   khz, sizeof khz);
        read_sysfs("/sys/class/thermal/thermal_zone6/temp", ttemp, sizeof ttemp);
        printf("  prompt %2d: %6.1fs  %7.3f tok/s  [cpu_khz=%s pkg_temp_mC=%s]%s\n",
               i, dt, tps[i], khz, ttemp,
               i == 0 ? "   (warm-up, excluded)" : "");
        fflush(stdout);
    }

    int n = n_prompts - 1;
    double sum = 0.0;
    for (int i = 1; i < n_prompts; i++) sum += tps[i];
    double mean = n > 0 ? sum / n : 0.0;
    double var = 0.0;
    for (int i = 1; i < n_prompts; i++) var += (tps[i] - mean) * (tps[i] - mean);
    var = n > 1 ? var / (n - 1) : 0.0;
    double sd = sqrt(var), se = n > 0 ? sd / sqrt((double)n) : 0.0;
    double tv = t95(n - 1);
    double lo = mean - tv * se, hi = mean + tv * se;

    printf("\nbench summary: n=%d (prompt 0 discarded)  mean=%.3f tok/s  "
           "sd=%.3f  se=%.3f  t=%.3f\n", n, mean, sd, se, tv);
    printf("  95%% CI [%.3f, %.3f]   bar is >=7.0 at CP_BUDGET=512MB\n", lo, hi);
    printf("BENCH_RESULT mean=%.4f sd=%.4f ci_lo=%.4f ci_hi=%.4f n=%d file=%s\n",
           mean, sd, lo, hi, n, path);
    pager_report();

    runstate_free(&rs, &m);
    free(tps); free(offs);
    munmap(ids, nbytes);
    close(fd);
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
    int step_check = 0;
    const char *sc_bin = NULL;
    long sc_start = 0;
    int sc_T = 0;
    int gen_check = 0;
    const char *gc_bin = NULL;
    long gc_start = 0;
    int gc_prompt = 0, gc_N = 0;
    int bench = 0;
    const char *bn_bin = NULL;
    long bn_start = 0;
    int bn_prompt = 0, bn_gen = 0, bn_prompts = 0;
    int pager_sim = 0;
    const char *ps_trace = NULL;
    long ps_budget = 512L * 1024 * 1024, ps_region = 0, ps_expert = 1183488L;
    long ps_core = 340L * 1024 * 1024, ps_limit = 0, ps_reset = 1;
    const char *ps_sweep = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc)
            g_temp = atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc)
            g_top_p = atof(argv[++i]);
        else if (strcmp(argv[i], "--max-new") == 0 && i + 1 < argc)
            g_max_new = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            rng_seed(strtoull(argv[++i], NULL, 10));
        else if (strcmp(argv[i], "--pager-sim") == 0 && i + 1 < argc) {
            ps_trace = argv[++i];
            pager_sim = 1;
        }
        else if (strcmp(argv[i], "--ps-budget") == 0 && i + 1 < argc)
            ps_budget = atol(argv[++i]) * 1024L * 1024L;
        else if (strcmp(argv[i], "--ps-region") == 0 && i + 1 < argc)
            ps_region = atol(argv[++i]);          /* bytes; 0 = one expert */
        else if (strcmp(argv[i], "--ps-expert") == 0 && i + 1 < argc)
            ps_expert = atol(argv[++i]);
        else if (strcmp(argv[i], "--ps-core") == 0 && i + 1 < argc)
            ps_core = atol(argv[++i]) * 1024L * 1024L;
        else if (strcmp(argv[i], "--ps-limit") == 0 && i + 1 < argc)
            ps_limit = atol(argv[++i]);
        else if (strcmp(argv[i], "--ps-reset") == 0 && i + 1 < argc)
            ps_reset = atol(argv[++i]);
        else if (strcmp(argv[i], "--ps-sweep") == 0 && i + 1 < argc)
            ps_sweep = argv[++i];
        else if (strcmp(argv[i], "--greedy") == 0)
            g_temp = 0.0;
        else if (strcmp(argv[i], "--bench") == 0 && i + 5 < argc) {
            bn_bin = argv[++i];
            bn_start = atol(argv[++i]);
            bn_prompt = atoi(argv[++i]);
            bn_gen = atoi(argv[++i]);
            bn_prompts = atoi(argv[++i]);
            bench = 1;
        }
        else if (strcmp(argv[i], "--gen-check") == 0 && i + 4 < argc) {
            gc_bin = argv[++i];
            gc_start = atol(argv[++i]);
            gc_prompt = atoi(argv[++i]);
            gc_N = atoi(argv[++i]);
            gen_check = 1;
        }
        else if (strcmp(argv[i], "--step-check") == 0 && i + 3 < argc) {
            sc_bin = argv[++i];
            sc_start = atol(argv[++i]);
            sc_T = atoi(argv[++i]);
            step_check = 1;
        }
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

    if (pager_sim)
        return pager_sim_mode(ps_trace, ps_budget, ps_region, ps_expert, ps_core,
                              (int)ps_limit, (int)ps_reset, ps_sweep);

    if (bench)
        return bench_mode(path, bn_bin, bn_start, bn_prompt, bn_gen, bn_prompts);

    if (gen_check)
        return gen_check_mode(path, gc_bin, gc_start, gc_prompt, gc_N);

    if (step_check)
        return step_check_mode(path, sc_bin, sc_start, sc_T);

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
