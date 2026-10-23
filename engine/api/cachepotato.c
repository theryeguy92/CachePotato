#define _GNU_SOURCE
/* cachepotato.c — the public API (cachepotato.h) over the Qwen3-MoE block and the pager.
 *
 * embed_rows / apply_head / run_layers and the turn loop are MOVED from tools/qwen3_chat.c,
 * not rewritten: the refactor's gate is that the chat driver's stdout is byte-identical before
 * and after, so the arithmetic and its order are the driver's, line for line.
 */
#include "api/cachepotato.h"

#include "blocks/qwen3_moe.h"
#include "kernels/kernels.h"
#include "kernels/parallel.h"
#include "pager/pager.h"
#include "tokenizers/tokenizer.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

int tokenizer_qwen_load_unicode(const char *path);
int tokenizer_qwen_encode(const Tokenizer *t, const char *text, int *ids, int max_ids);

#define IM_END 151645        /* <|im_end|> */
#define DEFAULT_CTX 2048
#define TOK_FILE "qwen2.5-tokenizer.bin"
#define UNI_FILE "qwen2.5-unicode.bin"

struct cp_model {
    QW3Layer base;           /* the mapping; layers[] are per-layer copies of it */
    QW3Layer *layers;
    QW3KV *kv;
    Tokenizer tk;
    float *x, *logits;
    int *ids, *turn;
    int L, max_ctx, hist, threads;
    long budget_mb;
    long long file_bytes;
};

static int g_opened;         /* the pager and the pool are process-global */

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void set_err(char *err, size_t n, const char *fmt, ...) {
    if (!err || !n) return;
    va_list ap; va_start(ap, fmt); vsnprintf(err, n, fmt, ap); va_end(ap);
}

/* ---- machine probes ---------------------------------------------------------------- */

int cp_physical_cores(void) {
    /* distinct (physical id, core id) pairs: HT siblings share a core, and 8 threads on this
       4-core part measured slower than 4 */
    FILE *f = fopen("/proc/cpuinfo", "r");
    int n = 0, pkg = 0, seen[512][2];
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            int v;
            if (sscanf(line, "physical id : %d", &v) == 1) pkg = v;
            else if (sscanf(line, "core id : %d", &v) == 1) {
                int dup = 0;
                for (int i = 0; i < n; i++) if (seen[i][0] == pkg && seen[i][1] == v) dup = 1;
                if (!dup && n < 512) { seen[n][0] = pkg; seen[n][1] = v; n++; }
            }
        }
        fclose(f);
    }
    if (n > 0) return n;
    long c = sysconf(_SC_NPROCESSORS_ONLN);
    return c > 0 ? (int)c : 1;
}

long cp_mem_available_mb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    fclose(f);
    return kb < 0 ? -1 : kb / 1024;
}

long cp_auto_budget_mb(long long model_file_bytes) {
    /* MemAvailable minus room for everything that is not a weight region: the KV cache
       (~400 MB at 2048 tokens), activations, the tokenizer, and the rest of the desktop.
       A budget past that swaps, and a swapping run looks exactly like a broken model. */
    const long headroom_mb = 2048;
    const long min_mb = 2048;   /* below this the core tier alone thrashes */
    long avail = cp_mem_available_mb();
    if (avail < 0) return -1;
    long b = avail - headroom_mb;
    long file_mb = (long)(model_file_bytes / (1024 * 1024)) + 1;
    if (model_file_bytes > 0 && b > file_mb) b = file_mb;
    return b < min_mb ? -1 : b;
}

/* ---- the forward pieces, from tools/qwen3_chat.c ---------------------------------- */

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

/* ---- open / close ------------------------------------------------------------------ */

void cp_options_default(cp_options *opt) {
    memset(opt, 0, sizeof *opt);
    opt->budget_mb = CP_BUDGET_AUTO;
    opt->pin_cores = -1;
}

static int file_exists(const char *p) { return access(p, R_OK) == 0; }

/* The tokenizer travels with the model (the HF repo ships it under tokenizer/); weights/ is
   where the repo's own conversion tools put it. */
static int has_tokenizer(const char *dir) {
    char a[4352], b[4352];
    if (snprintf(a, sizeof a, "%s/" TOK_FILE, dir) >= (int)sizeof a) return 0;
    if (snprintf(b, sizeof b, "%s/" UNI_FILE, dir) >= (int)sizeof b) return 0;
    return file_exists(a) && file_exists(b);
}

static int find_tokenizer(const char *model_path, const char *dir, char *out, size_t n) {
    char mdir[4096];
    if (snprintf(mdir, sizeof mdir, "%s", model_path) >= (int)sizeof mdir) return -1;
    char *slash = strrchr(mdir, '/');
    if (slash) *slash = 0; else snprintf(mdir, sizeof mdir, ".");
    char cand[3][4224];
    int nc = 0;
    if (dir) snprintf(cand[nc++], sizeof cand[0], "%s", dir);
    else {
        snprintf(cand[nc++], sizeof cand[0], "%s/tokenizer", mdir);
        snprintf(cand[nc++], sizeof cand[0], "%s", mdir);
        snprintf(cand[nc++], sizeof cand[0], "weights");
    }
    for (int i = 0; i < nc; i++)
        if (has_tokenizer(cand[i])) return snprintf(out, n, "%s", cand[i]) < (int)n ? 0 : -1;
    return -1;
}

cp_model *cp_open(const char *model_path, const cp_options *opt_in, char *err, size_t errlen) {
    cp_options opt;
    if (opt_in) opt = *opt_in; else cp_options_default(&opt);
    if (g_opened) { set_err(err, errlen, "one model per process: the pager is process-global"); return NULL; }

    struct stat st;
    if (stat(model_path, &st) != 0) { set_err(err, errlen, "cannot open model file %s", model_path); return NULL; }

    char tdir[4224];
    if (find_tokenizer(model_path, opt.tokenizer_dir, tdir, sizeof tdir)) {
        set_err(err, errlen, "tokenizer files (" TOK_FILE ", " UNI_FILE ") not found %s%s",
                opt.tokenizer_dir ? "in " : "next to the model, in its tokenizer/ folder or in weights/",
                opt.tokenizer_dir ? opt.tokenizer_dir : "");
        return NULL;
    }

    long budget_bytes, budget_mb = opt.budget_mb;
    if (opt.budget_mb == CP_BUDGET_AUTO) {
        budget_mb = cp_auto_budget_mb((long long)st.st_size);
        if (budget_mb < 0) {
            set_err(err, errlen, "not enough free memory for a pager budget (%ld MB available, "
                    "need at least 4096); close other programs or set a budget", cp_mem_available_mb());
            return NULL;
        }
        budget_bytes = budget_mb * 1024L * 1024L;
    } else if (opt.budget_mb == CP_BUDGET_ENV) {
        budget_bytes = QW3_BUDGET_ENV;
        const char *pb = getenv("CP_BUDGET"), *pg = getenv("CP_PAGER");
        budget_mb = pb ? atol(pb) : (pg && pg[0] == '1') ? LONG_MAX / (1024L * 1024L) : -1;
    } else if (opt.budget_mb == CP_BUDGET_NONE) {
        budget_bytes = QW3_BUDGET_NONE; budget_mb = -1;
    } else {
        budget_bytes = opt.budget_mb * 1024L * 1024L;
    }

    cp_model *m = calloc(1, sizeof *m);
    if (!m) { set_err(err, errlen, "out of memory"); return NULL; }
    m->max_ctx = opt.max_ctx > 0 ? opt.max_ctx : DEFAULT_CTX;
    m->file_bytes = (long long)st.st_size;
    m->budget_mb = budget_mb;

    char a[8192], b[8192];
    snprintf(a, sizeof a, "%s/" UNI_FILE, tdir);
    snprintf(b, sizeof b, "%s/" TOK_FILE, tdir);
    if (tokenizer_qwen_load_unicode(a) || tokenizer_load(&m->tk, b)) {
        set_err(err, errlen, "cannot load tokenizer from %s", tdir); free(m); return NULL;
    }

    /* the pool starts here, before any kernel runs, so the options rather than CP_THREADS
       decide its size; the thread count never changes a value (parallel.h) */
    m->threads = cp_threads_init(opt.threads > 0 ? opt.threads : cp_physical_cores(), opt.pin_cores);

    g_opened = 1;
    QW3Layer *base = &m->base;
    if (qwen3_map_ex(base, model_path, budget_bytes)) {
        set_err(err, errlen, "cannot map %s (not a CPW1 v4 Qwen3 file?)", model_path); goto fail;
    }
    if (qwen3_bind_layer(base, 0)) { set_err(err, errlen, "cannot bind layer 0"); goto fail; }
    m->L = base->n_layers;
    if (m->L <= 0) { set_err(err, errlen, "the file declares no attention blocks"); goto fail; }
    const int E = base->n_experts, D = base->d_model, V = base->vocab;

    /* bind every layer once (the 48x-per-token walk cost 11% back when it was per token) */
    m->layers = calloc((size_t)m->L, sizeof(QW3Layer));
    m->kv = calloc((size_t)m->L, sizeof(QW3KV));
    if (!m->layers || !m->kv) { set_err(err, errlen, "out of memory"); goto fail; }
    for (int l = 0; l < m->L; l++) {
        m->layers[l] = *base;
        m->layers[l].exp = calloc((size_t)E * 3, sizeof(QW3T));
        m->layers[l].exp_owned = 0;
        m->layers[l].reg_exp = calloc((size_t)E, sizeof(int));
        if (!m->layers[l].exp || !m->layers[l].reg_exp || qwen3_bind_layer(&m->layers[l], l)) {
            set_err(err, errlen, "bind layer %d failed", l); goto fail;
        }
        if (qwen3_kv_alloc(&m->kv[l], &m->layers[l], m->max_ctx)) {
            set_err(err, errlen, "KV cache allocation failed"); goto fail;
        }
    }
    m->x = malloc(sizeof(float) * (size_t)m->max_ctx * D);
    m->logits = malloc(sizeof(float) * (size_t)V);
    m->ids = malloc(sizeof(int) * (size_t)m->max_ctx);   /* the conversation, tokenised once */
    m->turn = malloc(sizeof(int) * (size_t)m->max_ctx);  /* just the new fragment */
    if (!m->x || !m->logits || !m->ids || !m->turn) { set_err(err, errlen, "out of memory"); goto fail; }
    return m;

fail:
    cp_close(m);
    return NULL;
}

void cp_close(cp_model *m) {
    /* The mapping stays: the process-global pager still describes its regions, and a second
       model cannot be opened in this process anyway. */
    if (!m) return;
    if (m->layers) {
        for (int l = 0; l < m->L; l++) { free(m->layers[l].exp); free(m->layers[l].reg_exp); }
        free(m->layers);
    }
    if (m->kv) { for (int l = 0; l < m->L; l++) qwen3_kv_free(&m->kv[l]); free(m->kv); }
    qwen3_layer_free_weights(&m->base);
    free(m->base.reg_exp);
    if (m->tk.vocab) tokenizer_free(&m->tk);
    free(m->x); free(m->logits); free(m->ids); free(m->turn);
    free(m);
}

void cp_reset(cp_model *m) { m->hist = 0; }

/* ---- a turn ------------------------------------------------------------------------ */

int cp_chat(cp_model *m, const char *user_msg, int max_new,
            cp_token_fn on_token, void *user, cp_turn *out) {
    cp_turn t;
    memset(&t, 0, sizeof t);
    QW3Layer *layers = m->layers;
    const int D = layers[0].d_model, V = layers[0].vocab, L = m->L, MAX_CTX = m->max_ctx;
    float *x = m->x, *logits = m->logits;
    int *ids = m->ids, *turn = m->turn;

    if (m->hist > MAX_CTX - max_new - 64) { m->hist = 0; t.context_reset = 1; }

    /* ChatML, with /no_think: Qwen3 is a thinking model, and without this the whole
       visible budget goes into a think block before the answer starts. The fragment is
       APPENDED to the conversation, so earlier turns stay in the KV cache instead of being
       re-prefilled — that is what makes follow-ups work (and what keeps the prefill cost
       proportional to the new message, not the whole history). */
    size_t plen = strlen(user_msg) + 128;
    char *prompt = malloc(plen);
    if (!prompt) return -1;
    snprintf(prompt, plen,
             "<|im_start|>user\n%s /no_think<|im_end|>\n<|im_start|>assistant\n", user_msg);
    int Tn = tokenizer_qwen_encode(&m->tk, prompt, turn, MAX_CTX - m->hist - 8);
    free(prompt);
    if (Tn <= 0) return -2;
    memcpy(ids + m->hist, turn, sizeof(int) * (size_t)Tn);
    const int pos0 = m->hist;              /* where this fragment starts in the cache */
    m->hist += Tn;
    t.prompt_tokens = Tn;

    double t0 = now();
    embed_rows(&layers[0], ids + pos0, Tn, pos0, x, D);
    if (run_layers(layers, x, Tn, pos0, m->kv, L)) return -3;
    double t1 = now();

    /* the last prefill position gives the first token */
    apply_head(&layers[0], x + (size_t)(Tn - 1) * D, 1, m->hist - 1, logits, V);
    int pos = m->hist, produced = 0, stop_req = 0, rc = 0;
    t.stop = CP_STOP_EOS;
    double t2 = now();
    for (;;) {
        int best = 0;
        for (int j = 1; j < V; j++) if (logits[j] > logits[best]) best = j;
        if (best == IM_END) { t.stop = CP_STOP_EOS; break; }
        if (produced >= max_new) { t.stop = CP_STOP_MAX_TOKENS; break; }
        if (pos >= MAX_CTX - 1) { t.stop = CP_STOP_CONTEXT; break; }
        if (on_token && best >= 0 && (uint32_t)best < m->tk.n_vocab)
            stop_req = on_token(m->tk.vocab[best], m->tk.vocab_len[best], user);
        if (pos < MAX_CTX - 2) ids[pos] = best, m->hist = pos + 1;
        /* the emitted token always goes through the forward pass, even when the caller
           stops here, so the KV cache holds every token hist counts */
        embed_rows(&layers[0], &best, 1, pos, x, D);
        if (run_layers(layers, x, 1, pos, m->kv, L)) { rc = -3; break; }
        apply_head(&layers[0], x, 1, pos, logits, V);
        pos++; produced++;
        if (stop_req) { t.stop = CP_STOP_CALLER; break; }
    }
    double t3 = now();
    t.generated_tokens = produced;
    t.context_tokens = m->hist;
    t.prefill_s = t1 - t0;
    t.decode_s = t3 - t2;
    if (out) *out = t;
    return rc;
}

/* ---- counters ---------------------------------------------------------------------- */

void cp_get_stats(const cp_model *m, cp_stats *s) {
    memset(s, 0, sizeof *s);
    const QW3Layer *b = &m->base;
    s->n_layers = m->L; s->d_model = b->d_model; s->n_experts = b->n_experts;
    s->topk = b->topk; s->vocab = b->vocab; s->max_ctx = m->max_ctx;
    s->threads = m->threads; s->budget_mb = b->pager ? m->budget_mb : -1;
    s->file_bytes = m->file_bytes;
    s->n_pinned = cp_pinned_cpus(s->pinned_cpus, 64);
    if (b->pager) {
        PagerTotals p;
        pager_totals(&p);
        s->hits = p.hits; s->misses = p.misses; s->evictions = p.evictions;
        s->bytes_fetched = p.bytes_fetched; s->resident_bytes = p.resident_bytes;
    }
}
