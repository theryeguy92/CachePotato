#ifndef CACHEPOTATO_H
#define CACHEPOTATO_H
/* cachepotato.h — the engine's public API: open a model, chat with it, read its counters.
 *
 * This is the surface front-ends link against (tools/qwen3_chat.c, the Rust CLI in cli/).
 * Everything a chat needs that used to live in a driver — ChatML, the embedding lookup, the
 * head, the KV-cached decode loop — lives behind it, so there is ONE copy of the code that
 * produces tokens and the gates that check it cover every front-end.
 *
 * Supported models: Qwen3-MoE CPW1 v4 files (Qwen3-30B-A3B).
 *
 * ONE MODEL PER PROCESS. The pager and the thread pool are process-global, so a second
 * cp_open() fails, including after cp_close(). A server runs one model in one process.
 *
 * Greedy decoding only, by design: the output is a deterministic function of the weights,
 * which is what the gates check. There is no sampler to configure.
 */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cp_model cp_model;

/* budget_mb: CP_BUDGET_AUTO sizes the pager from MemAvailable (cp_auto_budget_mb),
   CP_BUDGET_ENV reads CP_PAGER / CP_BUDGET as the old drivers did, CP_BUDGET_NONE attaches
   no pager (the model must then fit, or the OS page cache does the paging), > 0 is MB. */
#define CP_BUDGET_AUTO 0L
#define CP_BUDGET_ENV  (-2L)
#define CP_BUDGET_NONE (-1L)

typedef struct {
    long budget_mb;             /* see above */
    int threads;                /* 0 = physical cores (cp_physical_cores) */
    int pin_cores;              /* 1/0 = pin each thread to its own physical core or not;
                                   -1 = CP_PIN_CORES. Never changes a value (parallel.h) */
    int max_ctx;                /* 0 = 2048 tokens */
    const char *tokenizer_dir;  /* NULL = search: <model dir>/tokenizer, <model dir>, weights */
} cp_options;

/* Defaults: auto budget, physical cores, pinning from the environment, 2048 tokens. */
void cp_options_default(cp_options *opt);

/* Opens and binds the model. On failure returns NULL and writes a one-line reason to err. */
cp_model *cp_open(const char *model_path, const cp_options *opt, char *err, size_t errlen);
void cp_close(cp_model *m);

/* Called with the raw bytes of each generated token, in order. A token's bytes can end in the
   middle of a UTF-8 sequence; a front-end that needs whole characters must buffer. Return
   nonzero to stop generating after this token. */
typedef int (*cp_token_fn)(const char *bytes, size_t len, void *user);

typedef enum { CP_STOP_EOS = 0, CP_STOP_MAX_TOKENS, CP_STOP_CONTEXT, CP_STOP_CALLER } cp_stop;

typedef struct {
    int prompt_tokens;          /* tokens prefilled for this turn */
    int context_tokens;         /* tokens in the KV cache after the turn */
    int generated_tokens;
    int context_reset;          /* 1 if the conversation was full and was restarted first */
    double prefill_s, decode_s;
    cp_stop stop;
} cp_turn;

/* One user turn: appended to the conversation (earlier turns stay in the KV cache), then
   greedy decode until <|im_end|>, max_new tokens, the context limit, or the callback stops it.
   When the context cannot hold the turn plus max_new, the conversation restarts first and
   turn->context_reset says so. Returns 0, or < 0 on error (tokenizer or forward failure). */
int cp_chat(cp_model *m, const char *user_msg, int max_new,
            cp_token_fn on_token, void *user, cp_turn *turn);

/* Forget the conversation (the KV cache is reused, not reallocated). */
void cp_reset(cp_model *m);

typedef struct {
    /* model */
    int n_layers, d_model, n_experts, topk, vocab, max_ctx;
    int threads;
    long budget_mb;             /* -1 = no pager */
    long long file_bytes;
    /* pager, cumulative since cp_open (all zero without a pager) */
    long hits, misses, evictions;
    long long bytes_fetched, resident_bytes;
    /* thread placement: pool thread i runs on logical CPU pinned_cpus[i] (one per physical
       core); n_pinned is 0 when pinning is off or the topology could not be read */
    int n_pinned;
    int pinned_cpus[64];
} cp_stats;

void cp_get_stats(const cp_model *m, cp_stats *out);

/* Machine probes, also used by the defaults. */
int  cp_physical_cores(void);                       /* distinct (package, core); >= 1 */
long cp_mem_available_mb(void);                     /* MemAvailable, or -1 */
long cp_auto_budget_mb(long long model_file_bytes); /* -1 when there is not enough memory */

#ifdef __cplusplus
}
#endif
#endif
