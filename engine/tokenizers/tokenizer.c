#include "tokenizer.h"

#include <stdlib.h>
#include <string.h>

#define CPT_MAGIC 0x31545043u /* "CPT1" little-endian */

static int cmp_merge(const void *a, const void *b) {
    uint64_t ka = ((const Merge *)a)->key, kb = ((const Merge *)b)->key;
    return (ka > kb) - (ka < kb);
}

int tokenizer_load(Tokenizer *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open tokenizer"); return -1; }

    uint32_t hdr[4];
    if (fread(hdr, 4, 4, f) != 4 || hdr[0] != CPT_MAGIC || hdr[1] != 1) {
        fprintf(stderr, "bad tokenizer file (magic/version)\n");
        fclose(f);
        return -1;
    }
    t->n_vocab = hdr[2];
    t->n_merges = hdr[3];

    t->vocab = malloc(t->n_vocab * sizeof(char *));
    t->vocab_len = malloc(t->n_vocab * sizeof(uint16_t));
    for (uint32_t i = 0; i < t->n_vocab; i++) {
        uint16_t len;
        if (fread(&len, 2, 1, f) != 1) goto truncated;
        char *s = malloc(len ? len : 1);
        if (len && fread(s, 1, len, f) != len) { free(s); goto truncated; }
        t->vocab[i] = s;
        t->vocab_len[i] = len;
    }
    for (int b = 0; b < 256; b++) t->byte_id[b] = -1;
    for (uint32_t i = 0; i < t->n_vocab; i++)
        if (t->vocab_len[i] == 1)
            t->byte_id[(unsigned char)t->vocab[i][0]] = (int32_t)i;

    t->merges = malloc(t->n_merges * sizeof(Merge));
    for (uint32_t r = 0; r < t->n_merges; r++) {
        uint32_t row[3];
        if (fread(row, 4, 3, f) != 3) goto truncated;
        t->merges[r].key = ((uint64_t)row[0] << 32) | row[1];
        t->merges[r].out = row[2];
        t->merges[r].rank = r;
    }
    fclose(f);
    qsort(t->merges, t->n_merges, sizeof(Merge), cmp_merge);
    return 0;

truncated:
    fprintf(stderr, "truncated tokenizer file\n");
    fclose(f);
    return -1;
}

static const Merge *find_merge(const Tokenizer *t, uint32_t a, uint32_t b) {
    uint64_t key = ((uint64_t)a << 32) | b;
    uint32_t lo = 0, hi = t->n_merges;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (t->merges[mid].key < key) lo = mid + 1; else hi = mid;
    }
    return (lo < t->n_merges && t->merges[lo].key == key) ? &t->merges[lo] : NULL;
}

/* byte classes (GPT-2 regex works on unicode chars; we work on bytes and
   treat high bytes as letters — fine for UTF-8 word interiors) */
static int is_sp(int c)  { return c == ' ' || c == '\t' || c == '\n' ||
                                  c == '\r' || c == '\v' || c == '\f'; }
static int is_let(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
static int is_dig(int c) { return c >= '0' && c <= '9'; }

static const char *contractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
#define N_CONTRACTIONS 7

/* BPE one chunk: bytes -> single-byte ids -> repeatedly merge the
   lowest-rank adjacent pair until nothing merges. Appends to ids. */
int bpe_chunk(const Tokenizer *t, const unsigned char *s, int n, int *ids, int max_ids) {
    int tmp[512];
    if (n > 512) n = 512; /* ponytail: chunk cap; GPT-2 chunks are short */
    int len = 0;
    for (int i = 0; i < n; i++) {
        int id = t->byte_id[s[i]];
        if (id >= 0) tmp[len++] = id;
    }
    for (;;) {
        uint32_t best_rank = UINT32_MAX, best_out = 0;
        int best_i = -1;
        for (int i = 0; i + 1 < len; i++) {
            const Merge *m = find_merge(t, (uint32_t)tmp[i], (uint32_t)tmp[i + 1]);
            if (m && m->rank < best_rank) {
                best_rank = m->rank;
                best_out = m->out;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        tmp[best_i] = (int)best_out;
        memmove(tmp + best_i + 1, tmp + best_i + 2,
                (size_t)(len - best_i - 2) * sizeof(int));
        len--;
    }
    int wrote = 0;
    for (int i = 0; i < len && wrote < max_ids; i++) ids[wrote++] = tmp[i];
    return wrote;
}

/* GPT-2 pre-tokenization as a state machine:
   's|'t|'re|'ve|'m|'ll|'d | ?letters+ | ?digits+ | ?punct+ | \s+(?!\S) | \s+ */
int tokenizer_encode(const Tokenizer *t, const char *text, int *ids, int max_ids) {
    const unsigned char *s = (const unsigned char *)text;
    int n = (int)strlen(text), i = 0, out = 0;
    while (i < n && out < max_ids) {
        int c = s[i];
        int word = is_let(c) || (c == ' ' && i + 1 < n && is_let(s[i + 1]));
        int num  = is_dig(c) || (c == ' ' && i + 1 < n && is_dig(s[i + 1]));
        int pnc  = (!is_sp(c) && !is_let(c) && !is_dig(c)) ||
                   (c == ' ' && i + 1 < n && !is_sp(s[i + 1]) &&
                    !is_let(s[i + 1]) && !is_dig(s[i + 1]));
        if (word || num || pnc) {
            int end = (c == ' ') ? i + 1 : i;
            if (word)      while (end < n && is_let(s[end])) end++;
            else if (num)  while (end < n && is_dig(s[end])) end++;
            else           while (end < n && !is_sp(s[end]) && !is_let(s[end]) && !is_dig(s[end])) end++;
            out += bpe_chunk(t, s + i, end - i, ids + out, max_ids - out);
            i = end;
            if (word && i < n && s[i] == '\'') {
                for (int k = 0; k < N_CONTRACTIONS; k++) {
                    int L = (int)strlen(contractions[k]);
                    if (i + L <= n && memcmp(s + i, contractions[k], (size_t)L) == 0) {
                        out += bpe_chunk(t, s + i, L, ids + out, max_ids - out);
                        i += L;
                        break;
                    }
                }
            }
        } else {
            /* whitespace run: if something follows, all-but-last space is one
               chunk and the last space joins the next chunk (handled by the
               word/num/pnc branches above on the next iteration) */
            int j = i;
            while (j < n && is_sp(s[j])) j++;
            if (j == n) {
                out += bpe_chunk(t, s + i, j - i, ids + out, max_ids - out);
                i = j;
            } else if (j - i > 1) {
                out += bpe_chunk(t, s + i, j - 1 - i, ids + out, max_ids - out);
                i = j - 1;
            }
            /* single space before a word: emit nothing here — the next
               iteration's word/num/pnc branch captures it as the leading
               space of that chunk (GPT-2's " ?" rule) */
        }
    }
    return out;
}

void tokenizer_decode(const Tokenizer *t, const int *ids, int n, FILE *out) {
    for (int i = 0; i < n; i++)
        if (ids[i] >= 0 && (uint32_t)ids[i] < t->n_vocab)
            fwrite(t->vocab[ids[i]], 1, t->vocab_len[ids[i]], out);
}

void tokenizer_free(Tokenizer *t) {
    for (uint32_t i = 0; i < t->n_vocab; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_len);
    free(t->merges);
}
