/* tokenizer_qwen.c — Qwen2 pre-tokenization + specials, reusing the CPT1 BPE core.
 *
 * The BPE loop is unchanged: Qwen uses the same bytes-to-unicode mapping as GPT-2, so the vocab
 * file already stores raw bytes and merges by id. What differs is the SPLIT, and the split is
 * where a "close" tokenizer silently changes the question:
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)      contractions, case-insensitive
 *  |[^\r\n\p{L}\p{N}]?\p{L}+          one optional non-letter prefix (space, quote, ...) + letters
 *  |\p{N}                             ONE number character — NOT a run
 *  | ?[^\s\p{L}\p{N}]+[\r\n]*         punctuation run, optional leading space, trailing newlines
 *  |\s*[\r\n]+                        whitespace ending in newlines
 *  |\s+(?!\S)                         trailing whitespace
 *  |\s+                               any whitespace run
 *
 * Alternatives are tried IN ORDER and the first that matches wins, which is what "|" means here.
 *
 * NOT IMPLEMENTED: NFC normalization. The recorded corpus is already NFC, so the gate cannot
 * detect the omission — stated here rather than discovered later.
 */
#include "tokenizer.h"

/* the BPE core lives in tokenizer.c and is shared by both pre-tokenizers */
int bpe_chunk(const Tokenizer *t, const unsigned char *s, int n, int *ids, int max_ids);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNI_MAGIC 0x31494E55u          /* "UNI1" */
#define QWEN_SPECIAL_FIRST 151643u     /* <|endoftext|>; everything >= is a special token */

typedef struct { uint32_t lo, hi; uint32_t cls; } UniRange;

static UniRange *g_ranges = NULL;
static uint32_t g_nranges = 0;

int tokenizer_qwen_load_unicode(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    uint32_t h[2];
    if (fread(h, 4, 2, f) != 2 || h[0] != UNI_MAGIC) { fclose(f); return -1; }
    g_nranges = h[1];
    g_ranges = malloc(sizeof(UniRange) * g_nranges);
    for (uint32_t i = 0; i < g_nranges; i++) {
        uint32_t v[3];
        if (fread(v, 4, 3, f) != 3) { fclose(f); return -1; }
        g_ranges[i].lo = v[0]; g_ranges[i].hi = v[1]; g_ranges[i].cls = v[2];
    }
    fclose(f);
    return 0;
}

/* 0 other, 1 letter, 2 number */
static int uni_cls(uint32_t cp) {
    uint32_t lo = 0, hi = g_nranges;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (cp < g_ranges[mid].lo) hi = mid;
        else if (cp > g_ranges[mid].hi) lo = mid + 1;
        else return (int)g_ranges[mid].cls;
    }
    return 0;
}

typedef struct { uint32_t cp; int off, len; } UChar;

/* decode UTF-8; invalid bytes are treated as one-codepoint-of-class-other so the encoder never
   reads past the end of the buffer */
static int utf8_decode(const unsigned char *s, int n, UChar *out) {
    int i = 0, k = 0;
    while (i < n) {
        unsigned char c = s[i];
        int len = 1;
        uint32_t cp = c;
        if (c >= 0xF0 && i + 3 < n) { len = 4; cp = c & 0x07u; }
        else if (c >= 0xE0 && i + 2 < n) { len = 3; cp = c & 0x0Fu; }
        else if (c >= 0xC0 && i + 1 < n) { len = 2; cp = c & 0x1Fu; }
        for (int j = 1; j < len; j++) cp = (cp << 6) | (s[i + j] & 0x3Fu);
        out[k].cp = cp; out[k].off = i; out[k].len = len; k++;
        i += len;
    }
    return k;
}

static int is_sp_cp(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f';
}
static int is_nl(uint32_t cp) { return cp == '\n' || cp == '\r'; }

/* try to match a special token (id >= QWEN_SPECIAL_FIRST) at byte position i */
static int match_special(const Tokenizer *t, const unsigned char *s, int i, int n) {
    for (uint32_t id = QWEN_SPECIAL_FIRST; id < t->n_vocab; id++) {
        int L = t->vocab_len[id];
        if (L > 0 && i + L <= n && memcmp(s + i, t->vocab[id], (size_t)L) == 0)
            return (int)id;
    }
    return -1;
}

int tokenizer_qwen_encode(const Tokenizer *t, const char *text, int *ids, int max_ids) {
    const unsigned char *s = (const unsigned char *)text;
    int n = (int)strlen(text);
    UChar *u = malloc(sizeof(UChar) * (size_t)(n > 0 ? n : 1));
    int nu = utf8_decode(s, n, u);
    int out = 0, k = 0;
    const char *contr[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

    while (k < nu && out < max_ids) {
        int sp = match_special(t, s, u[k].off, n);
        if (sp >= 0) { ids[out++] = sp; k += (int)strlen(t->vocab[sp]); continue; }

        int start = k, end = -1;
        uint32_t c = u[k].cp;

        /* 1: contractions */
        if (c == '\'') {
            for (int q = 0; q < 7; q++) {
                int L = (int)strlen(contr[q]);
                int j = 0;
                while (j < L && k + j < nu) {
                    uint32_t cc = u[k + j].cp;
                    uint32_t want = (uint32_t)(unsigned char)contr[q][j];
                    if (j == 0 ? cc != '\'' : (cc | 0x20) != want) break;
                    j++;
                }
                if (j == L) { end = k + L; break; }
            }
        }
        /* 2: [^\r\n\p{L}\p{N}]? \p{L}+ */
        if (end < 0) {
            int j = k;
            if (!is_nl(c) && uni_cls(c) != 1 && uni_cls(c) != 2 && j + 1 < nu &&
                uni_cls(u[j + 1].cp) == 1)
                j++;
            if (j < nu && uni_cls(u[j].cp) == 1) {
                while (j < nu && uni_cls(u[j].cp) == 1) j++;
                end = j;
            }
        }
        /* 3: single \p{N} */
        if (end < 0 && uni_cls(c) == 2) end = k + 1;
        /* 4: ' '? [^\s\p{L}\p{N}]+ [\r\n]* */
        if (end < 0) {
            int j = k;
            if (c == ' ' && j + 1 < nu && !is_sp_cp(u[j + 1].cp) && uni_cls(u[j + 1].cp) == 0)
                j++;
            int q = j;
            while (q < nu && !is_sp_cp(u[q].cp) && uni_cls(u[q].cp) == 0) q++;
            if (q > j) { while (q < nu && is_nl(u[q].cp)) q++; end = q; }
        }
        /* 5: \s* [\r\n]+ */
        if (end < 0) {
            int j = k;
            while (j < nu && is_sp_cp(u[j].cp) && !is_nl(u[j].cp)) j++;
            if (j < nu && is_nl(u[j].cp)) {
                while (j < nu && is_sp_cp(u[j].cp)) j++;
                end = j;
            }
        }
        /* 6: \s+(?!\S), then 7: \s+.
           This one backtracks: greedy \s+ then "next is not non-space". For a run followed by a
           word, the greedy match fails and it gives back ONE character, so the LAST space pairs
           with the following token. Matching the whole run instead splits "  word" as
           ["  ", "word"] rather than [" ", " word"] — two ids in the wrong place, and the same
           mistake appeared twice in the corpus (leading whitespace, and "  #"). */
        if (end < 0 && is_sp_cp(c)) {
            int j = k;
            while (j < nu && is_sp_cp(u[j].cp)) j++;
            int m = (j < nu) ? j - 1 : j;      /* leave the last space when a word follows */
            end = (m > k) ? m : j;             /* alt 6 if it consumes >=1, else alt 7 */
        }
        if (end < 0) end = k + 1;    /* unreachable for valid input; never stall */

        int boff = u[start].off;
        int blen = (end < nu ? u[end].off : n) - boff;
        out += bpe_chunk(t, s + boff, blen, ids + out, max_ids - out);
        k = end;
    }
    free(u);
    return out;
}
