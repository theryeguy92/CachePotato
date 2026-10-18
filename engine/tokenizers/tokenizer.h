#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stdio.h>

/* CPT1 tokenizer: GPT-2 byte-level BPE, vocab as RAW BYTES (the unicode
   mapping was inverted at export time by tools/convert_tokenizer.py). */

typedef struct {
    uint64_t key;  /* (id1 << 32) | id2 */
    uint32_t out;  /* merged token id */
    uint32_t rank; /* position in the original merge table (lower = earlier) */
} Merge;

typedef struct {
    uint32_t n_vocab, n_merges;
    char **vocab;        /* id -> byte string (not NUL-terminated) */
    uint16_t *vocab_len;
    int32_t byte_id[256]; /* raw byte value -> single-byte token id */
    Merge *merges;        /* sorted by key for binary search */
} Tokenizer;

int tokenizer_load(Tokenizer *t, const char *path); /* 0 ok, -1 fail */
/* encode text into ids (at most max_ids), returns count */
int tokenizer_encode(const Tokenizer *t, const char *text, int *ids, int max_ids);
/* write decoded bytes for ids to out */
void tokenizer_decode(const Tokenizer *t, const int *ids, int n, FILE *out);
void tokenizer_free(Tokenizer *t);

#endif
