#include "instr.h"

#include <stdio.h>
#include <stdlib.h>

static FILE *log_file;
static long n_touches = -1; /* -1 = disabled (CP_INSTR unset) */

void instr_touch(const char *tensor, int layer, int token, size_t bytes) {
    if (n_touches == -2)
        return; /* permanently disabled (fopen failed once) */
    if (n_touches < 0) {
        const char *path = getenv("CP_INSTR");
        if (!path)
            return; /* stays disabled */
        log_file = fopen(path, "a");
        if (!log_file) { perror("CP_INSTR fopen"); n_touches = -2; return; }
        n_touches = 0;
    }
    fprintf(log_file, "%s,%d,%d,%zu\n", tensor, layer, token, bytes);
    n_touches++;
}

void instr_report(void) {
    if (n_touches > 0) {
        fprintf(stderr, "instr: %ld tensor touches logged\n", n_touches);
        fclose(log_file);
    }
}
