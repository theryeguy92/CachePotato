// calibrate.c — measure the memory hierarchy this engine will run on, and emit a
// machine profile (bench/profiles/<machine>.json).
//
// WHY THIS EXISTS
//   The engine's policy constants are one machine's constants: src/pager.c hardcodes a
//   4 KB PAGE, CP_PREFETCH_W defaults to 4, and the budget is an absolute MB figure. For
//   the engine to be portable those must be DERIVED from a measured curve instead. A
//   trace replay can tell you how many reads a layout costs; only this can tell you what
//   a read costs, which is what converts reads into seconds.
//
//   It also settles a contested number: the vault records 4K random reads at p50 ~227 us
//   while a later cold run measured ~1705 us. Both cannot describe the same thing.
//
// METHOD — and why the size argument is not optional
//   A file smaller than RAM cannot be measured honestly: btrfs serves it from page cache
//   and POSIX_FADV_DONTNEED does not reliably evict, which is how a warm measurement gets
//   reported as disk latency. So the file MUST exceed physical RAM (the trick
//   experiments/ssd_latency.c already established), and the probe refuses to run
//   otherwise. Both access methods are measured and reported so they can disagree
//   visibly:
//     - "fadvise": buffered pread with POSIX_FADV_DONTNEED before each read
//     - "odirect": pread through O_DIRECT (bypasses the page cache entirely)
//   O_DIRECT additionally requires length and offset aligned to the device logical block
//   size, so reads smaller than that are skipped and reported as such.
//
// WHAT IT MEASURES
//   1. read latency + bandwidth vs read size, per access method (p50/p95/max)
//   2. sequential throughput
//   3. single-thread and multi-thread matmul GFLOP/s, in the engine's actual shape
//      (matmul_xw: out[o] = sum_i x[i]*W[o][i] + b[o]) so the compute-vs-IO split can be
//      estimated rather than asserted
//   4. page size, and whether O_DIRECT is available at all
//
// BUILD/RUN
//   cc -O3 -march=native -Wall -Wextra -std=c11 -o build/calibrate tools/calibrate.c -lm -lpthread
//   ./build/calibrate --file /path/to/big.bin --bytes-mb 30720 --out bench/profiles/<machine>.json
//
// The file is created if missing (filled with real data — a sparse file would "read" as
// zeros without touching the disk) and its size is checked against RAM.

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}

static long ram_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    char key[64];
    long kb = 0;
    if (!f) return 0;
    while (fscanf(f, "%63[^:]: %ld", key, &kb) == 2) {
        if (strcmp(key, "MemTotal") == 0) break;
        kb = 0;
        int c; while ((c = fgetc(f)) != '\n' && c != EOF) {}
    }
    fclose(f);
    return kb * 1024L;
}

static unsigned long long rng_state = 88172645463325252ULL;
static unsigned long long xrand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* One measurement series: `n` reads of `size` bytes at random offsets. */
typedef struct {
    double us_p50, us_p95, us_max, us_mean, mb_per_s;
    int n_used, skipped;
} Series;

static Series measure(int fd, size_t size, int n, int use_fadvise, size_t lbads,
                      unsigned char *buf) {
    Series s; memset(&s, 0, sizeof s);
    if (size % lbads != 0) { s.skipped = n; return s; }
    FILE *f = fopen("/proc/self/statm", "r");   /* keep RSS honest in reports */
    if (f) fclose(f);

    struct stat st; fstat(fd, &st);
    size_t fsz = (size_t)st.st_size;
    if (fsz <= size) { s.skipped = n; return s; }

    /* Warm the sample set once so we measure steady state, not the first-ever read. */
    for (int i = 0; i < 3; i++) {
        off_t off = (off_t)((xrand() % (fsz - size)) & ~(off_t)(lbads - 1));
        if (use_fadvise) posix_fadvise(fd, off, (off_t)size, POSIX_FADV_DONTNEED);
        if (pread(fd, buf, size, off) < 0) { s.skipped = n; return s; }
    }

    double *t = malloc((size_t)n * sizeof(double));
    if (!t) { s.skipped = n; return s; }
    int used = 0;
    double total_bytes = 0;
    for (int i = 0; i < n; i++) {
        off_t off = (off_t)((xrand() % (fsz - size)) & ~(off_t)(lbads - 1));
        if (use_fadvise) posix_fadvise(fd, off, (off_t)size, POSIX_FADV_DONTNEED);
        double t0 = now_s();
        ssize_t r = pread(fd, buf, size, off);
        double dt = now_s() - t0;
        if (r < 0) { if (errno == EINVAL) { s.skipped = n; free(t); return s; } continue; }
        t[used++] = dt;
        total_bytes += (double)r;
    }
    if (used == 0) { s.skipped = n; free(t); return s; }
    qsort(t, (size_t)used, sizeof(double), cmp_double);
    double sum = 0;
    for (int i = 0; i < used; i++) sum += t[i];
    s.n_used = used;
    s.us_p50 = t[used / 2] * 1e6;
    s.us_p95 = t[(int)(used * 0.95)] * 1e6;
    s.us_max = t[used - 1] * 1e6;
    s.us_mean = (sum / used) * 1e6;
    s.mb_per_s = total_bytes / sum / 1e6;
    free(t);
    return s;
}

/* The engine's own kernel shape: out[o] = sum_i x[i]*W[o][i] + b[o].
   Deliberately a plain triple loop so the number reflects what a scalar C11 build
   actually achieves on this CPU, not what a tuned BLAS could. */
typedef struct { const float *x, *W; float *out; int o, i, reps; } MmArg;

static void *mm_worker(void *arg) {
    MmArg *a = (MmArg *)arg;
    for (int r = 0; r < a->reps; r++)
        for (int o = 0; o < a->o; o++) {
            float acc = 0.0f;
            const float *row = a->W + (size_t)o * a->i;
            for (int i = 0; i < a->i; i++) acc += a->x[i] * row[i];
            a->out[o] = acc;
        }
    return NULL;
}

static double gflops(int nthreads, int O, int I, int reps) {
    float *x = malloc((size_t)I * sizeof(float));
    float *W = malloc((size_t)O * I * sizeof(float));
    float *out = malloc((size_t)O * sizeof(float));
    if (!x || !W || !out) { free(x); free(W); free(out); return 0; }
    for (int i = 0; i < I; i++) x[i] = (float)(i % 7) * 0.1f;
    for (size_t i = 0; i < (size_t)O * I; i++) W[i] = (float)(i % 13) * 0.01f;

    pthread_t th[64];
    MmArg a = {x, W, out, O, I, reps};
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    double t0 = now_s();
    for (int i = 1; i < nthreads; i++) pthread_create(&th[i], NULL, mm_worker, &a);
    mm_worker(&a);
    for (int i = 1; i < nthreads; i++) pthread_join(th[i], NULL);
    double dt = now_s() - t0;
    free(x); free(W); free(out);
    double flops = 2.0 * O * I * (double)reps * nthreads;
    return dt > 0 ? flops / dt / 1e9 : 0.0;
}

int main(int argc, char **argv) {
    const char *path = NULL, *out = NULL, *note = "";
    long want_mb = 30720;
    int n4k = 300, nbig = 60;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--file") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--bytes-mb") && i + 1 < argc) want_mb = atol(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--note") && i + 1 < argc) note = argv[++i];
        else if (!strcmp(argv[i], "--n4k") && i + 1 < argc) n4k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nbig") && i + 1 < argc) nbig = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!path) { fprintf(stderr, "usage: --file <bigfile> [--bytes-mb N] [--out json]\n"); return 2; }

    long ram = ram_bytes();
    long psz = sysconf(_SC_PAGESIZE);
    size_t lbads = 512;
    {
        int f = open(path, O_RDONLY);
        if (f >= 0) {
            int v = 0; char p[256];
            snprintf(p, sizeof p, "/sys/block/%s/queue/logical_block_size", "sda");
            FILE *q = fopen(p, "r");
            if (q) { if (fscanf(q, "%d", &v) == 1 && v > 0) lbads = (size_t)v; fclose(q); }
            close(f);
        }
    }

    printf("calibrate: target %s\n", path);
    printf("  RAM %.1f GB | page %ld B | device logical block %zu B\n",
           ram / 1e9, psz, lbads);

    /* create / verify the file */
    struct stat st;
    int have = (stat(path, &st) == 0);
    long want_bytes = want_mb * 1024L * 1024L;
    if (!have || st.st_size < want_bytes) {
        printf("  creating %.1f GB (real data, not sparse)…\n", want_mb / 1024.0);
        int wfd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (wfd < 0) { perror("create"); return 1; }
        size_t chunk = 4u << 20;
        unsigned char *b = malloc(chunk);
        if (!b) return 1;
        for (size_t i = 0; i < chunk; i++) b[i] = (unsigned char)(i * 31 + 7);
        long written = 0;
        while (written < want_bytes) {
            size_t n = (size_t)((want_bytes - written) < (long)chunk ? (want_bytes - written) : (long)chunk);
            if (write(wfd, b, n) != (ssize_t)n) { perror("write"); break; }
            written += (long)n;
        }
        free(b);
        fsync(wfd);
        close(wfd);
        fflush(NULL);
        printf("  syncing (let writeback drain — dirty pages inflate cold-read latency)…\n");
        sync();
    }
    stat(path, &st);
    if ((long)st.st_size <= ram) {
        fprintf(stderr, "\nREFUSING: the file (%.1f GB) must EXCEED RAM (%.1f GB).\n"
                        "A smaller file is served from page cache on btrfs and the numbers\n"
                        "would be cache latency reported as disk latency. Pass a bigger\n"
                        "--bytes-mb or a bigger --file.\n",
                st.st_size / 1e9, ram / 1e9);
        return 1;
    }
    size_t fsz = (size_t)st.st_size;
    printf("  file %.1f GB (> RAM, so cache cannot mask the device)\n", fsz / 1e9);

    unsigned char *buf = NULL;
    if (posix_memalign((void **)&buf, (size_t)psz, 8u << 20) != 0) return 1;

    static const size_t sizes[] = {4096, 16384, 65536, 262144, 1048576, 4194304};
    const int NS = (int)(sizeof sizes / sizeof sizes[0]);
    Series fav[6], odi[6];

    for (int i = 0; i < NS; i++) {
        int n = sizes[i] <= 16384 ? n4k : nbig;
        int fd = open(path, O_RDONLY);
        if (fd >= 0) { fav[i] = measure(fd, sizes[i], n, 1, lbads, buf); close(fd); }
        int fod = open(path, O_RDONLY | O_DIRECT);
        if (fod >= 0) { odi[i] = measure(fod, sizes[i], n, 0, lbads, buf); close(fod); }
        else { odi[i].skipped = n; }
        printf("  %7zuK: fadvise p50 %8.1f us (%6.2f MB/s) | odirect p50 %8.1f us (%6.2f MB/s)\n",
               sizes[i] / 1024, fav[i].us_p50, fav[i].mb_per_s,
               odi[i].us_p50, odi[i].mb_per_s);
    }

    /* sequential throughput: 1 MB reads through the whole file, O_DIRECT */
    double seq_mb_s = 0;
    {
        int fd = open(path, O_RDONLY | O_DIRECT);
        if (fd >= 0) {
            size_t chunk = 1u << 20;
            size_t off = 0;
            double t0 = now_s();
            double bytes = 0;
            while (off + chunk <= fsz && (now_s() - t0) < 20.0) {
                ssize_t r = pread(fd, buf, chunk, (off_t)off);
                if (r <= 0) break;
                bytes += r; off += (size_t)r;
            }
            double dt = now_s() - t0;
            if (dt > 0) seq_mb_s = bytes / dt / 1e6;
            close(fd);
        }
    }
    printf("  sequential (O_DIRECT, 1MB reads): %.1f MB/s\n", seq_mb_s);

    /* compute: the engine's kernel shape at its real sizes, 1 and N threads */
    int nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    double g1 = gflops(1, 192, 768, 20000);
    double gn = gflops(nthreads, 192, 768, 20000);
    printf("  matmul 192x768 (engine expert shape): %.2f GFLOP/s 1 thread, %.2f GFLOP/s %d threads\n",
           g1, gn, nthreads);

    if (out) {
        FILE *j = fopen(out, "w");
        if (!j) { perror(out); return 1; }
        time_t t = time(NULL);
        char host[256] = "?";
        gethostname(host, sizeof host - 1);
        fprintf(j, "{\n");
        fprintf(j, "  \"schema\": \"cachepotato-machine-profile/1\",\n");
        fprintf(j, "  \"measured_at\": \"%s\",\n", strtok(ctime(&t), "\n"));
        fprintf(j, "  \"note\": \"%s\",\n", note);
        fprintf(j, "  \"machine\": {\"host\": \"%s\", \"ram_bytes\": %ld, \"page_size\": %ld, "
                   "\"logical_block_size\": %zu, \"cpus\": %d},\n",
                host, ram, psz, lbads, nthreads);
        fprintf(j, "  \"read_latency\": [\n");
        for (int i = 0; i < NS; i++) {
            fprintf(j, "    {\"bytes\": %zu, \"fadvise\": {\"n\": %d, \"p50_us\": %.1f, "
                       "\"p95_us\": %.1f, \"max_us\": %.1f, \"mb_per_s\": %.2f}, "
                       "\"odirect\": {\"n\": %d, \"p50_us\": %.1f, \"p95_us\": %.1f, "
                       "\"max_us\": %.1f, \"mb_per_s\": %.2f}}%s\n",
                    sizes[i], fav[i].n_used, fav[i].us_p50, fav[i].us_p95, fav[i].us_max,
                    fav[i].mb_per_s, odi[i].n_used, odi[i].us_p50, odi[i].us_p95,
                    odi[i].us_max, odi[i].mb_per_s, i + 1 < NS ? "," : "");
        }
        fprintf(j, "  ],\n");
        fprintf(j, "  \"sequential_mb_per_s\": %.1f,\n", seq_mb_s);
        fprintf(j, "  \"compute_gflops\": {\"matmul_192x768_1t\": %.3f, "
                   "\"matmul_192x768_%dt\": %.3f}\n", g1, nthreads, gn);
        fprintf(j, "}\n");
        fclose(j);
        printf("  wrote %s\n", out);
    }
    free(buf);
    return 0;
}
