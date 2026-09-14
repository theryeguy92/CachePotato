// ssd_latency.c — measure the target SSD's actual latency profile:
//   sequential 1MB reads  (throughput, what prefetch wants)
//   random 4K reads       (latency, what a cache miss costs)
//
// build: make ssd_latency      (from experiments/)
// run:   ./ssd_latency [path_on_target_ssd] [size_mb]
//
// Uses O_DIRECT + POSIX_FADV_DONTNEED to fight the page cache — but on
// this machine (btrfs) neither fully works for files smaller than RAM.
// RELIABLE METHOD: use size_mb LARGER than physical RAM (e.g. 30720 on a
// 16GB box); the random-4K numbers are then real disk latency. Sequential
// throughput stays partly cache-contaminated unless you drop_caches first:
//   sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
// Answers the open question in the vault's Risks-and-Open-Questions;
// log the output there when done.

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define SEQ_CHUNK  (1024 * 1024)   // 1MB sequential reads
#define RAND_CHUNK 4096            // 4K random reads
#define RAND_READS 2000

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "ssd_test.bin";
    size_t mb = argc > 2 ? strtoul(argv[2], NULL, 10) : 512;
    size_t len = mb * 1024 * 1024;

    // write real data first (a sparse file would "read" as zeros without disk I/O)
    int wfd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (wfd < 0) { perror("open for write"); return 1; }
    char *wbuf = malloc(SEQ_CHUNK);
    memset(wbuf, 0x5a, SEQ_CHUNK);
    for (size_t off = 0; off < len; off += SEQ_CHUNK)
        if (write(wfd, wbuf, SEQ_CHUNK) != SEQ_CHUNK) { perror("write"); return 1; }
    fsync(wfd);
    // evict the file from page cache so reads actually hit disk — without
    // root we can't drop_caches, and O_DIRECT alone is unreliable on btrfs
    posix_fadvise(wfd, 0, 0, POSIX_FADV_DONTNEED);
    close(wfd);
    free(wbuf);
    printf("test file: %s (%zu MB), written + fsync'd\n", path, mb);

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open O_DIRECT (filesystem may not support it)"); unlink(path); return 1; }

    char *buf = aligned_alloc(4096, SEQ_CHUNK);
    if (!buf) { perror("aligned_alloc"); return 1; }

    // --- sequential 1MB reads ---
    double t0 = now_s();
    ssize_t n;
    size_t off = 0;
    while ((n = pread(fd, buf, SEQ_CHUNK, (off_t)off)) > 0)
        off += (size_t)n;
    double seq_s = now_s() - t0;
    printf("sequential 1MB: %.1f MB/s (%.3fs total)\n", mb / seq_s, seq_s);

    // --- random 4K reads, per-read latency ---
    double *lat = malloc(RAND_READS * sizeof(double));
    size_t max_off = (len - RAND_CHUNK) / 4096;
    srand(42);  // fixed seed: comparable across runs
    for (int i = 0; i < RAND_READS; i++) {
        off_t roff = (off_t)(rand() % max_off) * 4096;
        t0 = now_s();
        if (pread(fd, buf, RAND_CHUNK, roff) != RAND_CHUNK) { perror("pread"); return 1; }
        lat[i] = now_s() - t0;
    }
    qsort(lat, RAND_READS, sizeof(double), cmp_double);
    double sum = 0;
    for (int i = 0; i < RAND_READS; i++) sum += lat[i];
    printf("random 4K (n=%d): avg %.0f us | p50 %.0f us | p95 %.0f us | p99 %.0f us\n",
           RAND_READS,
           sum / RAND_READS * 1e6,
           lat[RAND_READS / 2] * 1e6,
           lat[(int)(RAND_READS * 0.95)] * 1e6,
           lat[(int)(RAND_READS * 0.99)] * 1e6);

    free(lat);
    free(buf);
    close(fd);
    unlink(path);
    return 0;
}
