// madvise_hints.c — the two hints CachePotato's paging engine is built on:
//   MADV_DONTNEED = evict ("this is cold, reclaim the RAM")
//   MADV_WILLNEED = prefetch ("this will be hot soon, start reading")
//
// build: make madvise_hints    (from experiments/)
// run:   ./madvise_hints [size_mb] [path]
//
// What to look for: RSS drops after DONTNEED without munmap, and after
// WILLNEED the next touch of that range is much faster — readahead warms
// the page cache (RSS doesn't rise until you actually touch the pages).

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long rss_kb(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    long total, resident;
    if (!f || fscanf(f, "%ld %ld", &total, &resident) != 2) { perror("statm"); exit(1); }
    fclose(f);
    return resident * (sysconf(_SC_PAGESIZE) / 1024);
}

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? strtoul(argv[1], NULL, 10) : 256;
    const char *path = argc > 2 ? argv[2] : "madvise_test.bin";
    size_t len = mb * 1024 * 1024;

    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) { perror("open"); return 1; }
    if (ftruncate(fd, (off_t)len) < 0) { perror("ftruncate"); return 1; }

    char *p = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }

    // fault everything in
    long page = sysconf(_SC_PAGESIZE);
    volatile unsigned long sink = 0;
    for (size_t i = 0; i < len; i += (size_t)page)
        sink += p[i];
    printf("RSS after full touch:  %8ld KB\n", rss_kb());

    // evict: "we won't need this soon" — the engine's cold-expert mechanism
    if (madvise(p, len, MADV_DONTNEED) < 0) { perror("madvise DONTNEED"); return 1; }
    printf("RSS after DONTNEED:    %8ld KB   <- evicted, RAM reclaimed\n", rss_kb());

    // prefetch the first half: "we'll need this soon" — the engine's prefetch mechanism.
    // WILLNEED warms the kernel page cache; it does NOT populate our page
    // tables, so RSS won't move — but the next touch should skip the disk wait.
    if (madvise(p, len / 2, MADV_WILLNEED) < 0) { perror("madvise WILLNEED"); return 1; }
    struct timespec ts = {0, 500 * 1000 * 1000}; // give the kernel 500ms to readahead
    nanosleep(&ts, NULL);

    double t0 = now_s();
    for (size_t i = 0; i < len / 2; i += (size_t)page)
        sink += p[i];
    double warm = now_s() - t0;

    t0 = now_s();
    for (size_t i = len / 2; i < len; i += (size_t)page)
        sink += p[i];
    double cold = now_s() - t0;

    size_t pages_half = (len / 2) / (size_t)page;
    printf("touch after WILLNEED: %6.1f us/page (prefetched half)\n", warm * 1e6 / pages_half);
    printf("touch, no hint:       %6.1f us/page (cold half)\n", cold * 1e6 / pages_half);
    printf("(sink=%lu)\n", sink);
    munmap(p, len);
    close(fd);
    unlink(path);
    return 0;
}
