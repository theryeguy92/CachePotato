// mmap_basics.c — observe that mmap loads pages LAZILY, on first touch.
//
// build: make mmap_basics      (from experiments/)
// run:   ./mmap_basics [size_mb] [path]
//
// What to look for: RSS barely moves after mmap, then grows as pages
// are touched. The OS reads from disk per page fault, not upfront.

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static long rss_kb(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    long total, resident;
    if (!f || fscanf(f, "%ld %ld", &total, &resident) != 2) { perror("statm"); exit(1); }
    fclose(f);
    return resident * (sysconf(_SC_PAGESIZE) / 1024);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? strtoul(argv[1], NULL, 10) : 256;
    const char *path = argc > 2 ? argv[2] : "mmap_test.bin";
    size_t len = mb * 1024 * 1024;

    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) { perror("open"); return 1; }
    if (ftruncate(fd, (off_t)len) < 0) { perror("ftruncate"); return 1; }

    printf("file: %s (%zu MB)\n", path, mb);
    printf("RSS before mmap:  %8ld KB\n", rss_kb());

    char *p = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    printf("RSS after mmap:   %8ld KB   <- address space mapped, nothing read yet\n", rss_kb());

    long page = sysconf(_SC_PAGESIZE);
    volatile unsigned long sink = 0;
    double t0 = now_s();
    for (size_t i = 0; i < len; i += (size_t)page)
        sink += p[i];   // one byte per page = one page fault each
    double dt = now_s() - t0;

    printf("RSS after touch:  %8ld KB   <- pages faulted in on demand\n", rss_kb());
    printf("touched %zu pages in %.3fs (%.1f us/fault, sink=%lu)\n",
           len / (size_t)page, dt, dt * 1e6 / (len / (size_t)page), sink);

    munmap(p, len);
    close(fd);
    unlink(path);
    return 0;
}
