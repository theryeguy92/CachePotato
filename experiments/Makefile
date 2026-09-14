CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
PROGS   := mmap_basics madvise_hints ssd_latency

all: $(PROGS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: all clean
clean:
	rm -f $(PROGS)
