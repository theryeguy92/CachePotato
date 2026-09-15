CC      ?= cc
CFLAGS  ?= -O3 -march=native -Wall -Wextra -std=c11
LDLIBS  ?= -lm
BUILD   := build
SRCS    := $(wildcard src/*.c)
BIN     := $(BUILD)/cachepotato

$(BIN): $(SRCS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDLIBS)

.PHONY: run clean
run: $(BIN)
	./$(BIN)

clean:
	rm -rf $(BUILD)
