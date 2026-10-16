CC      ?= cc
CFLAGS  ?= -O3 -march=native -Wall -Wextra -std=c11
LDLIBS  ?= -lm
BUILD   := build
ENGINE  := src/weights.c src/kernels.c src/model.c src/tokenizer.c src/instr.c src/pager.c src/replay.c
TRAINER := src/weights.c src/kernels.c src/train.c

all: $(BUILD)/cachepotato $(BUILD)/train

$(BUILD)/cachepotato: src/main.c $(ENGINE)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/train: $(TRAINER)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: run clean
run: $(BUILD)/cachepotato
	./$(BUILD)/cachepotato

clean:
	rm -rf $(BUILD)
