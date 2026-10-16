CC      ?= cc
CFLAGS  ?= -O3 -march=native -Wall -Wextra -std=c11 -Iengine
LDLIBS  ?= -lm
BUILD   := build
ENGINE  := engine/core/weights.c engine/kernels/kernels.c engine/core/model.c engine/tokenizers/tokenizer.c engine/core/instr.c engine/pager/pager.c engine/core/replay.c
TRAINER := engine/core/weights.c engine/kernels/kernels.c src/train.c

.PHONY: all calibrate

all: $(BUILD)/cachepotato $(BUILD)/train

# Host calibration probe (Phase 8 T3): measures the memory hierarchy and emits a
# machine profile. Not part of the engine — a separate diagnostic binary.
calibrate: tools/calibrate.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/calibrate $< -lm -lpthread

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
