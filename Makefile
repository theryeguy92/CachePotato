CC      ?= cc
CFLAGS  ?= -O3 -march=native -Wall -Wextra -std=c11 -Iengine
LDLIBS  ?= -lm -lpthread
BUILD   := build
ENGINE  := engine/core/weights.c engine/core/weights_v4.c engine/kernels/kernels.c engine/core/model.c engine/tokenizers/tokenizer.c engine/core/instr.c engine/pager/pager.c engine/core/replay.c engine/blocks/attention_mha.c engine/blocks/ffn_dense.c engine/blocks/moe_routed.c engine/kernels/parallel.c
TRAINER := engine/core/weights.c engine/core/weights_v4.c engine/kernels/kernels.c src/train.c engine/kernels/parallel.c

.PHONY: all calibrate

all: $(BUILD)/cachepotato $(BUILD)/train $(BUILD)/qwen3_eval

# Host calibration probe (Phase 8 T3): measures the memory hierarchy and emits a
# machine profile. Not part of the engine — a separate diagnostic binary.
calibrate: tools/calibrate.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $(BUILD)/calibrate $< -lm -lpthread

$(BUILD)/cachepotato: src/main.c $(ENGINE)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Qwen2.5 chat CLI. The v4 weight file is built separately (see models/qwen2.5-0.5b/MODEL.md).
QWEN_SRC := tools/qwen_chat.c engine/blocks/qwen2.c engine/tokenizers/tokenizer.c \
            engine/tokenizers/tokenizer_qwen.c engine/kernels/kernels.c engine/kernels/parallel.c
$(BUILD)/qwen_chat: $(QWEN_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Decode benchmark for A/B work (tools/ab.py drives it).
BENCH_SRC := tools/bench_decode.c engine/blocks/qwen2.c engine/kernels/kernels.c \
             engine/kernels/parallel.c
$(BUILD)/bench_decode: $(BENCH_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Qwen3-30B-A3B eval driver (B5 + C1 paging). Built but not run by `make`: it wants the
# 32.5 GB v4 file, which only the paging work has. pager.c is linked because the block
# demand-touches file regions; it is a no-op unless CP_PAGER is set at run time.
QWEN3_SRC := tools/qwen3_eval.c engine/blocks/qwen3_moe.c engine/kernels/kernels.c \
             engine/kernels/parallel.c engine/pager/pager.c engine/core/weights.c \
             engine/core/weights_v4.c
$(BUILD)/qwen3_eval: $(QWEN3_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/train: $(TRAINER)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: run clean
run: $(BUILD)/cachepotato
	./$(BUILD)/cachepotato

clean:
	rm -rf $(BUILD)
