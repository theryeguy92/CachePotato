#!/bin/sh
# FFI gate: the Rust CLI must generate exactly the bytes the C chat driver generates.
#
# Both are clients of engine/api/cachepotato.h, so this checks the boundary between them:
# struct layouts, the token callback, option passing. A two-turn conversation is used because
# the second turn runs from the KV cache the first one left, which is where a client that
# mishandled the conversation state would diverge.
#
#   tools/verify_cli_chat.sh [model.v4cp]      (default scratch/qwen3_q4attn.v4cp)
#
# Output is budget-independent (paging is gated bit-identical at every budget), so a small
# budget is used to leave memory for the rest of the machine.
set -u
cd "$(dirname "$0")/.."
MODEL=${1:-scratch/qwen3_q4attn.v4cp}
[ -f "$MODEL" ] || { echo "CLI CHAT GATE: SKIP (no $MODEL)"; exit 0; }
command -v cargo >/dev/null || { echo "CLI CHAT GATE: SKIP (no cargo)"; exit 0; }

make -s build/qwen3_chat || { echo "CLI CHAT GATE: FAIL (C build)"; exit 1; }
(cd cli && cargo build --release -q) || { echo "CLI CHAT GATE: FAIL (cargo build)"; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf 'What is the capital of France? Answer in one sentence.\nWrite a C function that returns the length of a string.\n' > "$tmp/in"

CP_BUDGET=6144 CP_THREADS=4 ./build/qwen3_chat "$MODEL" 24 < "$tmp/in" > "$tmp/c" 2>/dev/null \
    || { echo "CLI CHAT GATE: FAIL (qwen3_chat exited $?)"; exit 1; }
./cli/target/release/cachepotato run "$MODEL" --budget 6144 --threads 4 --max-tokens 24 \
    < "$tmp/in" > "$tmp/rust" 2>/dev/null \
    || { echo "CLI CHAT GATE: FAIL (cachepotato exited $?)"; exit 1; }

if [ ! -s "$tmp/c" ]; then echo "CLI CHAT GATE: FAIL (no output)"; exit 1; fi
if cmp -s "$tmp/c" "$tmp/rust"; then
    echo "CLI CHAT GATE: PASS ($(wc -c < "$tmp/c") bytes identical, 2 turns)"
else
    echo "CLI CHAT GATE: FAIL (outputs differ)"; diff "$tmp/c" "$tmp/rust" | head -20; exit 1
fi
