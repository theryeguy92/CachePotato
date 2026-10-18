#!/bin/sh
# Gate suite for the engine/ refactor. Every phase must reproduce these EXACTLY.
set -u
cd "$(dirname "$0")/.."
fail=0
# a stale binary would silently 'pass' every behavioural gate, so build first and abort
if ! make -s >/dev/null 2>&1; then echo "BUILD FAILED — refusing to test a stale binary"; make 2>&1 | tail -5; exit 1; fi
run() { printf '  %-22s ' "$1"; shift; out=$("$@" 2>&1); rc=$?;
        echo "$out" | grep -E "$EXPECT" | tail -1 | sed 's/^/ /'; [ $rc -eq 0 ] || fail=1; }
EXPECT="PASS|GATE|max abs diff|routing|logits|hit_pct"
echo "== gate suite =="
run "verify_toy"       python3 tools/verify_toy.py
run "verify_cpw1_v3"   python3 tools/verify_cpw1_v3.py --ckpt weights/moe-scale-run/l05/model_only.pt --cpw weights/moe-scale-run/l05.cpw1v3 --trace data/traces/l05_fp32.npz --windows 1
run "gate1_step_check" ./build/cachepotato weights/moe-scale-run/l05.cpw1v3 --step-check data/fineweb_edu_valid.bin 0 32
run "gate2_gen_check"  ./build/cachepotato weights/moe-scale-run/l05.cpw1v3 --gen-check data/fineweb_edu_valid.bin 0 16 8
run "pager_sim_736"    ./build/cachepotato --pager-sim scratch/bench_l05.rtp1 --ps-budget 736
run "perm_gate"        python3 tools/verify_permutation_cpw1_v3.py --base weights/moe-scale-run/l05.cpw1v3 --permuted scratch/l05_permuted.cpw1v3 --ckpt weights/moe-scale-run/l05/model_only.pt --perm-json scratch/l05_permutation.json
# CPW1 v4 acceptance (format/SPEC.md 9.2/9.3). The v4 files are ~736MB each and
# gitignored, so they are generated on demand and reused. Set CP_SKIP_V4=1 to skip.
if [ "${CP_SKIP_V4:-0}" != "1" ] && [ -f weights/moe-scale-run/l05.cpw1v3 ]; then
    [ -f scratch/l05.v4cp ]      || python3 tools/v3_to_v4.py \
        --src weights/moe-scale-run/l05.cpw1v3 --dst scratch/l05.v4cp >/dev/null
    [ -f scratch/l05_shuf.v4cp ] || python3 tools/v3_to_v4.py \
        --src weights/moe-scale-run/l05.cpw1v3 --dst scratch/l05_shuf.v4cp --shuffle >/dev/null
    printf '  %-22s ' "v4_acceptance"
    out=$(python3 tools/verify_v4_roundtrip.py --v3 weights/moe-scale-run/l05.cpw1v3 \
        --v4 scratch/l05.v4cp --v4-shuffled scratch/l05_shuf.v4cp \
        --engine ./build/cachepotato --tokens 64 2>&1)
    rc=$?
    echo "$out" | grep -E "^V4 ACCEPTANCE" | tail -1 | sed 's/^/ /'
    [ $rc -eq 0 ] || fail=1
fi

# First port (models/gpt2-moe). Skipped unless its checkpoint exists, since weights are
# gitignored. Converts the checkpoint to v4, then runs the port's own gate.
if [ -f weights/qwen2.5-0.5b-instruct/model.safetensors ]; then
    printf '  %-22s ' "qwen_v4_forward"
    out=$(python3 models/qwen2.5-0.5b/verify/c_forward_v4.py scratch/qwen.v4cp 2>&1); rc=$?
    echo "$out" | grep -E "^C FORWARD \(v4\)" | tail -1 | sed 's/^/ /'
    [ $rc -eq 0 ] || fail=1

    printf '  %-22s ' "qwen_v4_values"
    python3 models/qwen2.5-0.5b/convert.py --out scratch/qwen.v4cp >/dev/null 2>&1 || fail=1
    out=$(python3 models/qwen2.5-0.5b/verify/v4_values.py scratch/qwen.v4cp 2>&1); rc=$?
    echo "$out" | grep -E "^V4 VALUES" | tail -1 | sed 's/^/ /'
    [ $rc -eq 0 ] || fail=1

    printf '  %-22s ' "qwen_c_forward"
    out=$(python3 models/qwen2.5-0.5b/verify/c_forward.py 2>&1); rc=$?
    echo "$out" | grep -E "^C FORWARD" | tail -1 | sed 's/^/ /'
    [ $rc -eq 0 ] || fail=1
fi

printf '  %-22s ' "qwen_kernels"
out=$(python3 tools/verify_qwen_kernels.py 2>&1); rc=$?
echo "$out" | grep -E "^QWEN KERNELS" | tail -1 | sed 's/^/ /'
[ $rc -eq 0 ] || fail=1

printf '  %-22s ' "q8_file"
out=$(python3 models/gpt2-moe/convert.py --ckpt weights/moe-scale-run/l05/model_only.pt \
        --out scratch/l05_q8.v4cp --quant q8_0 2>&1); rc=$?
if [ $rc -eq 0 ]; then
    v=$(python3 tools/verify_v4_file.py --file scratch/l05_q8.v4cp \
          --v3 weights/moe-scale-run/l05.cpw1v3 2>&1)
    echo "$v" | grep -q "V4 FORMAT: PASS" || rc=1
    run=$(CP_V4_LAMBDA=0.5 ./build/cachepotato scratch/l05_q8.v4cp --moe-sel \
            data/fineweb_edu_valid.bin 0 16 scratch/q8_gate.bin 2>&1)
    echo "$run" | grep -q "next-argmax" || rc=1
    # the experts-only file must take the PARTIAL IN-PLACE path: f32 tensors pointed into the
    # file, quantized experts decoded on the fly. If it silently falls back to the layout copy,
    # resident memory goes UP for a quantized file, which is the regression this exists to stop.
    python3 models/gpt2-moe/convert.py --ckpt weights/moe-scale-run/l05/model_only.pt \
        --out scratch/l05_q8x.v4cp --quant q8_experts >/dev/null 2>&1 || rc=1
    inl=$(CP_V4_LAMBDA=0.5 ./build/cachepotato scratch/l05_q8x.v4cp --moe-sel \
            data/fineweb_edu_valid.bin 0 8 scratch/q8x_gate.bin 2>&1)
    echo "$inl" | grep -q "in place" || { echo "      (experts-only file did NOT load in place)"; rc=1; }
    echo "$inl" | grep -q "quantized experts decoded on the fly" || rc=1
fi
if [ $rc -eq 0 ]; then echo " PASS — Q8_0 file validates and the engine runs it"
else echo " FAIL — see scratch/l05_q8.v4cp"; fail=1; fi

printf '  %-22s ' "q8_kernels"
out=$(python3 tools/verify_q8_kernels.py 2>&1); rc=$?
echo "$out" | grep -E "^Q8 KERNELS" | tail -1 | sed 's/^/ /'
[ $rc -eq 0 ] || fail=1

printf '  %-22s ' "q8_decode"
out=$(python3 tools/verify_q8_decode.py 2>&1); rc=$?
echo "$out" | grep -E "^Q8 DECODER" | tail -1 | sed 's/^/ /'
[ $rc -eq 0 ] || fail=1

if [ "${CP_SKIP_PORT:-0}" != "1" ] && [ -f weights/moe-scale-run/l05/model_only.pt ]; then
    printf '  %-22s ' "port_gpt2_moe"
    [ -f scratch/port_l05.v4cp ] || python3 models/gpt2-moe/convert.py \
        --ckpt weights/moe-scale-run/l05/model_only.pt --out scratch/port_l05.v4cp >/dev/null 2>&1
    [ -f scratch/port_l05_shuf.v4cp ] || python3 models/gpt2-moe/convert.py \
        --ckpt weights/moe-scale-run/l05/model_only.pt --out scratch/port_l05_shuf.v4cp \
        --shuffle >/dev/null 2>&1
    out=$(python3 models/gpt2-moe/verify/gate.py --windows 1 --tokens 64 2>&1)
    rc=$?
    echo "$out" | grep -E "^PORT GATE" | tail -1 | sed 's/^/ /'
    [ $rc -eq 0 ] || fail=1
fi

echo "== done (fail=$fail) =="
exit $fail
