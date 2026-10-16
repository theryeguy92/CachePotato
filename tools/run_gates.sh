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
echo "== done (fail=$fail) =="
exit $fail
