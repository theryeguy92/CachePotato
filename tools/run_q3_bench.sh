#!/bin/sh
# Q3 benchmark driver — RESUMABLE.
#
# Re-running this script skips any config that already has a BENCH_RESULT in
# raw.txt, so an interrupted run (laptop closed, power, crash) costs only the
# config that was in flight. The priming amendment makes resumed runs comparable
# across sessions, which is what makes this safe.
#
# Protocol: vault Phase-7-Integration Part 3 + Amendment 1.
#   CP_PAGER_SYS=1 is MANDATORY (without it evictions are free and the curve is
#   flat), primed is primary, both arms get the same prefetch W.
#
# Usage:  tools/run_q3_bench.sh
set -u
cd "$(dirname "$0")/.."

OUT=bench/results/q3
RAW="$OUT/raw.txt"
BIN=./build/cachepotato
TOK=data/fineweb_edu_valid.bin
mkdir -p "$OUT"

# has this label already produced a BENCH_RESULT?
completed() {
    [ -f "$RAW" ] || return 1
    awk -v lbl="=== $1 ===" '
        $0 == lbl      { inblk = 1; next }
        /^=== /        { inblk = 0 }
        inblk && /BENCH_RESULT/ { found = 1 }
        END            { exit !found }
    ' "$RAW"
}

state() {
    ac=$(cat /sys/class/power_supply/AC/online 2>/dev/null)
    bs=$(awk -F= '/POWER_SUPPLY_STATUS/{print $2}' \
         /sys/class/power_supply/BAT1/uevent 2>/dev/null)
    f=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq 2>/dev/null)
    t=$(cat /sys/class/thermal/thermal_zone6/temp 2>/dev/null)
    echo "  state: ac=$ac bat=$bs cpu_khz=$f pkg_temp_mC=$t"
}

run() {
    f="$1"; b="$2"; w="$3"; lbl="$4"; shift 4
    if completed "$lbl"; then
        echo "SKIP (already done): $lbl"
        return 0
    fi
    # AC is a precondition, not a note in a log. Battery costs ~24% on this
    # machine (identical config: 8.13 tok/s on battery vs 10.71 on AC), so a
    # config measured on battery is not comparable with one measured on AC.
    # Refuse loudly rather than silently record an incomparable number.
    ac=$(cat /sys/class/power_supply/AC/online 2>/dev/null)
    if [ "$ac" != "1" ]; then
        echo "ABORT: AC offline (ac=$ac) before '$lbl' — refusing to measure on battery"
        echo "ABORT: AC offline before $lbl" >> "$RAW"
        exit 1
    fi
    echo "=== $lbl ==="
    { echo "=== $lbl ==="; state; } >> "$RAW"
    tmp="$OUT/.tmp.$$"
    t0=$(date +%s)
    env CP_BUDGET="$b" CP_PAGER_SYS=1 CP_PREFETCH_W="$w" "$@" \
        "$BIN" "$f" --bench "$TOK" 0 48 512 16 > "$tmp" 2>&1
    t1=$(date +%s)

    # Integrity check. A suspend freezes the process, but CLOCK_MONOTONIC does
    # not advance during S3, so the per-prompt times come out looking perfectly
    # normal while the wall clock runs on. Three suspends hit the 2026-09-28
    # sweep and nothing in the output revealed them. Wall time vs the sum of the
    # measured prompt times makes it visible; >60s is treated as suspect.
    wall=$(( t1 - t0 ))
    psum=$(awk '/^  prompt/{ v=$3; sub(/s$/,"",v); s+=v } END{ printf "%d", s+0 }' "$tmp")
    drift=$(( wall - psum ))
    note="ok"
    if [ "$drift" -gt 60 ]; then
        note="SUSPECT: wall exceeds measured prompt time by ${drift}s — suspend or interference"
    fi
    {
        echo "  wall=${wall}s  prompt_sum=${psum}s  drift=${drift}s  $note"
        state
    } | tee -a "$RAW"

    cat "$tmp" >> "$RAW"
    grep -E "BENCH_RESULT|experts |core |evictions," "$tmp"
    rm -f "$tmp"
}

if [ ! -s "$RAW" ]; then
    {
        echo "=== Q3 benchmark $(date -Is) ==="
        echo "git: $(git rev-parse HEAD)"
        md5sum weights/moe-scale-run/l05.cpw1v3 weights/moe-scale-run/l00.cpw1v3 weights/gpt2.bin
        echo "protocol: Part 3 + Amendment 1 (primed primary, CP_PAGER_SYS=1 mandatory)"
        echo
    } | tee "$RAW"
fi

# Headline first: this is the only config the primary decision rule uses.
run weights/moe-scale-run/l05.cpw1v3 512 4 "HEADLINE l05 B=512 W=4"
run weights/moe-scale-run/l00.cpw1v3 512 4 "CONTROL l00 B=512 W=4"

for b in 384 448 576 640 736; do
    run weights/moe-scale-run/l05.cpw1v3 "$b" 4 "l05 B=$b W=4"
    run weights/moe-scale-run/l00.cpw1v3 "$b" 4 "l00 B=$b W=4"
done

run weights/moe-scale-run/l05.cpw1v3 512 1 "l05 B=512 W=1 spec-contract"
run weights/moe-scale-run/l00.cpw1v3 512 1 "l00 B=512 W=1 spec-contract"

run weights/moe-scale-run/l05.cpw1v3 512 4 "SENSITIVITY l05 B=512 W=4 UNPRIMED" CP_BENCH_NOPRIME=1
run weights/moe-scale-run/l00.cpw1v3 512 4 "SENSITIVITY l00 B=512 W=4 UNPRIMED" CP_BENCH_NOPRIME=1

for b in 384 512 736; do
    run weights/gpt2.bin "$b" 4 "DENSE GPT-2 B=$b"
done

echo "ALLDONE — results in $RAW"
