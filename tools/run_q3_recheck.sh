#!/bin/sh
# Q3 ARM RECHECK — paired, interleaved, single session.
#
# WHY THIS EXISTS
#   The 2026-09-27 session measured l05 (headline) at 8.13 tok/s and, in the same
#   session, l00 at 8.05/7.75 tok/s. The 2026-09-28 session measures l00 at
#   ~12.3 tok/s for the IDENTICAL configuration: same binary (build/cachepotato
#   unchanged since 09-27 08:48), same weights, same flags, same 701 MB priming.
#
#   A 53% session-to-session gap in a single config dwarfs any plausible arm
#   effect. So cross-session comparison cannot resolve lambda, and the resume
#   detector's "HEADLINE = DONE" is now a liability: it would leave the headline
#   as the only Sep-27 number in an otherwise Sep-28 matrix.
#
# DESIGN
#   Blocks alternate l05, l00, l05, l00... within one session, so slow drift
#   lands on both arms. Every block re-primes 701 MB, so every block starts from
#   the same cache state and drift cannot accumulate inside a config.
#   Power/thermal/frequency/MemAvailable are logged per block.
#
#   REPS=3, N=6 gives n=15 usable prompts per arm (prompt 0 of each block is
#   discarded as warm-up) — the same n as the headline.
#
# Usage:  tools/run_q3_recheck.sh
#   REPS=2 N=4 tools/run_q3_recheck.sh     # shorter run
set -u
cd "$(dirname "$0")/.."

OUT=bench/results/sweeps
RAW=${RAW:-"$OUT/2026-09-28-q3-recheck.txt"}
BIN=./build/cachepotato
TOK=data/fineweb_edu_valid.bin
REPS=${REPS:-3}
N=${N:-6}

mkdir -p "$OUT"
if [ ! -s "$RAW" ]; then
    {
        echo "=== Q3 arm recheck $(date -Is) — paired/interleaved, single session ==="
        echo "git: $(git rev-parse HEAD)"
        echo "binary: $(TZ=UTC stat -c '%y' "$BIN")  md5: $(md5sum "$BIN" | cut -d' ' -f1)"
        echo "design: $REPS reps, arm order counterbalanced across reps, $N prompts/block, prompt 0 discarded"
        echo "budgets: $BUDGETS   flags: CP_PAGER_SYS=1 CP_PREFETCH_W=4"
        echo
    } | tee "$RAW"
fi

state() {
    ac=$(cat /sys/class/power_supply/AC/online 2>/dev/null)
    gov=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor 2>/dev/null)
    f=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq 2>/dev/null)
    t=$(cat /sys/class/thermal/thermal_zone6/temp 2>/dev/null)
    m=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    # -F= is required: uevent lines contain no whitespace, so awk's default field
    # splitting leaves $2 empty. This field silently read empty for the whole
    # 2026-09-28 recheck, which hid whether the battery was charging or full.
    # STATUS matters as much as the wattage: "Charging" and "Full" are different
    # power states even though AC is online in both.
    bs=$(awk -F= '/POWER_SUPPLY_STATUS/{print $2}' \
         /sys/class/power_supply/BAT1/uevent 2>/dev/null)
    pw=$(awk -F= '/POWER_SUPPLY_POWER_NOW/{print $2}' \
         /sys/class/power_supply/BAT1/uevent 2>/dev/null)
    echo "  state: ac=$ac bat=$bs bat_uW=$pw gov=$gov cpu_khz=$f pkg_temp_mC=$t memavail_kb=$m"
}

# Each rep must sample DIFFERENT documents. bench_mode picks prompts at
# start + i*(n_tok/(n_prompts+1)) snapped forward to the next EOS, so a fixed
# start makes every rep reuse the same N-1 prompts — that measures repeatability,
# not prompt diversity, and would understate the per-prompt spread the headline
# saw (5.76-9.40 tok/s). Offsetting start per rep gives each rep its own
# documents while keeping the two arms inside a rep paired on identical content.
#
# Bounds: the highest offset probed is start + (N-1)*stride, and bench_mode only
# guards `p < n_tok - prompt_T - 2` — if start is too large the guard never fires
# and offs[i] lands out of bounds. The rep stride must satisfy
#     (REPS-1)/NSTART_DIV + (N-1)/(N+1) < 1
# so div=8 permits at most 3 reps and div=16 permits at most 5. Counterbalancing
# the arm order needs an EVEN rep count, which is why div=16 (REPS=4) exists.
NSTART_DIV=${NSTART_DIV:-8}
NTOK=$(( $(stat -c '%s' "$TOK") / 2 ))
STRIDE=$(( NTOK / (N + 1) ))
NSTART=$(( NTOK / NSTART_DIV ))
MAXIDX=$(( (REPS - 1) * NSTART + (N - 1) * STRIDE ))
if [ "$MAXIDX" -ge "$NTOK" ]; then
    echo "FATAL: prompt offsets would run past the end of $TOK" >&2
    echo "  n_tok=$NTOK stride=$STRIDE rep_start=$NSTART max_idx=$MAXIDX" >&2
    exit 1
fi
echo "  n_tok=$NTOK stride=$STRIDE rep_start=$NSTART max_idx=$MAXIDX (limit $NTOK)"

# Budgets to interleave. Every (arm, budget) cell is measured in every rep, so
# the pairing survives across the whole run.
#
# Default is 512 only: that is the pre-registered decision point, it is the cell
# the resume detector skipped, and it is the shorter exposure to another suspend.
# The 384 MB pair measured on 09-28 is confounded (l05 ran while charging, l00
# straddled a 93-minute suspend and ran fully-charged), so re-measure it with:
#     BUDGETS="512 384" tools/run_q3_recheck.sh
BUDGETS=${BUDGETS:-"512"}

rep=0
while [ "$rep" -lt "$REPS" ]; do
    start=$(( rep * NSTART ))
    # Counterbalance the arm order across reps. The measured machine noise floor
    # (8-17%: three l05@512 runs with byte-identical pager stats spanned
    # 8.13/10.17/11.60 tok/s) is the same size as the effect under test, so a
    # FIXED order lets any within-rep drift accumulate as an apparent arm
    # advantage instead of cancelling it. Even reps run l05 first, odd reps l00
    # first; the pooled paired difference then has order bias removed.
    if [ $(( rep % 2 )) -eq 0 ]; then arms="l05 l00"; else arms="l00 l05"; fi
    echo "  rep$rep arm order: $arms"
    for budget in $BUDGETS; do
      for arm in $arms; do
        lbl="RECHECK rep$rep $arm B=$budget W=4 start=$start"
        # AC is a precondition, checked per block, not per run. Battery costs
        # ~24% on this machine (identical config: 8.13 tok/s battery vs 10.71 AC)
        # and battery state can change mid-session — on 2026-09-28 the machine
        # was unplugged between the header and the first block. Refuse before
        # writing anything, so an aborted block leaves no partial record.
        ac=$(cat /sys/class/power_supply/AC/online 2>/dev/null)
        if [ "$ac" != "1" ]; then
            echo "ABORT: AC offline (ac=$ac) before '$lbl' — refusing to measure on battery" >&2
            echo "ABORT: AC offline before $lbl" >> "$RAW"
            exit 1
        fi
        echo "=== $lbl ==="
        {
            echo "=== $lbl ==="
            state
        } >> "$RAW"
        tmp="$OUT/.recheck.$$"
        t0=$(date +%s)
        env CP_BUDGET="$budget" CP_PAGER_SYS=1 CP_PREFETCH_W=4 "$BIN" \
            "weights/moe-scale-run/$arm.cpw1v3" --bench "$TOK" "$start" 48 512 "$N" > "$tmp" 2>&1
        t1=$(date +%s)

        # Integrity check. A suspend freezes the process, but CLOCK_MONOTONIC does
        # not advance during S3, so the per-prompt times come out looking perfectly
        # normal while the wall clock runs on. That is exactly what happened on
        # 2026-09-28: a 93-minute suspend (09:01:22 -> 10:34:56) landed inside the
        # l00@384 run and nothing in the output revealed it — the mean was quoted
        # as if the run had been continuous. Comparing wall time against the sum of
        # the measured prompt times makes it visible. Anything above ~30s is
        # priming plus process start; above 60s is treated as suspect.
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
        grep -E "BENCH_RESULT|experts |evictions," "$tmp"
        rm -f "$tmp"
      done
    done
    rep=$((rep + 1))
done
echo "RECHECK DONE — results in $RAW" | tee -a "$RAW"
