#!/usr/bin/env bash
# Hand-check run (chenyi9 order 2026-09-04): ONE model, the seven rungs
# A1 A2 A3b A4 A4b A5 A6, each with the layout probe on, so the real
# per-chunk addresses and the per-channel reduction terms land on disk.
#
#   usage: bash experiments/run_layout_handcheck.sh [OUT_DIR]
#
# Model: LLAMA3-8B.  Chosen because it is the simplest of the six to check by
# hand -- --ngpu 1 --num-hbm 1 means no tensor parallelism and a single HBM
# stack, and GQA leaves 8 KV heads, so heads_per_hbm = 8 divides the 16
# channels exactly (A3b stripe = 2).  LLAMA-7B is smaller still but its 32 KV
# heads clamp A3b's stripe to 1, which makes A3b degenerate to A3 and useless
# as a hand-check of head slicing.
#
# Budget (chenyi9 2026-09-04): AT MOST 64 cores and 500 GB RAM.
#   7 rungs x RAMU_WORKERS(8) + 7 construction processes = 63 <= 64.
#
# Scratch MUST live on /data2: this host's / has ~3 GB free.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

MODEL=${MODEL:-LLAMA3-8B}
NGPU=${NGPU:-1}
NUM_HBM=${NUM_HBM:-1}
K=${EPIC_K:-8}
WL=$REPO/workload/sweep/wl_baseline_alltoall_N16_C16_D2.json
OUT=${1:-$REPO/output/layout_handcheck_20260904/$MODEL}
SCRATCH=${SCRATCH:-/data2/chenyi9/kvpim_run_scratch/handcheck}
RUNGS=${RUNGS:-"A1 A2 A3b A4c A4e A5 A6"}
WORKERS=${RAMU_WORKERS:-8}

mkdir -p "$OUT" "$SCRATCH"
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export KVPIM_CPPCORE=${KVPIM_CPPCORE:-1}
export KVPIM_NOGC=0
# Probe scope: layer 0 only, and stop after 400 scans -- enough to see every
# rung's placement, small enough to read.
export KVPIM_LAYOUT_DUMP_LAYER=${KVPIM_LAYOUT_DUMP_LAYER:-0}
export KVPIM_LAYOUT_DUMP_MAX=${KVPIM_LAYOUT_DUMP_MAX:-400}

echo "=== layout hand-check: $MODEL  ngpu=$NGPU num_hbm=$NUM_HBM k=$K  $(date)"
echo "    rungs: $RUNGS   workers/rung: $WORKERS   out: $OUT"

declare -A PID
for A in $RUNGS; do
    REUSE=recompute
    EXTRA=(--epic-prefix-recompute-tokens "$K")
    # A1 is the no-reuse baseline: it takes neither --reuse recompute nor k.
    if [ "$A" = A1 ]; then REUSE=no-reuse; EXTRA=(); fi
    RD=$SCRATCH/ram_$A
    rm -rf "$RD"; mkdir -p "$RD"
    ln -sf "$REPO/ramulator2/ramulator2" "$RD/ramulator2"
    ln -sf "$REPO/ramulator2/trace_gen"  "$RD/trace_gen"
    cp "$REPO/ramulator.out" "$RD/ramulator.out" 2>/dev/null || :
    rm -f "$OUT/layout_$A.jsonl"
    echo "=== launch $A (reuse=$REUSE) $(date +%H:%M:%S)"
    (
      export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
      export KVPIM_LAYOUT_DUMP="$OUT/layout_$A.jsonl" KVPIM_LAYOUT_DUMP_TAG="$A"
      python3 main.py --system dgx-attacc --model "$MODEL" \
        --workload "$WL" --reuse "$REUSE" ${EXTRA[@]+"${EXTRA[@]}"} \
        --ablation "$A" --engine dag --pipeopt \
        --workload-report "$OUT/dag_$A.json" \
        --workload-report-events none \
        --cacheblend-batch-size 8 \
        --num-hbm "$NUM_HBM" --ngpu "$NGPU" \
        --ramulator-workers "$WORKERS"
    ) > "$OUT/dag_$A.log" 2>&1 &
    PID[$A]=$!
done

FAILED=0
for A in $RUNGS; do
    t=$(date +%s)
    if wait "${PID[$A]}"; then
        echo "=== done $A $(date +%H:%M:%S)"
        grep -h REPORT_SUMMARY "$OUT/dag_$A.log" || true
    else
        echo "!!! $A FAILED -- see $OUT/dag_$A.log" >&2
        tail -5 "$OUT/dag_$A.log" >&2
        FAILED=1
    fi
    echo "    layout dump: $(wc -l < "$OUT/layout_$A.jsonl" 2>/dev/null || echo 0) records"
done
rm -rf "$SCRATCH"/ram_*
echo "=== ALL_DONE failed=$FAILED $(date)"
