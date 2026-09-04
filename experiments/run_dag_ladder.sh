#!/usr/bin/env bash
# One-click DAG ladder (chenyi9 order 2026-08-26): point this script at ONE
# workload JSON and it runs all six A-rungs on the physical event-DAG engine
# (--engine dag) and emits a CSV with per-part energy and the share of
# prefill attention placed on the PIM.
#
#   usage: run_dag_ladder.sh <workload.json> [MODEL] [OUTDIR]
#     MODEL  default LLAMA-7B   (GPT-175B, LLAMA-65B, ... as in main.py)
#     OUTDIR default <workload_dir>/dag_ladder_<stem>_<MODEL>/
#
# Fixed choices (paper ladder conventions):
#   * A1 runs --reuse no-reuse; A2-A6 run --reuse epic with k=8
#     (--epic-prefix-recompute-tokens 8, the matrix口径).
#   * --ramulator-workers 64 and the persistent signature cache
#     (ruling 2026-08-26: cache first with up to 64 cores, then run).
#   * per-rung placement/layout/batch-command knobs come from the presets.
set -euo pipefail
WL=${1:?usage: run_dag_ladder.sh <workload.json> [model] [outdir]}
MODEL=${2:-LLAMA-7B}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
WL=$(readlink -f "$WL")
STEM=$(basename "$WL" .json)
# Output convention (chenyi9 2026-08-26): results live in <repo>/output/,
# one folder per run named <timestamp>_<workload>_<model>.
OUT=${3:-$REPO/output/$(date +%Y%m%d-%H%M%S)_${STEM}_${MODEL}_k${EPIC_K:-8}}
mkdir -p "$OUT"

# All six rungs launch IN PARALLEL (ruling chenyi9 2026-08-26).  Each rung
# runs its own warm phase; the persistent signature cache on disk is shared,
# so rungs feed each other whatever they finish first.
# Core budget (ruling chenyi9 2026-08-26): AT MOST 96 CPU cores total ->
# SEVEN rungs since A3a (2026-08-26): 6 PIM rungs x 14 Ramulator
# workers + 7 construction processes = 91 <= 96.  Override per-rung width with RAMU_WORKERS if the budget
# changes.
RAMU_WORKERS=${RAMU_WORKERS:-14}
# Recompute-ratio knob (chenyi9 2026-08-26: run several ratios, prefer
# lower recompute): EPIC prefix tokens per shifted segment for A2-A6.
EPIC_K=${EPIC_K:-8}
# A1 dedup (chenyi9 2026-08-27): A1 is no-reuse -- its invocation is
# LITERALLY the same command at every EPIC_K, so k!=2 batches may skip it
# and copy the k=2 batch's dag_A1.json (bit-identical by determinism).
# SKIP_A1=1 requires A1_JSON=<path to an existing dag_A1.json>.
# NINE rungs since the placement re-cut (chenyi9 2026-08-29): + A3b (head
# slicing) and A4b (global co-read placement table).
# RUNGS may be set in the environment to run a SUBSET -- used to backfill the
# rungs a task lost to an infrastructure failure without re-running the ones it
# already has.  Unset (the normal case) means the full nine, unchanged.  Note
# that collect_dag_ladder.py at the end reads whatever dag_A*.json are in OUT,
# so a backfill that completes the set still produces the full CSV.
RUNGS=${RUNGS:-"A1 A2 A3 A3a A3b A4 A4b A5 A6"}
if [ -n "${SKIP_A1:-}" ]; then
    RUNGS="A2 A3 A3a A3b A4 A4b A5 A6"
    : "${A1_JSON:?SKIP_A1=1 requires A1_JSON=<path to dag_A1.json>}"
    [ -f "$A1_JSON" ] || { echo "A1_JSON not found: $A1_JSON" >&2; exit 1; }
fi
declare -A PID
for A in $RUNGS; do
    REUSE=recompute
    EXTRA=(--epic-prefix-recompute-tokens "$EPIC_K")
    if [ "$A" = A1 ]; then REUSE=no-reuse; EXTRA=(); fi
    echo "=== launch $A (reuse=$REUSE) $(date +%H:%M:%S) ==="
    (cd "$REPO" && python3 main.py \
        --system dgx-attacc --model "$MODEL" \
        --workload "$WL" --reuse "$REUSE" ${EXTRA[@]+"${EXTRA[@]}"} \
        --ablation "$A" --engine dag \
        --workload-report "$OUT/dag_${A}.json" \
        --workload-report-events none \
        --cacheblend-batch-size 8 \
        ${NUM_HBM:+--num-hbm "$NUM_HBM"} \
        ${NGPU:+--ngpu "$NGPU"} \
        ${NO_WARM:+--no-warm} \
        --ramulator-workers "$RAMU_WORKERS") > "$OUT/dag_${A}.log" 2>&1 &
    PID[$A]=$!
done
FAILED=0
for A in $RUNGS; do
    if wait "${PID[$A]}"; then
        echo "=== done $A $(date +%H:%M:%S) ==="
        grep -h REPORT_SUMMARY "$OUT/dag_${A}.log" || true
    else
        echo "$A FAILED -- see $OUT/dag_${A}.log" >&2
        FAILED=1
    fi
done
[ "$FAILED" = 0 ] || exit 1

if [ -n "${SKIP_A1:-}" ]; then
    cp "$A1_JSON" "$OUT/dag_A1.json"
fi
# SKIP_COLLECT=1 when several single-rung invocations share one OUT (the
# per-rung fan-out of output/_orch2/baseline_rung_worker.sh): they would all
# rewrite dag_ladder.csv at once, racing each other over a file that is only
# meaningful once the whole ladder is on disk.  Collect once at the end
# instead -- collect_dag_ladder.py emits whichever rungs it finds.
if [ -n "${SKIP_COLLECT:-}" ]; then
    echo "SKIP_COLLECT set -- not writing dag_ladder.csv"
else
    python3 "$SCRIPT_DIR/collect_dag_ladder.py" "$OUT" "$WL" "$MODEL"
    echo "CSV: $OUT/dag_ladder.csv"
fi
