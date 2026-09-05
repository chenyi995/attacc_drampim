#!/usr/bin/env bash
# Run ladder points of workload/probe/sweep/manifest.csv, two ladders at a
# time, under the paper conventions (docs/README_run_guide.md):
#   --gpu-model flash  --pipeopt  --powerlimit  k=8  batch 8  1 GPU + 1 HBM
# Core budget <= 64: 2 ladders x (6 PIM rungs x RAMU_WORKERS + 7) = 62 at W=4.
#
#   usage: run_sweep.sh <outroot> [filter-regex] [MODEL]
#     filter-regex  selects manifest rows by file name, e.g. '^B0_' or 'S5_.*_turns'
#     MODEL         default CACHEBLEND-TINY
#   env: RAMU_WORKERS (4), EPIC_K (8), GPU_MODEL (flash), KVPIM_SCRATCH (Ramulator dir)
set -u
OUTROOT=${1:?usage: run_sweep.sh <outroot> [filter-regex] [MODEL]}
FILTER=${2:-.}
MODEL=${3:-CACHEBLEND-TINY}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
SWEEP=$REPO/workload/probe/sweep
export RAMU_WORKERS=${RAMU_WORKERS:-4} EPIC_K=${EPIC_K:-8} GPU_MODEL=${GPU_MODEL:-flash}
export NUM_HBM=${NUM_HBM:-1} NGPU=${NGPU:-1} KVPIM_CPPCORE=1 PYTHONPATH=$REPO
export RUNGS=${RUNGS:-"A1 A2 A3b A4c A4e A5 A6"}
if [ -n "${KVPIM_SCRATCH:-}" ]; then
    export ATTACC_RAMULATOR_DIR=$KVPIM_SCRATCH ATTACC_RAMULATOR_LOG=$KVPIM_SCRATCH/ramulator.out
fi
mkdir -p "$OUTROOT"
mapfile -t FILES < <(tail -n +2 "$SWEEP/manifest.csv" | cut -d, -f1 | grep -E "$FILTER")
echo "$(date +%T) sweep start: ${#FILES[@]} workloads, model $MODEL, gpu_model $GPU_MODEL" >> "$OUTROOT/sweep.log"
run_one() {
    local file=$1 tag=${1%.json}
    KVPIM_PREFILL_SIDE_LOG=$OUTROOT/$tag.sides.jsonl \
    bash "$SCRIPT_DIR/run_dag_ladder.sh" "$SWEEP/$file" "$MODEL" "$OUTROOT/$tag" \
        > "$OUTROOT/$tag.log" 2>&1
    echo "$(date +%T) $tag exit $?" >> "$OUTROOT/sweep.log"
}
i=0
while [ $i -lt ${#FILES[@]} ]; do
    run_one "${FILES[$i]}" &
    if [ $((i + 1)) -lt ${#FILES[@]} ]; then run_one "${FILES[$((i + 1))]}" & fi
    wait
    i=$((i + 2))
done
echo "$(date +%T) sweep done" >> "$OUTROOT/sweep.log"
