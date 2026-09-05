#!/usr/bin/env bash
# Run ladder points of workload/probe/sweep/manifest.csv under the paper
# conventions (docs/README_run_guide.md):
#   --gpu-model flash  --pipeopt  --powerlimit  k=8  batch 8  1 GPU + 1 HBM
# Rungs per point (chenyi9 ruling 2026-09-05): the baseline (B0_*) runs all
# seven, every sweep point runs only A3b and A6.  Core budget <= 64:
#   B0    2 ladders x (6 PIM rungs x W + 7) = 62 at W=4
#   sweep 4 ladders x (2 PIM rungs x W + 2) = 64 at W=7    (A3b, A6)
#
#   usage: run_sweep.sh <outroot> [filter-regex] [MODEL]
#     filter-regex  selects manifest rows by file name, e.g. '^B0_' or 'S5_.*_turns'
#     MODEL         default CACHEBLEND-TINY
#   env: RUNGS (override the per-point rule), RAMU_WORKERS, PARALLEL, EPIC_K (8),
#        GPU_MODEL (flash), KVPIM_SCRATCH (Ramulator dir)
set -u
OUTROOT=${1:?usage: run_sweep.sh <outroot> [filter-regex] [MODEL]}
FILTER=${2:-.}
MODEL=${3:-CACHEBLEND-TINY}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
SWEEP=$REPO/workload/probe/sweep
export EPIC_K=${EPIC_K:-8} GPU_MODEL=${GPU_MODEL:-flash}
export NUM_HBM=${NUM_HBM:-1} NGPU=${NGPU:-1} KVPIM_CPPCORE=1 PYTHONPATH=$REPO
RUNGS_OVERRIDE=${RUNGS:-}
if [ -n "${KVPIM_SCRATCH:-}" ]; then
    export ATTACC_RAMULATOR_DIR=$KVPIM_SCRATCH ATTACC_RAMULATOR_LOG=$KVPIM_SCRATCH/ramulator.out
fi
mkdir -p "$OUTROOT"
mapfile -t FILES < <(tail -n +2 "$SWEEP/manifest.csv" | cut -d, -f1 | grep -E "$FILTER")
echo "$(date +%T) sweep start: ${#FILES[@]} workloads, model $MODEL, gpu_model $GPU_MODEL" >> "$OUTROOT/sweep.log"
rungs_for() {   # baseline: all seven; sweep point: A3b and A6
    if [ -n "$RUNGS_OVERRIDE" ]; then echo "$RUNGS_OVERRIDE";
    elif [[ $1 == B0_* ]]; then echo "A1 A2 A3b A4c A4e A5 A6";
    else echo "A3b A6"; fi
}
run_one() {
    local file=$1 tag=${1%.json} rungs workers
    rungs=$(rungs_for "$file")
    if [[ $rungs == *A1* ]]; then workers=${RAMU_WORKERS:-4}; else workers=${RAMU_WORKERS:-7}; fi
    RUNGS="$rungs" RAMU_WORKERS=$workers KVPIM_PREFILL_SIDE_LOG=$OUTROOT/$tag.sides.jsonl \
    bash "$SCRIPT_DIR/run_dag_ladder.sh" "$SWEEP/$file" "$MODEL" "$OUTROOT/$tag" \
        > "$OUTROOT/$tag.log" 2>&1
    echo "$(date +%T) $tag [$rungs] exit $?" >> "$OUTROOT/sweep.log"
}
i=0
while [ $i -lt ${#FILES[@]} ]; do
    if [[ ${FILES[$i]} == B0_* ]]; then width=${PARALLEL:-2}; else width=${PARALLEL:-4}; fi
    for ((j = 0; j < width && i + j < ${#FILES[@]}; j++)); do
        run_one "${FILES[$((i + j))]}" &
    done
    wait
    i=$((i + width))
done
echo "$(date +%T) sweep done" >> "$OUTROOT/sweep.log"
