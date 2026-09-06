#!/usr/bin/env bash
# Run the workload protocol of workload/probe/sweep/manifest.csv on ONE model
# (docs/README_run_protocol.md, chenyi9 2026-09-05):
#   * the BASELINE workload (W1_turns.json) runs every combo
#     A1 A2 A3b A4c A4e A5 A6;
#   * a SWEEP point (W1_S*_turns.json) runs A3b and A6 only;
#   * flash, pipeopt, powerlimit, k=8, batch 8, and the model's DGX-A100 +
#     AttAcc geometry: 5 HBM stacks per GPU, ngpu from the table below.
# Core budget <= 64 (chenyi9 2026-09-05):
#   baseline  1 ladder  x (6 PIM combos x 8 workers + 7) = 55
#   sweep     3 ladders x (2 PIM combos x 8 workers + 2) = 54    (A3b, A6)
#
#   usage: run_sweep.sh <outroot> [filter-regex] [MODEL]
#     filter-regex  selects manifest rows by file name, e.g. '^W1_turns' or 'W1_S3_'
#     MODEL         default LLAMA3-8B
#   env: RUNGS (override the per-point rule), RAMU_WORKERS (8), PARALLEL,
#        EPIC_K (8), GPU_MODEL (flash), NGPU / NUM_HBM (override the table),
#        EVENTS (none), BATCH (8), KVPIM_SCRATCH (Ramulator dir)
set -u
OUTROOT=${1:?usage: run_sweep.sh <outroot> [filter-regex] [MODEL]}
FILTER=${2:-.}
MODEL=${3:-LLAMA3-8B}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
SWEEP=$REPO/workload/probe/sweep
# Tensor-parallel GPUs per model (DGX-A100 has 8; AttAcc puts 5 HBM3 stacks
# behind each GPU, so NUM_HBM = 5 x NGPU).  The busiest stack then holds
# ceil(local KV heads / 5) heads and a head is striped over 16 // that many
# channels: LLAMA3-8B and LLAMA-65B 8 channels per head, GPT-13B 4,
# LLAMA-33B and GPT-175B 5, LLAMA-7B 2, TINY 8 (flow check only).
gpus_for() {
    case $1 in
        CACHEBLEND-TINY|LLAMA3-8B|LLAMA-7B) echo 1 ;;
        GPT-13B) echo 2 ;;
        LLAMA-33B) echo 4 ;;
        LLAMA-65B|GPT-175B) echo 8 ;;
        *) echo 1 ;;
    esac
}
export NGPU=${NGPU:-$(gpus_for "$MODEL")}
export NUM_HBM=${NUM_HBM:-$((5 * NGPU))}
export EPIC_K=${EPIC_K:-8} GPU_MODEL=${GPU_MODEL:-flash} EVENTS=${EVENTS:-none} BATCH=${BATCH:-8}
export KVPIM_CPPCORE=1 PYTHONPATH=$REPO OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUNGS_OVERRIDE=${RUNGS:-}
if [ -n "${KVPIM_SCRATCH:-}" ]; then
    export ATTACC_RAMULATOR_DIR=$KVPIM_SCRATCH ATTACC_RAMULATOR_LOG=$KVPIM_SCRATCH/ramulator.out
fi
mkdir -p "$OUTROOT"
mapfile -t FILES < <(tail -n +2 "$SWEEP/manifest.csv" | cut -d, -f1 | grep -E "$FILTER")
echo "$(date +%T) sweep start: ${#FILES[@]} workloads, model $MODEL (ngpu $NGPU, hbm $NUM_HBM), gpu_model $GPU_MODEL, k $EPIC_K, batch $BATCH" >> "$OUTROOT/sweep.log"
is_baseline() { [[ $1 =~ ^W[0-9]+_turns\.json$ ]]; }
rungs_for() {   # baseline: every combo; sweep point: A3b and A6
    if [ -n "$RUNGS_OVERRIDE" ]; then echo "$RUNGS_OVERRIDE";
    elif is_baseline "$1"; then echo "A1 A2 A3b A4c A4e A5 A6";
    else echo "A3b A6"; fi
}
run_one() {
    local file=$1 tag=${1%.json} rungs
    rungs=$(rungs_for "$file")
    RUNGS="$rungs" RAMU_WORKERS=${RAMU_WORKERS:-8} \
    KVPIM_PREFILL_SIDE_LOG=$OUTROOT/$tag.sides.jsonl \
    bash "$SCRIPT_DIR/run_dag_ladder.sh" "$SWEEP/$file" "$MODEL" "$OUTROOT/$tag" \
        > "$OUTROOT/$tag.log" 2>&1
    echo "$(date +%T) $tag [$rungs] exit $?" >> "$OUTROOT/sweep.log"
    python3 "$SCRIPT_DIR/summarize_ladder.py" "$OUTROOT/$tag" "$SWEEP/$file" A3b \
        > "$OUTROOT/$tag/summary.md" 2>&1
}
i=0
while [ $i -lt ${#FILES[@]} ]; do
    if is_baseline "${FILES[$i]}"; then width=${PARALLEL:-1}; else width=${PARALLEL:-3}; fi
    for ((j = 0; j < width && i + j < ${#FILES[@]}; j++)); do
        run_one "${FILES[$((i + j))]}" &
    done
    wait
    i=$((i + width))
done
echo "$(date +%T) sweep done" >> "$OUTROOT/sweep.log"
