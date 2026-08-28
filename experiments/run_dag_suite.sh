#!/usr/bin/env bash
# Batch-level parallel suite driver (chenyi9 2026-08-27 restructure).
#
# One JOB per WORKLOAD (a chain): the k=2 batch runs first with the warm
# pass and all seven rungs; the other ratios follow inside the same chain
# with --no-warm and WITHOUT A1 (A1 is no-reuse, its invocation is the same
# command at every k, so the chain copies its own k=2 dag_A1.json --
# bit-identical by determinism; ruling chenyi9 2026-08-27).  N_PAR chains
# run in flight, so at most N_PAR batches execute at any moment (K_PAR>=2
# doubles a chain's k-sweep concurrency -- see run_chain) -- the same
# core/RAM envelope as the old 25-job pool: N_PAR x 6 PIM rungs x
# RAMU_WORKERS + N_PAR x 7 builders.
#   usage: run_dag_suite.sh [MODEL]
#   env:   N_PAR (default 3), RAMU_WORKERS (default 4), NUM_HBM (optional),
#          SUITE_WORKLOADS (subset), K_PAR (parallel k-sweep, default 1)
set -uo pipefail
MODEL=${1:-LLAMA-7B}
N_PAR=${N_PAR:-3}
export RAMU_WORKERS=${RAMU_WORKERS:-4}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
WORKLOADS=(star_repair_r5w3k47 pipeline_repair_c5k50 debate_d3r5k49
           mapreduce_sum_m8 multisource_rag_n12s96)
# Optional subset override (space-separated stems), e.g. resuming after an
# operator intervention: SUITE_WORKLOADS="mapreduce_sum_m8 multisource_rag_n12s96"
if [ -n "${SUITE_WORKLOADS:-}" ]; then
    read -r -a WORKLOADS <<< "$SUITE_WORKLOADS"
fi

run_batch() {
    local K=$1 W=$2 OUT=$3 warm=$4 a1json=$5
    echo "##### start k=$K $W $(date +%H:%M:%S)"
    local env_no_warm="" env_skip="" env_a1=""
    if [ "$warm" != warm ]; then env_no_warm=1; fi
    if [ -n "$a1json" ]; then env_skip=1; env_a1=$a1json; fi
    EPIC_K=$K NO_WARM=${env_no_warm} SKIP_A1=${env_skip} A1_JSON=${env_a1} \
        bash "$SCRIPT_DIR/run_dag_ladder.sh" \
        "$REPO/workload/workload_${W}.json" "$MODEL" "$OUT" \
        2>&1 | grep -E "REPORT_SUMMARY|CSV:|FAILED" | sed "s|^|[k$K $W] |"
    local rc=${PIPESTATUS[0]}
    echo "##### end k=$K $W $(date +%H:%M:%S)"
    return "$rc"
}

run_chain() {
    local W=$1
    local STEM="workload_${W}"
    local k2out="$REPO/output/$(date +%Y%m%d-%H%M%S)_${STEM}_${MODEL}_k2"
    if ! run_batch 2 "$W" "$k2out" warm ""; then
        echo "##### chain $W ABORTED: k=2 batch FAILED" >&2
        return 1
    fi
    local K
    # Ratio axis k in {2, 8, 32} (chenyi9 2026-08-27: 降为三点).
    # K_PAR=2 (W4) runs the two k-sweep batches concurrently -- use ONLY
    # when the operator has RAM headroom for two of this workload's
    # construction footprints (560-GB-cap note: big workloads do not fit).
    if [ "${K_PAR:-1}" -ge 2 ]; then
        local KPIDS=()
        for K in 8 32; do
            run_batch "$K" "$W" \
                "$REPO/output/$(date +%Y%m%d-%H%M%S)_${STEM}_${MODEL}_k${K}" \
                no-warm "$k2out/dag_A1.json" &
            KPIDS+=($!)
        done
        local KP
        for KP in "${KPIDS[@]}"; do wait "$KP" || true; done
    else
        for K in 8 32; do
            run_batch "$K" "$W" \
                "$REPO/output/$(date +%Y%m%d-%H%M%S)_${STEM}_${MODEL}_k${K}" \
                no-warm "$k2out/dag_A1.json" || true
        done
    fi
}

active=0
for W in "${WORKLOADS[@]}"; do
    run_chain "$W" &
    active=$((active + 1))
    if [ "$active" -ge "$N_PAR" ]; then
        wait -n
        active=$((active - 1))
    fi
done
wait
echo "SUITE DONE $(date +%H:%M:%S)"
