#!/usr/bin/env bash
# Batch-level parallel suite driver (chenyi9 2026-08-27: "能并行的都并行").
#
# Runs the 5-workload x 5-ratio ladder suite with N_PAR batches in flight at
# once.  Core budget stays under 96: N_PAR x 6 PIM rungs x RAMU_WORKERS
# simulations + N_PAR x 7 construction processes; the default 3 x 6 x 4 + 21
# = 93.  k=2 batches keep the cache-first warm pass (chart-grade firsts);
# every other ratio runs --no-warm (signatures largely on disk, rare misses
# simulate inline).
#   usage: run_dag_suite.sh [MODEL]
set -uo pipefail
MODEL=${1:-LLAMA-7B}
N_PAR=${N_PAR:-3}
export RAMU_WORKERS=${RAMU_WORKERS:-4}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKLOADS=(star_repair_r5w3k47 pipeline_repair_c5k50 debate_d3r5k49
           mapreduce_sum_m8 multisource_rag_n12s96)

run_one() {
    local K=$1 W=$2
    local warm_env=""
    [ "$K" != 2 ] && warm_env=1
    echo "##### start k=$K $W $(date +%H:%M:%S)"
    EPIC_K=$K NO_WARM=${warm_env} bash "$SCRIPT_DIR/run_dag_ladder.sh" \
        "$SCRIPT_DIR/../workload/workload_${W}.json" "$MODEL" \
        2>&1 | grep -E "REPORT_SUMMARY|CSV:|FAILED" | sed "s|^|[k$K $W] |"
    echo "##### end k=$K $W $(date +%H:%M:%S)"
}

# k=2 first (chart data), then the other ratios.
JOBS=()
for W in "${WORKLOADS[@]}"; do JOBS+=("2 $W"); done
for K in 4 8 16 32; do for W in "${WORKLOADS[@]}"; do JOBS+=("$K $W"); done; done

active=0
for job in "${JOBS[@]}"; do
    run_one $job &
    active=$((active + 1))
    if [ "$active" -ge "$N_PAR" ]; then
        wait -n
        active=$((active - 1))
    fi
done
wait
echo "SUITE DONE $(date +%H:%M:%S)"
