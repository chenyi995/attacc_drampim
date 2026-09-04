#!/usr/bin/env bash
# Local (no-Slurm) re-run of rungs A3b..A6 on the striped-append layout
# (chenyi9 2026-09-03).  One model at a time; the five rungs of that model run
# in parallel inside run_dag_ladder.sh.
#
#   usage: bash experiments/run_local_a3b_a6.sh [OUT_ROOT]
#
# Core budget on a 128-core host: 5 rungs x RAMU_WORKERS(20) + 5 construction
# processes = 105 <= 128.
#
# Scratch MUST live on /data2: this host's / (which carries /tmp) has ~3 GB
# free and a trace-heavy rung needs far more.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
OUT_ROOT=${1:-$REPO/output/sweep_a3b_a6_20260903_append}
SCRATCH=${SCRATCH:-/data2/chenyi9/kvpim_run_scratch}
WL=$REPO/workload/sweep/wl_baseline_alltoall_N16_C16_D2.json
K=8

# Same model table as output/_orch2/common.sh (ngpu, num_hbm).
declare -A NGPU=( [LLAMA-7B]=1 [LLAMA3-8B]=1 [GPT-13B]=2 [LLAMA-33B]=2 [GPT-175B]=8 [LLAMA-65B]=8 )
declare -A NHBM=( [LLAMA-7B]=1 [LLAMA3-8B]=1 [GPT-13B]=10 [LLAMA-33B]=10 [GPT-175B]=40 [LLAMA-65B]=40 )
MODELS=${MODELS:-"GPT-13B LLAMA-33B LLAMA3-8B LLAMA-65B GPT-175B LLAMA-7B"}

export PYTHONPATH=$REPO
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export KVPIM_CPPCORE=${KVPIM_CPPCORE:-1}      # .so rebuilt with gcc-toolset-11
export KVPIM_NOGC=0

mkdir -p "$OUT_ROOT"
echo "=== A3b..A6 local re-run -> $OUT_ROOT   $(date)"
echo "    engine: striped-append layout; cppcore=$KVPIM_CPPCORE; scratch=$SCRATCH"

for model in $MODELS; do
  OUT=$OUT_ROOT/$model/baseline_k$K
  mkdir -p "$OUT"
  if [ "$(ls "$OUT"/dag_A*.json 2>/dev/null | wc -l)" -ge 5 ]; then
    echo "--- $(date +%T) $model already complete, skipping"; continue
  fi
  RD=$SCRATCH/ram_${model}
  rm -rf "$RD"; mkdir -p "$RD"
  ln -sf "$REPO/ramulator2/ramulator2" "$RD/ramulator2"
  ln -sf "$REPO/ramulator2/trace_gen"  "$RD/trace_gen"
  cp "$REPO/ramulator.out" "$RD/ramulator.out" 2>/dev/null || :
  export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
  echo "--- $(date +%T) $model  ngpu=${NGPU[$model]} num_hbm=${NHBM[$model]}"
  t0=$(date +%s)
  RUNGS="A3b A4 A4b A5 A6" NUM_HBM=${NHBM[$model]} NGPU=${NGPU[$model]} \
    RAMU_WORKERS=${RAMU_WORKERS:-20} EPIC_K=$K \
    bash "$REPO/experiments/run_dag_ladder.sh" "$WL" "$model" "$OUT" \
    > "$OUT/ladder.log" 2>&1 < /dev/null
  rc=$?
  n=$(ls "$OUT"/dag_A*.json 2>/dev/null | wc -l)
  echo "--- $(date +%T) $model rc=$rc jsons=$n/5 elapsed=$(( $(date +%s) - t0 ))s"
  rm -rf "$RD"
done
echo "=== ALL_DONE $(date)"
