#!/usr/bin/env bash
# Validate pipe=False PIM-channel phases: 3 small workloads x A3--A6.
# Submit as one array with no more than six 8-core jobs at once.
#SBATCH --job-name=kvpim-ch-par
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-11%6
#SBATCH --output=experiments/channel_parallel_validation/logs/%A_%a.log
#SBATCH --error=experiments/channel_parallel_validation/logs/%A_%a.log

set -euo pipefail

REPO=/home/xw338/attacc/attacc_drampim
OUT_ROOT="$REPO/output/channel_parallel_validation_20260902"
CASES=(baseline broadcast reduce)
RUNGS=(A3 A4 A5 A6)
CASE_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
RUNG_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
CASE=${CASES[$CASE_INDEX]}
RUNG=${RUNGS[$RUNG_INDEX]}

case "$CASE" in
  baseline) WORKLOAD=wl_baseline_alltoall_N16_C32_D2.json ;;
  broadcast) WORKLOAD=wl_broadcast.json ;;
  reduce) WORKLOAD=wl_reduce.json ;;
esac

OUT="$OUT_ROOT/LLAMA-7B/${CASE}_k8"
mkdir -p "$OUT"
cd "$REPO"

# Do not pass --pipeopt: this is the conservative no-cross-device-overlap
# validation.  The patched scheduler still runs the lanes of each PIM scan
# concurrently and charges their maximum duration.
KVPIM_CPPCORE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 main.py --system dgx-attacc --model LLAMA-7B --ngpu 1 --num-hbm 1 \
  --workload "workload/sweep/$WORKLOAD" --reuse recompute \
  --epic-prefix-recompute-tokens 8 --ablation "$RUNG" --engine dag \
  --cacheblend-batch-size 8 --ramulator-workers 8 \
  --workload-report "$OUT/dag_${RUNG}.json" \
  --workload-report-events none
