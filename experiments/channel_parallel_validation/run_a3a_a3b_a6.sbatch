#!/usr/bin/env bash
# Baseline LLAMA-7B validation of the restored channel-parallel DAG and
# head-folded A6 placement probe.
#SBATCH --job-name=kvpim-a3a3ba6
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=experiments/channel_parallel_validation/logs/%A_%a.log
#SBATCH --error=experiments/channel_parallel_validation/logs/%A_%a.log

set -euo pipefail

REPO=/home/xw338/attacc/attacc_drampim
OUT="$REPO/output/channel_parallel_validation_20260902/LLAMA-7B/baseline_k8"
RUNGS=(A3a A3b A6)
RUNG=${RUNGS[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$OUT"
cd "$REPO"

# No --pipeopt: only the sixteen lanes of each PIM KV scan are parallel.
KVPIM_CPPCORE=1 KVPIM_DEBUG_A6=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 main.py --system dgx-attacc --model LLAMA-7B --ngpu 1 --num-hbm 1 \
  --workload workload/sweep/wl_baseline_alltoall_N16_C32_D2.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 --ablation "$RUNG" --engine dag \
  --cacheblend-batch-size 8 --ramulator-workers 8 \
  --workload-report "$OUT/dag_${RUNG}.json" \
  --workload-report-events none
