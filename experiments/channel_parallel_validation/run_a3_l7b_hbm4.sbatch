#!/usr/bin/env bash
# LLAMA-7B placement ladder with enough HBM stacks for A3b head slicing:
# 32 KV heads / 4 HBM = 8 heads per HBM, 16 channels / 8 heads = 2 lanes/head.
#
# Four stacks is the MINIMUM that makes A3b mean anything for this model:
# below it the stripe clamps to 1 and A3b is A3.  src/workload_runner.py's
# placement_degeneracy_warning() prints to stderr when that happens, so check
# the log before trusting an A3b number.
#SBATCH --job-name=kvpim-a3-hbm4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=experiments/channel_parallel_validation/logs/%A_%a.log
#SBATCH --error=experiments/channel_parallel_validation/logs/%A_%a.log

set -euo pipefail

REPO=/home/xw338/attacc/attacc_drampim
OUT="$REPO/output/channel_parallel_validation_20260902/LLAMA-7B/baseline_k8_hbm4"
RUNGS=(A3 A3a A3b)
RUNG=${RUNGS[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$OUT"
cd "$REPO"

# No --pipeopt: GPU/link/PIM macro phases remain serial; the 16 channel
# lanes of each KV scan run concurrently.  With 4 HBM, A3b has 2 lanes/head.
KVPIM_CPPCORE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 main.py --system dgx-attacc --model LLAMA-7B --ngpu 1 --num-hbm 4 \
  --workload workload/sweep/wl_baseline_alltoall_N16_C32_D2.json \
  --reuse recompute --epic-prefix-recompute-tokens 8 --ablation "$RUNG" --engine dag \
  --cacheblend-batch-size 8 --ramulator-workers 8 \
  --workload-report "$OUT/dag_${RUNG}.json" \
  --workload-report-events none
