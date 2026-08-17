#!/usr/bin/env bash
#SBATCH --job-name=attacc-llama7b-noreuse
#SBATCH --partition=athena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/cacheblend_tier_batch/results/llama7b_workers4/no_reuse_report_%j.log
#SBATCH --error=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/cacheblend_tier_batch/results/llama7b_workers4/no_reuse_report_%j.log

set -euo pipefail
cd /zpool-00/home/xw338/attacc/attacc_drampim
mkdir -p experiments/cacheblend_tier_batch/results/llama7b_workers4

srun --cpu-bind=cores python3 main.py \
  --system dgx-attacc --model LLAMA-7B \
  --workload workload/workload_relay_s400w4t1.json --reuse no-reuse \
  --ramulator-workers 4 \
  --workload-report experiments/cacheblend_tier_batch/results/llama7b_workers4/no_reuse_complete.json
