#!/usr/bin/env bash
#SBATCH --job-name=attacc-cacheblend-w4
#SBATCH --partition=athena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/cacheblend_tier_batch/results/cacheblend_workers4_%j.log
#SBATCH --error=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/cacheblend_tier_batch/results/cacheblend_workers4_%j.log

set -euo pipefail

cd /zpool-00/home/xw338/attacc/attacc_drampim
mkdir -p experiments/cacheblend_tier_batch/results

srun --cpu-bind=cores python3 main.py \
  --system dgx-attacc --model GPT-175B \
  --workload workload/workload_relay_s400w4t1.json --reuse cacheblend \
  --cacheblend-full-layers 0-1 --cacheblend-partial-layers 2-95 \
  --cacheblend-recompute-ratio 0.15 --reuse-seed 7 \
  --cacheblend-batch-size 4 --ramulator-workers 4 \
  --workload-report experiments/cacheblend_tier_batch/results/cacheblend_workers4.json
