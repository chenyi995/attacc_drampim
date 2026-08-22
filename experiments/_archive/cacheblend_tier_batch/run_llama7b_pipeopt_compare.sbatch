#!/usr/bin/env bash
#SBATCH --job-name=attacc-llama7b-pipe
#SBATCH --partition=athena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/cacheblend_tier_batch/results/llama7b_workers4/pipeopt_%j.log
#SBATCH --error=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/cacheblend_tier_batch/results/llama7b_workers4/pipeopt_%j.log

set -euo pipefail
cd /zpool-00/home/xw338/attacc/attacc_drampim
out_dir=experiments/cacheblend_tier_batch/results/llama7b_workers4
mkdir -p "$out_dir"

common=(
  --system dgx-attacc --model LLAMA-7B
  --workload workload/workload_relay_s400w4t1.json
  --ramulator-workers 4 --pipeopt
)

srun --cpu-bind=cores python3 main.py "${common[@]}" \
  --reuse no-reuse \
  --workload-report "$out_dir/no_reuse_pipeopt.json" \
  > "$out_dir/no_reuse_pipeopt.log" 2>&1

srun --cpu-bind=cores python3 main.py "${common[@]}" \
  --reuse cacheblend \
  --cacheblend-full-layers 0-1 --cacheblend-partial-layers 2-31 \
  --cacheblend-recompute-ratio 0.15 --reuse-seed 7 \
  --cacheblend-batch-size 4 \
  --workload-report "$out_dir/cacheblend_pipeopt.json" \
  > "$out_dir/cacheblend_pipeopt.log" 2>&1
