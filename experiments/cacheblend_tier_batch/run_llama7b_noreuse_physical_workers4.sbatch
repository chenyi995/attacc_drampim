#!/usr/bin/env bash
#SBATCH --job-name=attacc-llama7b-nrp
#SBATCH --partition=athena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/cacheblend_tier_batch/results/llama7b_workers4/no_reuse_physical_%j.log
#SBATCH --error=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/cacheblend_tier_batch/results/llama7b_workers4/no_reuse_physical_%j.log

set -euo pipefail
cd /zpool-00/home/xw338/attacc/attacc_drampim
out_dir=experiments/cacheblend_tier_batch/results/llama7b_workers4
mkdir -p "$out_dir"

srun --cpu-bind=cores python3 main.py \
  --system dgx-attacc --model LLAMA-7B \
  --workload workload/workload_relay_s400w4t1.json \
  --reuse no-reuse --no-reuse-latency-model physical \
  --cacheblend-batch-size 4 --ramulator-workers 4 --pipeopt \
  --workload-report "$out_dir/no_reuse_physical_b4_pipeopt.json" \
  > "$out_dir/no_reuse_physical_b4_pipeopt.log" 2>&1
