#!/usr/bin/env bash
# Required exports: MODE={analytic,ramulator}, TAG, WORKLOAD (repo-relative).
#SBATCH --job-name=a1-existing
#SBATCH --partition=athena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/analytic_a1_0902/slurm_%x_%j.log
#SBATCH --error=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/analytic_a1_0902/slurm_%x_%j.log

set -euo pipefail
repo=/zpool-00/home/xw338/attacc/attacc_drampim
out="$repo/experiments/analytic_a1_0902/slurm_existing/$TAG"
mode="${MODE:?}"; workload="${WORKLOAD:?}"
analytic_model="${ANALYTIC_MODEL:-experiments/analytic_a1_0902/timing_models.json}"
mkdir -p "$out"; cd "$repo"

# Keep each reference cache private so parallel Slurm jobs cannot append to
# the repository cache or race each other.
private_ram="$out/ramulator2"
if [[ ! -d "$private_ram" ]]; then
  mkdir -p "$private_ram"
  ln -s "$repo/ramulator2/ramulator2" "$private_ram/ramulator2"
  ln -s "$repo/ramulator2/trace_gen" "$private_ram/trace_gen"
  # COLD by construction: seeding from the repo cache would make any
  # "cold cache" wall-clock claim meaningless.
  : > "$private_ram/signature_cache.jsonl"
fi

args=(--system dgx-attacc --model LLAMA-7B --workload "$workload"
      --reuse no-reuse --ablation A1 --engine dag --powerlimit --no-warm
      --ramulator-workers 4 --workload-report-events none)
if [[ "$mode" == analytic ]]; then
  args+=(--analytic-pim-model "$analytic_model"
         --workload-report "$out/analytic.json")
else
  args+=(--workload-report "$out/ramulator.json")
fi
echo "START $(date --iso-8601=seconds) tag=$TAG mode=$mode"
ATTACC_RAMULATOR_DIR="$private_ram" ATTACC_RAMULATOR_LOG="$out/ramulator.out" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/bin/time -f 'wall_s=%e' python3 main.py "${args[@]}"
echo "END $(date --iso-8601=seconds) tag=$TAG mode=$mode"
