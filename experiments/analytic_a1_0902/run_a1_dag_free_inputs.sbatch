#!/usr/bin/env bash
# Required exports: TAG, WORKLOAD, MODE={analytic,ramulator}.
#SBATCH --job-name=a1-dagfree-inputs
#SBATCH --partition=athena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/analytic_a1_0902/slurm_%x_%j.log
#SBATCH --error=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/analytic_a1_0902/slurm_%x_%j.log
set -euo pipefail
repo=/zpool-00/home/xw338/attacc/attacc_drampim
out="$repo/experiments/analytic_a1_0902/dag_free_inputs/${TAG}.json"
mode="${MODE:?}"
cd "$repo"
if [[ "$mode" == "ramulator" ]]; then
  private_ram="$repo/experiments/analytic_a1_0902/dag_free_inputs/${TAG}.ramulator2"
  mkdir -p "$private_ram"
  ln -sfn "$repo/ramulator2/ramulator2" "$private_ram/ramulator2"
  ln -sfn "$repo/ramulator2/trace_gen" "$private_ram/trace_gen"
  # Preserve completed signatures when a calibration job is requeued.
  if [[ ! -f "$private_ram/signature_cache.jsonl" ]]; then
    # COLD by construction: seeding from the repo cache would make any
  # "cold cache" wall-clock claim meaningless.
  : > "$private_ram/signature_cache.jsonl"
  fi
fi
echo "START $(date --iso-8601=seconds) tag=$TAG"
ATTACC_RAMULATOR_DIR="${private_ram:-$repo/ramulator2}" /usr/bin/time -f 'wall_s=%e' python3 experiments/analytic_a1_0902/run_a1_dag_free_inputs.py \
  --workload "$WORKLOAD" --timing-model experiments/analytic_a1_0902/timing_models.json \
  --power-constraint --mode "$mode" --output "$out"
echo "END $(date --iso-8601=seconds) tag=$TAG"
