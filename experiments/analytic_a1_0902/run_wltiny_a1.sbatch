#!/usr/bin/env bash
# Submit twice: MODE=analytic first, then MODE=ramulator after it succeeds.
# Both jobs build the same current A1 DAG.  Only its PIM timing backend varies.
#SBATCH --job-name=a1-analytic-check
#SBATCH --partition=athena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/analytic_a1_0902/slurm_%x_%j.log
#SBATCH --error=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/analytic_a1_0902/slurm_%x_%j.log

set -euo pipefail
repo=/zpool-00/home/xw338/attacc/attacc_drampim
out="$repo/experiments/analytic_a1_0902/slurm_wltiny"
mode="${MODE:?submit with --export=MODE=analytic or MODE=ramulator}"
mkdir -p "$out"
cd "$repo"

# A private persistent cache means the reference may use existing calibration
# entries but cannot append to the repository's dirty signature-cache file.
private_ram="$out/ramulator2"
if [[ ! -d "$private_ram" ]]; then
  mkdir -p "$private_ram"
  ln -s "$repo/ramulator2/ramulator2" "$private_ram/ramulator2"
  ln -s "$repo/ramulator2/trace_gen" "$private_ram/trace_gen"
  cp "$repo/ramulator2/signature_cache.jsonl" "$private_ram/signature_cache.jsonl"
fi

args=(--system dgx-attacc --model LLAMA-7B --workload workload/wl_tiny.json
      --reuse no-reuse --ablation A1 --engine dag --powerlimit --no-warm
      --ramulator-workers 4 --workload-report-events none)
if [[ "$mode" == analytic ]]; then
  args+=(--analytic-pim-model experiments/analytic_a1_0902/timing_models.json
         --workload-report "$out/a1_llama7b_wltiny_analytic.json")
else
  args+=(--workload-report "$out/a1_llama7b_wltiny_ramulator.json")
fi

echo "START $(date --iso-8601=seconds) mode=$mode host=$(hostname)"
ATTACC_RAMULATOR_DIR="$private_ram" ATTACC_RAMULATOR_LOG="$out/ramulator.out" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/bin/time -f 'wall_s=%e' python3 main.py "${args[@]}"
echo "END $(date --iso-8601=seconds) mode=$mode"
