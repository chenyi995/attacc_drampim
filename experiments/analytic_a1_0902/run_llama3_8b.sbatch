#!/usr/bin/env bash
# LLAMA3-8B A1 run, used two ways:
#   MODE=ramulator  -> ground truth AND the cold-cache cost baseline
#   MODE=analytic   -> the replacement, priced by a model calibrated on
#                      LLAMA-7B data only (a transfer test, not a refit)
#
# Both start from an EMPTY signature cache.  Comparing a warm-cache Ramulator
# run against the analytic model would flatter the model: the point of the
# comparison is the cost of producing a number that does not exist yet.
# The run report records dag_build_s, dag_finalize_s and
# ramulator_signature_cache.host_pricing_seconds separately, so DAG
# construction and PIM pricing can be compared on their own terms.
#
# Required exports: MODE={analytic,ramulator}, TAG, WORKLOAD (repo-relative).
#SBATCH --job-name=a1-llama3
#SBATCH --partition=athena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/analytic_a1_0902/slurm_%x_%j.log
#SBATCH --error=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/analytic_a1_0902/slurm_%x_%j.log

set -euo pipefail
repo=/zpool-00/home/xw338/attacc/attacc_drampim
out="$repo/experiments/analytic_a1_0902/llama3_8b/$TAG"
mode="${MODE:?}"; workload="${WORKLOAD:?}"
model="${MODEL:-LLAMA3-8B}"
analytic_model="${ANALYTIC_MODEL:-experiments/analytic_a1_0902/timing_models.json}"
workers="${WORKERS:-8}"
mkdir -p "$out"; cd "$repo"

# Private Ramulator directory with a COLD cache: no seeding from the repo
# cache, so the reference pays for every signature it needs and the two modes
# are compared on equal terms.  It is also what makes the resulting
# signature_cache.jsonl a clean LLAMA3-8B ground-truth set for validation.
private_ram="$out/ramulator2"
rm -rf "$private_ram"; mkdir -p "$private_ram"
ln -s "$repo/ramulator2/ramulator2" "$private_ram/ramulator2"
ln -s "$repo/ramulator2/trace_gen" "$private_ram/trace_gen"
: > "$private_ram/signature_cache.jsonl"

args=(--system dgx-attacc --model "$model" --workload "$workload"
      --reuse no-reuse --ablation A1 --engine dag --powerlimit --no-warm
      --ramulator-workers "$workers" --workload-report-events none)
if [[ "$mode" == analytic ]]; then
  args+=(--analytic-pim-model "$analytic_model"
         --workload-report "$out/analytic.json")
else
  args+=(--workload-report "$out/ramulator.json")
fi

echo "START $(date --iso-8601=seconds) tag=$TAG mode=$mode model=$model workload=$workload"
echo "cold cache: $(wc -l < "$private_ram/signature_cache.jsonl") entries"
ATTACC_RAMULATOR_DIR="$private_ram" ATTACC_RAMULATOR_LOG="$out/ramulator.out" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/bin/time -f 'wall_s=%e maxrss_kb=%M' python3 main.py "${args[@]}"
echo "cache after run: $(wc -l < "$private_ram/signature_cache.jsonl") entries"
echo "END $(date --iso-8601=seconds) tag=$TAG mode=$mode"
