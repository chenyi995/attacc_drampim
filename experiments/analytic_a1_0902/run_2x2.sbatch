#!/usr/bin/env bash
# One cell of the A1 2x2: {event DAG, DAG-free enumerator} x {Ramulator, analytic}.
#
# Every cell starts from an EMPTY signature cache, so the wall clock is the
# cost of producing a number that does not exist yet.  Every cell is given the
# same model, tensor parallel, num_hbm and power constraint -- the two entry
# points disagreed on FIVE of those by default before this script existed.
#
# What is comparable across all four cells:
#   * PIM command counts (mac/sfm/mvgb/mvsb/wrgb)
#   * PIM scan ENERGY, in nJ
#   * unoverlapped PIM TIME (DAG: pim_time_s_unoverlapped minus the
#     bandwidth-priced dram_store_master events; enumerator:
#     pim_time_s_unordered)
#   * wall clock, split into work-list construction and PIM pricing
# What is NOT: makespan.  The enumerator has no scheduler; comparing its
# unscheduled work against a DAG makespan would be meaningless.
#
# Required exports: TAG, ENGINE={dag,enum}, PRICER={ramulator,analytic},
#                   WORKLOAD (repo-relative).  Optional: MODEL, TP, NUM_HBM.
#SBATCH --job-name=a1-2x2
#SBATCH --partition=athena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/analytic_a1_0902/slurm_%x_%j.log
#SBATCH --error=/zpool-00/home/xw338/attacc/attacc_drampim/experiments/analytic_a1_0902/slurm_%x_%j.log

set -euo pipefail
repo=/zpool-00/home/xw338/attacc/attacc_drampim
out="$repo/experiments/analytic_a1_0902/matrix2x2/$TAG"
engine="${ENGINE:?}"; pricer="${PRICER:?}"; workload="${WORKLOAD:?}"
model="${MODEL:-LLAMA-7B}"; tp="${TP:-8}"; num_hbm="${NUM_HBM:-5}"
analytic_model="${ANALYTIC_MODEL:-experiments/analytic_a1_0902/timing_models.json}"
mkdir -p "$out"; cd "$repo"

private_ram="$out/ramulator2"
rm -rf "$private_ram"; mkdir -p "$private_ram"
ln -s "$repo/ramulator2/ramulator2" "$private_ram/ramulator2"
ln -s "$repo/ramulator2/trace_gen" "$private_ram/trace_gen"
: > "$private_ram/signature_cache.jsonl"          # COLD

echo "START $(date --iso-8601=seconds) tag=$TAG engine=$engine pricer=$pricer model=$model tp=$tp workload=$workload"
export ATTACC_RAMULATOR_DIR="$private_ram"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [[ "$engine" == dag ]]; then
  args=(--system dgx-attacc --model "$model" --workload "$workload"
        --reuse no-reuse --ablation A1 --engine dag --powerlimit --no-warm
        --num-hbm "$num_hbm" --ramulator-workers 1 --workload-report-events none
        --workload-report "$out/report.json")
  if [[ "$pricer" == analytic ]]; then
    args+=(--analytic-pim-model "$analytic_model")
  fi
  # Record the PIM invocation multiset so the enumerator cells can be checked
  # against the very same run rather than against a different one.
  export ATTACC_RECORD_PIM_SIGNATURES="$out/pim_signatures.json"
  export ATTACC_RAMULATOR_LOG="$out/ramulator.out"
  /usr/bin/time -f 'wall_s=%e maxrss_kb=%M' python3 main.py "${args[@]}"
else
  /usr/bin/time -f 'wall_s=%e maxrss_kb=%M' python3 \
    experiments/analytic_a1_0902/run_a1_dag_free_inputs.py \
    --workload "$workload" --model "$model" --tensor-parallel "$tp" \
    --num-hbm "$num_hbm" --power-constraint --mode "$pricer" \
    --ramulator-workers 1 --timing-model "$analytic_model" \
    --output "$out/report.json"
fi
echo "cache after run: $(wc -l < "$private_ram/signature_cache.jsonl") entries"
echo "END $(date --iso-8601=seconds) tag=$TAG"
