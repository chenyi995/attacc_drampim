#!/usr/bin/env bash
# Multi-model parametric sweep (chenyi9 2026-08-29): experiment1.
#
#   6 models x 14 workload-configs x 9 rungs (A1 A2 A3 A3a A3b A4 A4b A5 A6)
#   = 756 runs.
#
# Model tiers (system config -> heads_per_hbm, i.e. how many KV heads share one
# HBM's sixteen channels; > 1 makes the head-slice vs placement-table ablation
# A3b/A4/A4b visible):
#   small :  LLAMA-7B, LLAMA3-8B   1 GPU  /  1 HBM   (heads_per_hbm 32 / 8)
#   medium:  GPT-13B , LLAMA-33B   2 GPU  / 10 HBM   (heads_per_hbm  4 / ~5)
#   large :  GPT-175B, LLAMA-65B   8 GPU  / 40 HBM   (heads_per_hbm  3 / 2)
#
# The 14 workload-configs are model-agnostic (they set agent topology/tokens,
# not the model), so the SAME workload/sweep/*.json serve every model.  Sized
# for a ~300-core / ~3-TB machine: models run sequentially; inside a model the
# 14 configs run sequentially; inside a config the 9 rungs run in parallel
# (run_dag_ladder.sh).  Bump the per-model parallelism only if RAM allows --
# the large-model (8-GPU) runs are the heavy ones.
#
#   usage: bash experiments/run_sweep_models.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, any machine
WD=workload/sweep
TS=$(date +%Y%m%d-%H%M%S)
ROOT=output/sweep_models_${TS}
mkdir -p "$ROOT"
echo "multi-model sweep -> $ROOT   $(date)"

# tier  model      ngpu  num_hbm
MODELS=(
  "small   LLAMA-7B    1   1"
  "small   LLAMA3-8B   1   1"
  "medium  GPT-13B     2  10"
  "medium  LLAMA-33B   2  10"
  "large   GPT-175B    8  40"
  "large   LLAMA-65B   8  40"
)

# name  workload-file  k   (same 14 configs as run_sweep.sh)
CFG=(
  "baseline   wl_baseline_alltoall_N16_C16_D2.json  8"
  "N-lo       wl_N4.json                            8"
  "N-hi       wl_N64.json                           8"
  "C-lo       wl_C8.json                            8"
  "C-hi       wl_C40.json                           8"
  "D-lo       wl_D1.json                            8"
  "D-hi       wl_D4.json                            8"
  "k-lo       wl_baseline_alltoall_N16_C16_D2.json  2"
  "k-hi       wl_baseline_alltoall_N16_C16_D2.json  32"
  "broadcast  wl_broadcast.json                     8"
  "reduce     wl_reduce.json                        8"
  "supervisor wl_supervisor_D4.json                 8"
  "pipeline   wl_pipeline_D4.json                   8"
  "private    wl_private.json                       8"
)

for m in "${MODELS[@]}"; do
  read -r tier model ngpu hbm <<< "$m"
  echo "########## $(date +%T) model ${model} (tier ${tier}, ngpu ${ngpu}, hbm ${hbm})"
  for entry in "${CFG[@]}"; do
    read -r name wl k <<< "$entry"
    OUT="$ROOT/${model}/${name}_k${k}"
    mkdir -p "$OUT"
    echo "##### $(date +%T) ${model} ${name} (k=${k})  ->  ${OUT}"
    KVPIM_CPPCORE=1 NUM_HBM=${hbm} NGPU=${ngpu} RAMU_WORKERS=8 EPIC_K=${k} \
      bash experiments/run_dag_ladder.sh "$WD/$wl" "$model" "$OUT" \
      2>&1 | grep -aE "REPORT_SUMMARY|CSV:|FAILED|Killed|launch" \
      | sed "s|^|[${model} ${name} k${k}] |"
  done
done
echo "MODELS SWEEP DONE $(date +%T)  ->  $ROOT"
