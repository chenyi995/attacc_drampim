#!/usr/bin/env bash
# Run the parametric sweep: 14 configs x 7 rungs (A1-A6) = 98 runs.
# Each config = one run_dag_ladder call (its 7 rungs run in parallel inside);
# configs run sequentially to bound memory (the N64/D4 configs are the heavy
# ones). k (recompute) is per-config via EPIC_K. Model LLAMA3-8B, num-hbm 16.
#   usage: bash experiments/run_sweep.sh
set -uo pipefail
cd /data2/chenyi9/KV-PIM/attacc_drampim_xinyao
WD=workload/sweep
TS=$(date +%Y%m%d-%H%M%S)
ROOT=output/sweep_${TS}
mkdir -p "$ROOT"
echo "sweep -> $ROOT   $(date)"

# name  workload-file  k
CFG=(
  "baseline   wl_baseline_alltoall_N16_C32_D2.json  8"
  "N-lo       wl_N4.json                            8"
  "N-hi       wl_N64.json                           8"
  "C-lo       wl_C16.json                           8"
  "C-hi       wl_C64.json                           8"
  "D-lo       wl_D1.json                            8"
  "D-hi       wl_D4.json                            8"
  "k-lo       wl_baseline_alltoall_N16_C32_D2.json  2"
  "k-hi       wl_baseline_alltoall_N16_C32_D2.json  32"
  "broadcast  wl_broadcast.json                     8"
  "reduce     wl_reduce.json                        8"
  "supervisor wl_supervisor_D4.json                 8"
  "pipeline   wl_pipeline_D4.json                   8"
  "private    wl_private.json                       8"
)

for entry in "${CFG[@]}"; do
  read -r name wl k <<< "$entry"
  OUT="$ROOT/${name}_k${k}"
  mkdir -p "$OUT"
  echo "##### $(date +%T) start config ${name} (k=${k})  ->  ${OUT}"
  KVPIM_CPPCORE=1 NUM_HBM=16 RAMU_WORKERS=8 EPIC_K=${k} \
    bash experiments/run_dag_ladder.sh "$WD/$wl" LLAMA3-8B "$OUT" \
    2>&1 | grep -aE "REPORT_SUMMARY|CSV:|FAILED|Killed|launch" | sed "s|^|[${name} k${k}] |"
  echo "##### $(date +%T) end   config ${name}"
done
echo "SWEEP DONE $(date +%T)  ->  $ROOT"
