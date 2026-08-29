#!/usr/bin/env bash
# Fill the missing mapreduce k8/k32 ladders (chenyi9 2026-08-29): the mapreduce
# chain ran only k2 (bare run_dag_ladder, no k-sweep). Reuse the k2 A1
# (no-reuse is identical across k), NO_WARM, same env as the suite. k8 then
# k32 sequentially (each k runs its 7 rungs in parallel inside run_dag_ladder);
# avoids the K_PAR=2 memory peak.
set -uo pipefail
cd /data2/chenyi9/KV-PIM/attacc_drampim_xinyao
A1=output/20260828-125509_workload_mapreduce_sum_m8_LLAMA3-8B_k2/dag_A1.json
[ -f "$A1" ] || { echo "ABORT: k2 A1 missing $A1"; exit 1; }
for K in 8 32; do
  if pgrep -f "workload_mapreduce_sum_m8.json.*recompute-tokens ${K} " >/dev/null; then
    echo "ABORT: mapreduce k${K} already running, refusing duplicate $(date +%F-%T)"; continue
  fi
  OUT=output/$(date +%Y%m%d-%H%M%S)_workload_mapreduce_sum_m8_LLAMA3-8B_k${K}
  echo "##### mapreduce k=${K} start $(date +%F-%T) -> $OUT"
  KVPIM_CPPCORE=1 NUM_HBM=16 RAMU_WORKERS=12 EPIC_K=${K} NO_WARM=1 SKIP_A1=1 A1_JSON="$A1" \
    bash experiments/run_dag_ladder.sh workload/workload_mapreduce_sum_m8.json LLAMA3-8B "$OUT" \
    2>&1 | grep -E "launch|REPORT_SUMMARY|CSV:|FAILED|Killed" | sed "s|^|[mapreduce k${K}] |"
  echo "##### mapreduce k=${K} end $(date +%F-%T)"
done
echo "MAPREDUCE k8/k32 DONE $(date +%F-%T)"
