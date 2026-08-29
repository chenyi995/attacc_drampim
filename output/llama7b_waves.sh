#!/usr/bin/env bash
# LLAMA-7B waves; trigger = all five L3-8B k32 CSVs exist.  128-core box.
set -uo pipefail
cd /data2/chenyi9/KV-PIM/attacc_drampim_xinyao
echo "### 7B runner waiting for 5x L3-8B k32 CSVs $(date +%F-%T)"
until [ "$(ls output/*_LLAMA3-8B_k32/dag_ladder.csv 2>/dev/null | wc -l)" -ge 5 ]; do sleep 300; done
echo "### L3-8B complete -- starting LLAMA-7B waves $(date +%F-%T)"
KVPIM_CPPCORE=1 SUITE_WORKLOADS="star_repair_r5w3k47" NUM_HBM=16 N_PAR=1 RAMU_WORKERS=14 \
  bash experiments/run_dag_suite.sh LLAMA-7B
KVPIM_CPPCORE=1 SUITE_WORKLOADS="mapreduce_sum_m8 multisource_rag_n12s96" NUM_HBM=16 N_PAR=2 RAMU_WORKERS=8 \
  bash experiments/run_dag_suite.sh LLAMA-7B
KVPIM_CPPCORE=1 SUITE_WORKLOADS="pipeline_repair_c5k50 debate_d3r5k49" NUM_HBM=16 N_PAR=2 RAMU_WORKERS=8 \
  bash experiments/run_dag_suite.sh LLAMA-7B
echo "EXP1 ALL MODELS DONE $(date +%F-%T)"
