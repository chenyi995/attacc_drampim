#!/usr/bin/env bash
# Follow-on chains on the cppcore branch (chenyi9 "1", 2026-08-27 21:0x):
# ride the construction valleys next to the star chain; cpp allowance ~600G.
set -uo pipefail
cd /data2/chenyi9/KV-PIM/attacc_drampim_xinyao
for W in mapreduce_sum_m8 multisource_rag_n12s96 pipeline_repair_c5k50 debate_d3r5k49; do
  echo "### rest-chain $W $(date +%F-%T)"
  KVPIM_CPPCORE=1 SUITE_WORKLOADS="$W" NUM_HBM=16 N_PAR=1 RAMU_WORKERS=3 \
    bash experiments/run_dag_suite.sh LLAMA3-8B
done
echo "REST CHAINS DONE $(date +%F-%T)"
