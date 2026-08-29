#!/usr/bin/env bash
# LLAMA-7B full suite on the cppcore branch (chenyi9 2026-08-27 21:1x:
# experiment 1 = ALL models x five workloads on cppcore).  Starts after
# both L3-8B runners finish; wave plan mirrors the 822 v2 orchestrator.
set -uo pipefail
cd /data2/chenyi9/KV-PIM/attacc_drampim_xinyao
STAR_LOG=output/exp1_cppcore_direct.log
REST_LOG=output/l38b_rest_chains.log
echo "### 7B runner waiting for L3-8B completion $(date +%F-%T)"
until grep -aq "SUITE DONE" "$STAR_LOG" 2>/dev/null && \
      grep -aq "REST CHAINS DONE" "$REST_LOG" 2>/dev/null; do sleep 300; done
echo "### L3-8B complete -- starting LLAMA-7B waves $(date +%F-%T)"
echo "### 7B star solo $(date +%F-%T)"
KVPIM_CPPCORE=1 SUITE_WORKLOADS="star_repair_r5w3k47" NUM_HBM=16 N_PAR=1 RAMU_WORKERS=12 \
  bash experiments/run_dag_suite.sh LLAMA-7B
echo "### 7B mapreduce+multisource $(date +%F-%T)"
KVPIM_CPPCORE=1 SUITE_WORKLOADS="mapreduce_sum_m8 multisource_rag_n12s96" NUM_HBM=16 N_PAR=2 RAMU_WORKERS=6 \
  bash experiments/run_dag_suite.sh LLAMA-7B
echo "### 7B pipeline+debate $(date +%F-%T)"
KVPIM_CPPCORE=1 SUITE_WORKLOADS="pipeline_repair_c5k50 debate_d3r5k49" NUM_HBM=16 N_PAR=2 RAMU_WORKERS=6 \
  bash experiments/run_dag_suite.sh LLAMA-7B
echo "EXP1 ALL MODELS DONE $(date +%F-%T)"
