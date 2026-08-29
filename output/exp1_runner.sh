#!/usr/bin/env bash
# Experiment-1 re-run on the C++ core branch (chenyi9 2026-08-28): waits for
# the frozen 822 orchestrator to finish (560G discipline -- never run both),
# then runs the SAME star chain (LLAMA3-8B @16HBM PC, k in {2,8,32}) here
# with KVPIM_CPPCORE=1, using the suite's chain logic.
set -uo pipefail
REPO=/data2/chenyi9/KV-PIM/attacc_drampim_xinyao
DONE_LOG=/data2/chenyi9/KV-PIM/attacc_drampim_822/output/orchestrator_v2_20260828.log
cd "$REPO"
echo "### exp1 runner waiting for 822 ORCHESTRATION DONE $(date +%F-%T)"
until grep -aq "ORCHESTRATION DONE" "$DONE_LOG" 2>/dev/null; do sleep 300; done
echo "### 822 done -- starting experiment-1 on cppcore branch $(date +%F-%T)"
# Re-sync the signature cache: the branch snapshot predates the 822 star
# sims; taking the finished orchestrator's cache makes exp1's warm phase
# all-hits and the comparison a clean build-time A/B.
cp /data2/chenyi9/KV-PIM/attacc_drampim_822/ramulator2/signature_cache_v2_headhbm.jsonl \
   "$REPO/ramulator2/signature_cache_v2_headhbm.jsonl"
echo "### cache re-synced: $(wc -l < "$REPO/ramulator2/signature_cache_v2_headhbm.jsonl") lines"
KVPIM_CPPCORE=1 SUITE_WORKLOADS="star_repair_r5w3k47" NUM_HBM=16 N_PAR=1 RAMU_WORKERS=12 \
  bash experiments/run_dag_suite.sh LLAMA3-8B
echo "EXP1 CPPCORE DONE $(date +%F-%T)"
