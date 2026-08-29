#!/usr/bin/env bash
# Debate chain gate v2 (2026-08-28): wait for the star k=2 batch to END
# (its ~180G falls back) AND measured free > 280G before launching --
# v1 checked instantaneous free and fired into the four-chain build peak.
set -uo pipefail
cd /data2/chenyi9/KV-PIM/attacc_drampim_xinyao
until grep -aq "##### end k=2 star_repair" output/exp1_cppcore_direct.log 2>/dev/null && \
      [ "$(free -g | awk '/^Mem:/{print $4}')" -gt 280 ]; do sleep 120; done
echo "### star k=2 ended and free>280G -- launching debate $(date +%F-%T)"
KVPIM_CPPCORE=1 SUITE_WORKLOADS="debate_d3r5k49" NUM_HBM=16 N_PAR=1 RAMU_WORKERS=8 \
  bash experiments/run_dag_suite.sh LLAMA3-8B
