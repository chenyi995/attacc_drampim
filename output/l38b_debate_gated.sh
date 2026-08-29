#!/usr/bin/env bash
# Launch the debate chain when MEASURED free RAM > 280G (chenyi9
# 2026-08-28: concurrency by real usage, not estimates).
set -uo pipefail
cd /data2/chenyi9/KV-PIM/attacc_drampim_xinyao
until [ "$(free -g | awk '/^Mem:/{print $4}')" -gt 280 ]; do sleep 120; done
echo "### free>280G -- launching debate $(date +%F-%T)"
KVPIM_CPPCORE=1 SUITE_WORKLOADS="debate_d3r5k49" NUM_HBM=16 N_PAR=1 RAMU_WORKERS=8 \
  bash experiments/run_dag_suite.sh LLAMA3-8B
