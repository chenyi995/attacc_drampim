#!/usr/bin/env bash
# Debate chain gate v3 (2026-08-28, cy): v2's condition was right (the star
# k=2 end marker does land in exp1_cppcore_direct.log) but its watcher died
# with the session. v3 = same gate, launched survivable, plus:
#   * duplicate guard: abort if any debate main.py is already running;
#   * the gate must hold on TWO consecutive checks 120 s apart, so we do
#     not fire into a transient dip like v1 did (that OOM-killed 6 rungs).
set -uo pipefail
cd /data2/chenyi9/KV-PIM/attacc_drampim_xinyao
ok=0
while [ "$ok" -lt 2 ]; do
  if grep -aq "##### end k=2 star_repair" output/exp1_cppcore_direct.log 2>/dev/null \
     && [ "$(free -g | awk '/^Mem:/{print $4}')" -gt 280 ]; then
    ok=$((ok+1)); echo "gate ok ($ok/2) $(date +%T)"
  else
    ok=0
  fi
  [ "$ok" -lt 2 ] && sleep 120
done
if pgrep -f "main.py --system dgx-attacc.*workload_debate" >/dev/null; then
  echo "### ABORT: debate processes already running, refusing duplicate $(date +%F-%T)"; exit 1
fi
echo "### star k=2 ended and free>280G twice -- launching debate $(date +%F-%T)"
KVPIM_CPPCORE=1 SUITE_WORKLOADS="debate_d3r5k49" NUM_HBM=16 N_PAR=1 RAMU_WORKERS=8 \
  bash experiments/run_dag_suite.sh LLAMA3-8B
