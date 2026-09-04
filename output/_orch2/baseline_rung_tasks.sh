#!/usr/bin/env bash
# Write BOTH rung queues, one line per rung (ruling chenyi9 2026-09-03).
#
#   tasks_baseline_rungs.txt   6 models x 7 rungs (A1 A2 A3b A4 A4b A5 A6 --
#                              no A3, no A3a)                     =  42
#   tasks_points_rungs.txt     6 models x 12 sweep points x 4 rungs
#                              (A3b A4 A5 A6)                     = 288
#
# The fan-out is per RUNG because a rung is independent of its siblings, so
# these are 330 independently claimable units instead of 78 ladders.
# N-hi (wl_N64, 128 agents) stays abandoned, same as every round since.
#
# Heaviest first: A1 builds the largest graph and ran ~3x slower than the
# others on 2026-09-02, and the big models are the long pole, so they are
# queued ahead of everything to start them before the cheap rungs fill the
# cluster.  A slot reads the queue top-down and claims the first free line.
#
#   usage: baseline_rung_tasks.sh [ROOT]
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ROOT=${1:-$REPO/output/sweep_colpack_20260903}
WLB=wl_baseline_alltoall_N16_C16_D2.json
BIG="GPT-175B LLAMA-65B"
SMALL="LLAMA-33B GPT-13B LLAMA3-8B LLAMA-7B"
mkdir -p "$ROOT/slurm" "$ROOT/claims_rung" "$ROOT/drain" "$ROOT/cachepool"
Q=$ROOT/tasks_baseline_rungs.txt
: > "$Q"
# A1 on the big models first -- the two longest single units in the whole set.
for m in $BIG;   do echo "$m baseline $WLB 8 A1" >> "$Q"; done
for m in $SMALL; do echo "$m baseline $WLB 8 A1" >> "$Q"; done
for r in A5 A6 A4 A4b A3b A2; do
  for m in $BIG $SMALL; do echo "$m baseline $WLB 8 $r" >> "$Q"; done
done
[ -f "$REPO/workload/sweep/$WLB" ] || { echo "MISSING $WLB" >&2; exit 1; }
echo "wrote $Q: $(wc -l < "$Q") rung-tasks (6 models x 7 rungs)"

# ---- the other sweep points: A3b / A4 / A5 / A6 (ruling chenyi9) ----------
QP=$ROOT/tasks_points_rungs.txt
: > "$QP"
POINTS="N-lo:wl_N4.json:8 C-lo:wl_C8.json:8 C-hi:wl_C40.json:8
        D-lo:wl_D1.json:8 D-hi:wl_D4.json:8 k-lo:$WLB:2 k-hi:$WLB:32
        broadcast:wl_broadcast.json:8 reduce:wl_reduce.json:8
        supervisor:wl_supervisor_D4.json:8 pipeline:wl_pipeline_D4.json:8
        private:wl_private.json:8"
# Heaviest first again: the big models, and A5/A6 (prefill on the PIM, the
# largest graphs of the four) ahead of A3b/A4.
for r in A5 A6 A4 A3b; do
  for m in $BIG $SMALL; do
    for p in $POINTS; do
      cfg=${p%%:*}; rest=${p#*:}; wl=${rest%%:*}; k=${rest#*:}
      [ -f "$REPO/workload/sweep/$wl" ] || { echo "MISSING $wl" >&2; exit 1; }
      echo "$m $cfg $wl $k $r" >> "$QP"
    done
  done
done
echo "wrote $QP: $(wc -l < "$QP") rung-tasks (6 models x 12 points x 4 rungs)"
