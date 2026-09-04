#!/usr/bin/env bash
# Write the baseline queue with ONE LINE PER RUNG (ruling chenyi9 2026-09-03).
#
# Six models x seven rungs (A1 A2 A3b A4 A4b A5 A6 -- no A3, no A3a) = 42
# independently claimable units, so the fan-out is 42 wide instead of 6.
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
