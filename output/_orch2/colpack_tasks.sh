#!/usr/bin/env bash
# Write the column-packed re-run's task queue (ruling chenyi9 2026-09-03).
#
# The queue itself lives under output/ and is therefore NOT tracked -- raw run
# data is local-only (.gitignore) -- so it is generated from here instead, and
# the run stays reproducible from the repo alone.  backfill.sh reads the fifth
# field as the comma-separated rung list.
#
# Rungs (chenyi9): the BASELINE gets A1/A2/A3b/A4/A4b/A5/A6 -- no A3, no A3a --
# and every other sweep point gets A3b and A6 only.  N-hi (wl_N64, 128 agents)
# stays abandoned, same as the rungs5 round.
#
# Baseline tasks are written FIRST so the slots claim them before the points.
#
#   usage: colpack_tasks.sh [ROOT]
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ROOT=${1:-$REPO/output/sweep_colpack_20260903}
WLB=wl_baseline_alltoall_N16_C16_D2.json
MODELS="GPT-13B LLAMA-33B LLAMA3-8B LLAMA-65B GPT-175B LLAMA-7B"
mkdir -p "$ROOT/slurm" "$ROOT/claims_bf" "$ROOT/drain" "$ROOT/cachepool"
Q=$ROOT/tasks.txt
: > "$Q"
for m in $MODELS; do
  echo "$m baseline $WLB 8 A1,A2,A3b,A4,A4b,A5,A6" >> "$Q"
done
for m in $MODELS; do
  while read -r cfg wl k; do
    [ -z "${cfg:-}" ] && continue
    echo "$m $cfg $wl $k A3b,A6" >> "$Q"
  done <<EOF
N-lo       wl_N4.json            8
C-lo       wl_C8.json            8
C-hi       wl_C40.json           8
D-lo       wl_D1.json            8
D-hi       wl_D4.json            8
k-lo       $WLB                  2
k-hi       $WLB                  32
broadcast  wl_broadcast.json     8
reduce     wl_reduce.json        8
supervisor wl_supervisor_D4.json 8
pipeline   wl_pipeline_D4.json   8
private    wl_private.json       8
EOF
done
# Every workload named in the queue must exist, or a slot burns a claim to fail.
while read -r _m _cfg wl _k _r; do
  [ -f "$REPO/workload/sweep/$wl" ] || { echo "MISSING workload: $wl" >&2; exit 1; }
done < "$Q"
echo "wrote $Q: $(wc -l < "$Q") tasks ($(grep -c ' baseline ' "$Q") baseline x 7 rungs, rest A3b+A6)"
