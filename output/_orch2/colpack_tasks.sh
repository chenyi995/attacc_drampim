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
# THREE queues, not one, so the 72 two-rung point tasks run ALONGSIDE the six
# seven-rung baselines instead of queueing behind them, and so each class can
# ask for the memory it actually needs.  Measured peaks of the 2026-09-02
# nine-rung ladders on the OLD (C=32, 8464-token) workload were 461 / 386 /
# 297 / 194 / 169 / 170 GB for GPT-175B / LLAMA-65B / LLAMA-33B / GPT-13B /
# LLAMA3-8B / LLAMA-7B.  This round runs SEVEN rungs on a context of 4608, so
# the asks below are sized well under those and still generous -- a slot that
# asks 420G when it needs 40 does not get backfilled, which is what left three
# of six slots pending on the first attempt.
#
#   usage: colpack_tasks.sh [ROOT]
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ROOT=${1:-$REPO/output/sweep_colpack_20260903}
WLB=wl_baseline_alltoall_N16_C16_D2.json
MODELS="GPT-13B LLAMA-33B LLAMA3-8B LLAMA-65B GPT-175B LLAMA-7B"
mkdir -p "$ROOT/slurm" "$ROOT/claims_bf" "$ROOT/drain" "$ROOT/cachepool"
Q=$ROOT/tasks.txt
QB=$ROOT/tasks_baseline_big.txt      # GPT-175B / LLAMA-65B baselines
QS=$ROOT/tasks_baseline.txt          # the other four baselines
QP=$ROOT/tasks_points.txt            # every A3b+A6 point
: > "$Q"; : > "$QB"; : > "$QS"; : > "$QP"
for m in $MODELS; do
  line="$m baseline $WLB 8 A1,A2,A3b,A4,A4b,A5,A6"
  echo "$line" >> "$Q"
  case $m in GPT-175B|LLAMA-65B) echo "$line" >> "$QB";; *) echo "$line" >> "$QS";; esac
done
for m in $MODELS; do
  while read -r cfg wl k; do
    [ -z "${cfg:-}" ] && continue
    echo "$m $cfg $wl $k A3b,A6" >> "$Q"
    echo "$m $cfg $wl $k A3b,A6" >> "$QP"
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
echo "wrote $Q: $(wc -l < "$Q") tasks total"
echo "  $QB  $(wc -l < "$QB") big baselines   (7 rungs)"
echo "  $QS  $(wc -l < "$QS") baselines       (7 rungs)"
echo "  $QP  $(wc -l < "$QP") points          (A3b+A6)"
