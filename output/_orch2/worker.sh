#!/usr/bin/env bash
# Phase 2 worker slot: pull (model, config) tasks from a shared NFS queue and
# run each as one run_dag_ladder call (9 rungs in parallel).  Claiming is an
# atomic mkdir, so any number of slots on any number of nodes can share the
# queue without a lock server.
#
# Governor contract (added 2026-08-30):
#   * one slot per Slurm job now, so the governor can add/remove capacity at
#     27-core granularity instead of 54;
#   * $ROOT/drain/<jobid> makes this slot exit AFTER its current task finishes
#     -- graceful give-back, no work lost;
#   * the claim's owner file records the Slurm job id so the governor can tell
#     an abandoned claim (owner job gone, <9 jsons) from a live one.
#   usage: worker.sh <root> <W> <slot-id> [queue-file]
# The queue is split by model class: a GPT-175B task holds ~10x the RAM of
# a LLAMA-7B one, so they get separate slot sizes rather than one worst-case size.
set -uo pipefail
source /home/cw636/chenyi/attacc_drampim/output/_orch2/common.sh
ROOT=$1; W=$2; SLOT=$3; QUEUE=${4:-$ROOT/tasks.txt}
POOL=$ROOT/cachepool; CLAIMS=$ROOT/claims
JID=${SLURM_JOB_ID:-nojob}
DRAIN=$ROOT/drain/$JID
mkdir -p "$POOL" "$CLAIMS" "$ROOT/drain"
export_env
H=$(hostname -s)
# Repeat passes over the queue.  A claim released by the governor (its owner
# slot died mid-task) sits BEHIND the read offset of every worker already
# running, so a single pass would leave it stranded.  Loop until a whole pass
# claims nothing, then exit.
while true; do
CLAIMED_THIS_PASS=0
while read -r model cfg wl k <&3; do
  [ -z "${model:-}" ] && continue
  if [ -e "$DRAIN" ]; then
    echo "[$(date +%F' '%T)] DRAIN requested -- slot$SLOT on $H exiting (job $JID)"
    break
  fi
  tid="${model}__${cfg}_k${k}"
  mkdir "$CLAIMS/$tid" 2>/dev/null || continue      # someone else owns it
  echo "$H slot$SLOT jobid=$JID $(date +%F' '%T)" > "$CLAIMS/$tid/owner"
  CLAIMED_THIS_PASS=$((CLAIMED_THIS_PASS+1))
  OUT="$ROOT/${model}/${cfg}_k${k}"; mkdir -p "$OUT"
  RD=$(make_ramdir "run_${model}_${cfg}_k${k}" "$model" "$POOL")
  seed=$(wc -l < "$RD/signature_cache.jsonl" 2>/dev/null || echo 0)
  export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
  publish_loop "$RD" "$POOL" "$model" "run_${cfg}_k${k}" & PUB=$!
  echo "[$(date +%F' '%T)] START $model $cfg k=$k node=$H slot$SLOT job=$JID W=$W seed=$seed"
  t0=$(date +%s)
  NUM_HBM=${NHBM[$model]} NGPU=${NGPU[$model]} RAMU_WORKERS=$W EPIC_K=$k \
    bash "$REPO/experiments/run_dag_ladder.sh" "$REPO/workload/sweep/$wl" \
         "$model" "$OUT" > "$OUT/ladder.log" 2>&1 < /dev/null
  rc=$?
  kill $PUB 2>/dev/null
  publish_final "$RD" "$POOL" "$model" "run_${cfg}_k${k}"
  nj=$(ls "$OUT"/dag_A*.json 2>/dev/null | wc -l)
  echo "[$(date +%F' '%T)] END   $model $cfg k=$k rc=$rc jsons=$nj/9 $(( $(date +%s)-t0 ))s node=$H"
  echo "rc=$rc jsons=$nj/9 $(date +%F' '%T)" > "$CLAIMS/$tid/done"
  [ "$rc" = 0 ] || echo "$model $cfg $wl $k rc=$rc" >> "$ROOT/failed_tasks.txt"
  rm -rf "$RD"
done 3< "$QUEUE"
  [ "$CLAIMED_THIS_PASS" = 0 ] && break
  [ -e "$DRAIN" ] && break
  echo "[$(date +%F' '%T)] pass complete on $H slot$SLOT: claimed $CLAIMED_THIS_PASS, re-scanning for released claims"
done
echo "[$(date +%F' '%T)] WORKER DONE node=$H slot$SLOT job=$JID"
