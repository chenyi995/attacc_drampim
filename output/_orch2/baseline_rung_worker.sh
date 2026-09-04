#!/usr/bin/env bash
# One slot, claiming ONE RUNG at a time.
#
# backfill.sh claims a whole (model, config) task, so a baseline task is one
# model's seven rungs and only SIX slots can ever be busy on the six baselines
# -- the rungs inside a slot are parallel, but they are pinned to one node and
# one allocation.  A rung is completely independent of its siblings (separate
# process, separate report file, shared only by the read-only signature cache),
# so the natural unit of work is the RUNG, not the ladder.  Forty-two
# independently claimable rungs can spread over every node the cluster will
# give us.
#
# Claims live in claims_rung/<model>__<cfg>_k<k>__<rung> and are taken with an
# atomic mkdir, same discipline as backfill.sh.  A rung already on disk is
# skipped without claiming, so a stopped run resumes by just resubmitting.
# Stale claims (slot killed by walltime) are released by colpack_reap.sh.
#
#   usage: baseline_rung_worker.sh <root> <ramulator-workers> <slot-id> <queue>
set -uo pipefail
source /home/cw636/chenyi/attacc_drampim/output/_orch2/common.sh
ROOT=$1; W=$2; SLOT=$3; QUEUE=$4
POOL=$ROOT/cachepool; CLAIMS=$ROOT/claims_rung
JID=${SLURM_JOB_ID:-nojob}
DRAIN=$ROOT/drain/$JID
mkdir -p "$POOL" "$CLAIMS" "$ROOT/drain"
export_env
H=$(hostname -s)

while true; do
CLAIMED=0
while read -r model cfg wl k rung <&3; do
  [ -z "${rung:-}" ] && continue
  [ -e "$DRAIN" ] && { echo "[$(date +%F' '%T)] DRAIN -- rw$SLOT on $H exiting"; break; }
  OUT="$ROOT/${model}/${cfg}_k${k}"
  [ -f "$OUT/dag_${rung}.json" ] && continue          # already done
  tid="${model}__${cfg}_k${k}__${rung}"
  mkdir "$CLAIMS/$tid" 2>/dev/null || continue        # someone else owns it
  echo "$H rw$SLOT jobid=$JID $(date +%F' '%T)" > "$CLAIMS/$tid/owner"
  CLAIMED=$((CLAIMED+1))
  mkdir -p "$OUT"
  RD=$(make_ramdir "rw_${tid}" "$model" "$POOL")
  export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
  echo "[$(date +%F' '%T)] START $model $cfg k=$k rung=$rung node=$H rw$SLOT job=$JID"
  t0=$(date +%s)
  SKIP_COLLECT=1 RUNGS="$rung" NUM_HBM=${NHBM[$model]} NGPU=${NGPU[$model]} \
    RAMU_WORKERS=$W EPIC_K=$k \
    bash "$REPO/experiments/run_dag_ladder.sh" "$REPO/workload/sweep/$wl" \
         "$model" "$OUT" > "$OUT/rung_${rung}.log" 2>&1 < /dev/null
  rc=$?
  ok=$([ -f "$OUT/dag_${rung}.json" ] && echo yes || echo NO)
  echo "[$(date +%F' '%T)] END   $model $cfg k=$k rung=$rung rc=$rc json=$ok $(( $(date +%s)-t0 ))s node=$H"
  echo "rc=$rc json=$ok $(date +%F' '%T)" > "$CLAIMS/$tid/done"
  [ "$rc" = 0 ] || echo "$model $cfg $wl $k $rung rc=$rc" >> "$ROOT/failed_rungs.txt"
  rm -rf "$RD"
done 3< "$QUEUE"
[ "$CLAIMED" = 0 ] && break
[ -e "$DRAIN" ] && break
done
echo "[$(date +%F' '%T)] RUNG WORKER DONE node=$H rw$SLOT job=$JID"
