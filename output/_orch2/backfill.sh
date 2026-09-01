#!/usr/bin/env bash
# Backfill slot: re-run ONLY the rungs a task lost, not the whole ladder.
#
# Ten tasks came out of phase 2 with holes -- 37 rungs, every one of them lost
# to the node running out of disk or memory, none to an engine defect.  Seven of
# the ten are missing A1, which builds the largest graph and runs ~3x slower
# than the other rungs, so re-running complete ladders would cost far more than
# the holes are worth: three of the ten are missing one rung each.
#
# Differences from worker.sh, all deliberate:
#   * the queue carries a fifth field, the comma-separated rungs to run, and it
#     is passed to run_dag_ladder.sh as RUNGS;
#   * claims live in $ROOT/claims_bf so the phase-2 claim of the same task --
#     which still carries its done and damaged markers -- is left intact as the
#     record of what went wrong;
#   * the Slurm job is named bf_* rather than slot*, which is what governor.py
#     matches on, so the governor neither drains nor counts these.  That means
#     THEIR MEMORY IS NOT UNDER ITS CONTROL: keep the slot count small and sized
#     by hand.
#
#   usage: backfill.sh <root> <ramulator-workers> <slot-id> [queue-file]
set -uo pipefail
source /home/cw636/chenyi/attacc_drampim/output/_orch2/common.sh
ROOT=$1; W=$2; SLOT=$3; QUEUE=${4:-$ROOT/tasks_backfill.txt}
POOL=$ROOT/cachepool; CLAIMS=$ROOT/claims_bf
JID=${SLURM_JOB_ID:-nojob}
DRAIN=$ROOT/drain/$JID
mkdir -p "$POOL" "$CLAIMS" "$ROOT/drain"
export_env
H=$(hostname -s)

while true; do
CLAIMED_THIS_PASS=0
while read -r model cfg wl k rungs <&3; do
  [ -z "${model:-}" ] && continue
  [ -z "${rungs:-}" ] && continue
  if [ -e "$DRAIN" ]; then
    echo "[$(date +%F' '%T)] DRAIN requested -- bf$SLOT on $H exiting (job $JID)"
    break
  fi
  tid="${model}__${cfg}_k${k}"
  mkdir "$CLAIMS/$tid" 2>/dev/null || continue      # someone else owns it
  echo "$H bf$SLOT jobid=$JID $(date +%F' '%T)" > "$CLAIMS/$tid/owner"
  CLAIMED_THIS_PASS=$((CLAIMED_THIS_PASS+1))
  OUT="$ROOT/${model}/${cfg}_k${k}"; mkdir -p "$OUT"
  want=${rungs//,/ }
  # Re-check against the disk rather than trusting the queue: the file was
  # written once, and a rung may have been filled in since.
  todo=""
  for a in $want; do [ -f "$OUT/dag_$a.json" ] || todo="$todo $a"; done
  todo=${todo# }
  if [ -z "$todo" ]; then
    echo "[$(date +%F' '%T)] SKIP  $model $cfg k=$k -- all requested rungs already present"
    echo "nothing to do $(date +%F' '%T)" > "$CLAIMS/$tid/done"
    continue
  fi
  RD=$(make_ramdir "bf_${model}_${cfg}_k${k}" "$model" "$POOL")
  export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
  before=$(ls "$OUT"/dag_A*.json 2>/dev/null | wc -l)
  echo "[$(date +%F' '%T)] START $model $cfg k=$k node=$H bf$SLOT job=$JID have=$before/9 rungs='$todo'"
  t0=$(date +%s)
  RUNGS="$todo" NUM_HBM=${NHBM[$model]} NGPU=${NGPU[$model]} RAMU_WORKERS=$W EPIC_K=$k \
    bash "$REPO/experiments/run_dag_ladder.sh" "$REPO/workload/sweep/$wl" \
         "$model" "$OUT" > "$OUT/ladder_backfill.log" 2>&1 < /dev/null
  rc=$?
  nj=$(ls "$OUT"/dag_A*.json 2>/dev/null | wc -l)
  echo "[$(date +%F' '%T)] END   $model $cfg k=$k rc=$rc jsons=$nj/9 (was $before) $(( $(date +%s)-t0 ))s node=$H"
  echo "rc=$rc jsons=$nj/9 $(date +%F' '%T)" > "$CLAIMS/$tid/done"
  [ "$nj" = 9 ] && rm -f "$ROOT/claims/$tid/damaged"   # hole closed
  [ "$rc" = 0 ] || echo "$model $cfg $wl $k rungs='$todo' rc=$rc" >> "$ROOT/failed_backfill.txt"
  rm -rf "$RD"
done 3< "$QUEUE"
[ "$CLAIMED_THIS_PASS" = 0 ] && break
[ -e "$DRAIN" ] && break
done
echo "[$(date +%F' '%T)] BACKFILL WORKER DONE node=$H bf$SLOT job=$JID"
