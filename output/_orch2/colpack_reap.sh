#!/usr/bin/env bash
# Release claims whose owning slot is gone.
#
# backfill.sh claims a task with an atomic `mkdir claims_bf/<tid>` and only
# writes `done` when the task finishes.  If the slot dies first -- walltime,
# preemption, OOM -- the claim directory survives with an `owner` file and no
# `done`, and because mkdir now fails for everyone the task is never retried.
# That is fine when slots run to completion, and it is exactly what stops a
# short-walltime slot from being usable, so this reaper makes short walltimes
# safe: shorter jobs backfill far better on a busy partition, and anything
# they lose is picked back up.
#
# Re-running a reaped task is cheap and correct: backfill.sh re-checks every
# rung against the disk before running it, so the finished ones are skipped.
#
#   usage: colpack_reap.sh [ROOT] [--loop SECONDS]
set -uo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ROOT=${1:-$REPO/output/sweep_colpack_20260903}
CLAIMS=$ROOT/claims_bf

reap_once() {
  local n=0
  for dir in "$CLAIMS"/*/; do
    [ -d "$dir" ] || continue
    [ -f "$dir/done" ] && continue                 # finished, keep the record
    local owner=$dir/owner
    [ -f "$owner" ] || continue
    local jid; jid=$(sed -n 's/.*jobid=\([0-9]*\).*/\1/p' "$owner")
    [ -n "$jid" ] || continue
    # Still queued or running?  Leave it alone.
    squeue -h -j "$jid" -o "%i" 2>/dev/null | grep -q . && continue
    echo "[$(date +%F' '%T)] reap $(basename "$dir") -- job $jid is gone, releasing"
    rm -rf "$dir"; n=$((n+1))
  done
  echo "[$(date +%F' '%T)] reaped $n claim(s); $(ls "$CLAIMS" 2>/dev/null | wc -l) still held"
}

if [ "${2:-}" = "--loop" ]; then
  while true; do reap_once; sleep "${3:-600}"; done
else
  reap_once
fi
