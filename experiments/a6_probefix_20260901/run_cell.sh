#!/usr/bin/env bash
# One A6-only cell of the probe-fix verification sweep (2026-09-01).
#
#   usage: run_cell.sh <PRIO> <MODEL> <NAME> <WORKLOAD_FILE> <EPIC_K> <ROOT>
#
# Runs ONLY the A6 rung (the probe fix can move no other rung: A1-A5 use a
# fixed prefill side, A6 is the only one that calls the placement probe), with
# KVPIM_DEBUG_A6=1 so every side decision is on the record.
#
# Safe to run concurrently (2026-09-01, parallel driver):
#   * ATOMIC CLAIM -- `mkdir $OUT/.claim` is the lock.  The done-check alone is
#     not enough: it only sees a FINISHED cell, so two workers would happily
#     start the same unfinished one.  A worker that loses the race exits 0.
#   * PRIVATE ramulator.out -- `Ramulator.update_log_file` rewrites that CSV
#     whole (read DataFrame -> to_csv), so concurrent cells sharing one file
#     truncate each other.  Each cell gets its own copy, seeded from the
#     shared one.  The signature cache stays SHARED on purpose: it is
#     append-only JSONL with <=165-byte lines, i.e. atomic O_APPEND writes,
#     and sharing it is the whole point of running cells on one node.
set -uo pipefail
PRIO=$1 MODEL=$2 NAME=$3 WL=$4 K=$5 ROOT=$6
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=$REPO/.venv/bin/python

# Model tier table -- identical to experiments/run_sweep_models.sh.
case "$MODEL" in
  LLAMA-7B|LLAMA3-8B) NGPU=1;  NHBM=1  ;;
  GPT-13B|LLAMA-33B)  NGPU=2;  NHBM=10 ;;
  GPT-175B|LLAMA-65B) NGPU=8;  NHBM=40 ;;
  *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac

OUT=$ROOT/$MODEL/${NAME}_k${K}
mkdir -p "$OUT"
if [ -s "$OUT/dag_A6.json" ]; then
  echo "[$PRIO $MODEL $NAME k$K] SKIP (already done)"
  exit 0
fi
# Take the claim, REAPING one whose SLURM job is gone.  scancel SIGKILLs, so
# an EXIT trap does not always fire and a dead job leaves its claim behind;
# without reaping those cells would be skipped forever.  Only a numeric owner
# that squeue no longer knows is reaped -- anything unparseable is left alone.
take_claim() {
  mkdir "$OUT/.claim" 2>/dev/null && return 0
  local owner jid
  owner=$(cat "$OUT/.claim/owner" 2>/dev/null || echo "")
  jid=${owner%%@*}
  case "$jid" in
    ''|*[!0-9]*) return 1 ;;                       # unparseable: respect it
  esac
  [ -n "$(squeue -h -j "$jid" -o %i 2>/dev/null)" ] && return 1   # still alive
  echo "[$PRIO $MODEL $NAME k$K] REAP stale claim of dead job $jid"
  rm -f "$OUT/.claim/owner"
  rmdir "$OUT/.claim" 2>/dev/null
  mkdir "$OUT/.claim" 2>/dev/null            # still atomic: one reaper wins
}
if ! take_claim; then
  echo "[$PRIO $MODEL $NAME k$K] SKIP (claimed by $(cat "$OUT/.claim/owner" 2>/dev/null || echo '?'))"
  exit 0
fi
echo "${SLURM_JOB_ID:-$$}@$(hostname -s) $(date +%F' '%T)" > "$OUT/.claim/owner"
# Release the claim on ANY exit: a finished cell is protected by its
# dag_A6.json, a failed one must stay retryable.
trap 'rm -f "$OUT/.claim/owner"; rmdir "$OUT/.claim" 2>/dev/null' EXIT

# Private CSV shape cache for this cell (see header).
CELLLOG=$OUT/ramulator.out
if [ ! -s "$CELLLOG" ] && [ -s "${ATTACC_RAMULATOR_LOG:-}" ]; then
  cp "$ATTACC_RAMULATOR_LOG" "$CELLLOG" 2>/dev/null || :
fi

echo "=== $(date +%F' '%T) [$PRIO] $MODEL $NAME k=$K ngpu=$NGPU hbm=$NHBM -> $OUT"
START=$(date +%s)
cd "$REPO"
KVPIM_CPPCORE=1 KVPIM_DEBUG_A6=1 \
ATTACC_RAMULATOR_LOG="$CELLLOG" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
"$PY" main.py \
  --system dgx-attacc --model "$MODEL" \
  --workload "$REPO/workload/sweep/$WL" \
  --reuse recompute --epic-prefix-recompute-tokens "$K" \
  --ablation A6 --engine dag \
  --workload-report "$OUT/dag_A6.json" \
  --workload-report-events none \
  --cacheblend-batch-size 8 \
  --num-hbm "$NHBM" --ngpu "$NGPU" \
  --ramulator-workers "${RAMU_WORKERS:-8}" \
  > "$OUT/dag_A6.log" 2>&1
RC=$?
END=$(date +%s)
# Keep the probe records as their own file; the log itself stays complete.
grep -a '^A6_PROBE ' "$OUT/dag_A6.log" > "$OUT/a6_probe.jsonl" 2>/dev/null || :
if [ $RC -ne 0 ]; then
  echo "[$PRIO $MODEL $NAME k$K] FAILED rc=$RC after $((END-START))s -- see $OUT/dag_A6.log"
  exit $RC
fi
echo "[$PRIO $MODEL $NAME k$K] OK in $((END-START))s"
grep -ah REPORT_SUMMARY "$OUT/dag_A6.log" | sed "s|^|[$PRIO $MODEL $NAME k$K] |"
