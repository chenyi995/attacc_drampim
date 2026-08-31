#!/usr/bin/env bash
# Verification matrix v2.
#
# The full-event reports are ~25 GB each and the shared volume is at 98%, so
# those are hashed through a FIFO and never touch disk: json.dump streams into
# the pipe while sha256sum consumes it.  json.dump uses sort_keys=True, so the
# byte stream is order-deterministic and a hash match IS a bit-identity proof
# over every event.  The small (events=none) reports are still kept as files.
#   usage: run_matrix2.sh <label> <tree-dir> [only-evfull]
set -uo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
LABEL=$1; TREE=${2:-$REPO}; ONLY=${3:-all}
OUT=$REPO/output/_verify/$LABEL; mkdir -p "$OUT"
SNAP=$REPO/output/_verify/snapshot.jsonl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 KVPIM_CPPCORE=1
RUNGS="A1 A2 A3 A3a A3b A4 A4b A5 A6"
CASES=(
  "LLAMA3-8B 1 1  wl_pipeline_D4.json none"
  "GPT-13B   2 10 wl_N4.json          none"
  "LLAMA-7B  1 1  wl_D1.json          none"
  "LLAMA3-8B 1 1  wl_pipeline_D4.json full"
)
one() {
  local MODEL=$1 NGPU=$2 HBM=$3 WL=$4 EV=$5 A=$6
  local tag="${MODEL}_${WL%.json}_ev${EV}"
  local RD=/tmp/kvpim_${USER}_v2_${LABEL}_${tag}_${A}
  rm -rf "$RD"; mkdir -p "$RD"
  ln -sf "$REPO/ramulator2/ramulator2" "$RD/ramulator2"
  ln -sf "$REPO/ramulator2/trace_gen"  "$RD/trace_gen"
  cp "$REPO/ramulator.out" "$RD/ramulator.out" 2>/dev/null || :
  cp "$SNAP" "$RD/signature_cache.jsonl"
  local REUSE=recompute; local EX=(--epic-prefix-recompute-tokens 8)
  [ "$A" = A1 ] && { REUSE=no-reuse; EX=(); }
  local t0=$(date +%s) rc target
  if [ "$EV" = full ]; then
    local F="$RD/report.fifo"; mkfifo "$F"
    ( sha256sum < "$F" | cut -d' ' -f1 > "$OUT/${tag}_${A}.sha" ) &
    local HASHER=$!
    target="$F"
  else
    target="$OUT/${tag}_${A}.json"
  fi
  ( cd "$TREE" && ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out" \
    timeout "${VER_TIMEOUT:-28800}" python3 main.py --system dgx-attacc --model "$MODEL" \
      --ngpu "$NGPU" --num-hbm "$HBM" --workload "workload/sweep/$WL" \
      --reuse "$REUSE" "${EX[@]}" --ablation "$A" --engine dag \
      --ramulator-workers 2 --cacheblend-batch-size 8 \
      --workload-report-events "$EV" --workload-report "$target" ) \
    > "$OUT/${tag}_${A}.log" 2>&1
  rc=$?
  [ "$EV" = full ] && wait $HASHER 2>/dev/null
  echo "[$(date +%F' '%T)] $LABEL $tag $A rc=$rc $(( $(date +%s)-t0 ))s$([ $rc = 0 ] || echo '  <-- FAILED')"
  rm -rf "$RD"
}
export -f one; export REPO LABEL OUT SNAP TREE
for case in "${CASES[@]}"; do
  set -- $case
  [ "$ONLY" = evfull ] && [ "$5" != full ] && continue
  for A in $RUNGS; do one "$1" "$2" "$3" "$4" "$5" "$A" & done
done
wait
echo "[$(date +%F' '%T)] MATRIX2 DONE $LABEL (json=$(ls "$OUT"/*.json 2>/dev/null|wc -l) sha=$(ls "$OUT"/*.sha 2>/dev/null|wc -l))"
