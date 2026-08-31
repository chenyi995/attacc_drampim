#!/usr/bin/env bash
# Reference matrix for the before/after identity check.
#
# Every run gets its OWN ramdir seeded from the SAME frozen signature snapshot,
# so runs are independent and deterministic -- running them in parallel cannot
# change a result, it only changes how long we wait.  A cache hit returns the
# stored numbers; a miss re-simulates from identical inputs.
#   usage: run_matrix.sh <label>
set -uo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
LABEL=$1
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
one() {   # model ngpu hbm wl events rung
  local MODEL=$1 NGPU=$2 HBM=$3 WL=$4 EV=$5 A=$6
  local tag="${MODEL}_${WL%.json}_ev${EV}"
  local RD=/tmp/kvpim_${USER}_ver_${LABEL}_${tag}_${A}
  rm -rf "$RD"; mkdir -p "$RD"
  ln -sf "$REPO/ramulator2/ramulator2" "$RD/ramulator2"
  ln -sf "$REPO/ramulator2/trace_gen"  "$RD/trace_gen"
  cp "$REPO/ramulator.out" "$RD/ramulator.out" 2>/dev/null || :
  cp "$SNAP" "$RD/signature_cache.jsonl"
  local REUSE=recompute; local EX=(--epic-prefix-recompute-tokens 8)
  [ "$A" = A1 ] && { REUSE=no-reuse; EX=(); }
  local t0=$(date +%s)
  ( cd "$REPO" && ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out" \
    timeout "${VER_TIMEOUT:-21600}" python3 main.py --system dgx-attacc --model "$MODEL" \
      --ngpu "$NGPU" --num-hbm "$HBM" --workload "workload/sweep/$WL" \
      --reuse "$REUSE" "${EX[@]}" --ablation "$A" --engine dag \
      --ramulator-workers 2 --cacheblend-batch-size 8 \
      --workload-report-events "$EV" \
      --workload-report "$OUT/${tag}_${A}.json" ) > "$OUT/${tag}_${A}.log" 2>&1
  local rc=$?          # read first: a $(...) later in the line would reset $?
  echo "[$(date +%F' '%T)] $LABEL $tag $A rc=$rc $(( $(date +%s)-t0 ))s$([ $rc = 0 ] || echo '  <-- FAILED')"
  rm -rf "$RD"
}
export -f one; export REPO LABEL OUT SNAP
for case in "${CASES[@]}"; do
  set -- $case
  for A in $RUNGS; do one "$1" "$2" "$3" "$4" "$5" "$A" & done
done
wait
echo "[$(date +%F' '%T)] MATRIX DONE $LABEL ($(ls "$OUT"/*.json 2>/dev/null | wc -l)/36 reports)"
