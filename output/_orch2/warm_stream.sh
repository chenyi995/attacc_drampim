#!/usr/bin/env bash
# Phase 1 -- pre-warm one model's Ramulator signatures (experiments/warm_cache.sh
# semantics: --no-warm, 9 rungs SEQUENTIAL so a shape has a single writer).
#   usage: warm_stream.sh <root> <model> <W> <tag> <workload.json> [<workload.json>...]
set -uo pipefail
source /home/cw636/chenyi/attacc_drampim/output/_orch2/common.sh
ROOT=$1; MODEL=$2; W=$3; TAG=$4; shift 4
POOL=$ROOT/cachepool; mkdir -p "$POOL"
export_env
RD=$(make_ramdir "warm_${MODEL}_${TAG}" "$MODEL" "$POOL")
export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
publish_loop "$RD" "$POOL" "$MODEL" "warm_$TAG" & PUB=$!
cd "$REPO"
for WL in "$@"; do
  for A in $RUNGS9; do
    REUSE=recompute; EX=(--epic-prefix-recompute-tokens 8)
    [ "$A" = A1 ] && { REUSE=no-reuse; EX=(); }
    t0=$(date +%s)
    python3 main.py --system dgx-attacc --model "$MODEL" --ngpu "${NGPU[$MODEL]}" \
      --workload "workload/sweep/$WL" --reuse "$REUSE" "${EX[@]}" \
      --ablation "$A" --engine dag --num-hbm "${NHBM[$MODEL]}" \
      --ramulator-workers "$W" --cacheblend-batch-size 8 \
      --workload-report-events none --no-warm \
      --workload-report "$RD/warm_${A}.json" > "$RD/warm_${A}.log" 2>&1
    rc=$?
    sigs=$(wc -l < "$RD/signature_cache.jsonl" 2>/dev/null || echo 0)
    echo "[$(date +%F' '%T)] warm $MODEL $WL $A rc=$rc $(( $(date +%s)-t0 ))s sigs=$sigs node=$(hostname -s)"
  done
  publish_final "$RD" "$POOL" "$MODEL" "warm_$TAG"
  echo "[$(date +%F' '%T)] WARMED $MODEL $WL (tag $TAG)"
done
kill $PUB 2>/dev/null
publish_final "$RD" "$POOL" "$MODEL" "warm_$TAG"
echo "[$(date +%F' '%T)] WARM STREAM DONE $MODEL tag=$TAG node=$(hostname -s)"
rm -rf "$RD"
