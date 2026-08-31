#!/usr/bin/env bash
# Smoke: do the two NEW rungs (A3b head-slicing, A4b global co-read table) run
# at all on this machine?  A3/A4 are run alongside as the known-good controls.
# Cheap: tiny workload, rungs SEQUENTIAL, one model at a time.
set -uo pipefail
source /home/cw636/chenyi/attacc_drampim/output/_orch2/common.sh
ROOT=$(cat /home/cw636/chenyi/attacc_drampim/output/_orch2/CURRENT_ROOT)
OUT=$REPO/output/_smoke_a3b; mkdir -p "$OUT"
export_env
W=6
for MODEL in LLAMA-7B GPT-13B; do
  RD=$(make_ramdir "smoke_${MODEL}" "$MODEL" "$ROOT/cachepool")
  export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
  for A in A3 A3b A4 A4b; do
    t0=$(date +%s)
    (cd "$REPO" && timeout 5400 python3 main.py --system dgx-attacc --model "$MODEL" \
      --ngpu "${NGPU[$MODEL]}" --num-hbm "${NHBM[$MODEL]}" \
      --workload workload/wl_tiny.json --reuse recompute \
      --epic-prefix-recompute-tokens 8 --ablation "$A" --engine dag \
      --ramulator-workers $W --cacheblend-batch-size 8 \
      --workload-report-events none \
      --workload-report "$OUT/smoke_${MODEL}_${A}.json") \
      > "$OUT/smoke_${MODEL}_${A}.log" 2>&1
    rc=$?
    echo "[$(date +%F' '%T)] SMOKE $MODEL $A rc=$rc $(( $(date +%s)-t0 ))s node=$(hostname -s)"
  done
  rm -rf "$RD"
done
echo "[$(date +%F' '%T)] SMOKE DONE"
