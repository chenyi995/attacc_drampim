#!/usr/bin/env bash
# Real-path smoke: the exact phase-2 call (run_dag_ladder.sh, 9 rungs parallel)
# on the smallest sweep workload.  Answers the handoff's #1 open question --
# do the new A3b / A4b rungs run at all on this machine -- and measures one
# task's wall time and peak RSS so the slot budget can be sized from data.
set -uo pipefail
source /home/cw636/chenyi/attacc_drampim/output/_orch2/common.sh
ROOT=$(cat /home/cw636/chenyi/attacc_drampim/output/_orch2/CURRENT_ROOT)
MODEL=${MODEL:-GPT-13B}; WL=${WL:-wl_pipeline_D4.json}
OUT=$REPO/output/_smoke9/${MODEL}_${WL%.json}; mkdir -p "$OUT"
export_env
RD=$(make_ramdir "smoke9_${MODEL}" "$MODEL" "$ROOT/cachepool")
export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
( while true; do
    rss=$(ps -u "$USER" -o rss= 2>/dev/null | awk '{s+=$1} END{printf "%d", s/1048576}')
    nproc_mine=$(pgrep -u "$USER" -c -f "main.py|ramulator2" 2>/dev/null || echo 0)
    echo "$(date +%F' '%T) rss=${rss}G procs=${nproc_mine}"; sleep 30
  done ) > "$OUT/usage.log" 2>&1 & MON=$!
t0=$(date +%s)
NUM_HBM=${NHBM[$MODEL]} NGPU=${NGPU[$MODEL]} RAMU_WORKERS=2 EPIC_K=8 \
  bash "$REPO/experiments/run_dag_ladder.sh" "$REPO/workload/sweep/$WL" "$MODEL" "$OUT" \
  > "$OUT/ladder.log" 2>&1
rc=$?; kill $MON 2>/dev/null
echo "[$(date +%F' '%T)] SMOKE9 $MODEL $WL rc=$rc $(( $(date +%s)-t0 ))s jsons=$(ls "$OUT"/dag_A*.json 2>/dev/null|wc -l)/9"
for A in A1 A2 A3 A3a A3b A4 A4b A5 A6; do
  if [ -s "$OUT/dag_${A}.json" ]; then echo "  $A OK"; else
    echo "  $A MISSING -- last log lines:"; tail -3 "$OUT/dag_${A}.log" 2>/dev/null | sed 's/^/      /'; fi
done
echo "peak rss: $(awk -F'rss=' '{split($2,a,"G");if(a[1]+0>m)m=a[1]+0} END{print m"G"}' "$OUT/usage.log" 2>/dev/null)"
rm -rf "$RD"
