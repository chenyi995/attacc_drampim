#!/usr/bin/env bash
# A/B: is --no-warm the reason the warm phase never uses its workers?
# Same model / workload / rung, both arms starting from an EMPTY signature
# cache, differing only in --no-warm.  Records wall time and the peak number
# of concurrent ramulator2 processes.
set -uo pipefail
source /home/cw636/chenyi/attacc_drampim/output/_orch2/common.sh
OUT=$REPO/output/_ab_nowarm; mkdir -p "$OUT"
export_env
MODEL=LLAMA3-8B; WL=wl_N4.json; A=A4b; W=8
for ARM in nowarm warm; do
  RD=/tmp/kvpim_${USER}_ab_${ARM}_$$; rm -rf "$RD"; mkdir -p "$RD"
  ln -sf "$REPO/ramulator2/ramulator2" "$RD/ramulator2"
  ln -sf "$REPO/ramulator2/trace_gen"  "$RD/trace_gen"
  cp "$REPO/ramulator.out" "$RD/ramulator.out" 2>/dev/null || :
  : > "$RD/signature_cache.jsonl"            # both arms start COLD
  export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
  NW=(); [ "$ARM" = nowarm ] && NW=(--no-warm)
  ( peak=0; while true; do
      c=$(pgrep -u "$USER" -c -f "$RD/ramulator2" 2>/dev/null || echo 0)
      [ "$c" -gt "$peak" ] && { peak=$c; echo "$(date +%T) concurrent ramulator2 = $peak"; }
      sleep 2
    done ) > "$OUT/${ARM}_conc.log" 2>&1 & MON=$!
  t0=$(date +%s)
  (cd "$REPO" && timeout 7200 python3 main.py --system dgx-attacc --model $MODEL \
     --ngpu "${NGPU[$MODEL]}" --num-hbm "${NHBM[$MODEL]}" \
     --workload "workload/sweep/$WL" --reuse recompute \
     --epic-prefix-recompute-tokens 8 --ablation $A --engine dag \
     --ramulator-workers $W --cacheblend-batch-size 8 \
     --workload-report-events none "${NW[@]}" \
     --workload-report "$OUT/${ARM}.json") > "$OUT/${ARM}.log" 2>&1
  rc=$?; kill $MON 2>/dev/null
  sigs=$(wc -l < "$RD/signature_cache.jsonl" 2>/dev/null || echo 0)
  peak=$(awk '{print $NF}' "$OUT/${ARM}_conc.log" 2>/dev/null | sort -n | tail -1)
  echo "[AB] arm=$ARM rc=$rc $(( $(date +%s)-t0 ))s sigs_built=$sigs peak_concurrent_ramulator=${peak:-0}"
  rm -rf "$RD"
done
echo "[AB] DONE"
