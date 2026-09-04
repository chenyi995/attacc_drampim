#!/usr/bin/env bash
# experiment1 Phase 1 -- PRE-WARM the Ramulator signature cache (chenyi9
# 2026-08-29).
#
# WHY: a fresh clone has a COLD cache.  Every new (model, context-size, rung)
# shape runs a full Ramulator DRAM simulation; the head-aware placement scan
# prices each channel's real run, and the 1-HBM small tier has the LARGEST
# single-channel traces, so cold shapes there are the slow part.  If you launch
# the whole 756-run sweep on a cold cache, many runs race to build the SAME
# shape and write the same signature file at once -- wasteful and slow.
#
# HOW: build the cache ONCE, in parallel, WITHOUT same-shape contention --
#   * different MODELS run in PARALLEL (they touch disjoint shapes: different
#     n_heads / heads_per_hbm), and
#   * inside a model the 9 RUNGS run SEQUENTIALLY (single writer per shape).
# A few shape-diverse configs (largest / mid / many-agent contexts) cover the
# bulk of the distinct shapes; the remaining sweep configs are then cache hits.
#
# WATCH  ramulator2/signature_cache_v2_headhbm.jsonl grow; when it stops
# growing the cache is basically built.  THEN run Phase 2:
#     bash experiments/run_sweep_models.sh
# which reruns everything with the 9 rungs in PARALLEL off the warm cache (fast)
# and is the pass whose dag_A*.json you extract.
#
#   usage: bash experiments/warm_cache.sh      # then run_sweep_models.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, any machine
WD=workload/sweep
RUNGS="A1 A2 A3 A3a A3b A4 A4b A5 A6"

# shape-diverse warm set: biggest context (C64), baseline, most agents (N64)
WARM_CFG=(
  "wl_C40.json                           8"
  "wl_baseline_alltoall_N16_C16_D2.json  8"
  "wl_N64.json                           8"
)
# tier  model      ngpu  num_hbm  (same as run_sweep_models.sh)
MODELS=(
  "LLAMA-7B    1   1"
  "LLAMA3-8B   1   1"
  "GPT-13B     2  10"
  "LLAMA-33B   2  10"
  "GPT-175B    8  40"
  "LLAMA-65B   8  40"
)

warm_model() {
  read -r model ngpu hbm <<< "$1"
  for entry in "${WARM_CFG[@]}"; do
    read -r wl k <<< "$entry"
    for A in $RUNGS; do
      REUSE=recompute; EX=(--epic-prefix-recompute-tokens "$k")
      [ "$A" = A1 ] && { REUSE=no-reuse; EX=(); }
      KVPIM_CPPCORE=1 python3 main.py --system dgx-attacc --model "$model" --ngpu "$ngpu" \
        --workload "$WD/$wl" --reuse "$REUSE" "${EX[@]}" --ablation "$A" --engine dag \
        --num-hbm "$hbm" --ramulator-workers 8 --cacheblend-batch-size 8 \
        --workload-report-events none --no-warm \
        --workload-report "/tmp/warm_${model}_${A}.json" >/dev/null 2>&1 || true
    done
    echo "[$(date +%T)] warmed ${model}  ${wl}"
  done
}

echo "CACHE WARM start $(date)  (6 models in parallel, 9 rungs each sequential)"
for m in "${MODELS[@]}"; do warm_model "$m" & done
wait
echo "CACHE WARM DONE $(date)  ->  ramulator2/signature_cache_v2_headhbm.jsonl"
echo "next: bash experiments/run_sweep_models.sh"
