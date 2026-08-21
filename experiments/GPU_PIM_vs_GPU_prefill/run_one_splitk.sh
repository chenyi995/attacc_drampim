#!/usr/bin/env bash
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
S=$1; wl=$2; cfg=$3; link=$4; policy=$5; extra=${6:-}; model=${MODEL:-LLAMA-7B}
case "$model" in LLAMA-7B) last=31;; LLAMA-65B) last=79;; GPT-175B) last=95;; esac
job="$S/.ram/${model}_${wl}_${cfg}_${link}_${policy}$(echo $extra | tr -d ' -')"; mkdir -p "$job/ramulator2"
for f in ramulator2 libramulator.so trace_gen; do ln -sfn "$REPO/ramulator2/$f" "$job/ramulator2/$f"; done
[ -f "$job/ramulator.out" ] || cp -f "$REPO/ramulator.out" "$job/ramulator.out"
export ATTACC_RAMULATOR_DIR="$job/ramulator2" ATTACC_RAMULATOR_LOG="$job/ramulator.out"
tag=${model}_${wl}_${cfg}_${link}_${policy}$(echo $extra | tr -d ' -')
args=(--system dgx-attacc --model $model --workload $S/wl/$wl.json --tier-batch-size 1 --ramulator-workers 2 --pipeopt --ffopt --ablation $cfg --reuse $policy --gpu-model flash --attn-splitk --pim-link $link)
case "$policy" in cacheblend) args+=(--cacheblend-full-layers 0-1 --cacheblend-partial-layers 2-$last --reuse-seed 7 $extra);; epic) args+=($extra);; esac
"$REPO/.venv/bin/python" main.py "${args[@]}" --workload-report "$S/$tag.json" > "$S/$tag.log" 2>&1
echo "$tag rc=$?"
