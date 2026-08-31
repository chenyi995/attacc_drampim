#!/usr/bin/env bash
ROOT=$(cat /home/cw636/chenyi/attacc_drampim/output/_orch2/CURRENT_ROOT)
echo "root: $ROOT"
tot=$(find "$ROOT" -name 'dag_A*.json' 2>/dev/null | wc -l)
echo "RUNS: $tot / 756"
echo; printf "%-12s %s\n" MODEL "configs complete (9/9)"
for m in GPT-175B LLAMA-65B LLAMA-33B GPT-13B LLAMA3-8B LLAMA-7B; do
  [ -d "$ROOT/$m" ] || { printf "%-12s -\n" "$m"; continue; }
  n=$(find "$ROOT/$m" -name 'dag_A*.json' 2>/dev/null | wc -l)
  c=0; for d in "$ROOT/$m"/*/; do [ "$(ls $d/dag_A*.json 2>/dev/null|wc -l)" = 9 ] && c=$((c+1)); done
  printf "%-12s %2d/14 configs, %3d/126 runs\n" "$m" "$c" "$n"
done
echo; echo "claimed: $(ls "$ROOT/claims" 2>/dev/null | wc -l)/84   failed: $(wc -l < "$ROOT/failed_tasks.txt" 2>/dev/null || echo 0)"
echo; echo "--- slurm ---"; squeue -u "$USER" -o "%.8i %.13P %.11j %.2t %.9M %R"
echo; echo "--- node usage (latest) ---"; for f in "$ROOT"/usage_node*.log; do [ -s "$f" ] && tail -1 "$f"; done
echo; echo "--- cache pool ---"; for m in GPT-175B LLAMA-65B LLAMA-33B GPT-13B LLAMA3-8B LLAMA-7B; do
  n=$(cat "$ROOT"/cachepool/${m}__*.jsonl 2>/dev/null | wc -l); echo "  $m: $n signatures"; done
