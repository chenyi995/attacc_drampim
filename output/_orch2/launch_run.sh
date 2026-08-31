#!/usr/bin/env bash
# Phase 2: drain the 84-task queue.  Every node runs N worker slots; each slot
# takes one (model, config) task at a time and runs its 9 rungs in parallel.
# Slot budget: 9 rungs x (1 + RAMU_WORKERS) processes must fit the node's 60%.
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ORCH=$REPO/output/_orch2
ROOT=$(cat "$ORCH/CURRENT_ROOT")
mkdir -p "$ROOT/slurm" "$ROOT/claims"
echo "sweep root: $ROOT   tasks: $(wc -l < "$ROOT/tasks.txt")"

sub() {  # node part cores mem slots W [dep]
  local node=$1 part=$2 cores=$3 mem=$4 slots=$5 w=$6 dep=${7:--}
  local sb=$ROOT/slurm/run_${node}.sbatch
  cat > "$sb" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=run_${node}
#SBATCH --partition=${part}
#SBATCH --nodelist=${node}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=${cores} --mem=${mem}
#SBATCH --time=7-00:00:00 --open-mode=append
#SBATCH --output=${ROOT}/slurm/run_${node}.out
#SBATCH --error=${ROOT}/slurm/run_${node}.out
set -uo pipefail
bash $ORCH/node_watch.sh $ROOT & WATCH=\$!
echo "=== \$(date) run on \$(hostname -s): ${slots} slots x 9 rungs x (1+${w}) = \$(( ${slots}*9*(1+${w}) )) procs, cap ${cores}"
PIDS=()
for s in \$(seq 1 ${slots}); do
  bash $ORCH/worker.sh $ROOT ${w} \$s & PIDS+=(\$!)
  sleep 20
done
for p in "\${PIDS[@]}"; do wait \$p; done
kill \$WATCH 2>/dev/null
echo "RUN NODE DONE \$(hostname -s) \$(date)"
SB
  if [ "$dep" = "-" ]; then sbatch "$sb"; else sbatch --dependency=afterany:"$dep" "$sb"; fi
}
# Request exactly the peak we spawn: slots x 9 rungs x (1 + RAMU_WORKERS).
# 2 slots x 9 x 3 = 54 cores.  Steady state is the 18 single-core builds; the
# Ramulator workers only burst on a cache miss, and after Phase 1 most shapes
# are hits.  Memory is a measured estimate -- re-check usage_node*.log after
# the first configs land and shrink if it is over.
#   node   partition    cores mem  slots W   dep
sub node5 athena-genai 54 256G 2 2 "${DEP_node5:--}"
sub node6 athena-genai 54 256G 2 2 "${DEP_node6:--}"
sub node1 athena-mini  54 192G 2 2 "${DEP_node1:--}"
sub node2 athena       54 192G 2 2 "${DEP_node2:--}"
sub node3 athena       54 192G 2 2 "${DEP_node3:--}"
sub node4 athena-small 54 192G 2 2 "${DEP_node4:--}"
