#!/usr/bin/env bash
# Phase 1: pre-warm the signature cache -- ONE MODEL PER NODE (models touch
# disjoint shapes), two streams per node.  Stream A warms the C64 context
# sizes, stream B the C32 ones (baseline then N64) -- so no two concurrent
# writers ever chase the same shape, which is the whole point of warm_cache.sh.
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ORCH=$REPO/output/_orch2
TS=${SWEEP_TS:-$(date +%Y%m%d-%H%M%S)}
ROOT=$REPO/output/sweep_models_${TS}
mkdir -p "$ROOT/cachepool" "$ROOT/slurm" "$ROOT/claims"
cp "$ORCH/tasks_master.txt" "$ROOT/tasks.txt"
echo "$ROOT" > "$ORCH/CURRENT_ROOT"
echo "sweep root: $ROOT"

sub() {  # node part cores mem W model
  local node=$1 part=$2 cores=$3 mem=$4 w=$5 model=$6
  local sb=$ROOT/slurm/warm_${node}.sbatch
  cat > "$sb" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=warm_${node}
#SBATCH --partition=${part}
#SBATCH --nodelist=${node}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=${cores} --mem=${mem}
#SBATCH --time=7-00:00:00 --open-mode=append
#SBATCH --output=${ROOT}/slurm/warm_${node}.out
#SBATCH --error=${ROOT}/slurm/warm_${node}.out
set -uo pipefail
bash $ORCH/node_watch.sh $ROOT & WATCH=\$!
echo "=== \$(date) warm on \$(hostname -s): model ${model}, ${cores} cores, ${mem} RAM, W=${w}"
bash $ORCH/warm_stream.sh $ROOT ${model} ${w} c64  wl_C64.json & P1=\$!
bash $ORCH/warm_stream.sh $ROOT ${model} ${w} c32  wl_baseline_alltoall_N16_C32_D2.json wl_N64.json & P2=\$!
wait \$P1; wait \$P2
kill \$WATCH 2>/dev/null
echo "WARM NODE DONE \$(hostname -s) \$(date)"
SB
  sbatch "$sb"
}
#   node   partition      cores(60%)  mem(60%)  W   model
sub node5 athena-genai 76 600G 37 GPT-175B
sub node6 athena-genai 76 600G 37 LLAMA-65B
sub node1 athena-mini  57 600G 27 LLAMA-33B
sub node2 athena       57 600G 27 GPT-13B
sub node3 athena       57 600G 27 LLAMA-7B
sub node4 athena-small 57 600G 27 LLAMA3-8B
