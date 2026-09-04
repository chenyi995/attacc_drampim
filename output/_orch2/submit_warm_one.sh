#!/usr/bin/env bash
# usage: submit_warm_one.sh <node> <part> <cores> <mem> <W> <model>
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim; ORCH=$REPO/output/_orch2
ROOT=$(cat "$ORCH/CURRENT_ROOT")
node=$1 part=$2 cores=$3 mem=$4 w=$5 model=$6
sb=$ROOT/slurm/warm_${node}.sbatch
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
bash $ORCH/warm_stream.sh $ROOT ${model} ${w} c64 wl_C40.json & P1=\$!
bash $ORCH/warm_stream.sh $ROOT ${model} ${w} c32 wl_baseline_alltoall_N16_C16_D2.json wl_N64.json & P2=\$!
wait \$P1; wait \$P2
kill \$WATCH 2>/dev/null
echo "WARM NODE DONE \$(hostname -s) \$(date)"
SB
sbatch "$sb"
