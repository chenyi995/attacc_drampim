#!/usr/bin/env bash
# Submit ONE phase-2 worker slot.  Two classes, because a task's cost is set by
# the model, not by the slot:
#   big   = GPT-175B / LLAMA-65B / LLAMA-33B  -- ~30G per rung, so ~300G a task
#   small = GPT-13B / LLAMA-7B / LLAMA3-8B    -- ~4-8G per rung
# Cores are the same for both: a task is 9 rungs of SINGLE-THREADED graph build
# plus one shared Ramulator worker pool, measured at ~10 cores, not the 27 the
# process count suggests.
#   usage: slot_submit.sh <node> <big|small> [dependency-jobid-list]
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ORCH=$REPO/output/_orch2
ROOT=$(cat "$ORCH/CURRENT_ROOT")
node=$1; class=$2; dep=${3:-}
declare -A PART=( [node1]=athena-mini [node2]=athena [node3]=athena
                  [node4]=athena-small [node5]=athena-genai [node6]=athena-genai )
if [ "$class" = big ]; then CORES=12; MEM=420G; QUEUE=$ROOT/tasks_big.txt;   NOGC=0
else                        CORES=12; MEM=140G; QUEUE=$ROOT/tasks_small.txt; NOGC=1; fi
W=${W:-2}
mkdir -p "$ROOT/slurm" "$ROOT/claims" "$ROOT/drain"
sb=$(mktemp "$ROOT/slurm/slot_${class}_${node}_XXXX.sbatch")
cat > "$sb" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=slot${class:0:1}_${node}
#SBATCH --partition=${PART[$node]}
#SBATCH --nodelist=${node}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=${CORES} --mem=${MEM}
#SBATCH --time=7-00:00:00 --open-mode=append
#SBATCH --output=${ROOT}/slurm/slot_${node}_%j.out
#SBATCH --error=${ROOT}/slurm/slot_${node}_%j.out
set -uo pipefail
# A big-model task holds ~400G with the cyclic collector off but ~300G with it
# on.  On a node capped at 80% that difference is one whole extra slot, which is
# worth more than the 16% the collector costs.
export KVPIM_NOGC=${NOGC}
bash $ORCH/node_watch.sh $ROOT & WATCH=\$!
echo "=== \$(date) ${class} slot on \$(hostname -s) job \$SLURM_JOB_ID: ${CORES} cores, ${MEM}, queue $(basename $QUEUE)"
bash $ORCH/worker.sh $ROOT ${W} \$SLURM_JOB_ID $QUEUE
kill \$WATCH 2>/dev/null
echo "SLOT DONE \$(hostname -s) job \$SLURM_JOB_ID \$(date)"
SB
if [ -n "$dep" ]; then sbatch --dependency=afterany:"$dep" "$sb"; else sbatch "$sb"; fi
