#!/usr/bin/env bash
# Submit ONE backfill slot.  Sized like slot_submit.sh, with two differences:
#
#   * the job is named bf_* so governor.py -- which matches ^slot -- neither
#     drains nor counts it.  Nothing else is watching this memory, so the caller
#     is responsible for keeping the slot count sane;
#   * a backfill runs at most 8 rungs, usually far fewer, so it is cheaper than
#     a phase-2 slot of the same class.  The memory ask is kept at the phase-2
#     figure anyway: --mem is informational on this cluster (CR_CORE, memory is
#     not a consumable resource) and the real guard is how many slots exist.
#
#   usage: backfill_submit.sh <node> <big|small>
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ORCH=$REPO/output/_orch2
ROOT=$(cat "$ORCH/CURRENT_ROOT")
node=$1; class=$2
declare -A PART=( [node1]=athena-mini [node2]=athena [node3]=athena
                  [node4]=athena-small [node5]=athena-genai [node6]=athena-genai )
if [ "$class" = big ]; then CORES=12; MEM=420G; NOGC=0
else                        CORES=12; MEM=140G; NOGC=1; fi
W=${W:-2}
mkdir -p "$ROOT/slurm" "$ROOT/claims_bf" "$ROOT/drain"
sb=$(mktemp "$ROOT/slurm/bf_${class}_${node}_XXXX.sbatch")
cat > "$sb" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=bf_${node}
#SBATCH --partition=${PART[$node]}
#SBATCH --nodelist=${node}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=${CORES} --mem=${MEM}
#SBATCH --time=2-00:00:00 --open-mode=append
#SBATCH --output=${ROOT}/slurm/bf_${node}_%j.out
#SBATCH --error=${ROOT}/slurm/bf_${node}_%j.out
set -uo pipefail
export KVPIM_NOGC=${NOGC}
bash $ORCH/node_watch.sh $ROOT & WATCH=\$!
echo "=== \$(date) backfill ${class} slot on \$(hostname -s) job \$SLURM_JOB_ID: ${CORES} cores, ${MEM}"
bash $ORCH/backfill.sh $ROOT ${W} \$SLURM_JOB_ID $ROOT/tasks_backfill.txt
kill \$WATCH 2>/dev/null
echo "BACKFILL SLOT DONE \$(hostname -s) job \$SLURM_JOB_ID \$(date)"
SB
sbatch "$sb"
