#!/usr/bin/env bash
# Submit ONE worker slot for the post-fix 5-rung sweep (A3/A4/A4b/A5/A6 on the
# twelve configs other than baseline; N-hi is abandoned).
#
# Reuses backfill.sh unchanged: it already takes ROOT and a queue whose fifth
# field is the rung list, claims with an atomic mkdir, and re-checks each rung
# against the disk before running it -- so a slot that dies mid-task leaves the
# finished rungs in place and the next slot picks up only what is missing.
#
#   usage: rungs5_submit.sh <node> <big|small>
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ORCH=$REPO/output/_orch2
ROOT=$REPO/output/sweep_postfix_20260903_rungs5
node=$1; class=$2
declare -A PART=( [node1]=athena-mini [node2]=athena [node3]=athena
                  [node4]=athena-small [node5]=athena-genai [node6]=athena-genai )
if [ "$class" = big ]; then CORES=12; MEM=420G; NOGC=0
else                        CORES=12; MEM=160G; NOGC=1; fi
W=${W:-2}
mkdir -p "$ROOT/slurm" "$ROOT/claims_bf" "$ROOT/drain"
sb=$(mktemp "$ROOT/slurm/r5_${node}_XXXX.sbatch")
cat > "$sb" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=r5_${node}
#SBATCH --partition=${PART[$node]}
#SBATCH --nodelist=${node}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=${CORES} --mem=${MEM}
#SBATCH --time=3-00:00:00 --open-mode=append
#SBATCH --output=${ROOT}/slurm/r5_${node}_%j.out
#SBATCH --error=${ROOT}/slurm/r5_${node}_%j.out
set -uo pipefail
export KVPIM_NOGC=${NOGC}
echo "=== \$(date) 5-rung slot on \$(hostname -s) job \$SLURM_JOB_ID  .so \$(stat -c %y $REPO/src/cppcore/libeventcore.so | cut -c1-19)"
bash $ORCH/backfill.sh $ROOT ${W} \$SLURM_JOB_ID $ROOT/tasks_rungs5.txt
echo "SLOT DONE \$(hostname -s) \$SLURM_JOB_ID \$(date)"
SB
sbatch "$sb"
