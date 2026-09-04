#!/usr/bin/env bash
# Submit ONE worker slot for the column-packed re-run (ruling chenyi9
# 2026-09-03): the baseline's A1/A2/A3b/A4/A4b/A5/A6, and A3b + A6 alone on
# every other sweep point.
#
# Reuses backfill.sh unchanged -- it already takes ROOT and a queue whose fifth
# field is the rung list, claims each task with an atomic mkdir, and re-checks
# every rung against the disk before running it, so a slot that dies mid-task
# leaves the finished rungs in place and the next slot picks up only the holes.
# The queue is ordered baseline-first, so the six baseline ladders are claimed
# before any of the two-rung points.
#
#   usage: colpack_submit.sh <node> <big|small>
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ORCH=$REPO/output/_orch2
ROOT=$REPO/output/sweep_colpack_20260903
node=$1; class=$2
declare -A PART=( [node1]=athena-mini [node2]=athena [node3]=athena
                  [node4]=athena-small [node5]=athena-genai [node6]=athena-genai )
# A baseline task runs seven rungs at once; W Ramulator workers each plus one
# construction process per rung is 7*(W+1) cores.  W=2 -> 21, which fits the
# 24-core ask with room for the collector.
if [ "$class" = big ]; then CORES=24; MEM=420G; NOGC=0
else                        CORES=24; MEM=180G; NOGC=1; fi
W=${W:-2}
mkdir -p "$ROOT/slurm" "$ROOT/claims_bf" "$ROOT/drain" "$ROOT/cachepool"
sb=$(mktemp "$ROOT/slurm/cp_${node}_XXXX.sbatch")
cat > "$sb" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cp_${node}
#SBATCH --partition=${PART[$node]}
#SBATCH --nodelist=${node}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=${CORES} --mem=${MEM}
#SBATCH --time=3-00:00:00 --open-mode=append
#SBATCH --output=${ROOT}/slurm/cp_${node}_%j.out
#SBATCH --error=${ROOT}/slurm/cp_${node}_%j.out
set -uo pipefail
export KVPIM_NOGC=${NOGC}
echo "=== \$(date) column-packed slot on \$(hostname -s) job \$SLURM_JOB_ID"
echo "    layout: 8-token columns, 32 col = 256-token row, charged whole"
echo "    .so \$(stat -c %y $REPO/src/cppcore/libeventcore.so | cut -c1-19)"
bash $ORCH/backfill.sh $ROOT ${W} \$SLURM_JOB_ID $ROOT/tasks.txt
echo "SLOT DONE \$(hostname -s) \$SLURM_JOB_ID \$(date)"
SB
sbatch "$sb"
