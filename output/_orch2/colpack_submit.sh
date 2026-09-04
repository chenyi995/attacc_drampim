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
# Three classes, each with its OWN queue, so the 72 two-rung point tasks run
# ALONGSIDE the baselines instead of behind them, and each asks the memory it
# needs.  Asking 420G for a slot that peaks at 40 does not get backfilled --
# that is what left three of six slots pending on the first attempt.
#
#   usage: colpack_submit.sh <node> <big|base|points>
#     big     GPT-175B / LLAMA-65B baselines, 7 rungs    9 cpu / 300G
#     base    the other four baselines,       7 rungs    9 cpu / 170G
#     points  A3b+A6 on one sweep point,      2 rungs    3 cpu /  48G
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ORCH=$REPO/output/_orch2
ROOT=$REPO/output/sweep_colpack_20260903
node=$1; class=$2
declare -A PART=( [node1]=athena-mini [node2]=athena [node3]=athena
                  [node4]=athena-small [node5]=athena-genai [node6]=athena-genai )
# A task runs its rungs at once; W Ramulator workers plus one construction
# process per rung is rungs*(W+1) cores.  Seven rungs at W=2 is 21, which fits
# the 24-core ask with room for the collector; two rungs is 6, which fits 8.
# Memory is sized from the measured peaks of the 2026-09-02 nine-rung ladders
# (461/386/297/194/169/170 GB) scaled for SEVEN rungs on a 4608-token context
# instead of nine on 8464 -- generous, but not so generous that Slurm cannot
# backfill the job.
# WALLTIME is the other half of getting scheduled.  Slurm's backfill only
# starts a job that provably finishes before the top-priority job's reserved
# start, so a 3-day limit -- what this asked at first -- almost never
# backfills on a partition with 55 jobs queued ahead.  These limits are sized
# to the work (a two-rung point on a 4608-token context is well under an hour
# per rung; the 2026-09-02 NINE-rung ladders on a context twice this size took
# 5.6-9.8 h) and losing one is cheap: colpack_reap.sh releases the claim and
# the next slot resumes it, skipping the rungs already on disk.
case "$class" in
  big)    CORES=9;  MEM=300G; NOGC=0; TIME=1-00:00:00; QUEUE=tasks_baseline_big.txt ;;
  base)   CORES=9;  MEM=170G; NOGC=0; TIME=16:00:00;   QUEUE=tasks_baseline.txt ;;
  points) CORES=3;  MEM=48G;  NOGC=1; TIME=6:00:00;    QUEUE=tasks_points.txt ;;
  *) echo "class must be big|base|points" >&2; exit 1 ;;
esac
W=${W:-2}
mkdir -p "$ROOT/slurm" "$ROOT/claims_bf" "$ROOT/drain" "$ROOT/cachepool"
sb=$(mktemp "$ROOT/slurm/cp_${class}_${node}_XXXX.sbatch")
cat > "$sb" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=cp${class:0:1}_${node}
#SBATCH --partition=${PART[$node]}
#SBATCH --nodelist=${node}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=${CORES} --mem=${MEM}
#SBATCH --time=${TIME} --open-mode=append
#SBATCH --output=${ROOT}/slurm/cp_${class}_${node}_%j.out
#SBATCH --error=${ROOT}/slurm/cp_${class}_${node}_%j.out
set -uo pipefail
export KVPIM_NOGC=${NOGC}
echo "=== \$(date) column-packed ${class} slot on \$(hostname -s) job \$SLURM_JOB_ID"
echo "    layout: 8-token columns, 32 col = 256-token row, charged whole"
echo "    .so \$(stat -c %y $REPO/src/cppcore/libeventcore.so | cut -c1-19)"
bash $ORCH/backfill.sh $ROOT ${W} \$SLURM_JOB_ID $ROOT/${QUEUE}
echo "SLOT DONE \$(hostname -s) \$SLURM_JOB_ID \$(date)"
SB
sbatch "$sb"
