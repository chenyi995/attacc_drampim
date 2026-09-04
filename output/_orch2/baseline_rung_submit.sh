#!/usr/bin/env bash
# Submit ONE per-rung worker slot for the baseline re-run.
#
# A slot now runs ONE rung at a time, so it needs one core for the build plus
# room for the W Ramulator workers of the warm phase.  Measured live on
# 2026-09-03: a rung process is single-threaded and averages 0.7 of a core
# while it builds, which is where the runtime goes.  Small ask = schedules
# even on the congested partitions, and 42 of them can be in flight at once.
#
# Memory is per RUNG, not per ladder.  The 2026-09-02 NINE-rung ladders peaked
# at 461/386/297/194/169/170 GB for GPT-175B/LLAMA-65B/LLAMA-33B/GPT-13B/
# LLAMA3-8B/LLAMA-7B on a context twice this one; per rung that is roughly a
# ninth, and A1 (the no-reuse graph) is the outlier.  These asks give A1 room
# on the worst model and still let Slurm backfill them.
#
# QUEUE picks which set the slot works on:
#   baseline (default)  tasks_baseline_rungs.txt   42 rungs, 7 per model
#   points              tasks_points_rungs.txt    288 rungs, A3b/A4/A5/A6 on
#                                                 the twelve other sweep points
#
#   usage: baseline_rung_submit.sh <node> <big|small> [baseline|points]
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ORCH=$REPO/output/_orch2
ROOT=$REPO/output/sweep_colpack_20260903
node=$1; class=${2:-small}; set_=${3:-baseline}
case "$set_" in
  baseline) QUEUE=tasks_baseline_rungs.txt ;;
  points)   QUEUE=tasks_points_rungs.txt ;;
  *) echo "third arg must be baseline|points" >&2; exit 1 ;;
esac
declare -A PART=( [node1]=athena-mini [node2]=athena [node3]=athena
                  [node4]=athena-small [node5]=athena-genai [node6]=athena-genai )
case "$class" in
  big)   CORES=4; MEM=120G; NOGC=0; TIME=12:00:00 ;;
  small) CORES=4; MEM=60G;  NOGC=1; TIME=8:00:00 ;;
  *) echo "class must be big|small" >&2; exit 1 ;;
esac
W=${W:-2}
mkdir -p "$ROOT/slurm" "$ROOT/claims_rung" "$ROOT/drain" "$ROOT/cachepool"
sb=$(mktemp "$ROOT/slurm/rw_${set_}_${class}_${node}_XXXX.sbatch")
cat > "$sb" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=r${set_:0:1}${class:0:1}_${node}
#SBATCH --partition=${PART[$node]}
#SBATCH --nodelist=${node}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=${CORES} --mem=${MEM}
#SBATCH --time=${TIME} --open-mode=append
#SBATCH --output=${ROOT}/slurm/rw_${set_}_${class}_${node}_%j.out
#SBATCH --error=${ROOT}/slurm/rw_${set_}_${class}_${node}_%j.out
set -uo pipefail
export KVPIM_NOGC=${NOGC}
echo "=== \$(date) per-rung worker (${set_}/${class}) on \$(hostname -s) job \$SLURM_JOB_ID"
bash $ORCH/baseline_rung_worker.sh $ROOT ${W} \$SLURM_JOB_ID \\
     $ROOT/${QUEUE}
echo "SLOT DONE \$(hostname -s) \$SLURM_JOB_ID \$(date)"
SB
sbatch "$sb"
