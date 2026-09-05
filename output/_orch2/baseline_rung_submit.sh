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
# NODE may be a node name (pins the slot with --nodelist, the original
# behaviour) or the literal `any`, which drops --nodelist and offers the job
# EVERY partition at once.  Prefer `any`.  Pinning was costing far more than
# the footprint: measured 2026-09-04, node2 sat with 48 idle cores and 309 GB
# free while thirty of our jobs waited on (Priority) -- for node2.  A job that
# can land anywhere takes the first gap that opens in any partition.
#
#   usage: baseline_rung_submit.sh <node|any> <big|small> [baseline|points]
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ORCH=$REPO/output/_orch2
ROOT=${ROOT:-$REPO/output/sweep_20260904_final}
node=$1; class=${2:-small}; set_=${3:-baseline}
case "$set_" in
  baseline) QUEUE=tasks_baseline_rungs.txt ;;
  points)   QUEUE=$([ "$class" = big ] && echo tasks_points_big_rungs.txt \
                                       || echo tasks_points_small_rungs.txt) ;;
  *) echo "third arg must be baseline|points" >&2; exit 1 ;;
esac
declare -A PART=( [node1]=athena-mini [node2]=athena [node3]=athena
                  [node4]=athena-small [node5]=athena-genai [node6]=athena-genai )
ALL_PARTS=athena,athena-mini,athena-small,athena-genai
if [ "$node" = any ]; then
  PARTITION=$ALL_PARTS; NODELINE=""
else
  PARTITION=${PART[$node]}; NODELINE="#SBATCH --nodelist=${node}"
fi
case "$class" in
  # Sized from live measurement (2026-09-04), not from the ladder totals: a
  # rung process is single-threaded at ~0.8 of a core and the heaviest points
  # rung observed was 30.3 GB (LLAMA-65B A3b), most 12-23 GB.  The previous
  # 4-core/60-120G asks ran at 20% CPU and 18% memory across the fleet.  The
  # queue is split by model size so a small slot can never be handed a
  # GPT-175B point.
  # WALLTIME, not cores or memory, is what the scheduler is actually gating
  # on.  Probed 2026-09-04 with identical 2-core/40G jobs on all four
  # partitions: 1h30, 2h, 3h, 4h and 6h ALL started immediately on node1,
  # while our 8h and 12h slots sat on (Priority) with node2 holding 48 idle
  # cores.  The backfill window is between 6h and 8h, so ask 6.  Rung
  # durations (144 samples, previous round): median 1h42, 90th pct 4h24, max
  # 9h54 -- 6h covers ~90% of them, and colpack_reap.sh releases the claim of
  # anything killed so a longer slot can finish it later.  Keep a few 12h
  # slots queued for that tail.
  # CORES is 1 for a small rung.  Measured 2026-09-04 across the fleet: a rung
  # process is single-threaded and sits at 0.7-0.8 of a core for the whole DAG
  # build, which is where the runtime goes.  On the nodes we hold, CORES is the
  # binding resource and memory is not -- node1 was 96/96 cores with 195 GB
  # free -- so one core per rung doubles how many rungs fit in the same
  # allocation.  The warm phase forks W Ramulator workers and will oversubscribe
  # that core; it is a minority of the runtime and its results are memoised.
  # big keeps 2 because its warm phase is heavier and it has the memory anyway.
  big)   CORES=${CORES:-2}; MEM=80G; NOGC=0; TIME=${TIME:-6:00:00} ;;
  small) CORES=${CORES:-1}; MEM=40G; NOGC=1; TIME=${TIME:-6:00:00} ;;
  *) echo "class must be big|small" >&2; exit 1 ;;
esac
W=${W:-2}
mkdir -p "$ROOT/slurm" "$ROOT/claims_rung" "$ROOT/drain" "$ROOT/cachepool"
sb=$(mktemp "$ROOT/slurm/rw_${set_}_${class}_${node}_XXXX.sbatch")
cat > "$sb" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=r${set_:0:1}${class:0:1}_${node}
#SBATCH --partition=${PARTITION}
${NODELINE}
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
