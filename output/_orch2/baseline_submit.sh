#!/usr/bin/env bash
# One job per model: the baseline config's full nine-rung ladder on the
# post-75da860 engine (PIM channel lanes overlap instead of summing).
#
# Deliberately NOT reusing slot_submit.sh/worker.sh: those read
# output/_orch2/CURRENT_ROOT, which still points at the archived 2026-08-30
# sweep.  Six tasks do not need a shared claim queue, and a one-off must not
# move a pointer other tooling reads.
#
#   usage: baseline_submit.sh <model> <node>
set -euo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
ORCH=$REPO/output/_orch2
ROOT=$REPO/output/sweep_baseline_20260902_postfix
model=$1; node=$2
WL=wl_baseline_alltoall_N16_C16_D2.json
K=8
source "$ORCH/common.sh"
declare -A PART=( [node1]=athena-mini [node2]=athena [node3]=athena
                  [node4]=athena-small [node5]=athena-genai [node6]=athena-genai )
OUT=$ROOT/$model/baseline_k$K
mkdir -p "$OUT" "$ROOT/slurm"
sb=$(mktemp "$ROOT/slurm/bl_${model}_XXXX.sbatch")
cat > "$sb" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=bl_${model}
#SBATCH --partition=${PART[$node]}
#SBATCH --nodelist=${node}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=420G
#SBATCH --time=2-00:00:00 --open-mode=append
#SBATCH --output=${ROOT}/slurm/bl_${model}_%j.out
#SBATCH --error=${ROOT}/slurm/bl_${model}_%j.out
set -uo pipefail
source $ORCH/common.sh
export_env
export KVPIM_NOGC=0
RD=\$(make_ramdir "bl_${model}" "${model}" "$ROOT/cachepool")
export ATTACC_RAMULATOR_DIR="\$RD" ATTACC_RAMULATOR_LOG="\$RD/ramulator.out"
echo "=== \$(date) ${model} baseline k=${K} on \$(hostname -s) job \$SLURM_JOB_ID"
echo "    engine: post-75da860 (channel lanes parallel);  .so \$(stat -c %y $REPO/src/cppcore/libeventcore.so | cut -c1-19)"
echo "    num_hbm=\${NHBM[$model]} ngpu=\${NGPU[$model]}"
t0=\$(date +%s)
NUM_HBM=\${NHBM[$model]} NGPU=\${NGPU[$model]} RAMU_WORKERS=2 EPIC_K=${K} \
  bash $REPO/experiments/run_dag_ladder.sh "$REPO/workload/sweep/$WL" \
       "${model}" "$OUT" > "$OUT/ladder.log" 2>&1 < /dev/null
rc=\$?
nj=\$(ls "$OUT"/dag_A*.json 2>/dev/null | wc -l)
echo "=== \$(date) END ${model} rc=\$rc jsons=\$nj/9 \$(( \$(date +%s)-t0 ))s"
rm -rf "\$RD"
SB
sbatch "$sb"
