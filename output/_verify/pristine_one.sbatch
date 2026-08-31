#!/usr/bin/env bash
#SBATCH --job-name=ver_pristine
#SBATCH --partition=athena
#SBATCH --nodelist=node3
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6 --mem=120G
#SBATCH --time=10:00:00 --open-mode=append
#SBATCH --output=/home/cw636/chenyi/attacc_drampim/output/_verify/pristine.out
#SBATCH --error=/home/cw636/chenyi/attacc_drampim/output/_verify/pristine.out
set -uo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
P=$REPO/output/_verify/pristine
OUT=$REPO/output/_verify/before
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 KVPIM_CPPCORE=1
RD=/tmp/kvpim_${USER}_pristine_A1; rm -rf "$RD"; mkdir -p "$RD"
ln -sf "$REPO/ramulator2/ramulator2" "$RD/ramulator2"
ln -sf "$REPO/ramulator2/trace_gen"  "$RD/trace_gen"
cp "$REPO/ramulator.out" "$RD/ramulator.out" 2>/dev/null || :
cp "$REPO/output/_verify/snapshot.jsonl" "$RD/signature_cache.jsonl"
export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
t0=$(date +%s)
( cd "$P" && timeout 32400 python3 main.py --system dgx-attacc --model LLAMA-7B \
    --ngpu 1 --num-hbm 1 --workload workload/sweep/wl_D1.json --reuse no-reuse \
    --ablation A1 --engine dag --ramulator-workers 2 --cacheblend-batch-size 8 \
    --workload-report-events none \
    --workload-report "$REPO/output/_verify/before_w2_A1.json" ) \
  > "$REPO/output/_verify/before_w2_A1.log" 2>&1
rc=$?
echo "[$(date +%F' '%T)] PRISTINE-W2 LLAMA-7B_wl_D1_evnone A1 rc=$rc $(( $(date +%s)-t0 ))s"
[ "$rc" = 0 ] && echo "W2 RERUN DONE" || echo "PRISTINE BASELINE FAILED rc=$rc"
rm -rf "$RD"
