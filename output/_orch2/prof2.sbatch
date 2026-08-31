#!/usr/bin/env bash
#SBATCH --job-name=prof_full
#SBATCH --partition=athena
#SBATCH --nodelist=node2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --time=12:00:00 --open-mode=append
#SBATCH --output=/home/cw636/chenyi/attacc_drampim/output/_prof/prof2.out
#SBATCH --error=/home/cw636/chenyi/attacc_drampim/output/_prof/prof2.out
set -uo pipefail
# Correct methodology this time:
#  (1) sample the WHOLE run, not a 900 s prefix -- the prefix over-weighted the
#      TLB binding, which is why token_offset looked like 14.6% when removing
#      it only bought 5.6%;
#  (2) mark phase boundaries in the log so the profile can be split into
#      TLB-binding / prefill-emission / decode-loop rather than read as one blob.
export PATH=$HOME/.local/bin:$PATH
REPO=/home/cw636/chenyi/attacc_drampim; cd $REPO
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KVPIM_CPPCORE=1
RD=/tmp/kvpim_${USER}_prof2_$$; rm -rf "$RD"; mkdir -p "$RD"
ln -sf "$REPO/ramulator2/ramulator2" "$RD/ramulator2"
ln -sf "$REPO/ramulator2/trace_gen"  "$RD/trace_gen"
cp "$REPO/ramulator.out" "$RD/ramulator.out" 2>/dev/null || :
cp "$REPO/output/_verify/snapshot.jsonl" "$RD/signature_cache.jsonl"
export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
echo "=== FULL-RUN sample: LLAMA-7B / wl_D1 / A1 (build was 3667 s after the patch) ==="
t0=$(date +%s)
py-spy record --format raw --rate 25 --duration 20000 --subprocesses \
  -o $REPO/output/_prof/full_raw.txt -- \
  python3 main.py --system dgx-attacc --model LLAMA-7B --ngpu 1 --num-hbm 1 \
    --workload workload/sweep/wl_D1.json --reuse no-reuse \
    --ablation A1 --engine dag --ramulator-workers 2 --cacheblend-batch-size 8 \
    --workload-report-events none --workload-report $REPO/output/_prof/full.json
rc=$?
echo "PROF2 rc=$rc $(( $(date +%s)-t0 ))s raw_lines=$(wc -l < $REPO/output/_prof/full_raw.txt 2>/dev/null)"
rm -rf "$RD"
