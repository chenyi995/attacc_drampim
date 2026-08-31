#!/usr/bin/env bash
#SBATCH --job-name=ver_checks
#SBATCH --partition=athena-small
#SBATCH --nodelist=node4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=06:00:00 --open-mode=append
#SBATCH --output=/home/cw636/chenyi/attacc_drampim/output/_verify/checks.out
#SBATCH --error=/home/cw636/chenyi/attacc_drampim/output/_verify/checks.out
set -uo pipefail
REPO=/home/cw636/chenyi/attacc_drampim; cd $REPO
export OMP_NUM_THREADS=1 KVPIM_CPPCORE=1
echo "=== unit tests (patched tree) ==="
python3 -m unittest discover -s tests > $REPO/output/_verify/unittest.txt 2>&1
echo "unittest rc=$?"; grep -aE "^Ran |^OK|^FAILED" $REPO/output/_verify/unittest.txt | tail -3
echo "=== rows invariant on real data (LLAMA3-8B / wl_pipeline_D4 / A1) ==="
RD=/tmp/kvpim_${USER}_inv_$$; rm -rf "$RD"; mkdir -p "$RD"
ln -sf "$REPO/ramulator2/ramulator2" "$RD/ramulator2"
ln -sf "$REPO/ramulator2/trace_gen"  "$RD/trace_gen"
cp "$REPO/ramulator.out" "$RD/ramulator.out" 2>/dev/null || :
cp "$REPO/output/_verify/snapshot.jsonl" "$RD/signature_cache.jsonl"
export ATTACC_RAMULATOR_DIR="$RD" ATTACC_RAMULATOR_LOG="$RD/ramulator.out"
python3 output/_verify/check_rows_invariant.py LLAMA3-8B 1 1 wl_pipeline_D4.json A1 \
  2>&1 | tail -4 | tee $REPO/output/_verify/rows.inv
rm -rf "$RD"
echo "CHECKS DONE"
