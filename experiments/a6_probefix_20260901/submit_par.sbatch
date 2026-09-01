#!/usr/bin/env bash
#SBATCH --job-name=a6par
#SBATCH --partition=athena
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=240G
#SBATCH --time=3-00:00:00 --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Parallel driver for the A6 probe-fix verification sweep.
#
# Runs PAR cells at a time instead of the strictly serial submit.sbatch.  It
# cooperates with the serial job (192054) and with any other copy of itself:
# run_cell.sh takes an ATOMIC `mkdir` CLAIM per cell, so no cell is ever run
# twice, and a cell already on disk is skipped.
#
#   sbatch --export=ALL,REPO=...,ROOT=...,PAR=3 submit_par.sbatch
set -uo pipefail
REPO=${REPO:?}
ROOT=${ROOT:?}
EXP=$REPO/experiments/a6_probefix_20260901
PAR=${PAR:-3}
export RAMU_WORKERS=${RAMU_WORKERS:-6}

scratch_root() {
  local base=/localdata/kvpim_${USER}
  if mkdir -p "$base" 2>/dev/null && [ -w "$base" ]; then echo "$base"; else echo /tmp; fi
}
RD=$(scratch_root)/a6par_${SLURM_JOB_ID:-$$}
mkdir -p "$RD" "$ROOT/sigcache"
ln -sf "$REPO/ramulator2/ramulator2" "$RD/ramulator2"
ln -sf "$REPO/ramulator2/trace_gen"  "$RD/trace_gen"
cp "$REPO/ramulator.out" "$RD/ramulator.out" 2>/dev/null || :

# Seed the signature cache from EVERY cache we know of: the repo's, the
# serial job's save, and any earlier parallel job's.  Concurrent cells then
# share this one file (append-only, atomic O_APPEND lines).
cat "$REPO/ramulator2/signature_cache.jsonl" \
    "$ROOT/signature_cache.jsonl" \
    "$ROOT"/sigcache/*.jsonl 2>/dev/null | sort -u > "$RD/signature_cache.jsonl"
export ATTACC_RAMULATOR_DIR=$RD
export ATTACC_RAMULATOR_LOG=$RD/ramulator.out   # seed only; run_cell.sh forks a private copy

# Own save path: the serial job owns $ROOT/signature_cache.jsonl and would be
# clobbered by an mv from here.
MINE=$ROOT/sigcache/par-${SLURM_JOB_ID:-$$}.jsonl
save_cache() { cp "$RD/signature_cache.jsonl" "$MINE.tmp" 2>/dev/null && mv -f "$MINE.tmp" "$MINE"; }
trap 'save_cache; rm -rf "$RD"' EXIT
( while sleep 300; do save_cache; done ) & SAVER=$!

echo "### a6 PARALLEL sweep  par=$PAR  workers=$RAMU_WORKERS  root=$ROOT  ramdir=$RD  node=$(hostname -s)  $(date)"
mkdir -p "$ROOT"

# Work order: the k-hi cells (k=32, the heaviest -- ~2 h each) start FIRST so
# they are not left as a tail; everything else keeps cells.txt order.
WORK=$RD/work.txt
norm() { awk 'NF && $1 !~ /^#/ {print $1, $2, $3, $4, $5}' "$EXP/cells.txt"; }
norm | awk '$3=="k-hi"' >  "$WORK"
norm | awk '$3!="k-hi"' >> "$WORK"
echo "### $(wc -l < "$WORK") cells queued (done/claimed ones exit immediately)"

export ROOT EXP
xargs -P "$PAR" -L1 bash -c 'bash "$EXP/run_cell.sh" "$@" "$ROOT"' _ < "$WORK"
RC=$?

kill "$SAVER" 2>/dev/null
save_cache
echo "### DONE $(date)  xargs_rc=$RC"
python3 "$EXP/collect.py" "$ROOT" > "$ROOT/SUMMARY.md" 2>&1 || :
