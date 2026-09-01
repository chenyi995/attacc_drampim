#!/usr/bin/env bash
#SBATCH --job-name=a6probefix
#SBATCH --partition=athena
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=3-00:00:00 --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# A6-only verification sweep for the placement-probe head-folding fix.
# Eight cores: cells run SERIALLY, each with 8 Ramulator workers, so the
# persistent signature cache accumulates and later cells hit it.
set -uo pipefail
REPO=${REPO:?}
ROOT=${ROOT:?}
EXP=$REPO/experiments/a6_probefix_20260901

# Ramulator traces are big and bursty: keep them on node-local disk, not on
# the 99%-full shared home.  Seed the persistent signature cache once (serial
# drain, one writer) and copy it back at the end.
scratch_root() {
  local base=/localdata/kvpim_${USER}
  if mkdir -p "$base" 2>/dev/null && [ -w "$base" ]; then echo "$base"; else echo /tmp; fi
}
RD=$(scratch_root)/a6probefix_${SLURM_JOB_ID:-$$}
mkdir -p "$RD"
ln -sf "$REPO/ramulator2/ramulator2" "$RD/ramulator2"
ln -sf "$REPO/ramulator2/trace_gen"  "$RD/trace_gen"
cp "$REPO/ramulator.out" "$RD/ramulator.out" 2>/dev/null || :
if [ -s "$ROOT/signature_cache.jsonl" ]; then
  cp "$ROOT/signature_cache.jsonl" "$RD/signature_cache.jsonl"       # resume
else
  cp "$REPO/ramulator2/signature_cache.jsonl" "$RD/signature_cache.jsonl" 2>/dev/null || :
fi
export ATTACC_RAMULATOR_DIR=$RD
export ATTACC_RAMULATOR_LOG=$RD/ramulator.out
export RAMU_WORKERS=8

save_cache() { cp "$RD/signature_cache.jsonl" "$ROOT/.sig.tmp" 2>/dev/null \
                 && mv -f "$ROOT/.sig.tmp" "$ROOT/signature_cache.jsonl"; }
trap 'save_cache; rm -rf "$RD"' EXIT

echo "### a6 probe-fix sweep  root=$ROOT  ramdir=$RD  node=$(hostname -s)  $(date)"
mkdir -p "$ROOT"
FAILED=0
while read -r PRIO MODEL NAME WL K; do
  [ -z "${PRIO:-}" ] && continue
  case "$PRIO" in \#*) continue ;; esac
  [ -n "${ONLY_PRIO:-}" ] && [ "$PRIO" != "$ONLY_PRIO" ] && continue
  bash "$EXP/run_cell.sh" "$PRIO" "$MODEL" "$NAME" "$WL" "$K" "$ROOT" || FAILED=$((FAILED+1))
  save_cache
done < "$EXP/cells.txt"
echo "### DONE $(date)  failed_cells=$FAILED"
python3 "$EXP/collect.py" "$ROOT" > "$ROOT/SUMMARY.md" 2>&1 || :
