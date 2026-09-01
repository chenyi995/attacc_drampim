#!/usr/bin/env bash
# Extract the sweep's measured results from the raw dag_*.json into CSV.
#
#   bash output/analysis/extract_sweep.sh [sweep-root] [jobs]
#
# Read-only with respect to the sweep.  Safe to run while the backfill is still
# going: it reports fewer complete tasks and sweep_completeness.csv records
# exactly which rungs were present at extraction time.
#
# HOW LONG: the reports total ~99 GB across 667 files and the floor is NFS read
# bandwidth (~78 MB/s), so a cold run is roughly 20 min at jobs=1 and a few
# minutes at the default jobs=6.  A cache keyed by (path, mtime, size) makes a
# re-run after the backfill nearly free -- only the rungs that changed are read
# again.  Run it under nohup if you would rather not hold a terminal:
#
#   nohup bash output/analysis/extract_sweep.sh > /tmp/extract.log 2>&1 &
#
# Refuses to finish if a sample of CSV values disagrees with the JSON they came
# from, so a mangled extraction cannot be committed by accident.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
ROOT=${1:-$(cat output/_orch2/CURRENT_ROOT 2>/dev/null)}
JOBS=${2:-6}
OUT=$REPO/output/analysis

if [ -z "${ROOT:-}" ] || [ ! -d "$ROOT" ]; then
  echo "usage: bash output/analysis/extract_sweep.sh [sweep-root] [jobs]" >&2
  echo "  no sweep root given and output/_orch2/CURRENT_ROOT is unusable" >&2
  exit 1
fi

echo "=== extracting $(date '+%F %T') ==="
python3 "$OUT/extract_sweep_csv.py" --root "$ROOT" --outdir "$OUT" --jobs "$JOBS" || exit 1

echo
echo "=== files ==="
for f in sweep_rungs.csv sweep_tiers.csv sweep_completeness.csv; do
  [ -f "$OUT/$f" ] || { echo "  MISSING $f" >&2; exit 1; }
  printf "  %-26s %10s bytes  sha256 %s\n" "$f" \
         "$(stat -c %s "$OUT/$f")" "$(sha256sum "$OUT/$f" | cut -c1-16)"
done

echo
echo "=== spot check: CSV values re-read from the JSON they came from ==="
python3 - "$ROOT" "$OUT/sweep_rungs.csv" <<'PY'
import csv, json, os, random, sys
root, csv_path = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(csv_path)))
random.seed(0)
# Sample rather than sweep: re-reading every report would cost another full
# pass over 99 GB, and a systematic extraction bug shows up in any sample.
sample = random.sample(rows, min(12, len(rows)))
checked = bad = 0
for r in sample:
    p = f"{root}/{r['model']}/{r['config']}_k{r['k']}/dag_{r['rung']}.json"
    if not os.path.exists(p):
        continue
    j = json.load(open(p))
    for col in ("makespan_s", "energy_nj", "link_bytes", "event_count"):
        a, b = r[col], j.get(col)
        if a == "" or b is None:
            continue
        checked += 1
        if abs(float(a) - float(b)) > 1e-9 * max(1.0, abs(float(b))):
            bad += 1
            print(f"  MISMATCH {p} {col}: csv={a} json={b}")
    tiers = (j.get("summary") or {}).get("tiers") or {}
    if str(len(tiers)) != r["tiers"]:
        bad += 1
        print(f"  MISMATCH {p} tiers: csv={r['tiers']} json={len(tiers)}")
    checked += 1
print(f"  {checked} values from {len(sample)} reports re-read, {bad} mismatched")
sys.exit(1 if bad else 0)
PY
rc=$?
[ "$rc" = 0 ] || { echo "  SPOT CHECK FAILED -- do not commit these CSVs" >&2; exit 1; }

echo
echo "=== done.  to commit ==="
echo "  git add output/analysis/sweep_rungs.csv output/analysis/sweep_tiers.csv \\"
echo "          output/analysis/sweep_completeness.csv"
echo "  git commit -m 'data: sweep results extracted from the raw dag_*.json'"
