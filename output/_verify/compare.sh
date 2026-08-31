#!/usr/bin/env bash
# Streaming identity check.  The full-event reports are ~25 GB each, so bytes
# are compared with cmp (constant memory).  Identical data structures serialise
# to identical bytes, so byte equality IS bit-identity; only a mismatch needs
# the Python walk, which applies the one documented exclusion
# (ramulator_signature_cache = memoisation counters, not a simulation result).
set -uo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
A=$REPO/output/_verify/${1:-before}; B=$REPO/output/_verify/${2:-after}
ok=0; diff=0; missing=0; total=0
for f in "$A"/*.json; do
  n=$(basename "$f"); b="$B/$n"; total=$((total+1))
  if [ ! -f "$b" ]; then echo "  MISSING in $(basename $B): $n"; missing=$((missing+1)); continue; fi
  sa=$(stat -c %s "$f"); sb=$(stat -c %s "$b")
  if [ "$sa" != "$sb" ]; then
    echo "  SIZE DIFF $n: $sa vs $sb"; diff=$((diff+1)); continue
  fi
  if cmp -s "$f" "$b"; then
    ok=$((ok+1))
  else
    off=$(cmp "$f" "$b" 2>&1 | head -1)
    echo "  BYTE DIFF $n: $off"
    if [ "$sa" -lt 209715200 ]; then
      python3 - "$f" "$b" <<'PY'
import json, sys
EXCL = {"ramulator_signature_cache"}
def walk(a, b, path, out, lim=8):
    if len(out) >= lim: return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in EXCL: continue
            if k not in a or k not in b: out.append(f"{path}.{k}: present in only one")
            else: walk(a[k], b[k], f"{path}.{k}", out, lim)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): out.append(f"{path}: len {len(a)} vs {len(b)}"); return
        for i,(x,y) in enumerate(zip(a,b)): walk(x, y, f"{path}[{i}]", out, lim)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); out=[]
walk(a,b,"",out)
print("      excluding the cache-stats block, real differences:", len(out))
for d in out[:8]: print("       ", d)
PY
    fi
    diff=$((diff+1))
  fi
done
echo
echo "byte-identical: $ok   differing: $diff   missing: $missing   (of $total in $(basename $A))"
if [ "$diff" = 0 ] && [ "$missing" = 0 ]; then echo "VERDICT: BIT-IDENTICAL"; else echo "VERDICT: NOT IDENTICAL"; fi
