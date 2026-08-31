#!/usr/bin/env bash
# Verify the REAL make_ramdir from common.sh (not a copy).  Written as a file
# rather than a nested bash -c: the nesting mangled the quoting last time and
# reported a false failure.
source /home/cw636/chenyi/attacc_drampim/output/_orch2/common.sh
R=$(cat /home/cw636/chenyi/attacc_drampim/output/_orch2/CURRENT_ROOT)
H=$(hostname -s)
RD=$(make_ramdir verifytest GPT-175B "$R/cachepool")
d=BAD; [ -d "$RD" ] && d=ok
r=BAD; [ -x "$RD/ramulator2" ] && r=ok
t=BAD; [ -e "$RD/trace_gen" ] && t=ok
c=$(wc -l < "$RD/signature_cache.jsonl" 2>/dev/null || echo 0)
o=BAD; [ -f "$RD/ramulator.out" ] && o=ok
echo "[$H] RD=$RD"
echo "[$H]   dir=$d ramulator=$r tracegen=$t ramulator.out=$o cache_lines=$c"
[ "$d$r$t" = "okokok" ] && [ "$c" -gt 0 ] && echo "[$H] MAKE_RAMDIR PASS" || echo "[$H] MAKE_RAMDIR FAIL"
rm -rf "$RD"
