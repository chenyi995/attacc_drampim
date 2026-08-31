#!/usr/bin/env python3
"""Apply the two zero-memory, provably-identical hot-path fixes.

Both are pure overhead removal: they compute the same values by a cheaper
route, so every number in the report must be bit-identical afterwards.
"""
import sys
p = "src/workload_runner.py"
s = open(p).read()

# ---- Fix 1: token_offset was a linear scan, making locate() quadratic -------
old1 = '''    def token_offset(self, owner_row: int) -> int:
        try:
            return self.rows.index(owner_row)
        except ValueError as exc:
            raise WorkloadValidationError(
                "TLB row {} is not reserved in {}".format(owner_row, self.block_id)) from exc
'''
new1 = '''    def token_offset(self, owner_row: int) -> int:
        # ``rows`` is ``tuple(sorted(<set>))`` at every construction site
        # (CacheBlendTLB.finalize, the no-reuse finalize, and the naive-page
        # finalize whose ``page_rows`` is a slice of one), so it is ascending
        # and duplicate-free.  ``tuple.index`` therefore returns the same index
        # a binary search does -- but it scans, and ``locate`` visits every row
        # of a block once, so the scan made block binding O(L^2) per
        # (request, layer).  A contiguous block needs no search at all.
        rows = self.rows
        count = len(rows)
        if count:
            if rows[count - 1] - rows[0] == count - 1:
                offset = owner_row - rows[0]
                if 0 <= offset < count:
                    return offset
            else:
                offset = bisect_left(rows, owner_row)
                if offset < count and rows[offset] == owner_row:
                    return offset
        raise WorkloadValidationError(
            "TLB row {} is not reserved in {}".format(owner_row, self.block_id))
'''
assert s.count(old1) == 1, "fix1 anchor not unique"
s = s.replace(old1, new1)

# ---- Fix 3: a fresh empty set was allocated per location --------------------
old3 = '''                    private = [location for location in reads_by_request[request_id][0]
                               if _address_key(location) not in (common_keys or set())]
'''
new3 = '''                    # ``(common_keys or set())`` built a throwaway set for
                    # every location; hoist it, and inline the address key so
                    # the membership test is not a function call per row.
                    common_addresses = common_keys or frozenset()
                    private = [location for location in reads_by_request[request_id][0]
                               if (location.key_address, location.value_address)
                               not in common_addresses]
'''
assert s.count(old3) == 1, "fix3 anchor not unique"
s = s.replace(old3, new3)

open(p, "w").write(s)
print("applied fix1 (token_offset) + fix3 (common_keys hoist)")
