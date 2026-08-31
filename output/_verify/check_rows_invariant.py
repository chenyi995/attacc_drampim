#!/usr/bin/env python3
"""Prove on REAL data that bisect_left == tuple.index for every KVBlock.

The token_offset fix is only valid if ``rows`` is ascending and duplicate-free
at every construction site.  That was read out of the code; this checks it on
the actual blocks a real run builds, without modifying the engine: KVBlock is
wrapped in this process only.

  usage: check_rows_invariant.py <model> <ngpu> <num_hbm> <workload> <rung>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bisect import bisect_left
from src import workload_runner as wr

_orig_init = wr.KVBlock.__init__
STATS = {"blocks": 0, "rows": 0, "unsorted": 0, "dup": 0, "mismatch": 0,
         "contiguous": 0}


def _checked_init(self, *a, **k):
    _orig_init(self, *a, **k)
    rows = self.rows
    STATS["blocks"] += 1
    STATS["rows"] += len(rows)
    if list(rows) != sorted(rows):
        STATS["unsorted"] += 1
    if len(set(rows)) != len(rows):
        STATS["dup"] += 1
    if rows and rows[-1] - rows[0] == len(rows) - 1:
        STATS["contiguous"] += 1
    # "ascending and duplicate-free" mathematically implies
    # bisect_left(rows, r) == rows.index(r) for every r, so the invariant is
    # the whole proof; an exhaustive per-row check would itself be O(L^2).
    # Spot-check a bounded sample anyway, to catch a wrong invariant test.
    step = max(1, len(rows) // 32)
    for r in rows[::step]:
        if bisect_left(rows, r) != rows.index(r):
            STATS["mismatch"] += 1


wr.KVBlock.__init__ = _checked_init

model, ngpu, hbm, wl, rung = sys.argv[1:6]
sys.argv = ["main.py", "--system", "dgx-attacc", "--model", model,
            "--ngpu", ngpu, "--num-hbm", hbm,
            "--workload", "workload/sweep/" + wl,
            "--reuse", "no-reuse" if rung == "A1" else "recompute",
            "--ablation", rung, "--engine", "dag",
            "--ramulator-workers", "2", "--cacheblend-batch-size", "8",
            "--workload-report-events", "none",
            "--workload-report", "/dev/null"]
if rung != "A1":
    sys.argv[sys.argv.index("--reuse") + 2:sys.argv.index("--reuse") + 2] = \
        ["--epic-prefix-recompute-tokens", "8"]
import main  # noqa: E402  (runs under __name__ != "__main__")
try:
    main.main()
except SystemExit:
    pass
print("\nROWS INVARIANT over {blocks} blocks / {rows} rows: "
      "unsorted={unsorted} duplicated={dup} bisect!=index={mismatch} "
      "contiguous_blocks={contiguous}".format(**STATS))
print("VERDICT: " + ("invariant holds -- bisect_left is exactly tuple.index"
                     if STATS["unsorted"] == STATS["dup"] == STATS["mismatch"] == 0
                     else "INVARIANT VIOLATED -- the fix would NOT be identical"))
