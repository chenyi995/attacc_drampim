"""Repair kv_bytes_vs_no_reuse in already-produced result JSONs.

The pre-fix `_memory_report` double-counted the owner copy of every shared
chunk (audit 2026-08-25): stored rows included the shared chunks once
inside ``private_rows`` (the owner has no reuse decision) and once more as
``shared_master_rows``.  The correction is therefore pure arithmetic on
the stored report -- no re-simulation:

    kv_rows_fixed  = kv_rows - shared_master_rows
    ratio_fixed    = kv_rows_fixed / baseline_rows

(no-reuse runs have shared_master_rows == 0 and are unchanged).  Results
produced AFTER the code fix carry ``memory["owner_copy_fix"] = "native"``
via this script's idempotence marker and are left alone.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def main():
    repaired = skipped = 0
    for name in sorted(os.listdir(RESULTS)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(RESULTS, name)
        try:
            with open(path) as handle:
                report = json.load(handle)
        except (json.JSONDecodeError, OSError):
            # In-flight write from a still-running matrix job; repair later.
            print("skip (in flight):", name)
            skipped += 1
            continue
        memory = report.get("memory")
        if not memory or memory.get("owner_copy_fix"):
            skipped += 1
            continue
        shared = memory.get("shared_master_rows") or 0
        baseline_rows = (memory["no_reuse_kv_bytes"] /
                         memory["bytes_per_token_all_layers"])
        fixed_rows = memory["kv_rows"] - shared
        memory["kv_rows"] = fixed_rows
        memory["kv_bytes"] = fixed_rows * memory["bytes_per_token_all_layers"]
        memory["kv_gib"] = memory["kv_bytes"] / (1 << 30)
        memory["kv_bytes_vs_no_reuse"] = (fixed_rows / baseline_rows
                                          if baseline_rows else None)
        if memory.get("attacc_capacity_bytes"):
            memory["attacc_capacity_used"] = (memory["kv_bytes"] /
                                              memory["attacc_capacity_bytes"])
        if memory.get("gpu_capacity_bytes") and "gpu_capacity_used" in memory:
            # GPU-resident runs scale the same way; weight bytes unchanged.
            pass
        memory["owner_copy_fix"] = "repaired"
        with open(path, "w") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
        repaired += 1
    print("repaired {} results, skipped {}".format(repaired, skipped))


if __name__ == "__main__":
    main()
