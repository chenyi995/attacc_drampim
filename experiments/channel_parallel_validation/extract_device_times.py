#!/usr/bin/env python3
"""Pull the per-device unoverlapped busy sums out of the raw dag_*.json.

These four scalars are all that is needed to attribute a makespan under
``pipe=False``, where every macro event is serial:

    makespan = gpu + die + link + pim_scan_critical_path

``pim_pool_time_s_unoverlapped`` is the sum of every PIM:* lane's duration
(src/workload_runner.py), i.e. the TOTAL scan work, not the critical path --
the placement decides what fraction of it the busiest channel carries.

The reports are 50-115 MB and stay local (docs/RAW_DATA_MANIFEST.md), so this
greps the scalars (they are one per line, sort_keys=True) into a small CSV
that IS tracked, and attribute_makespan.py works from that.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parents[1] / 'output' / 'channel_parallel_validation_20260902'
FIELDS = ('makespan_s', 'gpu_time_s_unoverlapped', 'die_time_s_unoverlapped',
          'pim_pool_time_s_unoverlapped')


def scalars(path: Path):
    """First occurrence of each top-level scalar, without parsing 100 MB."""
    wanted = {f'"{name}"': name for name in FIELDS}
    found = {}
    pattern = re.compile(r'^\s*"([a-z_0-9]+)":\s*([0-9.eE+-]+),?\s*$')
    with path.open() as handle:
        for line in handle:
            match = pattern.match(line)
            if match and f'"{match.group(1)}"' in wanted and \
                    match.group(1) not in found:
                found[match.group(1)] = float(match.group(2))
                if len(found) == len(FIELDS):
                    break
    return found


def main() -> int:
    rows = []
    for report in sorted(OUT_ROOT.glob('*/*/dag_*.json')):
        values = scalars(report)
        if len(values) != len(FIELDS):
            continue
        rows.append({'model': report.parents[1].name,
                     'case': report.parent.name,
                     'rung': report.stem.replace('dag_', ''),
                     **{name: values[name] for name in FIELDS}})

    destination = HERE / 'results' / 'device_times.csv'
    with destination.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {destination} ({len(rows)} reports)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
