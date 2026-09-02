#!/usr/bin/env python3
"""Collect REPORT_SUMMARY rows for the channel-parallel validation.

The raw ``dag_*.json`` event traces are 50-115 MB each and stay local (see
``docs/RAW_DATA_MANIFEST.md``); the one line that matters is the
``REPORT_SUMMARY`` the run prints at the end.  This reads that line, so it
costs a grep instead of a 100 MB JSON parse.

The pre-fix (serial-PIM) reference is the collaborator's 2026-08-30 sweep on
this cluster; the post-fix runs are this experiment's own output tree.  Rows
are matched on (case, rung); a cell with no comparable pre-fix run (the
``_hbm4`` cells -- the old sweep only ever ran one HBM) is emitted with the
old/speedup columns empty rather than compared across configurations.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Pre-fix reference: serial PIM scans, one HBM, same workloads and same
# --epic-prefix-recompute-tokens 8.
OLD_ROOT = Path('/home/cw636/chenyi/attacc_drampim/output/'
                'sweep_models_20260830-163226')
NEW_ROOT = Path(__file__).resolve().parents[2] / 'output' / \
    'channel_parallel_validation_20260902'
LOG_DIR = Path(__file__).resolve().parent / 'logs'

FIELDS = ('makespan_s', 'energy_nj', 'link_bytes', 'event_count',
          'kv_mapping', 'pim_prefill_mode')
SUMMARY_RE = re.compile(r'^REPORT_SUMMARY (\{.*\})\s*$', re.MULTILINE)


def summary_from_log(path: Path):
    """Last REPORT_SUMMARY in a run log, or None."""
    if not path.is_file():
        return None
    matches = SUMMARY_RE.findall(path.read_text(errors='replace'))
    return json.loads(matches[-1]) if matches else None


def new_summaries():
    """(case, rung) -> summary for this experiment's slurm logs.

    The new runs write ``dag_<rung>.json`` next to no log of their own, so the
    slurm log is the record; the 'Wrote ablation report to' line names the
    case directory and the rung, which is what the summary is keyed on.
    """
    wrote = re.compile(r'Wrote ablation report to .*/'
                       r'channel_parallel_validation_20260902/'
                       r'(?P<model>[^/]+)/(?P<case>[^/]+)/dag_(?P<rung>[^.]+)\.json')
    found = {}
    for log in sorted(LOG_DIR.glob('*.log')):
        text = log.read_text(errors='replace')
        target = wrote.search(text)
        summary = summary_from_log(log)
        if target is None or summary is None:
            continue                      # cancelled, crashed or still running
        key = (target['model'], target['case'], target['rung'])
        found[key] = (summary, log.name)  # a rerun's log sorts later and wins
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='LLAMA-7B')
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).resolve().parent / 'results' /
                        'channel_parallel_llama7b.csv')
    args = parser.parse_args()

    rows = []
    for (model, case, rung), (new, log_name) in sorted(new_summaries().items()):
        if model != args.model:
            continue
        # The pre-fix sweep only ever ran one HBM, so an _hbm4 cell has no
        # comparable old run: --num-hbm changes the placement, the event count
        # and the device energy, not just the schedule.  Leave those columns
        # empty rather than print a cross-configuration ratio.
        old = (None if case.endswith('_hbm4') else
               summary_from_log(OLD_ROOT / model / case / f'dag_{rung}.log'))
        row = {'case': case, 'rung': rung, 'slurm_log': log_name,
               'new_makespan_s': new['makespan_s'],
               'old_makespan_s': old['makespan_s'] if old else '',
               'speedup_x': (old['makespan_s'] / new['makespan_s']
                             if old and new['makespan_s'] else ''),
               'new_energy_nj': new['energy_nj'],
               'kv_mapping': new['kv_mapping'],
               'pim_prefill_mode': new['pim_prefill_mode'],
               'event_count': new['event_count']}
        rows.append(row)

    if not rows:
        print('no finished runs found under {}'.format(LOG_DIR), file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    width = max(len(f"{r['case']}/{r['rung']}") for r in rows)
    print(f"{'case/rung':<{width}}  {'old_s':>10}  {'new_s':>10}  {'x':>7}")
    for row in rows:
        speedup = f"{row['speedup_x']:.2f}" if row['speedup_x'] != '' else '-'
        old = f"{row['old_makespan_s']:.3f}" if row['old_makespan_s'] != '' else '-'
        print(f"{row['case'] + '/' + row['rung']:<{width}}  {old:>10}  "
              f"{row['new_makespan_s']:>10.3f}  {speedup:>7}")
    print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
