#!/usr/bin/env python3
"""Rebuild results/report_summaries.txt from the local slurm logs.

The logs themselves are raw run data and stay on the machine that ran them
(docs/RAW_DATA_MANIFEST.md); the two lines per run that carry the result --
where the report was written and the REPORT_SUMMARY -- are copied into the
repo so the numbers in RESULTS.md have a checked-in provenance.

An A3b run whose makespan equals the A3 of the same case bit for bit did not
run A3b: its slice stripe collapsed to one channel per head and the placement
became A3's.  Such an entry is kept, because it is the evidence for the
warning at the top of RESULTS.md, but it is marked INVALID so nobody lifts
the number out of this file as an A3b measurement.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_PREFIX = '/home/xw338/attacc/attacc_drampim/'
WROTE_RE = re.compile(r'Wrote ablation report to .*/'
                      r'channel_parallel_validation_20260902/'
                      r'(?P<model>[^/]+)/(?P<case>[^/]+)/dag_(?P<rung>[^.]+)\.json')

HEADER = """# REPORT_SUMMARY / target lines extracted from the slurm logs of this experiment.
# Full logs are raw run data and stay local (docs/RAW_DATA_MANIFEST.md):
#   experiments/channel_parallel_validation/logs/<jobid>_<task>.log
# Regenerate: python3 make_report_summaries.py
#
# An entry marked INVALID is kept as evidence of a defective RUN, not as a
# result: its A3b makespan equals the A3 of the same case bit for bit, which
# means the slice stripe collapsed and the run actually measured A3.
# See RESULTS.md, the warning at the top.
"""


def natural_key(path: Path):
    """193281_2.log after 193109_10.log, and _2 after _10 within a job."""
    return [int(part) if part.isdigit() else part
            for part in re.split(r'(\d+)', path.name)]


def main() -> int:
    entries = []
    for log in sorted((HERE / 'logs').glob('*.log'), key=natural_key):
        text = log.read_text(errors='replace')
        kept = [line for line in text.splitlines()
                if line.startswith('REPORT_SUMMARY') or
                'Wrote ablation report to' in line]
        if not kept:
            continue                  # cancelled, crashed or still running
        target = WROTE_RE.search(text)
        summary = json.loads(kept[-1].split(' ', 1)[1])
        entries.append({'log': log.name,
                        'case': target.group('case') if target else '?',
                        'rung': target.group('rung') if target else '?',
                        'makespan_s': summary['makespan_s'],
                        'lines': kept})

    a3 = {entry['case']: entry['makespan_s']
          for entry in entries if entry['rung'] == 'A3'}

    out = [HEADER]
    for entry in entries:
        invalid = (entry['rung'] == 'A3b' and
                   a3.get(entry['case']) == entry['makespan_s'])
        flag = ('  <-- INVALID: A3b degenerated to A3 (identical makespan); '
                'not a result' if invalid else '')
        out.append('## {}{}'.format(entry['log'], flag))
        out += [line.replace(REPO_PREFIX, '') for line in entry['lines']]
        out.append('')

    destination = HERE / 'results' / 'report_summaries.txt'
    destination.write_text('\n'.join(out))
    print('wrote {} ({} runs, {} marked INVALID)'.format(
        destination, len(entries),
        sum(1 for e in entries if e['rung'] == 'A3b' and
            a3.get(e['case']) == e['makespan_s'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
