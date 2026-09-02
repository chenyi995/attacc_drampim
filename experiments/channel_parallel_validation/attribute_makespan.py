#!/usr/bin/env python3
"""Why is A4 slower than A3?  Attribute each makespan to its four terms.

Without ``--pipeopt`` every macro event is serial, so

    makespan = gpu + die + link + pim_scan_critical

and the only term a placement policy can move is the last one.  The reports
record the TOTAL PIM lane work (``pim_pool_time_s_unoverlapped``, the sum over
every PIM:* lane); the placement decides what share of it the busiest channel
carries, which is exactly ``max(loads) / sum(loads)`` from the load model in
``src/workload_runner.py``.  So

    pim_scan_critical = pool_sum * max(loads) / sum(loads)

``link`` is not reported separately.  It is solved per case from the A3 row --
A3's ``single`` placement is perfectly balanced here (32 heads over 16
channels, two each; or 8 heads over 16, one each), so its critical share is
known exactly -- and then held fixed to PREDICT every other rung in that case.
The other rungs are therefore out-of-sample: the residual column is a real
test of the load model, not a fit.

Scan shape for these runs (workload/sweep/wl_*.json, 256-token chunks):
the reused corpus is 8192 doc tokens = 32 master chunks, and
``--epic-prefix-recompute-tokens 8`` over 32 reused blocks recomputes 256
tokens = 1 diff chunk.  Override with --c-master / --c-diff to see the
sensitivity.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.workload_runner import (_layout_channel_loads,  # noqa: E402
                                 _heads_per_hbm)

HERE = Path(__file__).resolve().parent
# (kv_mapping, channel_placement) per rung, from src/ablation.py PRESETS.
POLICY = {'A3': 'single', 'A3a': 'single', 'A3b': 'slice',
          'A4': 'master-diff-slice', 'A4b': 'master-diff-table'}
KV_HEADS = 32                     # LLAMA-7B is MHA: 32 KV heads
NUM_HBM = {'baseline_k8': 1, 'baseline_k8_hbm4': 4,
           'broadcast_k8': 1, 'reduce_k8': 1}


def critical_share(rung: str, num_hbm: int, c_master: int, c_diff: int):
    """max(loads)/sum(loads): the busiest channel's share of one scan."""
    heads = _heads_per_hbm(KV_HEADS, num_hbm)
    loads = _layout_channel_loads(POLICY[rung], c_master, c_diff, heads)
    return max(loads) / sum(loads), max(loads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--c-master', type=int, default=32)
    parser.add_argument('--c-diff', type=int, default=1)
    args = parser.parse_args()

    rows = list(csv.DictReader(
        (HERE / 'results' / 'device_times.csv').open()))
    by_case = {}
    for row in rows:
        if row['rung'] in POLICY:
            by_case.setdefault(row['case'], {})[row['rung']] = row

    print('c_master={} c_diff={}   (link solved from each case\'s A3, then held '
          'fixed)\n'.format(args.c_master, args.c_diff))
    header = ('case', 'rung', 'busiest', 'gpu', 'die', 'link', 'pim_crit',
              'predicted', 'measured', 'residual')
    print('{:<17}{:<5}{:>8}{:>7}{:>6}{:>7}{:>10}{:>11}{:>10}{:>11}'.format(*header))

    for case, rungs in sorted(by_case.items()):
        if 'A3' not in rungs:
            continue
        num_hbm = NUM_HBM[case]
        share, _ = critical_share('A3', num_hbm, args.c_master, args.c_diff)
        base = rungs['A3']
        link = (float(base['makespan_s']) - float(base['gpu_time_s_unoverlapped'])
                - float(base['die_time_s_unoverlapped'])
                - float(base['pim_pool_time_s_unoverlapped']) * share)
        for rung in sorted(rungs):
            row = rungs[rung]
            share, busiest = critical_share(rung, num_hbm,
                                            args.c_master, args.c_diff)
            gpu = float(row['gpu_time_s_unoverlapped'])
            die = float(row['die_time_s_unoverlapped'])
            crit = float(row['pim_pool_time_s_unoverlapped']) * share
            predicted = gpu + die + link + crit
            measured = float(row['makespan_s'])
            flag = ' <- link solved here' if rung == 'A3' else ''
            print('{:<17}{:<5}{:>8.0f}{:>7.2f}{:>6.2f}{:>7.2f}{:>10.2f}'
                  '{:>11.2f}{:>10.2f}{:>+10.1f}%{}'.format(
                      case, rung, busiest, gpu, die, link, crit, predicted,
                      measured, (predicted - measured) / measured * 100, flag))
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
