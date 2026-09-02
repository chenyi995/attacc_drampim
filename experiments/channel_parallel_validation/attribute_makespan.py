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

The scan shapes are DERIVED, not assumed: the workload and the reuse plan are
rebuilt here (same call main.py makes) and each request's context is turned
into its (master chunks, diff chunks) pair, then every class is weighted by
the work it contributes.  A3's ``single`` placement is exactly balanced for
any shape, which is why the solved link term does not depend on this at all.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.workload import (build_reuse_plan, load_workload,  # noqa: E402
                          workload_summary)
from src.workload_runner import (_layout_channel_loads,  # noqa: E402
                                 _heads_per_hbm)

HERE = Path(__file__).resolve().parent
# (kv_mapping, channel_placement) per rung, from src/ablation.py PRESETS.
POLICY = {'A3': 'single', 'A3a': 'single', 'A3b': 'slice',
          'A4': 'master-diff-slice', 'A4b': 'master-diff-table'}
KV_HEADS = 32                     # LLAMA-7B is MHA: 32 KV heads
PAGE_ROWS = 256                   # one chunk = one row on one channel
NUM_HBM = {'baseline_k8': 1, 'baseline_k8_hbm4': 4,
           'broadcast_k8': 1, 'reduce_k8': 1}
WORKLOAD = {'baseline_k8': 'wl_baseline_alltoall_N16_C32_D2',
            'baseline_k8_hbm4': 'wl_baseline_alltoall_N16_C32_D2',
            'broadcast_k8': 'wl_broadcast', 'reduce_k8': 'wl_reduce'}


def scan_shapes(case: str):
    """(c_master, c_diff) -> request count, for the case's workload.

    A decode scan reads the whole context, so master rows are everything the
    recompute plan did NOT mark as a correction.  ``build_reuse_plan`` is
    called exactly as main.py calls it for these runs.
    """
    root = Path(__file__).resolve().parents[2]
    workload = load_workload(
        root / 'workload' / 'sweep' / f'{WORKLOAD[case]}.json')
    plan = build_reuse_plan(workload, 'recompute', 0.0, 0, (), (), 8,
                            recompute_canonical=True)
    corrections = collections.Counter()
    for segment in workload_summary(workload, plan)['reuse']['reusable_segments']:
        corrections[segment['request']] += len(segment['epic_prefix_rows'])
    shapes = collections.Counter()
    for request in workload.requests:
        rows = request.total_length + request.history_len
        diff = corrections.get(request.request_id, 0)
        shapes[(-(-(rows - diff) // PAGE_ROWS), -(-diff // PAGE_ROWS))] += 1
    return shapes


def critical_share(rung: str, num_hbm: int, shapes):
    """Work-weighted max(loads)/sum(loads) over the case's request classes."""
    heads = _heads_per_hbm(KV_HEADS, num_hbm)
    busiest = total = 0.0
    for (c_master, c_diff), count in shapes.items():
        loads = _layout_channel_loads(POLICY[rung], c_master, c_diff, heads)
        busiest += count * max(loads)
        total += count * sum(loads)
    return busiest / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    rows = list(csv.DictReader(
        (HERE / 'results' / 'device_times.csv').open()))
    by_case = {}
    for row in rows:
        if row['rung'] in POLICY:
            by_case.setdefault(row['case'], {})[row['rung']] = row

    print('scan shapes derived from the workload + reuse plan; link solved '
          "from each case's A3 row, then held fixed\n")
    header = ('case', 'rung', 'share', 'gpu', 'die', 'link', 'pim_crit',
              'predicted', 'measured', 'residual')
    print('{:<17}{:<5}{:>8}{:>7}{:>6}{:>7}{:>10}{:>11}{:>10}{:>11}'.format(*header))

    for case, rungs in sorted(by_case.items()):
        if 'A3' not in rungs:
            continue
        num_hbm = NUM_HBM[case]
        shapes = scan_shapes(case)
        share = critical_share('A3', num_hbm, shapes)
        base = rungs['A3']
        link = (float(base['makespan_s']) - float(base['gpu_time_s_unoverlapped'])
                - float(base['die_time_s_unoverlapped'])
                - float(base['pim_pool_time_s_unoverlapped']) * share)
        for rung in sorted(rungs):
            row = rungs[rung]
            share = critical_share(rung, num_hbm, shapes)
            gpu = float(row['gpu_time_s_unoverlapped'])
            die = float(row['die_time_s_unoverlapped'])
            crit = float(row['pim_pool_time_s_unoverlapped']) * share
            predicted = gpu + die + link + crit
            measured = float(row['makespan_s'])
            flag = ' <- link solved here' if rung == 'A3' else ''
            print('{:<17}{:<5}{:>8.4f}{:>7.2f}{:>6.2f}{:>7.2f}{:>10.2f}'
                  '{:>11.2f}{:>10.2f}{:>+10.1f}%{}'.format(
                      case, rung, share, gpu, die, link, crit, predicted,
                      measured, (predicted - measured) / measured * 100, flag))
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
