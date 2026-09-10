#!/usr/bin/env python3
"""Check freshly computed request results and redraw only the published figures."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('TTFT_ms', 'TBT_ms', 'E2E_ms', 'simultaneous_peak_KV_GiB')
VARIANTS = ('F0', 'F1', 'F2', 'F4')


def table(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def csvout(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(path):
    rows = {}
    for file in sorted(Path(path).glob('runs/*/cases/*.json')):
        case = json.loads(file.read_text())
        for row in case['summary']:
            key = row['model'], row['workload_id'], row['variant']
            assert key not in rows, ('Duplicate request result', key)
            rows[key] = dict(row, source_file=str(file.resolve()),
                source_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    return rows


def generate(requests, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scope = json.loads((ROOT / 'artifact/paper-scope.json').read_text())
    models, workloads = scope['model_order'], scope['workload_order']
    actual = collect(requests)
    expected = {(m, w, v) for m in models for w in workloads for v in VARIANTS}
    assert set(actual) == expected, ('Request scope mismatch', expected - actual.keys(), actual.keys() - expected)
    reference = {(r['model'], r['workload_id'], r['variant']): r for r in table(ROOT / 'artifact/reference/requests.csv')}
    differences = []
    for key, row in actual.items():
        assert math.isclose(row['E2E_ms'], row['TTFT_ms'] + (row['output_tokens'] - 1) * row['TBT_ms'], rel_tol=1e-12)
        for field in METRICS:
            value, original = row[field], float(reference[key][field])
            if not math.isclose(value, original, rel_tol=1e-10, abs_tol=1e-9):
                differences.append(dict(model=key[0], workload_id=key[1], variant=key[2], metric=field,
                    actual=value, reference=original, relative_error=(value-original)/original))
    checks = dict(cases=len(models)*len(workloads), variants=list(VARIANTS),
                  reference_comparisons=len(actual)*len(METRICS), mismatches=differences,
                  passed=not differences, inputs_are_fresh_request_results=True)
    (output / 'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    if differences:
        raise RuntimeError('Fresh request replay differs from the paper: '+str(output/'checks.json'))
    tables = {'request-e2e': [], 'request-phase-breakdown': [], 'request-capacity': []}
    ratios = []
    for model in models:
        for workload in workloads:
            cell = {v: actual[model, workload, v] for v in VARIANTS}
            for variant, row in cell.items():
                tables['request-e2e'].append(dict(model=model, workload_id=workload, variant=variant,
                    TTFT_ms=row['TTFT_ms'], TBT_ms=row['TBT_ms'], E2E_ms=row['E2E_ms'],
                    F0_E2E_ms=cell['F0']['E2E_ms'], decode_steps=row['output_tokens']-1,
                    decode_total_ms=row['TBT_ms']*(row['output_tokens']-1), output_tokens=row['output_tokens'],
                    batch=row['batch'], valid='true', displayed='true'))
            for panel, baseline, metric in [('prefill','F2','TTFT_ms'), ('decode','F1','TBT_ms')]:
                tables['request-phase-breakdown'].append(dict(model=model, workload_id=workload, panel=panel,
                    reference_variant=baseline, method_variant='F4', reference_ms=cell[baseline][metric],
                    method_ms=cell['F4'][metric], speedup=cell[baseline][metric]/cell['F4'][metric], valid='true'))
            tables['request-capacity'].append(dict(model=model, group=workload, workload_id=workload,
                displayed='true', eligible='true', slot_marker='',
                raw_numerator=cell['F2']['simultaneous_peak_KV_GiB'],
                raw_denominator=cell['F4']['simultaneous_peak_KV_GiB'],
                plotted_value=cell['F2']['simultaneous_peak_KV_GiB']/cell['F4']['simultaneous_peak_KV_GiB']))
            ratios.append(dict(model=model, workload_id=workload,
                E2E_F1_over_F4=cell['F1']['E2E_ms']/cell['F4']['E2E_ms'],
                TTFT_F2_over_F4=cell['F2']['TTFT_ms']/cell['F4']['TTFT_ms'],
                TBT_F1_over_F4=cell['F1']['TBT_ms']/cell['F4']['TBT_ms'],
                capacity_F2_over_F4=cell['F2']['simultaneous_peak_KV_GiB']/cell['F4']['simultaneous_peak_KV_GiB']))
    csvout(output/'paper-benefits.csv', ratios)
    for name, rows in tables.items():
        target=output/name
        shutil.copytree(ROOT/'plots'/name, target)
        csvout(target/'data.csv', rows)
        subprocess.run([sys.executable, str(target/'plot.py')], check=True)
    (output/'README.md').write_text(
        '# Reproduced request figures\n\n'
        'These figures were generated from newly simulated request results. '
        '`checks.json` compares all four published metrics for all 100 configurations against the paper reference.\n\n'
        '* `request-e2e/figure.pdf`: Figure 5, E2E throughput relative to Full GPU.\n'
        '* `request-phase-breakdown/figure.pdf`: Figure 6, TTFT relative to Materialized PIM and TBT relative to GPU reuse.\n'
        '* `request-capacity/figure.pdf`: Figure 7, simultaneous peak KV capacity, Materialized PIM / KVChime.\n\n'
        'Each figure folder contains its generated CSV, Python plot script, style and template. '
        'Run `python3 plot.py` there to redraw it.\n')
    print(json.dumps(checks), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    generate(args.requests,args.output)
