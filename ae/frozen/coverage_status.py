"""Report complete-input replay coverage by verifying case files, not success flags.

Reads existing artifacts only. No simulator, trace generator or source mutation.
"""


import argparse


import ast


from collections import Counter, defaultdict


from copy import deepcopy


import hashlib


import json


import math


from pathlib import Path


import sys


sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[2]


BASE = ROOT/'archive/coverage-expansion'


STAGE = BASE/'attacc-fugue'


RUNNER = ROOT/'outline/analysis/run_coverage_expansion.py'


AUDIT = ROOT/'audit/full-coverage-progress.json'


GUIDE = ROOT/'outline/README_coverage_expansion.md'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def canonical(value):
    return digest(json.dumps(value, sort_keys=True).encode())


def source_rules():
    tree = ast.parse(RUNNER.read_text())
    variants = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VARIANTS' for t in n.targets))
    simulate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'simulate')
    counts = {n.targets[0].id: n.value for n in simulate.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ('pre_counts', 'decode_counts')}
    assert set(counts) == {'pre_counts', 'decode_counts'}
    return variants, counts


def check_case(data, case, spec, identity=None, rules=None):
    """Independent accounting and horizon checks; never execute cost operators."""
    variants, nodes = rules or source_rules()
    failures = []
    checks = 0

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(label)

    def close(a, b, label):
        check(isinstance(a, (int, float)) and isinstance(b, (int, float))
              and math.isfinite(a) and math.isfinite(b)
              and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), label)

    try:
        name = spec['name']; tp = spec['selected']['tensor_parallel']; geo = spec['geometry']
        env = {'Counter': Counter, 'L': geo['ndec'], 'case': case}
        layer_counts = {key: eval(compile(ast.Expression(node), str(RUNNER), 'eval'), env)
                        for key, node in nodes.items()}
        check(data['case_sha256'] == canonical(case), 'input case SHA')
        if identity is not None:
            check(data['runner_sha256'] == identity['runner_sha256'], 'runner SHA')
            check(data['run_identity_sha256'] == identity['_file_sha256'], 'run identity SHA')
        rows = data['summary']
        check(Counter(r['variant'] for r in rows) == Counter(variants), 'exactly one row per intended variant')
        by_variant = {r['variant']: r for r in rows}
        expected_blocks = Counter()
        for variant in variants:
            for lp, count in layer_counts['pre_counts'].items():
                expected_blocks[variant, 'prefill', 0, lp] = count
            for step in range(1, case['output']):
                for lp, count in layer_counts['decode_counts'].items():
                    expected_blocks[variant, 'decode', step, lp] = count
        actual_blocks = {}; totals = defaultdict(lambda: [[], []])
        for index, block in enumerate(data['event_blocks']):
            key = block['variant'], block['phase'], block['step'], block['layer_class']
            check(key not in actual_blocks, f'duplicate event block {index}')
            actual_blocks[key] = block['repetitions']
            for component in block['components']:
                check(len(component) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in component[1:]), f'finite component {index}')
            time = math.fsum(c[1] for c in block['components']) * block['repetitions']
            energy = math.fsum(c[2] for c in block['components']) * block['repetitions']
            close(block['time_us'], time, f'component time sum {index}')
            close(block['energy_uJ'], energy, f'component energy sum {index}')
            totals[key[:2]][0].append(time); totals[key[:2]][1].append(energy)
        check(actual_blocks == dict(expected_blocks), 'complete output horizon and all layer repetitions')
        profiles = data['decode_profiles']
        check([(p['step'], p['n']) for p in profiles] == [(s, case['n']+s) for s in range(1, case['output'])], 'decode profile horizon')
        profile_keys = {'F2'} | {f'{v}-layer{lp}' for v in ('F3', 'F4') for lp in layer_counts['decode_counts']}
        check(all(set(p['profiles']) == profile_keys for p in profiles), 'decode path profiles for every layer class')
        for variant in variants:
            row = by_variant[variant]
            expected = dict(case_id=name+'--'+case['id'], workload_id=case['workload_id'], model=name,
                            source=case['source'], dataset=case.get('dataset'), scope=case['scope'],
                            source_row=case.get('source_row'), GPUs=tp, batch=len(case['members']),
                            Q_heads=geo['num_heads'], KV_heads=geo['num_kv_heads'], gqa_size=geo['gqa_size'],
                            prompt_tokens=case['n'], partial_Q=case['q'], output_tokens=case['output'])
            for key, value in expected.items():
                check(row[key] == value, variant+' source '+key)
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    check(math.isfinite(value) and value >= 0, variant+' finite '+key)
            pre = [math.fsum(values) for values in totals[variant, 'prefill']]
            dec = [math.fsum(values) for values in totals[variant, 'decode']]
            for suffix, column in [('_ms', 0), ('_energy_mJ', 1)]:
                close(row['TTFT'+suffix], pre[column]/1000, variant+' TTFT'+suffix)
                close(row['TBT'+suffix], dec[column]/(case['output']-1)/1000, variant+' TBT'+suffix)
                close(row['E2E'+suffix], (pre[column]+dec[column])/1000, variant+' E2E'+suffix)
                close(row['TTFT'+suffix]+(case['output']-1)*row['TBT'+suffix], row['E2E'+suffix], variant+' horizon identity'+suffix)
            check(row['TTFT_ms'] > 0 and row['TBT_ms'] > 0, variant+' positive service time')
            snapshots = data['storage'][variant]['snapshots']
            expected_snapshots = [('prefill', lp) for lp in layer_counts['pre_counts']]
            expected_snapshots += [('decode', step) for step in range(1, case['output'])]
            actual_snapshots = [(s['phase'], s['layer_class'] if s['phase'] == 'prefill' else s['step']) for s in snapshots]
            check(actual_snapshots == expected_snapshots, variant+' complete storage horizon')
            for index, snap in enumerate(snapshots):
                close(snap['total_KV_bytes'], snap['GPU_KV_bytes']+snap['remote_shared_bytes']+snap['remote_private_bytes'], variant+f' simultaneous snapshot {index}')
            close(row['GPU_peak_KV_GiB'], max(s['GPU_KV_bytes'] for s in snapshots)/2**30, variant+' GPU KV peak')
            close(row['remote_peak_KV_GiB'], max(s['remote_shared_bytes']+s['remote_private_bytes'] for s in snapshots)/2**30, variant+' remote KV peak')
            close(row['simultaneous_peak_KV_GiB'], max(s['total_KV_bytes'] for s in snapshots)/2**30, variant+' simultaneous total KV peak')
            check(row['GPU_peak_total_per_device_GiB']*2**30 <= spec['selected']['GPU_budget_bytes_per_device'], variant+' GPU capacity')
            check(row['remote_peak_KV_GiB']*2**30 <= spec['selected']['remote_budget_bytes'], variant+' remote capacity')
            for key, warmup_key in [('warmup_ms', 'time_us'), ('warmup_energy_mJ', 'energy_uJ')]:
                close(row[key], 0 if variant == 'F0' else data['warmup'][warmup_key]/1000, variant+' '+key)
        if len(case['members']) == 1 and geo['gqa_size'] == 1:
            close(by_variant['F3']['TBT_ms'], by_variant['F4']['TBT_ms'], 'single-query MHA has no MQ change')
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as error:
        check(False, 'malformed or incomplete case: '+repr(error))
    return dict(passed=not failures, checks=checks, failures=failures)

