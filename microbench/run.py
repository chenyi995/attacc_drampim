#!/usr/bin/env python3
"""Reproduce the final crossover, frequency and area evidence in a new directory."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def table(path):
    return list(csv.DictReader(line for line in Path(path).read_text().splitlines() if not line.startswith('#')))


def csvout(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_module(name):
    path = HERE / 'source' / (name + '.py')
    spec = importlib.util.spec_from_file_location('ae_original_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_copied_sources():
    for record in read(HERE / 'copied-sources.json'):
        if sha(HERE / record['path']) != record['sha256']:
            raise RuntimeError('Copied source identity mismatch: ' + record['path'])


def verify_simulator(stage):
    recorded = read(HERE / 'simulator-source-sha256.json')
    direct = ['fugue/kvchime_model.py', 'fugue/kvchime_traces.py', 'fugue/attention.py',
              'fugue/runtime.py', 'fugue/kvchime_selector.py', 'fugue/official-gqa-geometry.json']
    direct += [p for p in recorded if p.startswith('src/') or p.startswith('pim_ramulator_src/')]
    for name in direct:
        if sha(stage / name) != recorded[name]:
            raise RuntimeError('Frozen microbenchmark source changed: ' + name)
    return {name: recorded[name] for name in direct}


def service_kernel():
    """Return the source-extracted original service equations."""
    return source_module('crossover_kernel').service


def fresh_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def plot_folder(kind, output):
    target = output / 'plot'
    shutil.copytree(HERE / 'plots' / kind, target)
    return target


def crossover(args):
    spec = read(HERE / 'crossover-input.json')
    if args.plan:
        print(json.dumps(dict(spec, execution='Real Ramulator required only without --plan'), indent=2))
        return
    stage = args.simulator.resolve()
    identities = verify_simulator(stage)
    runtime = args.runtime.resolve()
    for name in ['ramulator2', 'libramulator.so']:
        if not (runtime / name).is_file():
            raise FileNotFoundError(runtime / name)
    output = fresh_output(args.output)
    os.environ.update(FUGUE_OUTPUT=str(output / 'unused'), FUGUE_JOBS=str(args.jobs),
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    os.environ['LD_LIBRARY_PATH'] = str(runtime) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    sys.path.insert(0, str(stage))
    # Import before ExactTraceBank so its historical fallback path is never used.
    from fugue import kvchime_traces
    from fugue.kvchime_model import Model, geometry
    from fugue.attention import TCK, CAP
    cache = source_module('coverage_trace_cache')
    bank = cache.ExactTraceBank(output / 'profiles', runtime, geometry(spec['model']), output / 'exact-cache')
    model = Model(spec['model'], spec['tensor_parallel'], bank, prepare_profiles=False)
    original = source_module('crossover_kernel')
    keys = original.keys_for(spec, model, CAP)
    save(output / 'run.json', dict(complete=False, input=spec, profile_keys=keys,
        source_sha256=identities, runtime={name: sha(runtime / name) for name in ['ramulator2', 'libramulator.so']},
        service_source_sha256=sha(HERE / 'source/crossover_kernel.py')))
    for offset in range(0, len(keys), 16):
        bank.prepare(keys[offset:offset + 16], dhead=model.trace_dhead, dbyte=model.dbyte)
        print(json.dumps(dict(completed_profiles=min(offset + 16, len(keys)), total_profiles=len(keys))), flush=True)
    profiles, windows = {}, {}
    for n, h, r in keys:
        row = bank.get(n, h, r, dhead=model.trace_dhead, dbyte=model.dbyte)
        profiles[n, h, r] = row
        windows[n, h, r] = list(bank.iter_key_events(row))
    service = service_kernel()
    operands, display, crossings = [], [], []
    gpu = {}
    for panel in spec['panels']:
        subset = []
        for q in panel['query_samples']:
            row = service(panel, q, model, profiles, windows, Model, CAP, TCK)
            row['source_kind'] = 'fresh-native-replay'
            row['source_panel'] = panel['source_panel']
            operands.append(row)
            same = gpu.setdefault(q, row['gpu_service_us'])
            assert same == row['gpu_service_us']
            item = dict(panel=panel['panel'], source_panel=panel['source_panel'], q=q,
                model=model.name, cached_tokens=panel['cached'], link_GBps=panel['link_GBps'], mode=panel['mode'],
                gpu_service_us=row['gpu_service_us'], pim_service_us=row['pim_service_us'],
                gpu_throughput_tokens_per_second=q / row['gpu_service_us'] * 1e6,
                pim_throughput_tokens_per_second=q / row['pim_service_us'] * 1e6,
                original_marker=json.dumps(q in panel['marker_samples']))
            subset.append(item)
            display.append(item)
        for left, right in zip(subset, subset[1:]):
            a = left['gpu_service_us'] - left['pim_service_us']
            b = right['gpu_service_us'] - right['pim_service_us']
            if a * b < 0:
                crossings.append(dict(panel=panel['panel'], source_panel=panel['source_panel'],
                                      q_low=left['q'], q_high=right['q']))
    target = plot_folder('crossover', output)
    csvout(output / 'service-operands.csv', operands)
    csvout(target / 'panel_data.csv', display)
    csvout(target / 'gpu_data.csv', [dict(q=q, gpu_service_us=t,
        gpu_throughput_tokens_per_second=q / t * 1e6) for q, t in sorted(gpu.items())])
    csvout(target / 'crossings.csv', crossings, ['panel', 'source_panel', 'q_low', 'q_high'])
    csvout(target / 'panels.csv', [dict(panel=p['panel'], source_panel=p['source_panel'], mode=p['mode'],
        cached_tokens=p['cached'], link_GBps=p['link_GBps'], pim_label=p['pim_label']) for p in spec['panels']])
    reference = {(r['panel'], int(r['q'])): r for r in table(HERE / 'reference/crossover/panel_data.csv')}
    mismatches = []
    for row in display:
        for field in ['gpu_service_us', 'pim_service_us']:
            old = float(reference[row['panel'], row['q']][field])
            if not math.isclose(row[field], old, rel_tol=1e-10, abs_tol=1e-9):
                mismatches.append(dict(panel=row['panel'], q=row['q'], field=field, reference=old, actual=row[field]))
    physical = [read(p) for p in (output / 'exact-cache/profiles').glob('*/complete.json')]
    assert physical and all(p['complete'] and p['kind'] == 'ramulator' for p in physical)
    checks = dict(rows=len(display), profiles=len(profiles), fresh_physical_replays=len(physical),
                  all_physical_replays_complete=True, crossings=crossings,
                  all_reference_values_match=not mismatches, mismatches=mismatches,
                  actual_replay=True, source_identity_after=verify_simulator(stage))
    save(output / 'checks.json', checks)
    run = read(output / 'run.json'); run.update(complete=True, checks_pass=not mismatches)
    save(output / 'run.json', run)
    if mismatches:
        raise SystemExit('Fresh replay differs from paper evidence; inspect checks.json.')
    if not args.no_plot:
        os.environ['MPLCONFIGDIR'] = str(output / '.matplotlib-cache')
        subprocess.run([sys.executable, str(target / 'plot.py')], check=True)
    print(output)


def frequency(args):
    stage = args.simulator.resolve()
    recorded = read(HERE / 'simulator-source-sha256.json')
    for name in ['src/config.py', 'fugue/attention.py']:
        assert sha(stage / name) == recorded[name], name
    output = fresh_output(args.output)
    module = source_module('current_frequency')
    module.SNAPSHOT = stage
    module.source_records = lambda: [dict(source_relative_path=n, sha256=sha(stage / n)) for n in module.FILES]
    data = module.calculate()
    save(output / 'derivation.json', data)
    target = plot_folder('frequency', output)
    count = data['inputs']['query_capacity']; tck = data['inputs']['tck_ns']
    csvout(target / 'parameters.csv', [{k: data['inputs'][k] for k in
                                      ['query_capacity', 'tck_ns', 'dram_floor_tck']}])
    csvout(target / 'balance.csv', [data['balance']])
    csvout(target / 'intervals.csv', [{k: r[k] for k in
        ['frequency_ghz', 'compute_interval_tck', 'effective_interval_tck']} for r in data['curve']])
    csvout(target / 'curve.csv', [dict(sample=i, frequency_ghz=r['frequency_ghz'],
        effective_query_MAC_per_ns=r['effective_query_mac_per_ns'],
        compute_query_MAC_per_ns=count / (r['compute_interval_tck'] * tck),
        power_query_MAC_per_ns=data['balance']['effective_query_mac_per_ns']) for i, r in enumerate(data['curve'])])
    reference = table(HERE / 'reference/frequency/intervals.csv')
    assert len(reference) == len(data['curve'])
    for old, new in zip(reference, data['curve']):
        assert all(float(old[k]) == new[k] for k in old), (old, new)
    save(output / 'checks.json', dict(all_reference_intervals_match=True, rows=len(reference),
         actual_simulator_run=False, model_calculation=True, balance=data['balance']))
    if not args.no_plot:
        os.environ['MPLCONFIGDIR'] = str(output / '.matplotlib-cache')
        subprocess.run([sys.executable, str(target / 'plot.py')], check=True)
    print(output)


def area(args):
    output = fresh_output(args.output)
    components = table(HERE / 'area/data_components.csv')
    constants = {}
    for node in ast.parse((HERE / 'source/area_constants.py').read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant):
            constants[node.targets[0].id] = node.value.value
    import re
    density = float(re.search(r'areas by (\d+)', (HERE / 'area/data_overhead.csv').read_text()).group(1))
    totals = defaultdict(lambda: defaultdict(float)); reports = []
    measurements = {r['tag']: r for r in table(HERE / 'area/source-measurements.csv')}
    for row in components:
        measurement = measurements[row['tag']]
        assert measurement['top'] == row['top']
        measured = float(measurement['cell_area_um2'])
        period = float(measurement['period_ps'])
        slack = float(measurement['slack_ps'])
        violations = int(measurement['violations'])
        assert measured == float(row['area_um2']) and period == float(row['period_ps'])
        assert slack == float(row['slack_ps']) and slack >= 0 and violations == 0
        for config in row['used_by'].split('+'):
            raw = measured * int(row['count_per_stack'])
            totals['asap7_raw', config][row['level']] += raw
            totals['dram_equivalent', config][row['level']] += raw * (density if row['level'] in ['bank', 'bank_group'] else 1)
        reports.append(dict(tag=row['tag'], cell_area_um2=measured, period_ps=period,
            slack_ps=slack, violations=int(violations), area_report_sha256=measurement['area_report_sha256'], qor_report_sha256=measurement['qor_report_sha256']))
    module = source_module('area_data')
    data = module.physical_layers(components, totals, constants['DIE_MM2'], constants['DIES_PER_STACK'],
          constants['BANKS_PER_DIE'] * constants['BANK_STORE_MM2'], density)
    save(output / 'area.json', data)
    csvout(output / 'measured-components.csv', reports)
    dram = data['dram_die']; buffer = data['buffer_die']
    summary = [dict(metric='DRAM die area (mm2)', value=dram['die_area_mm2']),
        dict(metric='Bank footprint (% of full DRAM die)', value=dram['bank_footprint_pct_of_die']),
        dict(metric='KVChime added Bank/BG logic (% of full DRAM die)', value=dram['increment_pim_pct_of_die']),
        dict(metric='KVChime Bank/BG total (% of full DRAM die)', value=dram['configurations']['KVChime']['pim_pct_of_die']),
        dict(metric='Remaining accounting margin (% of full DRAM die)', value=dram['configurations']['KVChime']['remaining_margin_pct_of_die']),
        dict(metric='Added selected buffer/controller components (mm2)', value=buffer['increment_mm2']),
        dict(metric='KVChime selected buffer/controller components (mm2)', value=buffer['kvchime_total_mm2'])]
    csvout(output / 'paper-area.csv', summary)
    save(output / 'checks.json', dict(archived_synthesis_operands_match=True, components=len(components),
         synthesis_executed=False, scope='Arithmetic from exact scalar operands extracted from retained Genus reports; original raw reports remain local; fixed literature density/footprint inputs.'))
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['crossover', 'frequency', 'area']:
        p = sub.add_parser(name)
        p.add_argument('--output', type=Path)
        p.add_argument('--no-plot', action='store_true')
        if name != 'area': p.add_argument('--simulator', type=Path)
        if name == 'crossover':
            p.add_argument('--runtime', type=Path)
            p.add_argument('--jobs', type=int, default=1)
            p.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    verify_copied_sources()
    if not getattr(args, 'plan', False):
        if not args.output: parser.error('--output is required for an execution')
        if args.command != 'area' and not args.simulator: parser.error('--simulator is required')
        if args.command == 'crossover' and not args.runtime: parser.error('--runtime is required')
    if args.command == 'crossover' and not 1 <= args.jobs <= 24: parser.error('--jobs must be 1..24')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
