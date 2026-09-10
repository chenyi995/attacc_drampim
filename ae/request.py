#!/usr/bin/env python3
"""Run the current paper's 25 complete requests with the frozen final timing model.

The frozen source modules retain their original bytes. This adapter relocates
source/configuration discovery, verifies the final configuration, and removes
the undisplayed F3 path. It never loads historical timing profiles.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import csv
import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import runpy
import shutil
import sys
from types import SimpleNamespace

VARIANTS = ('F0', 'F1', 'F2', 'F4')
REPOSITORY = Path(__file__).resolve().parents[1]
FROZEN = Path(__file__).with_name('frozen')
CONFIG = REPOSITORY / 'artifact/request'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify_code(simulator):
    lock = read(CONFIG / 'code-lock.json')
    for row in lock['files']:
        relative = Path(row['path'])
        path = simulator.joinpath(*relative.parts[1:]) if relative.parts[0] == 'simulator' else REPOSITORY / relative
        assert path.is_file() and sha(path) == row['sha256'], ('Frozen source/configuration changed', path)
    native = read(simulator / 'artifact/native-source-lock.json')
    assert sha(simulator / 'artifact/native-source-lock.json') == lock['original_native_lock_sha256']
    for name, expected in native.items():
        assert sha(simulator / name) == expected, ('Native source changed', name)
    return lock


def scope_sources(backbone, checker, output):
    """Change only undisplayed variant enumeration and F3-only assertions."""
    simulation_source = Path(backbone.__file__)
    tree = ast.parse(simulation_source.read_text())
    counts = Counter()

    class Scope(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(target, ast.Name) and target.id == 'VARIANTS' for target in node.targets):
                assert ast.literal_eval(node.value) == ('F0', 'F1', 'F2', 'F3', 'F4')
                node.value = ast.parse(repr(VARIANTS), mode='eval').body
                counts['variant_enumeration'] += 1
            return self.generic_visit(node)

        def visit_For(self, node):
            target = ast.parse("[('F3', False), ('F4', True)]", mode='eval').body
            if ast.dump(node.iter) == ast.dump(target):
                node.iter = ast.parse("[('F4', True)]", mode='eval').body
                counts['F3_scan_loop'] += 1
            return self.generic_visit(node)

        def visit_If(self, node):
            target = ast.parse('b == 1 and m.gqa == 1', mode='eval').body
            if ast.dump(node.test) == ast.dump(target):
                node.test = ast.BoolOp(op=ast.And(), values=[ast.parse("'F3' in VARIANTS", mode='eval').body, node.test])
                counts['F3_only_assertion'] += 1
            return self.generic_visit(node)

    scoped = ast.fix_missing_locations(Scope().visit(tree))
    assert counts == dict(variant_enumeration=1, F3_scan_loop=1, F3_only_assertion=1), counts
    path = output / 'implementation/scoped_request_source.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ast.unparse(scoped) + '\n')
    backbone.__file__ = str(path)
    backbone.VARIANTS = VARIANTS
    checker_source = Path(checker.__file__)
    checker_tree = ast.parse(checker_source.read_text())
    check_counts = Counter()

    class CheckScope(ast.NodeTransformer):
        def visit_comprehension(self, node):
            target = ast.parse("('F3', 'F4')", mode='eval').body
            if ast.dump(node.iter) == ast.dump(target):
                node.iter = ast.parse("('F4',)", mode='eval').body
                check_counts['profile_scope'] += 1
            return self.generic_visit(node)

        def visit_If(self, node):
            target = ast.parse("len(case['members']) == 1 and geo['gqa_size'] == 1", mode='eval').body
            if ast.dump(node.test) == ast.dump(target):
                node.test = ast.BoolOp(op=ast.And(), values=[ast.parse("'F3' in variants", mode='eval').body, node.test])
                check_counts['F3_only_assertion'] += 1
            return self.generic_visit(node)

    check_tree = ast.fix_missing_locations(CheckScope().visit(checker_tree))
    assert check_counts == dict(profile_scope=1, F3_only_assertion=1), check_counts
    check_path = output / 'implementation/scoped_check_source.py'
    check_path.write_text(ast.unparse(check_tree) + '\n')
    checker.__file__ = str(check_path)
    checker.RUNNER = path
    save(output / 'implementation/scope-adapter.json', dict(
        source_sha256=sha(simulation_source), scoped_sha256=sha(path), changes=dict(counts),
        checker_source_sha256=sha(checker_source), checker_scoped_sha256=sha(check_path), checker_changes=dict(check_counts),
        scope='Only F3 enumeration/scan and F3-only assertions change. The original F2/staging AST adapter is then applied unchanged.'))


def initialize(simulator, output, jobs):
    simulator = Path(simulator).resolve()
    verify_code(simulator)
    assert 1 <= jobs <= 24
    os.environ.update(FUGUE_JOBS=str(jobs), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      PYTHONDONTWRITEBYTECODE='1', FUGUE_OUTPUT=str(output))
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(FROZEN), str(simulator)]
    final = importlib.import_module('hbm_2k_model')
    runner = importlib.import_module('hbm_2k_run')
    prior, traces, commands = final.prior, final.traces, final.prior.command_model
    backbone, checker = runner.backbone, runner.status_source
    assert Path(final.native.__file__).resolve() == simulator / 'fugue/kvchime_model.py'
    backbone.STAGE = simulator
    policies = {name: read(CONFIG / filename) for name, filename in (
        ('point', 'point-policy.json'), ('repair', 'repair-policy.json'), ('hbm', 'hbm-policy.json'))}
    digests = {name: sha(CONFIG / filename) for name, filename in (
        ('point', 'point-policy.json'), ('repair', 'repair-policy.json'), ('hbm', 'hbm-policy.json'), ('calibration', 'calibration.json'))}
    assert policies['hbm']['parent_policy_sha256'] == digests['repair']
    assert policies['repair']['parent_policy_sha256'] == digests['point']
    assert policies['hbm']['coefficient_parent_policy_sha256'] == digests['point']
    assert all(policy['calibration_sha256'] == digests['calibration'] for policy in policies.values())
    assert policies['point']['wrapper_sha256'] == sha(FROZEN / 'selector_point_model.py')
    assert policies['point']['base_prediction_sha256'] == sha(FROZEN / 'selector_cost_model.py')
    assert policies['hbm']['gpu_traffic_policy'] == final.GPU_TRAFFIC_POLICY
    specs = {row['name']: row for row in read(CONFIG / 'models.json')['models']}

    def geometry(name):
        # Replace source-document discovery with the frozen, checked final model declaration.
        expected = deepcopy(specs[name]['geometry'])
        expected['dtype'] = final.native.DataType.W16A16
        actual = final.native.geometry(name)
        if name in prior.CORRECTED_MT:
            actual['num_heads'] = actual['hdim'] // actual['dhead']
            actual['num_kv_heads'] = actual['num_heads']
        assert actual == expected, ('Final model geometry changed', name, actual, expected)
        return actual

    def capacity(dhead, dbyte=2):
        # The current five models all use 128-wide FP16 heads. The original
        # source read 512 from a historical README; this input is frozen in policy.
        assert dbyte == 2 and str(dhead) in policies['hbm']['capacity_by_dhead']
        frozen = deepcopy(policies['hbm']['capacity_by_dhead'][str(dhead)])
        g = traces.frozen_traces.LAYOUT_GEOMETRY
        lanes = g['column_bytes'] // dbyte
        columns = math.ceil(dhead / (g['banks_per_group'] * lanes))
        assert frozen['buffer_bytes'] == 512
        assert frozen['bank_query_columns'] == columns
        assert frozen['padded_query_bytes'] == columns * g['column_bytes']
        assert frozen['unpadded_query_bytes'] == dhead * dbyte / g['banks_per_group']
        assert frozen['capacity'] == min(traces.FROZEN_CAP, frozen['buffer_bytes'] // frozen['padded_query_bytes'])
        assert frozen['dbyte'] == dbyte and frozen['dhead'] == dhead
        return frozen

    prior.geometry = final.geometry = geometry
    traces.capacity_spec = prior.capacity_spec = final.capacity_spec = capacity

    class Selector(final.PointSelector):
        def __init__(self, model):
            # Preserve the final predict_views implementations exactly. Original
            # constructors validated author-machine paths; the portable code lock
            # and policy chain above validate the same executable/configuration bytes.
            commands.CommandAwareSelector.__init__(self, CONFIG / 'calibration.json', model.h,
                model.gqa, model.trace_dhead, model.dbyte)
            self.policy = policies['hbm']
            self.policy_path = str(CONFIG / 'hbm-policy.json')
            self.policy_sha256 = digests['hbm']
            self.parent_policy_sha256 = digests['repair']
            self.coefficient_parent_policy_sha256 = digests['point']
            self.capacity = capacity(model.trace_dhead, model.dbyte)['capacity']
            self.calls = []

        def choose_views(self, *args, **kwargs):
            prediction = self.predict_views(*args, **kwargs)
            self.calls.append(prediction)
            return prediction['choice'], prediction['service_us']

    scope_sources(backbone, checker, output)
    simulate = runner.repaired_simulate()
    checker.case_checker = runner.case_checker()
    return SimpleNamespace(Model=final.Model, Selector=Selector, TraceBank=traces.TraceBank,
        geometry=geometry, specs=specs, simulate=simulate, runner=runner, backbone=backbone,
        checker=checker, staging=runner.staging, native=final.native, policies=policies,
        identity=digests, simulator=simulator, impl=final)


def runtime_files(built, output):
    built = Path(built).resolve()
    target = output / 'runtime'
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for filename in ('ramulator2', 'libramulator.so'):
        source = built / filename
        assert source.is_file(), ('Missing built simulator', source)
        destination = target / filename
        if destination.exists():
            assert sha(source) == sha(destination), ('Runtime changed', destination)
        else:
            shutil.copy2(source, destination)
        files.append(dict(name=filename, sha256=sha(destination), source=str(source)))
    os.environ['LD_LIBRARY_PATH'] = str(target) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    save(target / 'build-provenance.json', dict(files=files, copied_at=now()))
    return target


def verify_case(context, value, case, spec):
    rules = context.checker.source_rules()
    assert rules[0] == VARIANTS
    check = context.checker.case_checker(value, case, spec, rules=rules)
    assert check['passed'], check['failures']
    staged = context.staging.verify(value, case, context.Model(spec['name'], spec['selected']['tensor_parallel'], None))
    assert staged['passed'], staged['failures']
    return dict(accounting=check, staging=staged)


def execute_cell(context, spec, case, catalog, output, runtime, model_hook=None, result_hook=None):
    folder = output / 'runs' / spec['name']
    bank = context.TraceBank(folder / 'profiles', runtime, context.geometry(spec['name']), output / 'exact-trace-cache')
    model = context.Model(spec['name'], spec['selected']['tensor_parallel'], bank, prepare_profiles=False)
    assert model.geometry_audit == spec['repair_geometry']
    model.coverage_remote_budget = spec['selected']['remote_budget_bytes']
    model.selector = context.Selector(model)
    if model_hook is not None:
        model_hook(model)
    started = now()
    value = context.staging.apply(context.simulate(case, model, catalog), case, model)
    value.update(case_sha256=context.checker.canonical(case), selector_predictions=model.selector.calls,
        calibration_sha256=context.identity['calibration'], policy_sha256=context.identity['hbm'],
        geometry_audit=model.geometry_audit, gpu_traffic_audit=deepcopy(model.gpu_traffic_audit),
        f2_service_timeline=context.runner.timeline_check(value, case, model),
        started_utc=started, finished_utc=now(), cache_stats=dict(bank.cache_stats),
        implementation=bank.implementation)
    assert len(value['decisions']) == len(model.selector.calls)
    assert all(decision['choice'] == prediction['choice'] for decision, prediction in zip(value['decisions'], model.selector.calls))
    value['checks']['portable_accounting'] = verify_case(context, value, case, spec)
    qualifications = {(r['model'], r['workload_id'], r['variant']): r
                      for r in read(CONFIG / 'timing-qualification.json')['rows']}
    for row in value['summary']:
        qualification = qualifications[row['model'], row['workload_id'], row['variant']]
        row['invalid_fields'] = qualification.get('invalid_fields', [])
        row['invalid_reasons'] = qualification.get('invalid_reasons', [])
    if result_hook is not None:
        result_hook(model, value, case, context.backbone.source_helpers(catalog), folder / 'profiles')
    save(folder / 'cases' / (case['id'] + '.json'), value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator', type=Path, default=REPOSITORY / 'simulator')
    parser.add_argument('--inputs', type=Path, default=REPOSITORY / 'artifact/inputs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-built-dir', type=Path)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--model', help='Run one current-scope model; the default is all five.')
    parser.add_argument('--case', help='Run one current-scope workload; the default is all five.')
    parser.add_argument('--fig1a-output', type=Path, help='Export current Fig. 1(a) accounting after the Llama 7B Readers 4 cell.')
    parser.add_argument('--prepare-only', action='store_true', help='Verify the portable code/configuration and model geometry without running requests or Ramulator.')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit('Use a fresh output directory: ' + str(output))
    output.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (5_000_000_000, 5_000_000_000))
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.jobs])
    context = initialize(args.simulator, output, args.jobs)
    input_receipt = runpy.run_path(str(REPOSITORY / 'artifact/inputs/prepare_inputs.py'))['verify'](args.inputs)
    selected = read(args.inputs / 'selected.json')
    cases = [item['case'] for item in selected]
    catalog = {row['chunk_id']: row for row in [json.loads(line) for line in (args.inputs / 'catalog.jsonl').read_text().splitlines()]}
    scope = read(args.inputs / 'scope.json')
    assert [case['id'] for case in cases] == scope['workload_order']
    assert list(context.specs) == scope['model_order']
    assert len(cases) == 5 and len(context.specs) == 5
    specs = list(context.specs.values())
    if args.model:
        assert args.model in context.specs
        specs = [context.specs[args.model]]
    if args.case:
        assert args.case in scope['workload_order']
        cases = [case for case in cases if case['id'] == args.case]
    for spec in specs:
        model = context.Model(spec['name'], spec['selected']['tensor_parallel'], None)
        assert model.geometry_audit == spec['repair_geometry']
    save(output / 'manifest.json', dict(schema='kvchime-ae-requests-v1', status='prepared',
        models=[s['name'] for s in specs], workloads=[c['id'] for c in cases], variants=VARIANTS,
        inputs={name: sha(args.inputs / name) for name in ('selected.json', 'catalog.jsonl', 'scope.json')},
        input_verification=input_receipt,
        policy=context.identity, started_utc=now(), trace_workers=args.jobs,
        resource_bound_bytes=(1 + args.jobs) * 5_000_000_000))
    if args.prepare_only:
        print(json.dumps(dict(status='prepared', simulated=False, output=str(output))))
        return
    if not args.runtime_built_dir:
        parser.error('--runtime-built-dir is required for request execution')
    runtime = runtime_files(args.runtime_built_dir, output)
    all_rows = []
    for spec in specs:
        for case in cases:
            print('START', spec['name'], case['id'], now(), flush=True)
            result_hook = None
            if args.fig1a_output and spec['name'] == 'LLAMA-7B' and len(case['members']) == 4:
                from microbench.fig1a import derive_prefill, export

                def result_hook(model, raw, request, helpers, profiles):
                    report = derive_prefill(context.impl, model, raw, request, *helpers, profiles)
                    export(report, args.fig1a_output.resolve(), plot=True)

            value = execute_cell(context, spec, case, catalog, output, runtime, result_hook=result_hook)
            all_rows.extend(value['summary'])
            print('FINISHED', spec['name'], case['id'], now(), flush=True)
    path = output / 'request-summary.csv'
    with path.open('w', newline='') as stream:
        fields = list(dict.fromkeys(key for row in all_rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    manifest = read(output / 'manifest.json')
    manifest.update(status='complete', completed_utc=now(), rows=len(all_rows), summary_sha256=sha(path))
    save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
