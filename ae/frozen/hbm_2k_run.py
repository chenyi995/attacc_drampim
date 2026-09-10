#!/usr/bin/env python3
"""Independent EPIC chunk and batch replay with native GPU HBM accounting.

Prepare freezes shape-only decisions before replay. Run preserves all raw native
Ramulator artifacts, including exact inherited recordings and new cache misses.
An explicit common-workload selection must pass the independent capacity review
before prepare can freeze anything. Model TP values remain the prior repair's.
Exact inherited profiles are hard-linked only after full identity and artifact
verification; complete inherited files are never opened for writing.
"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import datetime
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'archive/hbm-2k-batch'
PLAN = OUT/'plan.json'
SELECTION = OUT/'inputs-execution/execution-selection.json'
PARENT_PLAN = ROOT/'archive/repair-0908/plan.json'
REVISION = 'hbm-2k-batch-complete-request-v1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('FUGUE_JOBS', '6')
import run_coverage_expansion as backbone
import coverage_status as status_source
from coverage_status import canonical
import hbm_2k_staging as staging


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


sha = backbone.sha
save = backbone.save


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def f2_service(readback, attention, export):
    """Readback precedes attention; the ready complete KV can export in parallel."""
    assert all(math.isfinite(x) and x >= 0 for x in (readback, attention, export))
    return readback + max(attention, export)


def repaired_simulate():
    """Adapt F2 overlap and baseline remote-capacity eligibility, not GPU HBM."""
    tree = ast.parse(Path(backbone.__file__).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'simulate')
    target = ast.parse("d['gpu_kernel']+d['fetch']+write", mode='eval').body
    replacement = ast.parse("f2_service(d['fetch'], d['gpu_kernel'], write)", mode='eval').body
    class Change(ast.NodeTransformer):
        count = 0
        remote_count = 0
        GPU_count = 0
        def visit_Assign(self, n):
            if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'st':
                if ast.dump(n.value) == ast.dump(target):
                    n.value = deepcopy(replacement)
                    self.count += 1
            return n
        def visit_If(self, n):
            if ast.dump(n.test) == ast.dump(ast.parse("hasattr(m,'coverage_remote_budget')", mode='eval').body):
                n.test = ast.BoolOp(op=ast.And(), values=[n.test,
                    ast.parse("variant in ('F2','F3','F4')", mode='eval').body])
                self.remote_count += 1
            return self.generic_visit(n)
        def visit_Assert(self, n):
            target = ast.parse("per_gpu<=m.gconf['MEM_CAPACITY_PER_DEVICE']", mode='eval').body
            if ast.dump(n.test) == ast.dump(target):
                self.GPU_count += 1
                condition = ast.parse("not (baseline_route['mode']=='offloaded' and variant in ('F0','F1'))", mode='eval').body
                return ast.If(test=condition, body=[n], orelse=[])
            return self.generic_visit(n)
    change = Change()
    fixed = ast.fix_missing_locations(change.visit(deepcopy(node)))
    assert change.count == 1, 'Frozen F2 expression changed; do not silently adapt.'
    assert change.remote_count == 1, 'Frozen remote-capacity guard changed.'
    assert change.GPU_count == 1, 'Frozen GPU capacity assertion changed.'
    fixed.body.insert(0, ast.parse("baseline_route = staging.residency(case,m)\nassert baseline_route['passed'], ('GPU current-layer staging capacity', case['id'], baseline_route)").body[0])
    fixed.body.insert(1, ast.parse("assert baseline_route['passed'], ('GPU current-layer staging capacity', case['id'], baseline_route)").body[0])
    ast.fix_missing_locations(fixed)
    namespace = dict(vars(backbone), f2_service=f2_service, staging=staging)
    exec(compile(ast.Module(body=[fixed], type_ignores=[]), str(Path(__file__)), 'exec'), namespace)
    return namespace['simulate']


@lru_cache(maxsize=1)
def case_checker():
    """Keep all source accounting gates; exempt only F0/F1 remote capacity."""
    tree = ast.parse(Path(status_source.__file__).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_case')
    target = ast.parse("row['remote_peak_KV_GiB']*2**30 <= spec['selected']['remote_budget_bytes']", mode='eval').body
    class Change(ast.NodeTransformer):
        count = 0
        def visit_Compare(self, n):
            if ast.dump(n) == ast.dump(target):
                self.count += 1
                return ast.BoolOp(op=ast.Or(), values=[
                    ast.parse("variant in ('F0','F1')", mode='eval').body, n])
            return self.generic_visit(n)
    change = Change()
    fixed = ast.fix_missing_locations(change.visit(deepcopy(node)))
    assert change.count == 1, 'Frozen independent remote-capacity check changed.'
    namespace = dict(vars(status_source))
    exec(compile(ast.Module(body=[fixed], type_ignores=[]), str(Path(__file__)), 'exec'), namespace)
    return namespace['check_case']


def check_case(value, case, spec):
    check = case_checker()(value, case, spec)
    if 'baseline_staging' in value:
        from hbm_2k_model import Model
        model = Model(spec['name'], spec['selected']['tensor_parallel'], None)
        staged = staging.verify(value, case, model)
        check = dict(passed=check['passed'] and staged['passed'],
                     checks=check['checks']+staged['checks'],
                     failures=check['failures']+staged['failures'], staging=staged)
    return check


def load_plan():
    plan = read(PLAN)
    assert plan['status'] == 'frozen' and plan['correction_identity'] == REVISION
    for item in plan['sources']:
        assert sha(item['path']) == item['sha256'], item['path']
    for key in ('policy', 'calibration'):
        assert sha(plan[key+'_path']) == plan[key+'_sha256']
    assert sha(plan['catalog_path']) == plan['catalog_sha256']
    assert plan['common_workload_matrix'] is True
    intended = [selection['case']['id'] for selection in plan['selected']]
    assert intended and len(intended) == len(set(intended))
    assert plan['intended_case_count'] == len(plan['models'])*len(intended)
    return plan


def known_predictions(plan):
    from hbm_2k_model import Model, PointSelector
    catalog = {c['chunk_id']: c for c in backbone.read_lines(Path(plan['catalog_path']))}
    view, moved = backbone.source_helpers(catalog)
    rows = []
    for spec in plan['models']:
        m = Model(spec['name'], spec['selected']['tensor_parallel'], None, prepare_profiles=False)
        selector = PointSelector(plan['calibration_path'], m.h, m.gqa, m.trace_dhead, m.dbyte,
                                 policy=plan['policy_path'])
        for selected in plan['selected']:
            c = selected['case']; n, q, b = c['n'], c['q'], len(c['members'])
            for lp in ([0, 1, 2] if c['layer_policy'] == 'cacheblend' else [2]):
                aq = n if lp == 0 else q; updated = n if lp < 2 else q
                objects, views = view(c, 'full' if lp < 2 else 'partial', aq)
                gt, _ = m.gpu_attention(aq, n, b); sm, _ = m.softmax(aq, n, b)
                qi, desc = moved(m, views, aq); qin, _ = m.link(qi+desc)
                output, _ = m.link(aq*b*m.qbytes)
                fetch, _ = m.link((n-updated)*b*m.kvbytes)
                write, _ = m.link(updated*b*m.kvbytes)
                costs = dict(gpu_us=max(gt+fetch, write), softmax_us=sm, qin_us=qin+output, write_us=write)
                prediction = selector.predict_views(objects, views, **costs)
                rows.append(dict(model=m.name, case_id=c['id'], layer_class=lp,
                                 prediction=prediction, costs=costs, geometry=m.geometry_audit))
    return rows


def execution_selection(path=SELECTION):
    """Require a reviewed common matrix; never silently drop model/case cells."""
    import hbm_2k_execution_inputs as inputs
    from hbm_2k_model import Model
    receipt = inputs.verify()
    selected = read(path)
    assert selected['schema'] == inputs.SELECTION_SCHEMA and selected['status'] == 'frozen'
    assert selected['manifest_sha256'] == receipt['manifest_sha256']
    assert sha(path) == receipt['execution_selection_sha256']
    capacity_path = Path(selected['preflight_path'])
    assert sha(capacity_path) == selected['preflight_sha256'] == receipt['preflight_sha256']
    capacity = read(capacity_path)
    assert capacity['schema'] == inputs.PREFLIGHT_SCHEMA and capacity['status'] == 'complete'
    assert capacity['all_selected_model_cases_eligible'] and not capacity['performance_simulated']
    parent = read(PARENT_PLAN)
    assert selected['model_names'] == [model['name'] for model in parent['models']], 'Keep all parent models in source order.'
    assert capacity['models'] == parent['models'], 'TP, model geometry and budgets must remain unchanged.'
    ids = selected['workload_ids']
    by_id = {row['case']['id']: row for row in receipt['selected']}
    assert ids and len(ids) == len(set(ids)) and set(ids) == set(by_id)
    assert selected['model_cases'] == [{key: row[key] for key in
        ('model', 'workload_id', 'eligible', 'baseline_residency')} for row in capacity['model_cases']]
    pairs = {(row['model'], row['workload_id']): row for row in capacity['model_cases']}
    assert len(pairs) == len(capacity['model_cases']) == len(parent['models'])*len(ids)
    assert set(pairs) == {(spec['name'], identifier) for spec in parent['models'] for identifier in ids}
    for spec in parent['models']:
        model = Model(spec['name'], spec['selected']['tensor_parallel'], None)
        for identifier in ids:
            reviewed = pairs[spec['name'], identifier]
            assert reviewed['eligible'] and all(row['eligible'] for row in reviewed['variants'])
            actual = staging.residency(by_id[identifier]['case'], model)
            assert all(actual[key] == value for key, value in reviewed['baseline_residency'].items() if key != 'scope')
    assert selected['exclusions'] == capacity['exclusions']
    assert all(row.get('reason') for row in selected['exclusions'])
    return selected, receipt, [deepcopy(by_id[identifier]) for identifier in ids]


def prepare(selection_path=SELECTION):
    if PLAN.exists():
        return load_plan()
    from hbm_2k_model import geometry, Model, write_policy, CALIBRATION_PATH
    from repair_0908_traces import capacity_spec
    backbone.verify_native()
    selection, receipt, selected = execution_selection(selection_path)
    old = read(PARENT_PLAN)
    models = deepcopy(old['models'])
    for spec in models:
        spec['geometry'] = geometry(spec['name'])
        spec['geometry_consistent'] = True
        m = Model(spec['name'], spec['selected']['tensor_parallel'], None, prepare_profiles=False)
        cap = capacity_spec(m.trace_dhead, m.dbyte)
        spec['selected']['heads_per_HBM'] = m.h
        spec.update(resident_queries=cap['capacity'], query_buffer_bytes=cap['buffer_bytes'],
                    query_slice_bytes_per_query=cap['padded_query_bytes'], repair_geometry=m.geometry_audit)
        assert m.tp*m.remote_stacks_per_gpu == spec['selected']['PIM_HBM_stacks']
    evidence = ROOT/'docs/session/local/attention-shape-residency/megatron-source.json'
    source_table = read(evidence)
    for name, row in zip(('MT-76B', 'MT-146B'), source_table['rows']):
        g = geometry(name)
        assert (g['hdim'], g['num_heads'], g['ndec']) == (row['hidden'], row['heads'], row['layers'])
    policy = write_policy(additional_sources=[Path(__file__), Path(staging.__file__),
                                              ROOT/'docs/analysis/test_hbm_2k_model.py', evidence])
    input_manifest = read(receipt['manifest_path'])
    paths = [Path(__file__), Path(backbone.__file__), ROOT/'docs/analysis/coverage_status.py',
             ROOT/'docs/analysis/coverage_trace_cache.py', ROOT/'docs/analysis/repair_0908_model.py',
             ROOT/'docs/analysis/repair_0908_traces.py', ROOT/'docs/analysis/selector_point_model.py',
             ROOT/'docs/analysis/hbm_2k_model.py', ROOT/'docs/analysis/hbm_2k_inputs.py',
             ROOT/'docs/analysis/hbm_2k_execution_inputs.py', ROOT/'docs/analysis/hbm_2k_capacity.py',
             ROOT/'docs/analysis/hbm_2k_staging.py',
             ROOT/'docs/analysis/test_hbm_2k_model.py', ROOT/'docs/analysis/repair_0908.py',
             ROOT/'docs/analysis/selector_cost_model.py', Path(receipt['catalog_path']),
             PARENT_PLAN, Path(selection_path), Path(selection['preflight_path']), Path(receipt['manifest_path']),
             policy, CALIBRATION_PATH, evidence, source_table['path']]
    paths += [Path(item['path']) for item in input_manifest['sources']]
    paths += [Path(item['path']) for item in input_manifest['files'].values()]
    paths += sorted((backbone.STAGE/'fugue').glob('*.py'))
    paths += [backbone.STAGE/'fugue/official-gqa-geometry.json']
    paths = sorted({Path(path).resolve() for path in paths}, key=str)
    limits = deepcopy(old['resources'])
    assert limits['cpu_affinity_limit'] <= 96 and limits['memory_budget_bytes'] <= 600_000_000_000
    assert limits['model_workers']*(1+limits['trace_jobs_per_model']) <= limits['cpu_affinity_limit']
    assert limits['model_workers']*(1+limits['trace_jobs_per_model'])*limits['address_space_bytes_per_process'] < limits['memory_budget_bytes']
    plan = dict(status='frozen', schema=REVISION, correction_identity=REVISION, created_utc=now(),
        models=models, selected=selected, common_workload_matrix=True,
        intended_case_count=len(models)*len(selected),
        catalog_path=receipt['catalog_path'], catalog_sha256=sha(receipt['catalog_path']),
        input_manifest_path=receipt['manifest_path'], input_manifest_sha256=receipt['manifest_sha256'],
        execution_selection_path=str(Path(selection_path).resolve()), execution_selection_sha256=sha(selection_path),
        capacity_report_path=selection['preflight_path'], capacity_report_sha256=selection['preflight_sha256'],
        excluded_workloads=deepcopy(selection['exclusions']),
        sources=[record(p) for p in paths], policy_path=str(policy), policy_sha256=sha(policy),
        calibration_path=str(CALIBRATION_PATH), calibration_sha256=sha(CALIBRATION_PATH),
        scope='Source-derived EPIC chunk controls and synchronous reader batches, with unchanged MuSiQue. All parent models retain TP. Logical-head GPU operators use unchanged native HBM traffic/bandwidth/utilization with no GQA L2 discount. Corrected MT geometry, synchronous dimension-bounded QK/PV groups, and F2 readback plus overlapping attention/export are preserved. Native command timing and analytical GPU costs; no numerical inference or new hardware calibration.',
        selection_scope='Input and capacity selection frozen before performance replay; a common workload matrix across all models, with excluded input controls recorded explicitly.',
        execution='Frozen source simulate body with the same asserted F2 AST expression replacement; new independent GPU model and immutable prior PIM TraceBank.',
        remote_capacity_scope='F0/F1 have no remote capacity ceiling. A full-horizon GPU-capacity test selects either the original resident baseline or explicit one-layer GPU staging over a complete remote request cache. F2/F3/F4 retain their PIM remote budget. All variants retain the original GPU-local HBM budget and native transfer bandwidth/per-byte energy.',
        staging_scope='For offloaded F0/F1 only, prefill exports complete current-layer KV after QKV readiness, overlapping attention. F1 retained-KV readback precedes this fork. Decode reads old KV first, then overlaps attention with the one-token KV write. Common operators never enter this overlap, and all operation energies remain additive. The before-ledger is an invalid-capacity intermediate, never a completed result.',
        resources=limits,
        old_raw_preserved=True, new_raw_retained=True,
        limitations=['Ideal ready-KV export/attention overlap excludes general DMA and cache contention.',
                    'Geometry and resident checks do not establish numerical inference correctness.',
                    'Known native multiwave value-address defects retain independent publication qualification.'])
    predictions = known_predictions(plan)
    path = OUT/'frozen-predictions.json'
    save(path, dict(created_utc=now(), policy_sha256=sha(policy), records=predictions,
                   scope='Shape-only predictions frozen before repaired performance replay. Coefficients unchanged.'))
    plan['predictions_path'] = str(path); plan['predictions_sha256'] = sha(path)
    plan['sources'].append(record(path))
    save(PLAN, plan)
    print(json.dumps(dict(plan=str(PLAN), models=len(models), workloads=len(selected), predictions=len(predictions))), flush=True)
    return plan


def cache_bank():
    from repair_0908_traces import TraceBank
    from coverage_trace_cache import encoded, file_lock
    class InheritedExactBank(TraceBank):
        def run(self, name, lines, residents, metadata=None):
            data = ('\n'.join(lines)+'\n').encode()
            config = self._yaml(self.root/name, residents)
            identity = dict(implementation=self.implementation, trace_sha256=hashlib.sha256(data).hexdigest(),
                            trace_bytes=len(data), residents=list(residents), effective_yaml=config)
            identifier = hashlib.sha256(encoded(identity)).hexdigest()
            target = self.cache_root/'profiles'/identifier
            with file_lock(self.cache_root/'locks'/('profile-'+identifier+'.lock')):
                if not target.exists():
                    for parent_name, parent_root in [('repair-0908', ROOT/'archive/repair-0908/exact-trace-cache'),
                                                      ('coverage', backbone.BASE/'exact-trace-cache')]:
                        prior = parent_root/'profiles'/identifier
                        if not (prior/'complete.json').is_file():
                            continue
                        self._verify(prior, identity, 'ramulator')
                        # Only immutable canonical files are shared. Logical
                        # sidecars and metadata are always written under OUT.
                        temporary = target.with_name(identifier+'.inherit-'+str(os.getpid()))
                        shutil.copytree(prior, temporary, copy_function=os.link)
                        temporary.rename(target)
                        self.cache_stats['inherited_exact_profile'] += 1
                        self.cache_stats['inherited_from_'+parent_name] += 1
                        break
            return super().run(name, lines, residents, metadata)
    return InheritedExactBank


class RecordedSelector:
    def __init__(self, model, plan):
        from hbm_2k_model import PointSelector
        self.predictor = PointSelector(plan['calibration_path'], model.h, model.gqa, model.trace_dhead,
                                       model.dbyte, policy=plan['policy_path'])
        self.calls = []
    def choose_views(self, objects, views, gpu_us, softmax_us, qin_us, write_us):
        result = self.predictor.predict_views(objects, views, gpu_us, softmax_us, qin_us, write_us)
        self.calls.append(result)
        return result['choice'], result['service_us']


def timeline_check(value, case, model):
    rows = []
    for block in value['event_blocks']:
        if block['variant'] != 'F2' or block['phase'] != 'prefill':
            continue
        lp = block['layer_class']; n = case['n']; b = len(case['members'])
        q = n if lp == 0 else case['q']; updated = n if lp < 2 else case['q']
        attention, ae = model.gpu_attention(q, n, b)
        fetch, fe = model.link((n-updated)*b*model.kvbytes)
        export, we = model.link(n*b*model.kvbytes)
        component = next(c for c in block['components'] if c[0] == 'attention_service')
        expected = f2_service(fetch, attention, export)
        assert math.isclose(component[1], expected, rel_tol=1e-12)
        assert math.isclose(component[2], ae+fe+we, rel_tol=1e-12)
        rows.append(dict(layer_class=lp, readback_us=fetch, attention_us=attention,
                         export_us=export, service_us=expected, old_serialized_us=fetch+attention+export,
                         hidden_export_us=min(attention, export), energy_uJ=ae+fe+we))
    assert rows
    return rows


def profile_ledger(folder, bank):
    rows = []
    for path in sorted((folder/'profiles').glob('*/cache-reference.json')):
        ref = read(path); complete = Path(ref['canonical_path'])/'complete.json'
        assert sha(complete) == ref['complete_manifest_sha256']
        manifest = read(complete)
        assert manifest['complete'] and manifest['identity']['implementation'] == bank.implementation
        rows.append(dict(profile=ref['requested_profile'], canonical_path=str(complete.parent),
                         complete_path=str(complete), complete_sha256=sha(complete), trace_sha256=ref['trace_sha256'],
                         cache_hit=ref['cache_hit'], logical_files=[record(p) for p in sorted(path.parent.iterdir()) if p.is_file()]))
    return rows


def worker(name, only_case=None):
    plan = load_plan(); spec = next(m for m in plan['models'] if m['name'] == name)
    limit = plan['resources']['address_space_bytes_per_process']
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:plan['resources']['cpu_affinity_limit']])
    from hbm_2k_model import Model, geometry
    folder = OUT/'runs'/name; backbone.runtime(folder/'runtime')
    bank = cache_bank()(folder/'profiles', folder/'runtime', geometry(name), OUT/'exact-trace-cache')
    model = Model(name, spec['selected']['tensor_parallel'], bank, prepare_profiles=False)
    model.selector = RecordedSelector(model, plan)
    model.coverage_remote_budget = spec['selected']['remote_budget_bytes']
    identity = dict(model=name, plan_sha256=sha(PLAN), correction_identity=REVISION,
                    native_implementation=bank.implementation, geometry=model.geometry_audit)
    ident = folder/'identity.json'
    if ident.exists(): assert read(ident) == identity
    else: save(ident, identity)
    catalog = {c['chunk_id']: c for c in backbone.read_lines(Path(plan['catalog_path']))}
    frozen = {(r['case_id'], r['layer_class']): r['prediction'] for r in read(plan['predictions_path'])['records'] if r['model'] == name}
    simulate = repaired_simulate(); finished, failed = [], []
    intended_ids = [row['case']['id'] for row in plan['selected']]
    assert only_case is None or only_case in intended_ids, 'Requested case is outside the frozen matrix.'
    for selected in plan['selected']:
        case = selected['case']; path = folder/'cases'/(case['id']+'.json')
        if only_case is not None and case['id'] != only_case: continue
        if path.exists():
            value = read(path)
            assert value['plan_sha256'] == sha(PLAN) and value['run_identity_sha256'] == sha(ident)
            assert check_case(value, case, spec)['passed']
            assert all(prediction == frozen[case['id'], decision['layer_class']]
                       for prediction, decision in zip(value['selector_predictions'], value['decisions']))
            finished.append(case['id']); continue
        print('START', name, case['id'], now(), flush=True)
        try:
            model.selector.calls = []; before = Counter(bank.cache_stats)
            intermediate = simulate(case, model, catalog)
            value = staging.apply(intermediate, case, model)
            assert len(model.selector.calls) == len(value['decisions'])
            for decision, prediction in zip(value['decisions'], model.selector.calls):
                assert prediction == frozen[case['id'], decision['layer_class']]
                assert prediction['choice'] == decision['choice']
            assert all(not audit['shared_off_HBM'] and audit['original_off_HBM_operands'] == audit['charged_off_HBM_operands']
                       for audit in model.gpu_traffic_audit.values()), 'GQA HBM discount must remain disabled.'
            value.update(case_sha256=canonical(case), plan_sha256=sha(PLAN), correction_identity=REVISION,
                run_identity_sha256=sha(ident), policy_sha256=plan['policy_sha256'],
                calibration_sha256=plan['calibration_sha256'], selector_predictions=model.selector.calls,
                geometry_audit=model.geometry_audit, gpu_traffic_audit=deepcopy(model.gpu_traffic_audit),
                f2_service_timeline=timeline_check(value, case, model), finished_utc=now(), scope=plan['scope'],
                cache_stats=dict(Counter(bank.cache_stats)-before))
            check = check_case(value, case, spec)
            assert check['passed'], check['failures']
            value['repair_checks'] = dict(check, predictions_equal_frozen=True, synchronous_qk_pv_capacity=model.mq_capacity,
                                          geometry_checked=True, f2_overlap_energy_preserved=True,
                                          native_gpu_hbm_traffic_unchanged=True,
                                          baseline_remote_capacity_unbounded=True,
                                          GPU_local_and_PIM_remote_capacity_enforced=True)
            save(path, value); finished.append(case['id'])
            print('FINISHED', name, case['id'], now(), flush=True)
        except Exception:
            failed.append(case['id'])
            save(folder/'failures'/(case['id']+'.json'), dict(model=name, case_id=case['id'], utc=now(), error=traceback.format_exc()))
            print('FAILED', name, case['id'], traceback.format_exc(), flush=True)
    save(folder/'profiles.json', dict(implementation=bank.implementation, profiles=profile_ledger(folder, bank), cache_stats=dict(bank.cache_stats)))
    requested = intended_ids if only_case is None else [only_case]
    complete = not failed and Counter(finished) == Counter(requested)
    save(folder/('complete.json' if only_case is None else 'partial.json'),
         dict(status='completed' if complete else 'incomplete', model=name, plan_sha256=sha(PLAN),
              finished=finished, failed=failed, requested=requested,
              full_input_list=only_case is None, all_requested_cases_validated=complete,
              identity_sha256=sha(ident), utc=now()))
    return int(not complete)


def run():
    plan = load_plan(); limits = plan['resources']
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:limits['cpu_affinity_limit']])
    # Each worker has at most trace_jobs simultaneous native children. Even if
    # all processes exhaust their address-space limits, this is below the budget.
    bounded = limits['model_workers']*(1+limits['trace_jobs_per_model'])*limits['address_space_bytes_per_process']
    assert bounded < limits['memory_budget_bytes']
    status = dict(status='running', plan_sha256=sha(PLAN), started_utc=now(), models=[],
                  affinity=sorted(os.sched_getaffinity(0)), descendant_address_space_bound_bytes=bounded)
    save(OUT/'launch.json', status)
    def one(spec):
        folder = OUT/'runs'/spec['name']; folder.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, FUGUE_JOBS=str(limits['trace_jobs_per_model']))
        with (folder/'worker.log').open('a') as stream:
            result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', spec['name']],
                                    stdout=stream, stderr=subprocess.STDOUT, env=env)
        return dict(model=spec['name'], returncode=result.returncode, utc=now())
    with ThreadPoolExecutor(max_workers=limits['model_workers']) as pool:
        for future in as_completed([pool.submit(one, m) for m in plan['models']]):
            item = future.result(); status['models'].append(item)
            save(OUT/'launch.json', status); print(json.dumps(item), flush=True)
    status.update(status='completed' if all(x['returncode'] == 0 for x in status['models']) else 'incomplete', finished_utc=now())
    save(OUT/'launch.json', status)
    return int(status['status'] != 'completed')


def self_test():
    # Catches the two forbidden overlap interpretations and preserves energy.
    for r, a, w in ((7, 11, 3), (7, 3, 11), (0, 5, 5), (10, 0, 2)):
        t = f2_service(r, a, w)
        assert t >= r+a and t >= r+w
        assert math.isclose((r+a+w)-t, min(a, w))
    repaired_simulate()
    case_checker()
    staging.self_test(repaired_simulate())
    # Saved results provide an accounting fixture; changing only the supplied
    # remote ceiling demonstrates that PIM variants retain the capacity gate.
    parent = read(PARENT_PLAN)
    spec = deepcopy(next(m for m in parent['models'] if m['name'] == 'LLAMA-7B'))
    case = parent['selected'][0]['case']
    value = read(ROOT/'archive/repair-0908/runs'/spec['name']/'cases'/(case['id']+'.json'))
    assert check_case(value, case, spec)['passed']
    spec['selected']['remote_budget_bytes'] = 0
    failures = check_case(value, case, spec)['failures']
    assert {label for label in failures if label.endswith('remote capacity')} == {
        variant+' remote capacity' for variant in ('F2', 'F3', 'F4')}
    spec['selected']['GPU_budget_bytes_per_device'] = 0
    failures = check_case(value, case, spec)['failures']
    assert {label for label in failures if label.endswith('GPU capacity')} == {
        variant+' GPU capacity' for variant in ('F0', 'F1', 'F2', 'F3', 'F4')}
    print('F2 overlap, native-HBM capacity boundaries and frozen-source AST checks passed; no simulation or freeze.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--case')
    parser.add_argument('--selection', type=Path, default=SELECTION)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test: self_test()
    if args.prepare: prepare(args.selection)
    if args.worker: raise SystemExit(worker(args.worker, args.case))
    if args.run: raise SystemExit(run())
