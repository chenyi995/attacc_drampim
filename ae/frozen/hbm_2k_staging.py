#!/usr/bin/env python3
"""Capacity-triggered GPU baseline staging over immutable operator costs.

Only F0/F1 that cannot retain the complete configured KV horizon use this path.
The GPU keeps one complete current-layer KV buffer; the full request cache lives
remotely. This is an explicit analytical staging schedule, not a GPU kernel or
DMA-contention measurement. No function in this module runs a simulator.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math

SCHEMA = 'hbm-2k-capacity-triggered-gpu-staging-v1'
BASELINES = ('F0', 'F1')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def residency(case, model):
    """Choose one path using the known full configured output horizon."""
    batch, layers, tp = len(case['members']), model.layers, model.tp
    final_tokens = case['n'] + case['output'] - 1
    weights, _, temporary = model.capacity(batch, case['n'], case['output'])
    layer_bytes = final_tokens*batch*model.kvbytes*tp
    resident_bytes = layer_bytes*layers
    budget = model.gconf['MEM_CAPACITY_PER_DEVICE']
    resident_total = (weights+temporary+resident_bytes)/tp
    staging_total = (weights+temporary+layer_bytes)/tp
    mode = 'resident' if resident_total <= budget else 'offloaded'
    passed = (resident_total if mode == 'resident' else staging_total) <= budget
    return dict(schema=SCHEMA, mode=mode, passed=passed, batch=batch, layers=layers,
        tensor_parallel=tp, final_context_tokens=final_tokens,
        resident_GPU_KV_peak_bytes=resident_bytes, staging_GPU_KV_peak_bytes=layer_bytes,
        weights_bytes=weights, temp_bytes=temporary, GPU_budget_bytes_per_device=budget,
        resident_total_per_device_bytes=resident_total, staging_total_per_device_bytes=staging_total,
        decision_scope='One predeclared path for the known complete output horizon, without mid-request residency changes.',
        remote_scope='F0/F1 remote capacity is unbounded; PIM variants retain their budget. Remote bandwidth and per-byte energy remain unchanged.')


def snapshot_bundle(value):
    return dict(summary=[deepcopy(row) for row in value['summary'] if row['variant'] in BASELINES],
                event_blocks=[deepcopy(row) for row in value['event_blocks'] if row['variant'] in BASELINES],
                storage={variant: deepcopy(value['storage'][variant]) for variant in BASELINES})


def unaffected_bundle(value):
    return dict(summary=[row for row in value['summary'] if row['variant'] not in BASELINES],
                event_blocks=[row for row in value['event_blocks'] if row['variant'] not in BASELINES],
                storage={variant: state for variant, state in value['storage'].items() if variant not in BASELINES},
                decisions=value['decisions'], decode_profiles=value['decode_profiles'], warmup=value['warmup'])


def service_terms(case, model, block):
    """Costs per layer: read first, then attention and write in parallel."""
    variant, phase = block['variant'], block['phase']
    assert variant in BASELINES
    batch = len(case['members'])
    if phase == 'prefill':
        query = case['n'] if variant == 'F0' or block['layer_class'] == 0 else case['q']
        context = case['n']
        updated = case['n'] if block['layer_class'] < 2 else case['q']
        read_tokens = 0 if variant == 'F0' else context-updated
        write_tokens = context
    else:
        assert phase == 'decode' and block['step'] > 0
        query, context = 1, case['n']+block['step']
        read_tokens, write_tokens = context-1, 1
    read_bytes = read_tokens*batch*model.kvbytes
    write_bytes = write_tokens*batch*model.kvbytes
    attention, attention_energy = model.gpu_attention(query, context, batch)
    read_time, read_energy = model.link(read_bytes)
    write_time, write_energy = model.link(write_bytes)
    return dict(variant=variant, phase=phase, step=block['step'], layer_class=block['layer_class'],
        repetitions=block['repetitions'], query_tokens=query, context_tokens=context,
        batch=batch, read_tokens_per_request=read_tokens, write_tokens_per_request=write_tokens,
        readback_bytes_per_GPU=read_bytes, export_bytes_per_GPU=write_bytes,
        GPU_attention_us=attention, readback_us=read_time, export_us=write_time,
        GPU_attention_energy_uJ=attention_energy, readback_energy_uJ=read_energy,
        export_energy_uJ=write_energy, service_us=read_time+max(attention, write_time),
        service_energy_uJ=attention_energy+read_energy+write_energy,
        hidden_export_us=min(attention, write_time),
        scope='Times are per-layer wall time; bytes per GPU use KV heads once, without a GQA query-head multiplier. Energy includes all GPUs. Common operators are outside the overlap.')


def apply(value, case, model):
    """Return a new result; preserve the original intermediate baseline ledger."""
    route = residency(case, model)
    assert route['passed'], ('Current-layer staging exceeds GPU capacity', model.name, case['id'], route)
    result = deepcopy(value)
    for row in result['summary']:
        if row['variant'] in BASELINES:
            row['gpu_kv_mode'] = 'resident' if route['mode'] == 'resident' else 'offloaded-pending'
    before = snapshot_bundle(result)
    unchanged = fingerprint(unaffected_bundle(value))
    result['baseline_residency'] = route
    if route['mode'] == 'resident':
        result['baseline_staging'] = dict(schema=SCHEMA, applied=False,
            scope='The full configured KV horizon fits GPU HBM; original resident baseline values are retained exactly.',
            before=before, after=deepcopy(before), timeline=[], unaffected_sha256=unchanged)
        return result
    timeline = []
    for block in result['event_blocks']:
        if block['variant'] not in BASELINES:
            continue
        terms = service_terms(case, model, block)
        matches = [i for i, component in enumerate(block['components']) if component[0] == 'attention_service']
        assert len(matches) == 1
        index = matches[0]
        terms['before_attention_service_us'] = block['components'][index][1]
        terms['before_attention_service_energy_uJ'] = block['components'][index][2]
        terms['before_event_time_us'] = block['time_us']
        terms['before_event_energy_uJ'] = block['energy_uJ']
        block['components'][index] = ['attention_service', terms['service_us'], terms['service_energy_uJ']]
        block['time_us'] = math.fsum(component[1] for component in block['components'])*block['repetitions']
        block['energy_uJ'] = math.fsum(component[2] for component in block['components'])*block['repetitions']
        terms['after_event_time_us'], terms['after_event_energy_uJ'] = block['time_us'], block['energy_uJ']
        timeline.append(terms)
    batch = len(case['members'])
    unit = batch*model.kvbytes*model.tp
    for variant in BASELINES:
        state = result['storage'][variant]
        state['active'] = 0
        private = 0
        counts = {block['layer_class']: block['repetitions'] for block in result['event_blocks']
                  if block['variant'] == variant and block['phase'] == 'prefill'}
        for snapshot in state['snapshots']:
            if snapshot['phase'] == 'prefill':
                tokens = case['n']
                private += counts[snapshot['layer_class']]*tokens*unit
            else:
                tokens = case['n']+snapshot['step']
                private = model.layers*tokens*unit
            snapshot.update(GPU_KV_bytes=tokens*unit,
                remote_shared_bytes=state['remote_pool'], remote_private_bytes=private,
                total_KV_bytes=tokens*unit+state['remote_pool']+private)
        state['private'] = private
        row = next(row for row in result['summary'] if row['variant'] == variant)
        row['gpu_kv_mode'] = 'offloaded'
        for suffix, field in [('_ms', 'time_us'), ('_energy_mJ', 'energy_uJ')]:
            prefill = math.fsum(block[field] for block in result['event_blocks']
                               if block['variant'] == variant and block['phase'] == 'prefill')
            decode = math.fsum(block[field] for block in result['event_blocks']
                              if block['variant'] == variant and block['phase'] == 'decode')
            row['TTFT'+suffix] = prefill/1000
            row['TBT'+suffix] = decode/(case['output']-1)/1000
            row['E2E'+suffix] = (prefill+decode)/1000
        peak_gpu = max(snapshot['GPU_KV_bytes'] for snapshot in state['snapshots'])
        row['GPU_peak_KV_GiB'] = peak_gpu/2**30
        row['remote_peak_KV_GiB'] = max(snapshot['remote_shared_bytes']+snapshot['remote_private_bytes']
                                      for snapshot in state['snapshots'])/2**30
        row['simultaneous_peak_KV_GiB'] = max(snapshot['total_KV_bytes'] for snapshot in state['snapshots'])/2**30
        row['GPU_peak_total_per_device_GiB'] = (route['weights_bytes']+route['temp_bytes']+peak_gpu)/model.tp/2**30
    result['baseline_staging'] = dict(schema=SCHEMA, applied=True, before=before,
        after=snapshot_bundle(result), timeline=timeline, unaffected_sha256=unchanged,
        scope='Before contains a nonpublishable resident-path intermediate. After uses one current-layer GPU KV buffer and a remote complete request cache, with post-write/pre-release simultaneous snapshots. No interlayer prefetch, cache benefit, or DMA-contention model is added.')
    assert fingerprint(unaffected_bundle(result)) == unchanged
    checked = verify(result, case, model)
    assert checked['passed'], checked['failures']
    result['baseline_staging']['checks'] = checked
    return result


def verify(value, case, model):
    """Independent identities over the saved before/after events and snapshots."""
    failures, checks = [], 0
    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(label)
    def close(a, b, label):
        check(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-8), label)
    route = residency(case, model)
    ledger = value['baseline_staging']
    check(value['baseline_residency'] == route and route['passed'], 'Whole-horizon residency and current-layer GPU capacity')
    check(ledger['schema'] == SCHEMA, 'Staging ledger identity')
    check(fingerprint(unaffected_bundle(value)) == ledger['unaffected_sha256'], 'All PIM paths, decisions, scan profiles and warmup unchanged')
    check(snapshot_bundle(value) == ledger['after'], 'Saved after-ledger matches published baseline rows')
    if route['mode'] == 'resident':
        check(not ledger['applied'] and not ledger['timeline'], 'Resident path has no staging operations')
        check(ledger['before'] == ledger['after'], 'Resident baseline bytes unchanged')
        return dict(passed=not failures, checks=checks, failures=failures)
    check(ledger['applied'], 'Offloaded path explicitly applied')
    old_blocks = {(b['variant'], b['phase'], b['step'], b['layer_class']): b for b in ledger['before']['event_blocks']}
    new_blocks = {(b['variant'], b['phase'], b['step'], b['layer_class']): b for b in ledger['after']['event_blocks']}
    keys = [(t['variant'], t['phase'], t['step'], t['layer_class']) for t in ledger['timeline']]
    check(set(keys) == set(old_blocks) == set(new_blocks) and len(keys) == len(set(keys)), 'One timeline record per baseline event')
    batch, unit = len(case['members']), len(case['members'])*model.kvbytes*model.tp
    for record in ledger['timeline']:
        key = record['variant'], record['phase'], record['step'], record['layer_class']
        old, new = old_blocks[key], new_blocks[key]
        check([c for c in old['components'] if c[0] != 'attention_service'] ==
              [c for c in new['components'] if c[0] != 'attention_service'], 'Common and software selection costs unchanged')
        read, attention, write = record['readback_us'], record['GPU_attention_us'], record['export_us']
        close(record['service_us']-read, max(attention, write), 'Readback precedes attention/write fork')
        close(read+attention+write-record['service_us'], min(attention, write), 'Only the shorter attention/export branch is hidden')
        close(record['service_energy_uJ'], sum(record[k] for k in
            ('GPU_attention_energy_uJ', 'readback_energy_uJ', 'export_energy_uJ')), 'Overlap never removes energy')
        phase, variant = record['phase'], record['variant']
        if phase == 'decode':
            tokens = case['n']+record['step']
            expected_read, expected_write = (tokens-1)*batch*model.kvbytes, batch*model.kvbytes
            expected_query = 1
        else:
            tokens = case['n']
            expected_read = 0 if variant == 'F0' or record['layer_class'] < 2 else (tokens-case['q'])*batch*model.kvbytes
            expected_write = tokens*batch*model.kvbytes
            expected_query = tokens if variant == 'F0' or record['layer_class'] == 0 else case['q']
        close(record['readback_bytes_per_GPU'], expected_read, 'Old/retained KV read once without a GQA multiplier')
        close(record['export_bytes_per_GPU'], expected_write, 'Complete prefill export or one-token decode update')
        native_time, native_energy = model.gpu_attention(expected_query, tokens, batch)
        close(attention, native_time, 'Native GPU attention cost unchanged')
        close(record['GPU_attention_energy_uJ'], native_energy, 'Native GPU attention energy unchanged')
        for prefix, size in [('readback', expected_read), ('export', expected_write)]:
            time, energy = model.link(size)
            close(record[prefix+'_us'], time, 'Native link bandwidth '+prefix)
            close(record[prefix+'_energy_uJ'], energy, 'Native link energy '+prefix)
        actual = next(c for c in new['components'] if c[0] == 'attention_service')
        close(actual[1], record['service_us'], 'Event contains staged service time')
        close(actual[2], record['service_energy_uJ'], 'Event contains full staged energy')
    for variant in BASELINES:
        before = ledger['before']['storage'][variant]
        state = value['storage'][variant]
        check(state['remote_pool'] == before['remote_pool'], 'Immutable pool retained separately')
        completed_layers = 0
        for snap in state['snapshots']:
            if snap['phase'] == 'prefill':
                completed_layers += new_blocks[variant, 'prefill', 0, snap['layer_class']]['repetitions']
                tokens, layer_count = case['n'], completed_layers
            else:
                tokens, layer_count = case['n']+snap['step'], model.layers
            close(snap['GPU_KV_bytes'], tokens*unit, 'Exactly one complete current layer near GPU')
            close(snap['remote_private_bytes'], tokens*unit*layer_count, 'All processed complete layer caches retained remotely')
            close(snap['remote_shared_bytes'], before['remote_pool'], 'Shared pool neither replaced nor merged with private cache')
            close(snap['total_KV_bytes'], sum(snap[k] for k in ('GPU_KV_bytes', 'remote_shared_bytes', 'remote_private_bytes')), 'Simultaneous post-write/pre-release capacity')
        row = next(row for row in value['summary'] if row['variant'] == variant)
        close(row['GPU_peak_KV_GiB']*2**30, route['staging_GPU_KV_peak_bytes'], 'One-layer peak spans the complete output horizon')
        check(row['GPU_peak_total_per_device_GiB']*2**30 <= route['GPU_budget_bytes_per_device'], 'Original GPU HBM budget enforced')
    return dict(passed=not failures, checks=checks, failures=failures)


def self_test(simulate):
    """Tiny synthetic accounting fixture; no native timing/profile execution."""
    from types import SimpleNamespace
    class Model:
        name, tp, layers, heads, kvheads, gqa = 'UNIT-ACCOUNTING', 1, 2, 4, 1, 4
        qbytes, kvbytes = 4, 2
        m = dict(num_heads=heads, num_kv_heads=kvheads)
        coverage_remote_budget = 10000
        selector = SimpleNamespace(choose_views=lambda *args: ('GPU', 1000.))
        def __init__(self, budget):
            self.gconf = {'MEM_CAPACITY_PER_DEVICE': budget}
        def capacity(self, *args):
            return 5, 0, 1
        def common(self, *args):
            return [('common-fixture', 10., 100.)]
        def gpu_attention(self, q, n, b):
            return float(2+q), float((2+q)*11)
        def link(self, count):
            return count/4, count*7
        def softmax(self, *args):
            return 1., 1.
        def dense_scan(self, *args):
            return dict(scan_us=1., profile='synthetic-accounting-fixture')
        def shared_scan(self, *args):
            return dict(scan_us=1., first_private_k_us=1., profile='synthetic-accounting-fixture')
        def scan_energy(self, *args):
            return 1.
    case = dict(id='synthetic-accounting-fixture', workload_id='synthetic-accounting-fixture',
        source='unit-test', dataset='unit-test', scope='unit-test', n=12, q=3, output=3,
        members=[dict(chunk_ids=['a', 'b'], recomputed_indices=[0, 6, 11])],
        pool_ids=['a', 'b'], layer_policy='epic')
    catalog = {'a': dict(tokens=6), 'b': dict(tokens=6)}
    resident = Model(1000)
    original = simulate(case, resident, catalog)
    resident_result = apply(original, case, resident)
    assert resident_result['baseline_residency']['mode'] == 'resident'
    assert verify(resident_result, case, resident)['passed']
    for before, after in zip(original['summary'], resident_result['summary']):
        assert all(after[key] == value for key, value in before.items())
    offloaded = Model(50)
    intermediate = simulate(case, offloaded, catalog)
    result = apply(intermediate, case, offloaded)
    assert result['baseline_residency']['mode'] == 'offloaded'
    assert verify(result, case, offloaded)['passed']
    assert fingerprint(unaffected_bundle(intermediate)) == fingerprint(unaffected_bundle(result))
    by_event = {(r['variant'], r['phase'], r['step']): r for r in result['baseline_staging']['timeline']}
    # Explicit golden arithmetic: F0 compute-dominant, F1 export-dominant,
    # decode reads only old KV, and common costs remain entirely outside max.
    assert by_event['F0', 'prefill', 0]['service_us'] == 14
    assert by_event['F1', 'prefill', 0]['service_us'] == 10.5
    assert by_event['F1', 'prefill', 0]['service_energy_uJ'] == 55+126+168
    assert by_event['F1', 'prefill', 0]['after_event_time_us'] == 41
    assert by_event['F0', 'decode', 1]['readback_bytes_per_GPU'] == 24
    assert by_event['F1', 'decode', 1]['service_us'] == 9
    assert by_event['F1', 'decode', 2]['service_us'] == 9.5
    assert by_event['F1', 'decode', 1]['after_event_time_us'] == 38
    assert by_event['F1', 'decode', 1]['service_energy_uJ'] == 33+168+14
    for variant in BASELINES:
        snaps = result['storage'][variant]['snapshots']
        assert [s['GPU_KV_bytes'] for s in snaps] == [24, 26, 28]
        assert [s['remote_private_bytes'] for s in snaps] == [48, 52, 56]
        assert result['storage'][variant]['remote_pool'] == (0 if variant == 'F0' else 48)
        row = next(row for row in result['summary'] if row['variant'] == variant)
        assert row['GPU_peak_total_per_device_GiB']*2**30 == 34
    # Reject an incorrect new-token readback and an energy subtraction even
    # when ordinary total-event arithmetic could be adjusted to match them.
    broken = deepcopy(result)
    record = next(r for r in broken['baseline_staging']['timeline'] if r['phase'] == 'decode')
    record['readback_bytes_per_GPU'] += offloaded.kvbytes
    assert not verify(broken, case, offloaded)['passed']
    broken = deepcopy(result)
    broken['baseline_staging']['timeline'][0]['service_energy_uJ'] -= 1
    assert not verify(broken, case, offloaded)['passed']
    too_small = Model(33)
    assert not residency(case, too_small)['passed']
    try:
        simulate(case, too_small, catalog)
    except AssertionError:
        pass
    else:
        raise AssertionError('Current-layer GPU overflow must fail before any profile work.')
    return dict(passed=True, fixture='Synthetic small accounting unit test, not a performance experiment.',
                cases=['resident identity', 'capacity-selected staging', 'both overlap branches',
                       'old-token decode boundary', 'additive energy', 'simultaneous cache peaks',
                       'PIM path identity', 'corruption rejection', 'one-layer overflow rejection'])
