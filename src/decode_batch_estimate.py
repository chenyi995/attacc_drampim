"""Replay one measured decode structure at a larger virtual request batch.

GPU operations are repriced at the target batch. Each physical PIM channel's
duration is multiplied by target/base actual request count, as requested by
chenyi9. This is an extrapolation of repeated identical structures, not a
Ramulator measurement of a larger MQ sweep. No PIM simulator is invoked.
"""
from collections import defaultdict
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import make_model_config, make_xpu_config, SCALING_FACTOR
from src.devices import xPU
from src.model import Transformer
from src.type import DataType, DeviceType, GPUType
from src.workload_runner import _place_on_resource

METADATA = frozenset(('DIE', 'TLB', 'STORE'))
QKV = frozenset(('decode_qkv', 'decode_batch_qkv'))


def target_position(event, target):
    members = event.get('batch_members') or [event['request']]
    if target not in members or not event['query_positions']:
        return None
    positions = event['query_positions']
    return positions[members.index(target)] if len(positions) == len(members) else positions[0]


def extract_step(events, target, position):
    """Keep the full ancestor closure, including other real batch members."""
    by_id = {e['id']: e for e in events}
    own = [e for e in events if e['name'].startswith('decode_') and
           target_position(e, target) == position]
    roots = [e for e in own if e['name'] in QKV and e['time_s'] > 0 and e['transformer_layer'] == 0]
    if len(roots) != 1:
        raise ValueError('The representative decode step must have one complete layer-zero QKV')
    root = roots[0]
    final = max((e for e in own if e['device'] == 'GPU' and e['time_s'] > 0), key=lambda e: e['end_s'])
    keep, pending = set(), [final['id']]
    while pending:
        eid = pending.pop()
        if eid in keep:
            continue
        event = by_id[eid]
        if not event['name'].startswith('decode_'):
            raise ValueError('Decode step still depends on prefill; steady-state replay is unavailable')
        keep.add(eid)
        if eid != root['id']:
            pending.extend(event['depends_on'])
    base = root['rows']
    members = set(root.get('batch_members') or [root['request']])
    if len(members) != base:
        raise ValueError('QKV actual rows disagree with request batch membership')
    selected = []
    for event in events:
        if event['id'] not in keep:
            continue
        if event['name'] in QKV and event['time_s'] > 0:
            if event['rows'] != base or set(event.get('batch_members') or [event['request']]) != members:
                raise ValueError('Batch membership changes inside the step; use a stable representative group')
        copied = {k: v for k, v in event.items() if k != 'dram_addresses'}
        copied['depends_on'] = [] if event['id'] == root['id'] else list(event['depends_on'])
        selected.append(copied)
    return selected, base, final['end_s']-root['start_s']


class GPUPrice:
    def __init__(self, run_config):
        if run_config['pim_link'] != 'nvlink3':
            raise ValueError('Keep the original NVLink 3 GPU-PIM link')
        dtype = DataType.W16A16 if run_config['word'] == 2 else DataType.W8A8
        config = make_xpu_config(GPUType[run_config['gpu']], num_gpu=run_config['ngpu'],
            gpu_model=run_config['gpu_model'], pim_link_bw=600e9)['GPU']
        self.gpu = xPU(DeviceType.GPU, config, SCALING_FACTOR)
        self.model = Transformer(make_model_config(run_config['model'], dtype), tensor_parallel=run_config['ngpu'])
        self.model.build(1, 1, 2, True)
        self.templates = {op.name: op for op in self.model.sum_decoder}
        self.cache = {}

    def __call__(self, event, batch):
        if event['name'] in QKV:
            name = 'qkv'
        elif '_gpu_' in event['name']:
            name = event['name'].split('_gpu_', 1)[1]
        else:
            raise ValueError('Unsupported GPU operation: '+event['name'])
        key = name, batch
        if key not in self.cache:
            op = deepcopy(self.templates[name])
            op.m = batch
            self.cache[key] = self.gpu.get_time_and_energy(op)[0]
        return self.cache[key]


def replay_step(events, base_batch, target_batch, gpu_price):
    if base_batch < 1 or target_batch < base_batch or target_batch % base_batch:
        raise ValueError('Target batch must be an integer multiple of the actual baseline request batch')
    factor = target_batch // base_batch
    finish, busy, scheduled = {}, {}, []
    for event in events:
        e = dict(event)
        duration = event['time_s']
        e['source_time_s'] = duration
        if 'energy_nj' in e:
            e['source_energy_nj'] = e.pop('energy_nj')
        if event['device'] == 'GPU' and duration > 0:
            duration = gpu_price(event, target_batch)
            if target_batch == base_batch and abs(duration-event['time_s']) > 1e-12:
                raise ValueError('GPU formula does not reproduce the source report')
            e['virtual_gpu_rows'] = target_batch
        elif event['device'].startswith('PIM') or event['device'] == 'LINK':
            duration *= factor
        elif event['device'] in METADATA:
            if duration != 0:
                raise ValueError('Expected dependency-only metadata')
        elif duration != 0:
            raise ValueError('Unsupported resource: '+event['device'])
        if event['device'] == 'LINK':
            # Repeated small transfers: retain NVLink3 per-request semantics,
            # including the original absence of a fixed decode intercept.
            e['link_bytes'] = event['link_bytes']*factor
        ready = max((finish[dep] for dep in event['depends_on']), default=0.0)
        start = ready if event['device'] in METADATA else _place_on_resource(
            busy, event['device'], ready, duration)
        e.update(time_s=duration, start_s=start, end_s=start+duration,
            actual_base_batch=base_batch, virtual_batch=target_batch, structure_copies=factor)
        finish[e['id']] = e['end_s']
        scheduled.append(e)
    layers = defaultdict(list)
    for e in scheduled:
        if 'pim_kv_scan' in e['name']:
            layers[e['transformer_layer']].append(e)
    span = max(finish.values(), default=0.0)
    gpu = sum(e['time_s'] for e in scheduled if e['device'] == 'GPU')
    scan = sum(max(e['end_s'] for e in layer)-min(e['start_s'] for e in layer)
               for layer in layers.values())
    return scheduled, dict(tbt_s=span, gpu_service_s=gpu,
        scan_elapsed_s=scan, non_gpu_critical_s=span-gpu,
        link_service_s=sum(e['time_s'] for e in scheduled if e['device'] == 'LINK'))


def estimate(report, target, batches):
    if report.get('decode_pipeline_granularity') != 'whole_operation':
        raise ValueError('Use a source report with the repaired complete decode dependencies')
    if report.get('decode_new_kv_time_model') != 'zero_current_token_qk_pv':
        raise ValueError('Source must use all-PIM attention and zero current-token QK/PV time')
    price = GPUPrice(report['run_config'])
    positions = sorted({p for e in report['events'] if e['name'].startswith('decode_')
                        and (p := target_position(e, target)) is not None})
    if len(positions) < 2:
        raise ValueError('At least two decode positions are needed to exclude first-token latency')
    measurements, traces, baseline = [], {}, []
    actual_base = None
    for position in positions[1:]:
        source, base, source_span = extract_step(report['events'], target, position)
        actual_base = base if actual_base is None else actual_base
        if base != actual_base:
            raise ValueError('Actual request batch changes between measured decode positions')
        _, reference = replay_step(source, base, base, price)
        if abs(reference['tbt_s']-source_span) > 1e-10:
            raise ValueError('An external resource wait affects the source step; isolated replay is not equivalent')
        baseline.append(reference['tbt_s'])
        for batch in sorted(set(batches) | {base}):
            scheduled, metrics = replay_step(source, base, batch, price)
            measurements.append(dict(query_position=position, actual_base_batch=base,
                virtual_batch=batch, structure_copies=batch//base, **metrics))
            traces.setdefault(str(batch), []).append(dict(query_position=position, events=scheduled))
    measured = report['summary']['requests'][target]
    original_tbt = (measured['end_s']-measured['first_token_s'])/(len(positions)-1)
    if abs(sum(baseline)/len(baseline)-original_tbt) > 1e-10:
        raise ValueError('Source TBT includes scheduling outside the representative decode structure')
    summary = []
    for batch in sorted({m['virtual_batch'] for m in measurements}):
        rows = [m for m in measurements if m['virtual_batch'] == batch]
        average = lambda name: sum(m[name] for m in rows)/len(rows)
        summary.append(dict(actual_base_batch=actual_base, virtual_batch=batch,
            structure_copies=batch//actual_base, tbt_us=average('tbt_s')*1e6,
            gpu_us=average('gpu_service_s')*1e6, scan_elapsed_us=average('scan_elapsed_s')*1e6,
            other_critical_us=(average('non_gpu_critical_s')-average('scan_elapsed_s'))*1e6,
            link_service_us=average('link_service_s')*1e6,
            estimated_tokens_per_s=batch/average('tbt_s')))
    return dict(kind='simplified_repeated_structure_decode_batch_estimate',
        target_request=target, configured_batch_cap=report['cacheblend_batch_size'],
        actual_base_batch=actual_base, original_tbt_us=original_tbt*1e6,
        baseline_replay_passed=True, hardware=report['run_config'],
        rules=dict(gpu='Reprice the complete GPU operation with m=virtual_batch',
            pim='Multiply each measured physical-channel duration by virtual_batch/actual_base_batch',
            link='Repeat original small NVLink3 transfers; bytes and service time scale linearly',
            attention='Historical and current-token attention stay in PIM; current QK/PV remains zero',
            parallelism='Retain the source event dependencies and independent physical channels',
            scope='Steady decode only; no new MQ reuse between structure copies; no scaled prefill estimate'),
        summary=summary, steps=measurements, traces=traces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--target-request', required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[8, 64])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = args.report.read_bytes()
    result = estimate(json.loads(payload), args.target_request, args.batches)
    result['source_report'] = str(args.report.resolve())
    result['source_sha256'] = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
