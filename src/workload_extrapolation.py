"""Sample requests from a full workload, then reprice their representative DAG.

Physical scans and addresses are simulated once. A declared larger workload
scales channel service, while the GPU is repriced at its larger matrix sizes.
This is an explicit performance approximation, not additional Ramulator runs.
"""
from collections import Counter
from dataclasses import replace
import math

from .model import Layer
from .type import DataType, LayerType
from .workload import Workload, WorkloadValidationError
from .workload_runner import (SplitEvent, DEPENDENCY_ONLY_DEVICES,
                              _place_on_resource, summarize_cacheblend_schedule)


def sample_workload(workload):
    """Read the FULL input; select only the declared dependency-closed sample.

    Every full request maps to a materialized representative of the same
    shape. The complete input and mapping exist before any event construction.
    There is no implicit duplication of a small input to invent a full one.
    """
    raw = workload.raw
    config = raw.get('meta', {}).get('simulation_sample') if isinstance(raw, dict) else None
    if config is None:
        return workload, None
    mapping = config.get('representative_for')
    by_id = {q.request_id: q for q in workload.requests}
    if not isinstance(mapping, dict) or set(mapping) != set(by_id):
        raise WorkloadValidationError('simulation_sample must map every full request exactly once')
    representatives = set(mapping.values())
    if not representatives.issubset(by_id) or any(mapping[r] != r for r in representatives):
        raise WorkloadValidationError('Representatives must be full-input requests mapped to themselves')
    fingerprints = {}
    for rid, representative in mapping.items():
        q, r = by_id[rid], by_id[representative]
        if (q.tier, q.lout, q.history_len, [(s.role, s.length, s.position_delta) for s in q.segments]) != (
                r.tier, r.lout, r.history_len, [(s.role, s.length, s.position_delta) for s in r.segments]):
            raise WorkloadValidationError('Sample shape mismatch: '+rid+' -> '+representative)
        if (mapping.get(q.parent_id) if q.parent_id else None) != r.parent_id:
            raise WorkloadValidationError('Sample must preserve parent dependencies: '+rid)
        for a, b in zip(q.segments, r.segments):
            previous = fingerprints.setdefault(a.fingerprint, b.fingerprint)
            if previous != b.fingerprint:
                raise WorkloadValidationError('Shared content maps to inconsistent sampled fingerprints')
    selected = tuple(q for q in workload.requests if q.request_id in representatives)
    if any(q.parent_id is not None and q.parent_id not in representatives for q in selected):
        raise WorkloadValidationError('Sample is not closed over parent dependencies')
    weights = dict(Counter(mapping.values()))
    meta = dict(raw.get('meta', {})); meta.pop('simulation_sample', None)
    sample_raw = dict(raw, meta=meta, agents=[dict(q.raw) for q in selected])
    return Workload(workload.kind, selected, sample_raw), dict(
        method='sample_full_workload', full_requests=len(by_id), sampled_requests=len(selected),
        representative_for=dict(mapping), request_weights=weights,
        selection=config.get('selection', 'explicit input mapping'),
        full_input_materialized=True, full_event_graph_materialized=False)


def price_gpu(event, factor, gpu, cache, weights=None):
    """Batch weight-bearing work; keep attention contexts independent."""
    specs = event.gpu_pricing
    if not specs:
        raise WorkloadValidationError('Missing GPU shape for extrapolation: '+event.name)
    total = 0.0
    for spec in specs:
        op_factor = weights.get(spec.get('request_id'), factor) if weights else factor
        key = tuple(sorted(spec.items())), op_factor
        if key not in cache:
            op = Layer(spec['stage'], spec['name'], LayerType[spec['type']],
                       spec['has_weight'], DataType[spec['dtype']],
                       spec['m'], spec['n'], spec['k'], spec['numOp'])
            if op.type in (LayerType.MATMUL, LayerType.SOFTMAX):
                # Copies have separate KV, so scale the number of attention
                # matrices, not context length or queries sharing one KV.
                op.numOp *= op_factor
            else:
                # Fractional sampling weights use a padded integral GPU M.
                op.m = math.ceil(op.m * op_factor)
            seconds, _ = gpu.get_time_and_energy(op)
            cache[key] = seconds * spec.get('repeat', 1)
        total += cache[key]
    return total


def replay(events, weights, gpu):
    if not weights or any(not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in weights.values()):
        raise WorkloadValidationError('Every sampled request needs a positive integral weight')
    finishes, busy, priced, cache = {}, {}, [], {}
    for event in events:
        members = event.batch_members or (event.request_id,)
        counts = [weights[m] for m in members if m in weights]
        if not counts and event.time_s > 0:
            raise WorkloadValidationError('Unmapped priced event: '+event.name)
        factor = sum(counts)/len(counts) if counts else 1
        duration = event.time_s
        if event.device == 'GPU' and duration > 0:
            duration = price_gpu(event, factor, gpu, cache, weights)
            if factor == 1 and not math.isclose(duration, event.time_s, rel_tol=1e-9, abs_tol=1e-12):
                raise WorkloadValidationError('GPU source cost does not reproduce: '+event.name)
        elif event.device.startswith('PIM'):
            # A shared sweep with unequal strata is padded to the largest
            # multiplicity. Equal-sized strata give the requested N/n rule.
            factor = max(counts, default=1)
            duration *= factor
        elif event.device == 'LINK':
            duration *= factor
        elif event.device in DEPENDENCY_ONLY_DEVICES:
            if duration:
                raise WorkloadValidationError('Metadata unexpectedly has service time')
        elif duration:
            raise WorkloadValidationError('Unsupported extrapolation resource: '+event.device)
        ready = max((finishes[d] for d in event.depends_on), default=0.0)
        start = ready if event.device in DEPENDENCY_ONLY_DEVICES else _place_on_resource(busy, event.device, ready, duration)
        end = start + duration
        finishes[event.event_id] = end
        # Energy is deliberately omitted from extrapolation's reported metrics.
        priced.append(replace(event, time_s=duration, start_s=start, end_s=end,
                              link_bytes=event.link_bytes * factor))
    return priced


def events_from_report(report):
    if not isinstance(report.get('events'), list):
        raise WorkloadValidationError('Representative replay requires full source events')
    return [SplitEvent(e['id'], e['transformer_layer'], e['tier'], e['request'],
                       e['name'], e['device'], e['rows'], e['time_s'], e['energy_nj'],
                       link_bytes=e['link_bytes'], depends_on=tuple(e['depends_on']),
                       query_positions=tuple(e['query_positions']),
                       batch_members=tuple(e.get('batch_members', ())),
                       masked_rows=e.get('masked_rows', 0), start_s=e['start_s'], end_s=e['end_s'],
                       gpu_pricing=tuple(e.get('gpu_pricing', ()))) for e in report['events']]


def extrapolate_report(report, workload, gpu, sampling, *, pipe=True):
    if not pipe:
        raise WorkloadValidationError('Representative replay currently requires --pipeopt')
    if 'request_release_dependencies' not in report:
        raise WorkloadValidationError('Source report lacks request release dependencies')
    source = events_from_report(report)
    weights = sampling['request_weights']
    if set(weights) != {q.request_id for q in workload.requests} or sum(weights.values()) != sampling['full_requests']:
        raise WorkloadValidationError('Sampling weights disagree with the full workload')
    reference = replay(source, {r:1 for r in weights}, gpu)
    for a, b in zip(source, reference):
        if not math.isclose(a.start_s, b.start_s, rel_tol=1e-9, abs_tol=1e-10) or not math.isclose(a.end_s, b.end_s, rel_tol=1e-9, abs_tol=1e-10):
            raise WorkloadValidationError('Factor-one replay does not reproduce source schedule: '+a.event_id)
    scheduled = replay(source, weights, gpu)
    summary = summarize_cacheblend_schedule(scheduled, workload, release_deps=report['request_release_dependencies'])
    requests = summary['requests']
    intervals = sum(weights[q.request_id]*(q.lout-1) for q in workload.requests)
    tbt = sum(weights[q.request_id]*(requests[q.request_id]['end_s']-requests[q.request_id]['first_token_s'])
              for q in workload.requests if q.lout > 1) / intervals if intervals else None
    ttfts = [(weights[r],s['ttft_s']) for r,s in requests.items() if s['ttft_s'] is not None]
    batches = [sum(weights[r] for r in (e.batch_members or (e.request_id,)))
               for e in source if e.name in ('decode_qkv','decode_batch_qkv','decode_gpu_step_batch') and e.time_s > 0]
    return dict(method='sample_full_workload', source_schedule_reproduced=True,
        sampled_requests=len(workload.requests), full_requests=sampling['full_requests'],
        simulated_events=len(source), weighted_tbt_s=tbt,
        mean_ttft_s=sum(w*t for w,t in ttfts)/sum(w for w,t in ttfts) if ttfts else None,
        makespan_s=max((e.end_s for e in scheduled), default=0.0),
        gpu_service_s=sum(e.time_s for e in scheduled if e.device=='GPU'),
        link_bytes=sum(e.link_bytes for e in scheduled),
        logical_decode_batch_min=min(batches) if batches else None,
        logical_decode_batch_max=max(batches) if batches else None,
        summary=summary, events=[e.to_dict() for e in scheduled],
        assumptions={
            'workload':'Full input contains all requests and dependencies. Only mapped representative requests build an event DAG; each weight counts full-input requests mapped to it. Phases and shapes must match.',
            'gpu':'FC/normalization/activation/collective M is repriced at ceil(M*weight); attention matrix count scales with weight and retains its original context.',
            'pim':'Each physical channel service time times its request weight; shared sweeps use the largest member weight (padding if unequal). Preserve parallel channel resources; no extra cross-stratum MQ reuse or Ramulator simulation.',
            'link':'Original NVLink service and bytes times weight, including repeated startup costs already present in the source.',
            'selection':'Prefill side, admission order, placement and masks are frozen from the representative run; the dependency graph is rescheduled with scaled costs.',
            'scope':'Performance estimate only; no capacity/quality/energy validation and no production arrival replay.'})
