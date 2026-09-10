"""Attribute fresh Llama 7B / Readers 4 prefill using the paper's operators.

No simulation or profile preparation is called here. Pass the finished request
worker's model, raw result, case, view helpers and profile directory.
"""
import ast
from collections import Counter
from copy import copy, deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7), (a, b)


def inspected_type(impl):
    """Compile the original nested Inspect class without changing its AST."""
    source = HERE / 'source/operator_attribution.py'
    tree = ast.parse(source.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Inspect')
    namespace = dict(impl=impl, close=close)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[klass], type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['Inspect']


def derive_prefill(impl, model, raw, case, build_view, moved, profiles_root, workload_id=None):
    """Return only three current Fig1a rows plus their exact additive blocks.

    impl is the final hbm_2k_model module (Model and native attributes).
    model is the same configured Model used by the completed request worker.
    raw contains event_blocks, decisions, summary and baseline_residency.
    The source selection is fixed: LLAMA-7B, four readers, prefill, F0/F1/F4.
    """
    assert model.name == 'LLAMA-7B' and len(case['members']) == 4
    assert raw['baseline_residency']['mode'] == 'resident'
    profiles_root = Path(profiles_root)
    wid = workload_id or raw.get('case_id', case.get('id', 'Readers 4'))
    inspected = copy(model)
    inspected.__class__ = inspected_type(impl)
    inspected.captured = []
    calls = {}
    def operators(kind, args):
        key = kind, args
        if key not in calls:
            inspected.captured = []
            getattr(inspected, kind)(*args)
            calls[key] = deepcopy(inspected.captured)
        return deepcopy(calls[key])
    n, b = case['n'], len(case['members'])
    decisions = {d['layer_class']: d for d in raw['decisions']}
    summaries = {r['variant']: r for r in raw['summary']}
    rows, blocks = [], []
    for variant in ['F0', 'F1', 'F4']:
        totals = Counter(); refs = []; layers = 0
        selected = [(i, e) for i, e in enumerate(raw['event_blocks']) if e['variant'] == variant and e['phase'] == 'prefill']
        for index, event in selected:
            lp, reps = event['layer_class'], event['repetitions']
            assert event['step'] == 0 and isinstance(reps, int) and reps > 0
            layers += reps
            q = n if variant == 'F0' or lp == 0 else case['q']
            qkv = n if variant == 'F0' or lp < 2 else case['q']
            ops = operators('common', (qkv, q, n, b, False))
            components = {c[0]: c[1] for c in event['components']}
            close(sum(r['total_us'] for r in ops), components['common'])
            local = Counter(memory=0., compute=0., other=sum(t for k, t in components.items() if k not in ('common', 'attention_service')), fused_scan=0.)
            for op in ops: local.update(op['parts_us'])
            exposed = Counter(KV_readback=0., exposed_KV_write=0., query_input=0., descriptor_input=0., attention_output=0.)
            updated = n if lp < 2 else case['q']
            fused = softmax = full_write = hidden_write = 0.
            scan_refs = []
            if variant in ('F0', 'F1') or decisions[lp]['choice'] == 'GPU':
                device = 'GPU'
                attention_ops = operators('gpu_attention', (q, n, b))
                attention = sum(r['total_us'] for r in attention_ops)
                for op in attention_ops: local.update(op['parts_us'])
                ops += attention_ops
                if variant != 'F0': exposed['KV_readback'] = model.link((n - updated) * b * model.kvbytes)[0]
                if variant == 'F4':
                    full_write = model.link(updated * b * model.kvbytes)[0]
                    hidden_write = min(full_write, attention + exposed['KV_readback'])
                    exposed['exposed_KV_write'] = full_write - hidden_write
                close(attention + sum(exposed.values()), components['attention_service'])
            else:
                device = 'PIM'
                record = decisions[lp]['scan_repeats'] or decisions[lp]['scan_profile']
                profiles = json.loads(record) if record.startswith('{') else {record: 1}
                for name, repeat in profiles.items():
                    path = profiles_root / name / 'timing.json'
                    timing = json.loads(path.read_text())
                    assert timing['profile'] == name and isinstance(repeat, int) and repeat > 0
                    fused += timing['scan_us'] * repeat
                    scan_refs.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), repeats=repeat, scan_us=timing['scan_us'], profile=name))
                first_private = decisions[lp]['first_private_K_us']
                _, views = build_view(case, 'full' if lp < 2 else 'partial', q, 0)
                qi, desc = moved(model, views, q)
                soft_ops = operators('softmax', (q, n, b)); softmax = sum(r['total_us'] for r in soft_ops)
                for op in soft_ops: local.update(op['parts_us'])
                ops += soft_ops; local['fused_scan'] = fused
                full_write = model.link(updated * b * model.kvbytes)[0]
                hidden_write = min(full_write, first_private)
                exposed.update(exposed_KV_write=full_write-hidden_write, query_input=model.link(qi)[0],
                               descriptor_input=model.link(desc)[0], attention_output=model.link(q*b*model.qbytes)[0])
                close(exposed['exposed_KV_write'], decisions[lp]['KV_exposed_us'])
                assert qi == decisions[lp]['Q_variant_bytes'] and desc == decisions[lp]['descriptor_bytes']
                close(fused + softmax + sum(exposed.values()), components['attention_service'])
            gpu_memory = local['memory']; local['memory'] += sum(exposed.values())
            overlap = math.fsum(op['memory_compute_overlap_us'] for op in ops)
            memory_only = math.fsum(op['memory_only_us'] for op in ops) + sum(exposed.values())
            compute_bound = math.fsum(op['decomposed_GPU_compute_bound_us_not_additive'] for op in ops)
            close(memory_only + overlap, local['memory'])
            close(overlap + local['compute'], compute_bound)
            close(sum(local.values()) * reps, event['time_us'])
            for key, value in local.items(): totals[key] += value * reps
            for key, value in dict(gpu_local_memory=gpu_memory, external_exposed_memory=sum(exposed.values()),
                buffer_die_softmax_other=softmax, memory_only=memory_only, memory_compute_overlap=overlap,
                decomposed_GPU_compute_bound_not_additive=compute_bound).items(): totals[key] += value * reps
            bid = len(blocks); refs.append(bid)
            blocks.append(dict(id=bid, model=model.name, workload_id=wid, variant=variant, phase='prefill', step=0,
                layer_class=lp, repetitions=reps, device=device, parts_us_per_layer=dict(local),
                external_exposed_us_per_layer=dict(exposed), memory_only_us_per_layer=memory_only,
                memory_compute_overlap_us_per_layer=overlap, operators=ops, source_event_index=index,
                source_components=event['components'], source_event_time_us=event['time_us'],
                scan_timing_sources=scan_refs, full_KV_write_us_not_additive=full_write,
                hidden_KV_write_us_not_additive=hidden_write))
        assert layers == model.layers
        parts = {k+'_ms': totals[k]/1000 for k in ['memory', 'compute', 'fused_scan', 'other']}
        close(sum(parts.values()), summaries[variant]['TTFT_ms'])
        rows.append(dict(model=model.name, workload_id=wid, display='Readers 4', variant=variant, phase='prefill',
            total_ms=summaries[variant]['TTFT_ms'], **parts, memory_only_ms=totals['memory_only']/1000,
            memory_compute_overlap_ms=totals['memory_compute_overlap']/1000,
            decomposed_GPU_compute_bound_ms_not_additive=totals['decomposed_GPU_compute_bound_not_additive']/1000,
            parts_ms={k:v/1000 for k,v in totals.items()}, eligible=True, invalid_reasons=[], source_field='TTFT_ms',
            source_refs=dict(block_ids=refs), divisor_us_to_phase_ms=1000, prompt_tokens=n, query_tokens=case['q'],
            batch=b, output_tokens=case['output'], layers=model.layers, tensor_parallel=model.tp,
            gpu_kv_mode='resident', checks=dict(exclusive_parts_equal_complete_phase=True,
            full_horizon_layer_repetitions=True, source_summary_unchanged=True)))
    return dict(rows=rows, blocks=blocks, model=model.name, workload='Readers 4',
        scope='Fresh request prefill attribution only; the composite timeline is a mechanism illustration.',
        source_operator_attribution_sha256=hashlib.sha256((HERE/'source/operator_attribution.py').read_bytes()).hexdigest())


def export(report, output, plot=True):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/'attribution.json').write_text(json.dumps(report, indent=2)+'\n')
    for path in (HERE/'plots/fig1a').iterdir():
        if path.is_file(): shutil.copyfile(path, output/path.name)
    schema = json.loads((output/'columns.json').read_text())
    def encode(value, kind):
        if value is None: return ''
        return json.dumps(value, separators=(',', ':')) if kind in ['int','float','dict','list','bool'] else value
    with (output/'data.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(schema));writer.writeheader()
        for row in report['rows']: writer.writerow({k:encode(row.get(k),v) for k,v in schema.items()})
    flat=[]
    for block in report['blocks']:
        row={k:block[k] for k in ['id','model','workload_id','variant','phase','repetitions','memory_only_us_per_layer']}
        row.update({'external_exposed_us_per_layer.'+k:v for k,v in block['external_exposed_us_per_layer'].items()})
        flat.append(row)
    with (output/'event-blocks.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    if plot: subprocess.run([sys.executable,str(output/'plot.py')],check=True)
    return output
