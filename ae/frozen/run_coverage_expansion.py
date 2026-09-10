#!/usr/bin/env python3
"""Complete shape replay, one model/case at a time, with bounded audit storage.

Native AttAcc files are immutable. Per-request operators follow the existing
KVChime wrapper. Identical layers are represented by counted event blocks;
every requested output position is evaluated, without time extrapolation.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'archive/coverage-expansion'
STAGE = BASE/'attacc-fugue'
SOURCE = ROOT.parent/'attacc-fugue'
BUILD = SOURCE/'output/KVChime-multi-model-TP4-20260907/build/Fugue-asplos-build.json'
VARIANTS = ('F0', 'F1', 'F2', 'F3', 'F4')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(data, indent=2, default=str)+'\n')
    temporary.replace(path)

def read_lines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]

def verify_native():
    lock = json.loads((STAGE/'artifact/native-source-lock.json').read_text())
    assert (STAGE/'artifact/native-source-lock.json').read_bytes() == (SOURCE/'artifact/native-source-lock.json').read_bytes()
    for name, digest in lock.items():
        assert sha(STAGE/name) == sha(SOURCE/name) == digest, name
    # Also protect original files absent from the publication lock.
    source_manifest = json.loads((BASE/'source-manifest.json').read_text())
    for record in source_manifest['files']:
        relative = Path(record['source']).relative_to(SOURCE)
        if relative.parts[0] in ('src', 'pim_ramulator_src', 'vendor'):
            assert sha(record['target']) == sha(record['source']) == record['sha256'], relative
    return lock

def runtime(path):
    path.mkdir(parents=True, exist_ok=True)
    build = json.loads(BUILD.read_text())
    for source_key, name, digest_key in [('binary','ramulator2','binary_sha256'), ('library','libramulator.so','library_sha256')]:
        original = Path(build[source_key]); target = path/name
        assert sha(original) == build[digest_key]
        if target.exists():
            assert sha(target) == build[digest_key]
        else:
            shutil.copy2(original, target)
        assert sha(target) == sha(original)
    save(path/'build-provenance.json', {'source': str(BUILD), 'sha256': sha(BUILD), 'build': build})
    os.environ['LD_LIBRARY_PATH'] = str(path)+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')

def source_helpers(catalog):
    source = STAGE/'fugue/kvchime.py'
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('view','moved')]
    assert {n.name for n in nodes} == {'view','moved'}
    env = {'catalog': catalog}
    exec(compile(ast.Module(body=nodes,type_ignores=[]), str(source), 'exec'), env)
    original_view = env['view']
    def view(case, updated, query_count, step=0):
        objects, views = original_view(case, updated, query_count, step)
        for obj in objects:
            obj['capacity_tokens'] = ((case['n'] if updated == 'full' else case['q']) + case['output']-1
                                      if obj['private'] else obj['tokens'])
        return objects, views
    return view, env['moved']

class EventBlocks:
    """Audit additive costs at layer-class granularity, not individual operators."""
    def __init__(self):
        self.records=[]
        self.time=Counter(); self.energy=Counter()
    def add(self, variant, phase, step, layer_class, layers, components):
        assert layers > 0
        for name,t,e in components:
            assert math.isfinite(t) and t >= 0 and math.isfinite(e) and e >= 0, (name,t,e)
        t = math.fsum(c[1] for c in components)*layers
        e = math.fsum(c[2] for c in components)*layers
        self.time[variant,phase] += t; self.energy[variant,phase] += e
        self.records.append(dict(variant=variant,phase=phase,step=step,layer_class=layer_class,
                                 repetitions=layers,components=components,time_us=t,energy_uJ=e))

def simulate(case, m, catalog):
    view,moved = source_helpers(catalog)
    n,q,b,L = case['n'],case['q'],len(case['members']),m.layers
    assert case['output'] > 1
    pre_counts = Counter(min(l,2) if case['layer_policy']=='cacheblend' else 2 for l in range(L))
    decode_counts = Counter(0 if case['layer_policy']=='cacheblend' and l<2 else 2 for l in range(L))
    pool = sum(catalog[k]['tokens'] for k in set(case['pool_ids']))*m.kvbytes*L*m.tp
    warmup_time=warmup_energy=0.
    for oid in sorted(set(case['pool_ids'])):
        length=catalog[oid]['tokens']; common=m.common(length,length,length,1)
        gt,ge=m.gpu_attention(length,length,1);t,e=m.link(length*m.kvbytes)
        warmup_time+=(sum(x[1] for x in common)+gt+t)*L
        warmup_energy+=(sum(x[2] for x in common)+ge+e)*L
    pre={}; decisions=[]; audit=EventBlocks()
    for lp in pre_counts:
        aq=n if lp==0 else q;qkv=n if lp<2 else q;updated=n if lp<2 else q
        objects,views=view(case,'full' if lp<2 else 'partial',aq)
        scan=m.shared_scan(objects,views,True);sm,sme=m.softmax(aq,n,b);gt,ge=m.gpu_attention(aq,n,b)
        qi,desc=moved(m,views,aq);qin,qine=m.link(qi+desc);out,oute=m.link(aq*b*m.qbytes)
        fetch,fe=m.link((n-updated)*b*m.kvbytes);write,write_energy=m.link(updated*b*m.kvbytes)
        gpu=max(gt+fetch,write);exposed=max(0,write-scan.get('first_private_k_us',0))
        pim=qin+scan['scan_us']+sm+out+exposed
        pim_energy=qine+m.scan_energy(scan,aq,n,b)+sme+oute+write_energy
        choice,estimate=m.selector.choose_views(objects,views,gpu,sm,qin+out,write)
        selection=(0.,0.)
        if case['layer_policy']=='cacheblend' and lp==1:
            cached=sum(catalog[k]['tokens'] for k in case['members'][0]['chunk_ids'])
            selection=m.link(cached*m.kvbytes/2)
        pre[lp]=dict(q=aq,updated=updated,common=m.common(qkv,aq,n,b),gpu=gpu,gpu_energy=ge+fe+write_energy,
                     gpu_kernel=gt,gpu_kernel_energy=ge,fetch=fetch,fetch_energy=fe,write=write,
                     pim=pim,pim_energy=pim_energy,scan=scan['scan_us'],choice=choice,selection=selection)
        decisions.append(dict(layer_class=lp,q=aq,n=n,batch=b,GPU_us=gpu,PIM_us=pim,estimated_PIM_us=estimate,
                              choice=choice,oracle='PIM' if pim<gpu else 'GPU',
                              regret_us=(pim if choice=='PIM' else gpu)-min(pim,gpu),
                              scan_profile=scan['profile'],scan_repeats=scan.get('profile_repeats'),
                              Q_variant_bytes=qi,descriptor_bytes=desc,KV_write_us=write,KV_exposed_us=exposed,
                              first_private_K_us=scan.get('first_private_k_us',0)))
    full_common=m.common(n,n,n,b);full_gpu=m.gpu_attention(n,n,b)
    storage={};scan_totals=Counter()
    for variant in VARIANTS:
        active=private=0;remote_pool=0 if variant=='F0' else pool
        snapshots=[]
        for lp,count in pre_counts.items():
            if variant=='F0':
                detail=full_common;st,se=full_gpu;device='GPU';selection=(0.,0.);updated=n
            else:
                d=pre[lp];detail=d['common'];selection=d['selection'];updated=d['updated']
                device=d['choice'] if variant=='F4' else 'GPU'
                if variant=='F1':st=d['gpu_kernel']+d['fetch'];se=d['gpu_kernel_energy']+d['fetch_energy']
                elif variant=='F2':
                    write,e=m.link(n*b*m.kvbytes)
                    st=d['gpu_kernel']+d['fetch']+write;se=d['gpu_kernel_energy']+d['fetch_energy']+e
                else:st,se=(d['pim'],d['pim_energy']) if device=='PIM' else (d['gpu'],d['gpu_energy'])
            components=[('common',sum(x[1] for x in detail),sum(x[2] for x in detail)),
                        ('selection',*selection),('attention_service',st,se)]
            audit.add(variant,'prefill',0,lp,count,components)
            if variant in ('F0','F1'):active+=count*n*b*m.kvbytes*m.tp
            else:private+=count*(n if variant=='F2' else updated)*b*m.kvbytes*m.tp
            live=active if variant in ('F0','F1') else (n if device=='GPU' else updated)*b*m.kvbytes*m.tp
            snapshots.append(dict(phase='prefill',layer_class=lp,GPU_KV_bytes=live,
                                  remote_shared_bytes=remote_pool,remote_private_bytes=private,total_KV_bytes=live+remote_pool+private))
        storage[variant]=dict(active=active,private=private,remote_pool=remote_pool,snapshots=snapshots)
    decode_profiles=[]
    for step in range(1,case['output']):
        nn=n+step;common=m.common(1,1,nn,b,True);ga=m.gpu_attention(1,nn,b);sm,sme=m.softmax(1,nn,b)
        dense=m.dense_scan(1,nn,b);kt,ke=m.link(b*(2*m.qbytes+m.kvbytes))
        values={'F0':(ga[0],ga[1],0.),'F1':(ga[0],ga[1],0.),
                'F2':(dense['scan_us']+sm+kt,m.scan_energy(dense,1,nn,b)+sme+ke,dense['scan_us'])}
        profiles={'F2':dense.get('profile_repeats',dense['profile'])}
        for lp in decode_counts:
            objects,views=view(case,'full' if lp==0 else 'partial',1,step)
            qi,desc=moved(m,views,1);qin,qine=m.link(qi+desc);out,oute=m.link(b*m.qbytes);write,wre=m.link(b*m.kvbytes)
            for variant,mq in [('F3',False),('F4',True)]:
                scan=m.shared_scan(objects,views,mq);exposed=max(0,write-scan.get('first_private_k_us',0))
                t=qin+scan['scan_us']+sm+out+exposed
                e=qine+m.scan_energy(scan,1,nn,b)+sme+oute+wre
                values[variant,lp]=(t,e,scan['scan_us'])
                profiles[f'{variant}-layer{lp}']=scan.get('profile_repeats',scan['profile'])
        decode_profiles.append(dict(step=step,n=nn,profiles=profiles))
        for variant in VARIANTS:
            for lp,count in decode_counts.items():
                t,e,scan=values[variant,lp] if variant in ('F3','F4') else values[variant]
                audit.add(variant,'decode',step,lp,count,
                          [('common',sum(x[1] for x in common),sum(x[2] for x in common)),('attention_service',t,e)])
                scan_totals[variant]+=count*scan
            state=storage[variant]
            if variant in ('F0','F1'):state['active']+=L*b*m.kvbytes*m.tp
            else:state['private']+=L*b*m.kvbytes*m.tp
            live=state['active'] if variant in ('F0','F1') else b*m.kvbytes*m.tp
            state['snapshots'].append(dict(phase='decode',step=step,GPU_KV_bytes=live,
                                          remote_shared_bytes=state['remote_pool'],remote_private_bytes=state['private'],
                                          total_KV_bytes=live+state['remote_pool']+state['private']))
    rows=[]
    for variant in VARIANTS:
        ttft=audit.time[variant,'prefill'];decode=audit.time[variant,'decode']
        ttft_energy=audit.energy[variant,'prefill'];decode_energy=audit.energy[variant,'decode']
        snaps=storage[variant]['snapshots']
        peak=max(s['total_KV_bytes'] for s in snaps);peak_gpu=max(s['GPU_KV_bytes'] for s in snaps)
        peak_remote=max(s['remote_shared_bytes']+s['remote_private_bytes'] for s in snaps)
        weights,_,temp=m.capacity(b,n,case['output']);per_gpu=(weights+temp+peak_gpu)/m.tp
        assert per_gpu<=m.gconf['MEM_CAPACITY_PER_DEVICE'],(case['id'],variant,'GPU capacity',per_gpu)
        # Remote HBM capacity is checked by the preflight and is independently
        # recorded here; use native configuration values, never a renamed die.
        if hasattr(m,'coverage_remote_budget'):
            assert peak_remote<=m.coverage_remote_budget,(case['id'],variant,'remote capacity',peak_remote)
        row=dict(case_id=m.name+'--'+case['id'],source=case['source'],workload_id=case['workload_id'],
                 dataset=case.get('dataset'),scope=case.get('scope'),source_row=case.get('source_row'),
                 model=m.name,GPUs=m.tp,batch=b,variant=variant,gqa_size=m.gqa,
                 Q_heads=m.m['num_heads'],KV_heads=m.m['num_kv_heads'],prompt_tokens=n,partial_Q=q,
                 output_tokens=case['output'],TTFT_ms=ttft/1000,TBT_ms=decode/(case['output']-1)/1000,
                 E2E_ms=(ttft+decode)/1000,TTFT_energy_mJ=ttft_energy/1000,
                 TBT_energy_mJ=decode_energy/(case['output']-1)/1000,E2E_energy_mJ=(ttft_energy+decode_energy)/1000,
                 decode_scan_ms_per_token=scan_totals[variant]/(case['output']-1)/1000,
                 prefill_scan_ms=sum(pre[lp]['scan']*count for lp,count in pre_counts.items()
                                     if variant=='F4' and pre[lp]['choice']=='PIM')/1000,
                 GPU_peak_KV_GiB=peak_gpu/2**30,remote_peak_KV_GiB=peak_remote/2**30,simultaneous_peak_KV_GiB=peak/2**30,
                 remote_pool_GiB=storage[variant]['remote_pool']/2**30,GPU_weights_GiB=weights/2**30,
                 GPU_peak_total_per_device_GiB=per_gpu/2**30,warmup_ms=0 if variant=='F0' else warmup_time/1000,
                 warmup_energy_mJ=0 if variant=='F0' else warmup_energy/1000)
        assert math.isclose(row['TTFT_ms']+(case['output']-1)*row['TBT_ms'],row['E2E_ms'],rel_tol=1e-12)
        rows.append(row)
    if b==1 and m.gqa==1:
        assert math.isclose(rows[3]['TBT_ms'],rows[4]['TBT_ms'],rel_tol=1e-12)
    return dict(summary=rows,decisions=decisions,event_blocks=audit.records,storage=storage,decode_profiles=decode_profiles,
                checks=dict(all_output_steps_evaluated=True,layer_counts=dict(prefill=pre_counts,decode=decode_counts),
                            additive_event_costs=True,simultaneous_capacity_peaks=True,native_source_unchanged=True),
                warmup=dict(time_us=warmup_time,energy_uJ=warmup_energy),
                abstraction='Original per-layer KVChime cost operators; identical layer classes stored as counted blocks. Full configured generation horizon, not observed EOS or numerical inference.')

def worker(args):
    verify_native()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]='1'
    sys.path.insert(0,str(STAGE));sys.dont_write_bytecode=True
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['FUGUE_JOBS']=str(args.jobs)
    from fugue.kvchime_model import Model,geometry
    if args.trace_cache=='exact':
        from coverage_trace_cache import ExactTraceBank as TraceBank
    else:
        from fugue.kvchime_traces import TraceBank
    run=BASE/'runs'/args.run_name/args.model
    runtime(run/'runtime')
    plan=json.loads((BASE/'model-plan.json').read_text())
    assert plan['inputs_sha256']==sha(BASE/'inputs/manifest.json')
    specification=next(row for row in plan['models'] if row['name']==args.model)
    assert specification['status']=='ready' and args.tp==specification['selected']['tensor_parallel']
    identity=dict(runner_sha256=sha(__file__),model=args.model,tensor_parallel=args.tp,
                  model_plan_sha256=sha(BASE/'model-plan.json'),inputs_sha256=plan['inputs_sha256'],
                  catalog_sha256=sha(BASE/'inputs/catalog.jsonl'),cases_sha256=sha(BASE/'inputs/cases.jsonl'),
                  wrapper={str(path.relative_to(STAGE)):sha(path) for path in sorted((STAGE/'fugue').glob('*.py'))},
                  official_geometry_sha256=sha(STAGE/'fugue/official-gqa-geometry.json'),native_lock=verify_native())
    identity['trace_cache']=args.trace_cache
    if args.trace_cache=='exact':identity['trace_cache_source_sha256']=sha(ROOT/'outline/analysis/coverage_trace_cache.py')
    if (run/'identity.json').exists():assert json.loads((run/'identity.json').read_text())==identity
    else:save(run/'identity.json',identity)
    cases=read_lines(BASE/'inputs/cases.jsonl')
    if args.case_id:cases=[case for case in cases if case['id']==args.case_id]
    assert cases
    catalog={row['chunk_id']:row for row in read_lines(BASE/'inputs/catalog.jsonl')}
    bank=TraceBank(run/'profiles',run/'runtime',geometry(args.model))
    model=Model(args.model,args.tp,bank)
    model.coverage_remote_budget=specification['selected']['remote_budget_bytes']
    failures=[];finished=[]
    for case in cases:
        target=run/'cases'/(case['id']+'.json')
        if target.exists():
            previous=json.loads(target.read_text())
            assert previous['case_sha256']==hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()
            assert previous['runner_sha256']==sha(__file__)
            assert previous['run_identity_sha256']==sha(run/'identity.json')
            print('VERIFIED FINISHED',args.model,case['id'],flush=True)
            finished.append(case['id'])
            continue
        print('START',args.model,case['id'],datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
        try:
            result=simulate(case,model,catalog)
            if args.validate:
                from check_coverage_runner import compare_case
                comparison=compare_case(case,model,catalog,raise_on_error=False)
                save(run/'validation'/(case['id']+'.json'),comparison)
                assert comparison['status']=='passed',comparison['failures'][:3]
            result.update(case_sha256=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest(),
                          runner_sha256=sha(__file__),run_identity_sha256=sha(run/'identity.json'),
                          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save(target,result)
            finished.append(case['id'])
            print('FINISHED',args.model,case['id'],flush=True)
        except Exception:
            save(run/'failures'/(case['id']+'.json'),dict(case_id=case['id'],model=args.model,error=traceback.format_exc(),
                                                      runner_sha256=sha(__file__),native_source_unchanged=bool(verify_native())))
            failures.append(case['id'])
            print('FAILED',args.model,case['id'],traceback.format_exc(),flush=True)
            if args.fail_fast:raise
    verify_native()
    save(run/'checks.json',dict(cases=len(cases),model=args.model,native_source_unchanged=True,
                               requested_case_ids=[case['id'] for case in cases],finished_case_ids=finished,failed_case_ids=failures,
                               status='passed' if not failures else 'incomplete',full_input_list=not bool(args.case_id),
                               finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    if failures:raise RuntimeError(f'{args.model}: {len(failures)} requested cases failed; records retained')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--tp',required=True,type=int)
    parser.add_argument('--jobs',type=int,default=8)
    parser.add_argument('--case-id')
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--fail-fast',action='store_true')
    parser.add_argument('--trace-cache',choices=['exact','off'],default='off')
    parser.add_argument('--run-name',default='full-source-shape-replay')
    worker(parser.parse_args())
