"""KVChime additions: F0, shared decode MQ, native large model, simple selector."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import datetime,hashlib,json,math
from fugue.runtime import REPO,RUN,setup,check_sources,load,save,csvout,sha
from fugue.kvchime_traces import TraceBank
from fugue.kvchime_model import Model,geometry
from src.config import make_model_config,make_xpu_config,make_pim_config,SCALING_FACTOR,ENERGY_TABLE
from src.devices import xPU,PIM
from src.model import Layer,Transformer
from src.system import System
from src.type import DataType,DeviceType,GPUType,PIMType,InterfaceType,LayerType

ROOT=RUN/'kvchime';RAW,OUT,AUDIT,RUNTIME=setup(ROOT)
lock=check_sources();config=load(REPO/'artifact/inputs/kvchime.json')
epic=load(REPO/'artifact/inputs/epic.json');cb=load(REPO/'artifact/inputs/cacheblend.json')
catalog={c['chunk_id']:c for c in epic['cache_catalog']+cb['cache_catalog']}
cases=[]
for req in cb['requests']:
    c=req['cached_input_tokens'];r=req['recompute_old_tokens']
    # Upstream top-k outputs were not measured: deterministic count-preserving
    # positions are a mechanism input, as recorded explicitly in the manifest.
    changed=sorted({int(i*c/r) for i in range(r)})+list(range(c,req['total_prompt_tokens']))
    cases.append(dict(id='KVChime-CB-'+Path(req['source_input']).stem,source='CacheBlend',
        model='LLAMA-7B',tp=1,n=req['total_prompt_tokens'],q=req['selected_query_tokens'],
        output=req['generated_tokens'],members=[dict(chunk_ids=req['chunk_ids'],recomputed_indices=changed)],
        pool_ids=[x['chunk_id'] for x in cb['cache_catalog']],layer_policy='cacheblend',
        selection='Frozen selected count; evenly spaced old positions, not measured top-k output.'))
for case in epic['cases']:
    cases.append(dict(id=case['case_id'].replace('Fugue-asplos','KVChime'),source='EPIC',
        model='LLAMA-7B',tp=1,n=case['total_prompt_tokens'],q=case['selected_query_tokens'],
        output=case['generated_tokens'],members=case['members'],pool_ids=case['immutable_pool_chunk_ids'],
        layer_policy='epic',selection='Frozen exact kvlink-16 indices.'))
base_cases=cases
cases=[]
for spec in config['models']:
    for base in base_cases:
        case=deepcopy(base);case.update(model=spec['name'],tp=spec['tensor_parallel'])
        case['workload_id']=case['id'];case['id']=spec['name']+'--'+case['id']
        cases.append(case)
save(AUDIT/'plan.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    native_source_sha256=lock,config=config,cases=cases,
    scope='CPU timing/energy simulation with native AttAcc operators and commands. No request extrapolation or new placement.',
    energy='Original dynamic-energy coefficients and combined score convention. Not board power.',
    rope='Query rotation correctness is mathematical; arithmetic is fused/uncharged as in the native operator scope. Query variants and descriptors are charged on the link.'))
save(ROOT/'workload.json',dict(cases=cases,cache_catalog=[{k:v for k,v in c.items() if k!='token_ids'} for c in catalog.values()]))


bank=TraceBank(RAW/'profiles',RUNTIME,geometry('LLAMA-7B'));bank.shared_cache={}
models={(name,tp):Model(name,tp,bank) for name,tp in sorted({(c['model'],c['tp']) for c in cases})}

def view(case,updated,query_count,step=0,private_cache=False):
    objects={};views=[];n=case['n']
    for i,member in enumerate(case['members']):
        changed=set(range(n)) if updated=='full' else set(member['recomputed_indices'])
        refs=[];start=0
        for oid in member['chunk_ids']:
            length=catalog[oid]['tokens'];valid=[j for j in range(start,start+length) if j not in changed]
            key=oid if not private_cache else oid+'-reader'+str(i)
            if valid:
                objects[key]=dict(id=key,tokens=length,private=False,source_chunk_id=oid)
                refs.append(dict(object=key,logical_start=start,reference_start=0,query_shift=start,
                    valid_logical_positions=valid,invalid_old_positions=[j for j in range(start,start+length) if j in changed]))
            start+=length
        assert start<=n
        priv=sorted(changed)+list(range(n,n+step));key='private-'+str(i)
        if priv:
            objects[key]=dict(id=key,tokens=len(priv),private=True)
            refs.append(dict(object=key,query_shift=0,reference_frame='consumer global positions',valid_logical_positions=priv))
        positions=sorted(changed) if query_count>1 else [n+step-1]
        if query_count!=len(positions):positions=list(range(n-query_count,n))
        views.append(dict(agent=i,logical_length=n+step,query_count=query_count,
            query_positions=positions,references=refs))
    # Stable shared order followed by private objects; no ordering by performance.
    objs=sorted(objects.values(),key=lambda x:(x['private'],x['id']))
    return objs,views

def moved(m,views,q):
    variants=sum(len({r['query_shift'] for r in v['references']}) for v in views)*q
    desc=sum(32*len(v['references'])+4*sum(len(r['valid_logical_positions']) for r in v['references'] if r['object'].startswith('private')) for v in views)
    return variants*m.qbytes,desc

summary=[];events=[];decisions=[];storage=[];scan_rows=[];sensitivity=[];warmup=[];checks=[]
for case in cases:
    print('CASE',case['id'],case['model'],flush=True)
    m=models[case['model'],case['tp']];n,q,b=case['n'],case['q'],len(case['members']);L=m.layers
    # Prewarm is separate from online timing and charged once to reuse variants.
    pool=sum(catalog[k]['tokens'] for k in set(case['pool_ids']))*m.kvbytes*L*m.tp
    wt=we=0.
    for oid in sorted(set(case['pool_ids'])):
        length=catalog[oid]['tokens'];common=m.common(length,length,length,1);gt,ge=m.gpu_attention(length,length,1)
        t,e=m.link(length*m.kvbytes);wt+=(sum(x[1] for x in common)+gt+t)*L;we+=(sum(x[2] for x in common)+ge+e)*L
    warmup.append(dict(case_id=case['id'],time_ms=wt/1000,energy_mJ=we/1000,shared_pool_GiB=pool/2**30))
    pre={};decs={}
    policies=range(3) if case['layer_policy']=='cacheblend' else [2]
    for lp in policies:
        aq=n if lp==0 else q;qkv=n if lp<2 else q;updated=n if lp<2 else q
        objs,vs=view(case,'full' if lp<2 else 'partial',aq)
        sr=m.shared_scan(objs,vs,True);sm,sme=m.softmax(aq,n,b);gt,ge=m.gpu_attention(aq,n,b)
        qi,desc=moved(m,vs,aq);qin,qine=m.link(qi+desc);out,oute=m.link(aq*b*m.qbytes)
        fetch,fe=m.link((n-updated)*b*m.kvbytes);write,wr_e=m.link(updated*b*m.kvbytes)
        gc=max(gt+fetch,write);exposed=max(0,write-sr.get('first_private_k_us',0))
        pc=qin+sr['scan_us']+sm+out+exposed
        pe=qine+m.scan_energy(sr,aq,n,b)+sme+oute+wr_e
        choice,estimate=m.selector.choose_views(objs,vs,gc,sm,qin+out,write)
        sel=(0.,0.)
        if case['layer_policy']=='cacheblend' and lp==1:
            cached=sum(catalog[k]['tokens'] for k in case['members'][0]['chunk_ids']);sel=m.link(cached*m.kvbytes/2)
        d=dict(q=aq,updated=updated,common=m.common(qkv,aq,n,b),gpu=gc,gpu_energy=ge+fe+wr_e,
            gpu_kernel=gt,gpu_kernel_energy=ge,fetch=fetch,fetch_energy=fe,write=write,write_energy=wr_e,
            pim=pc,pim_energy=pe,scan=sr['scan_us'],choice=choice,estimate=estimate,selector=sel,
            qi_bytes=qi,descriptor_bytes=desc,scan_record=sr)
        pre[lp]=d
        decisions.append(dict(case_id=case['id'],layer_class=lp,q=aq,n=n,batch=b,GPU_us=gc,PIM_us=pc,
            estimated_PIM_us=estimate,choice=choice,oracle='PIM' if pc<gc else 'GPU',
            regret_us=(pc if choice=='PIM' else gc)-min(pc,gc),Q_variant_bytes=qi,descriptor_bytes=desc,
            pim_scan_us=sr['scan_us'],softmax_us=sm,KV_write_us=write,KV_exposed_us=exposed,
            first_private_K_us=sr.get('first_private_k_us',0),query_and_descriptor_input_us=qin,output_us=out))
    fullcommon=m.common(n,n,n,b);fullgpu=m.gpu_attention(n,n,b)
    for step in range(1,case['output']):
        nn=n+step;common=m.common(1,1,nn,b,True);ga=m.gpu_attention(1,nn,b);sm,sme=m.softmax(1,nn,b)
        dense=m.dense_scan(1,nn,b);kt,ke=m.link(b*(2*m.qbytes+m.kvbytes))
        values={'F0':(ga[0],ga[1],0.),'F1':(ga[0],ga[1],0.),
            'F2':(dense['scan_us']+sm+kt,m.scan_energy(dense,1,nn,b)+sme+ke,dense['scan_us'])}
        # CacheBlend's first two layers retain full private replacements.
        for lp in ([0,2] if case['layer_policy']=='cacheblend' else [2]):
            objs,vs=view(case,'full' if lp==0 else 'partial',1,step)
            qi,desc=moved(m,vs,1);qin,qine=m.link(qi+desc);out,oute=m.link(b*m.qbytes);write,wre=m.link(b*m.kvbytes)
            for v,mq in [('F3',False),('F4',True)]:
                r=m.shared_scan(objs,vs,mq);ex=max(0,write-r.get('first_private_k_us',0))
                t=qin+r['scan_us']+sm+out+ex;e=qine+m.scan_energy(r,1,nn,b)+sme+oute+wre
                values[v,lp]=(t,e,r['scan_us'])
                if step==1:scan_rows.append(dict(case_id=case['id'],variant=v,layer_class=lp,**r,Q_variant_bytes=qi,descriptor_bytes=desc))
        decs[step]=(common,values)
    for variant in ['F0','F1','F2','F3','F4']:
        time=energy=0.;private=active=0;remote_pool=0 if variant=='F0' else pool
        peak=remote_pool;peak_gpu=0;peak_remote=remote_pool;scan_total=0
        def event(phase,layer,op,device,t,e,step=0):
            events.append(dict(case_id=case['id'],variant=variant,phase=phase,layer=layer,step=step,
                operation=op,device=device,start_us=time,end_us=time+t,time_us=t,energy_uJ=e))
        def snap(phase,layer,step,scratch):
            live=active if variant in ['F0','F1'] else scratch
            storage.append(dict(case_id=case['id'],variant=variant,phase=phase,layer=layer,step=step,
                GPU_KV_bytes=live,remote_shared_bytes=remote_pool,remote_private_bytes=private,
                total_KV_bytes=live+remote_pool+private))
            return live,remote_pool+private,live+remote_pool+private
        for l in range(L):
            lp=min(l,2) if case['layer_policy']=='cacheblend' else 2
            if variant=='F0':
                detail=fullcommon;st,se=fullgpu;device='GPU';selection=(0.,0.);updated=n
            else:
                d=pre[lp];detail=d['common'];selection=d['selector'];updated=d['updated'];device=d['choice'] if variant=='F4' else 'GPU'
                if variant=='F1':st=d['gpu_kernel']+d['fetch'];se=d['gpu_kernel_energy']+d['fetch_energy']
                elif variant=='F2':
                    write,we=m.link(n*b*m.kvbytes);st=d['gpu_kernel']+d['fetch']+write;se=d['gpu_kernel_energy']+d['fetch_energy']+we
                else:st,se=(d['pim'],d['pim_energy']) if device=='PIM' else (d['gpu'],d['gpu_energy'])
            for name,t,e in detail:
                if name!='qkv':continue
                event('prefill',l,name,'GPU',t,e);time+=t;energy+=e
            if selection[0]:event('prefill',l,'software_selection_V_read','link',*selection);time+=selection[0];energy+=selection[1]
            event('prefill',l,'attention_service',device,st,se);time+=st;energy+=se
            for name,t,e in detail:
                if name=='qkv':continue
                event('prefill',l,name,'GPU',t,e);time+=t;energy+=e
            if variant in ['F0','F1']:active+=n*b*m.kvbytes*m.tp
            else:private+=(n if variant=='F2' else updated)*b*m.kvbytes*m.tp
            g,r,total=snap('prefill',l,0,(n if device=='GPU' else updated)*b*m.kvbytes*m.tp)
            peak=max(peak,total);peak_gpu=max(peak_gpu,g);peak_remote=max(peak_remote,r)
        ttft,ttfte=time,energy;decode_scan=0
        for step,(detail,values) in decs.items():
            for l in range(L):
                lp=0 if case['layer_policy']=='cacheblend' and l<2 else 2
                st,se,scan=values[variant,lp] if variant in ['F3','F4'] else values[variant]
                for name,t,e in detail:
                    if name!='qkv':continue
                    event('decode',l,name,'GPU',t,e,step);time+=t;energy+=e
                event('decode',l,'attention_service','GPU' if variant in ['F0','F1'] else 'PIM',st,se,step);time+=st;energy+=se;decode_scan+=scan
                for name,t,e in detail:
                    if name=='qkv':continue
                    event('decode',l,name,'GPU',t,e,step);time+=t;energy+=e
            if variant in ['F0','F1']:active+=L*b*m.kvbytes*m.tp
            else:private+=L*b*m.kvbytes*m.tp
            g,r,total=snap('decode','all',step,b*m.kvbytes*m.tp)
            peak=max(peak,total);peak_gpu=max(peak_gpu,g);peak_remote=max(peak_remote,r)
        weights,kv,temp=m.capacity(b,n,case['output'])
        # Native capacity function returns system totals; GPU budget is per device.
        per_gpu=(weights+temp+peak_gpu)/m.tp
        assert per_gpu<=m.gconf['MEM_CAPACITY_PER_DEVICE'],(case['id'],variant,per_gpu)
        row=dict(case_id=case['id'],source=case['source'],workload_id=case['workload_id'],model=m.name,GPUs=m.tp,batch=b,variant=variant,gqa_size=m.gqa,Q_heads=m.m['num_heads'],KV_heads=m.m['num_kv_heads'],
            prompt_tokens=n,partial_Q=q,output_tokens=case['output'],TTFT_ms=ttft/1000,
            TBT_ms=(time-ttft)/(case['output']-1)/1000,E2E_ms=time/1000,
            TTFT_energy_mJ=ttfte/1000,TBT_energy_mJ=(energy-ttfte)/(case['output']-1)/1000,E2E_energy_mJ=energy/1000,
            decode_scan_ms_per_token=decode_scan/(case['output']-1)/1000,
            prefill_scan_ms=sum(pre[min(l,2) if case['layer_policy']=='cacheblend' else 2]['scan'] for l in range(L)
                if variant=='F4' and pre[min(l,2) if case['layer_policy']=='cacheblend' else 2]['choice']=='PIM')/1000,
            GPU_peak_KV_GiB=peak_gpu/2**30,remote_peak_KV_GiB=peak_remote/2**30,simultaneous_peak_KV_GiB=peak/2**30,
            remote_pool_GiB=remote_pool/2**30,GPU_weights_GiB=weights/2**30,GPU_peak_total_per_device_GiB=per_gpu/2**30,
            warmup_ms=0 if variant=='F0' else wt/1000,warmup_energy_mJ=0 if variant=='F0' else we/1000)
        summary.append(row);print('RESULT',case['id'],variant,round(row['TTFT_ms'],4),round(row['TBT_ms'],4),flush=True)
    csvout(OUT/'summary.csv',summary)
    # F2 write-overlap control uses exactly the same transfer volume/capacity.
    f2=next(r for r in summary if r['case_id']==case['id'] and r['variant']=='F2')
    overlap=0
    for l in range(L):
        d=pre[min(l,2) if case['layer_policy']=='cacheblend' else 2];write,_=m.link(n*b*m.kvbytes)
        overlap+=sum(x[1] for x in d['common'])+d['selector'][0]+d['fetch']+max(d['gpu_kernel'],write)
    sensitivity.append(dict(case_id=case['id'],F2_serial_TTFT_ms=f2['TTFT_ms'],F2_overlap_TTFT_ms=overlap/1000))

# Small controlled fanout: identical EPIC content, explicit shared-object fractions.
# A private-cache control duplicates object identity, not a faster/slower token shape.
micro=[]
for (name,tp),m in models.items():
    source=next(c for c in cases if c['model']==name and c['id'].endswith('long-4000'))
    for b,fraction in [(1,1.),(2,1.),(4,1.),(8,1.),(4,0.),(4,.5)]:
        case=deepcopy(source);case['members']=[deepcopy(source['members'][0]) for _ in range(b)]
        objects,views=view(case,'partial',1,1)
        shared_ids=[o['id'] for o in objects if not o['private']]
        kept=set(shared_ids[:round(len(shared_ids)*fraction)])
        new_objects={o['id']:o for o in objects if o['private'] or o['id'] in kept}
        original={o['id']:o for o in objects}
        for i,v in enumerate(views):
            for ref in v['references']:
                oid=ref['object']
                if oid in shared_ids and oid not in kept:
                    key=oid+'-copy'+str(i);new_objects[key]=dict(original[oid],id=key)
                    ref['object']=key
        objects=sorted(new_objects.values(),key=lambda x:(x['private'],x['id']))
        qi,desc=moved(m,views,1);qin,qe=m.link(qi+desc);out,oe=m.link(b*m.qbytes);write,we=m.link(b*m.kvbytes)
        sm,sme=m.softmax(1,case['n']+1,b);common=m.common(1,1,case['n']+1,b,True)
        for mode in [False,True]:
            sr=m.shared_scan(objects,views,mode)
            attn=qin+sr['scan_us']+sm+out+max(0,write-sr.get('first_private_k_us',0))
            e=qe+m.scan_energy(sr,1,case['n']+1,b)+sme+oe+we
            micro.append(dict(model=m.name,GPUs=m.tp,gqa_size=m.gqa,agents=b,nominal_shared_fraction=fraction,
                shared_document_tokens=sum(original[k]['tokens'] for k in kept),mode='MQ' if mode else 'single_query',
                scan_us=sr['scan_us'],TBT_ms=(attn+sum(x[1] for x in common))*m.layers/1000,
                TBT_energy_mJ=(e+sum(x[2] for x in common))*m.layers/1000,
                mac_commands=sr['mac_commands'],represented_column_MACs=sr['represented_column_MACs'],
                dram_read_bytes=sr['dram_read_bytes_per_attacc'],profile=sr['profile']))
csvout(OUT/'shared-mq.csv',micro)

# A small independent shape grid for the four execution policies. The selector
# only uses the two native calibration scans, never these evaluated scan times.
selection=[]
for (name,tp),m in models.items():
    for c in config['selector_cached']:
        for q in config['selector_q']:
            case=dict(n=c+q,members=[dict(chunk_ids=[],recomputed_indices=list(range(c,c+q)))])
            objects=[];refs=[]
            if c:objects.append(dict(id='cached',tokens=c,private=False));refs.append(dict(object='cached',query_shift=0,valid_logical_positions=list(range(c))))
            objects.append(dict(id='new',tokens=q,private=True));refs.append(dict(object='new',query_shift=0,valid_logical_positions=list(range(c,c+q))))
            views=[dict(logical_length=c+q,query_count=q,query_positions=list(range(c,c+q)),references=refs)]
            sr=m.shared_scan(objects,views,True);gt,ge=m.gpu_attention(q,c+q,1);sm,sme=m.softmax(q,c+q,1)
            fetch,_=m.link(c*m.kvbytes);write,_=m.link(q*m.kvbytes);qi,_=m.link(q*m.qbytes)
            gc=max(gt+fetch,write);pc=2*qi+sr['scan_us']+sm+max(0,write-sr.get('first_private_k_us',0))
            choice,estimate=m.selector.choose_views(objects,views,gc,sm,2*qi,write)
            oracle=min(gc,pc)
            selection.append(dict(model=m.name,GPUs=m.tp,gqa_size=m.gqa,q=q,cached=c,n=c+q,fixed_GPU_us=gc,fixed_PIM_us=pc,
                algorithm_us=pc if choice=='PIM' else gc,oracle_us=oracle,algorithm_choice=choice,
                oracle_choice='PIM' if pc<gc else 'GPU',estimated_PIM_us=estimate,
                regret_percent=100*((pc if choice=='PIM' else gc)/oracle-1)))
        print('SELECTION',c,flush=True)

for name,rows in [('summary',summary),('events',events),('decisions',decisions),('storage',storage),('decode-scans',scan_rows),
                  ('F2-overlap-control',sensitivity),('warmup',warmup),('selection',selection)]:csvout(OUT/(name+'.csv'),rows)
csvout(OUT/'profiles.csv',[{k:v for k,v in r.items() if k!='key_events'} for r in bank.native_commands])
csvout(OUT/'selector-calibration.csv',[dict(model=name,tp=tp,**m.selector.calibration) for (name,tp),m in models.items()])
# Independent event/capacity checks and exact single-agent MQ degeneration.
for row in summary:
    es=[e for e in events if (e['case_id'],e['variant'])==(row['case_id'],row['variant'])]
    assert math.isclose(sum(e['time_us'] for e in es)/1000,row['E2E_ms'],rel_tol=1e-10)
    assert math.isclose(sum(e['energy_uJ'] for e in es)/1000,row['E2E_energy_mJ'],rel_tol=1e-10)
    assert all(math.isclose(a['end_us'],b['start_us'],rel_tol=1e-10,abs_tol=1e-7) for a,b in zip(es,es[1:]))
    assert math.isclose(row['TTFT_ms']+(row['output_tokens']-1)*row['TBT_ms'],row['E2E_ms'],rel_tol=1e-10)
    ss=[s for s in storage if (s['case_id'],s['variant'])==(row['case_id'],row['variant'])]
    assert max(s['total_KV_bytes'] for s in ss)/2**30==row['simultaneous_peak_KV_GiB']
for case in cases:
    if len(case['members'])==1 and models[case['model'],case['tp']].gqa==1:
        a,b=[next(r for r in summary if r['case_id']==case['id'] and r['variant']==v) for v in ['F3','F4']]
        assert math.isclose(a['TBT_ms'],b['TBT_ms'],rel_tol=1e-10)
assert all(r['oracle_us']<=min(r['fixed_GPU_us'],r['fixed_PIM_us'],r['algorithm_us'])+1e-9 for r in selection)
check_sources()
report=dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=len(cases),summary_rows=len(summary),
    profiles=len(bank.native_commands),selector_cases=len(selection),
    selector_mismatches=sum(r['algorithm_choice']!=r['oracle_choice'] for r in selection),
    max_selector_regret_percent=max(r['regret_percent'] for r in selection),
    native_source_unchanged=True,all_command_returncodes_zero=True,event_latency_energy_and_capacity_checks_pass=True,
    one_agent_MHA_MQ_decode_equals_single_query=True,models=list(dict.fromkeys(c['model'] for c in cases)),mathematical_view_partition_checks_pass=True,
    no_request_extrapolation=True,no_placement_optimization=True)
save(AUDIT/'checks.json',report);print('COMPLETE',json.dumps(report),flush=True)
