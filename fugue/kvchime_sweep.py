"""Multi-model Q/cache/link sweep, using fresh native DRAM command profiles."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import json,datetime,math
from fugue.runtime import REPO,RUN,JOBS,setup,check_sources,load,save,csvout
from fugue.kvchime_model import Model,geometry
from fugue.kvchime_traces import TraceBank

ROOT=RUN/'model-sweep';RAW,OUT,AUDIT,RUNTIME=setup(ROOT)
check_sources();cfg=load(REPO/'artifact/inputs/kvchime.json')
qs=[1,2,4,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048]
cs=[0,128,256,512,1024,2048,4096,8192]
links=[32,48,64,96,128,150,192,225,256,300,350,400,450]
bank=TraceBank(RAW/'profiles',RUNTIME,geometry('LLAMA-7B'))
models=[Model(s['name'],s['tensor_parallel'],bank) for s in cfg['models']]
save(AUDIT/'plan.json',dict(models=cfg['models'],q=qs,cached=cs,link_GBps=links,
    statement='Native complete rectangular operator query tiles; no request or layer extrapolation.',
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
keys=set()
for m in models:
    for c in cs:
        for q in qs:
            k=q*m.gqa
            keys|={(c+q,m.h,1),(c+q,m.h,min(k,8))}
            if k%8:keys.add((c+q,m.h,k%8))
print('MULTI-MODEL SWEEP',len(keys),'native signatures',flush=True)
bank.prepare(keys)
rows=[]
for m in models:
    for c in cs:
        for q in qs:
            gt,ge=m.gpu_attention(q,c+q,1);sm,sme=m.softmax(q,c+q,1)
            records={}
            for mode in ['plain','mq']:
                k=q*m.gqa
                residents=([1]*k) if mode=='plain' else ([8]*(k//8)+([k%8] if k%8 else []))
                first=bank.get(c+q,m.h,residents[0])
                window=0. if not c else min(clock for index,clock in first['key_events'] if index>=2*(c//16))*.769/1000
                records[mode]=(m.combine([bank.get(c+q,m.h,r) for r in residents]),window)
            for bw in links:
                qi=q*m.qbytes/bw/1000;kv=q*m.kvbytes/bw/1000;fetch=c*m.kvbytes/bw/1000
                gpu=max(gt+fetch,kv)
                row=dict(model=m.name,GPUs=m.tp,Q_heads=m.m['num_heads'],KV_heads=m.m['num_kv_heads'],gqa_size=m.gqa,
                    q=q,cached=c,total_kv=c+q,link_GBps=bw,
                    gpu_attention_us=gt,pim_softmax_us=sm,q_input_us=qi,output_us=qi,
                    q_input_bytes=q*m.qbytes,new_kv_bytes=q*m.kvbytes,cached_kv_bytes=c*m.kvbytes,
                    cached_kv_readback_us=fetch,new_kv_transfer_us=kv,gpu_service_us=gpu)
                for mode,(scan,window) in records.items():
                    cost=2*qi+scan['scan_us']+sm+max(0,kv-window)
                    row.update({mode+'_scan_us':scan['scan_us'],mode+'_service_us':cost,
                        mode+'_new_kv_exposed_us':max(0,kv-window),mode+'_overlap_window_us':window,
                        mode+'_speedup_over_GPU':gpu/cost,mode+'_profile_repeats':scan['profile_repeats']})
                rows.append(row)
    print('SWEEP MODEL',m.name,flush=True)
csvout(OUT/'sweep.csv',rows)
csvout(OUT/'profiles.csv',[{k:v for k,v in r.items() if k!='key_events'} for r in bank.native_commands])
assert len(rows)==len(models)*len(qs)*len(cs)*len(links)
assert all(math.isfinite(r[k]) and r[k]>0 for r in rows for k in ['gpu_service_us','plain_service_us','mq_service_us'])
check_sources();save(AUDIT/'checks.json',dict(models=[m.name for m in models],rows=len(rows),
    native_profiles=len(bank.native_commands),all_commands_succeeded=True,positive_finite_latencies=True,
    all_models_cover_identical_q_cache_bandwidth_grid=True,native_source_unchanged=True))
print('MULTI-MODEL SWEEP COMPLETE',len(rows),flush=True)
