#!/usr/bin/env python3
"""EPIC long-context and synchronous shared-reader replay using this checkout's AttAcc."""
import ast,csv,datetime,glob,hashlib,io,json,math,os,resource,shutil,subprocess,sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import Dict
from fugue.runtime import REPO,RUN,JOBS,setup,check_sources
ROOT=RUN/'experiment45'
RAW,OUT,AUDIT,RUNTIME=setup(ROOT)
from src.config import make_model_config,make_xpu_config,make_pim_config,SCALING_FACTOR,ENERGY_TABLE
from src.devices import xPU,PIM
from src.model import Layer,Transformer
from src.system import System
from src.type import DataType,DeviceType,LayerType,GPUType,PIMType,InterfaceType
from src.ramulator_wrapper import Ramulator
SOFTWARE=REPO/'artifact/inputs'

MODEL='LLAMA-7B';LAYERS=32;KV_BYTES=16384;Q_BYTES=8192;GENERATED=16;TCK=.769;CAP=8
E_COL=ENERGY_TABLE['PIM'][PIMType.BA]['mem']*32
E_OP=16*ENERGY_TABLE['PIM'][PIMType.BA]['alu']+32*ENERGY_TABLE['PIM'][PIMType.BA]['sram']
model=make_model_config(MODEL,DataType.W16A16)
gpu_config=make_xpu_config(GPUType.A100a,num_gpu=1)['GPU']
pim_config=make_pim_config(PIMType.BA,InterfaceType.NVLINK3,num_attacc=1,num_hbm=5,power_constraint=True)
gpu=xPU(DeviceType.GPU,gpu_config,SCALING_FACTOR)
pim=PIM(pim_config,SCALING_FACTOR,None)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,default=str)+'\n')
def csvout(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def same(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
base_plan={'source_sha256':check_sources(),'base_commit':'c60005143a6b492d7ef83231723386478b59a506'}
work=json.loads((SOFTWARE/'epic.json').read_text())
cases=work['cases'];catalog={r['chunk_id']:r for r in work['cache_catalog']}
(ROOT/'Fugue-asplos-workload.json').write_text(json.dumps(work,indent=2)+'\n')
for p in SOFTWARE.glob('Fugue-asplos-context-*.json'):shutil.copy2(p,RAW/p.name)
method=REPO/'fugue/attention.py'
plan=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),driver_sha256=sha(Path(__file__)),
    native_source_sha256=base_plan['source_sha256'],native_commit=base_plan['base_commit'],
    input_sha256=sha(SOFTWARE/'epic.json'),input_provenance=json.loads((SOFTWARE/'epic-provenance.json').read_text()),
    method_sha256=sha(method),cwd=str(Path.cwd()),repo=str(REPO),argv=sys.argv,cases=cases,
    methodology='Frozen EPIC kvlink-16 token IDs and indices; exact final experiment4/5 numerical evaluation.')
save(AUDIT/'Fugue-asplos-plan.json',plan)
print('WORKLOAD',[(c['case_id'],c['total_prompt_tokens'],c['selected_query_tokens'],c['batch']) for c in cases],flush=True)
needed={}
for c in cases:
    b,n,q=c['batch'],c['total_prompt_tokens'],c['selected_query_tokens']
    needed.setdefault((b,n),set()).update([1,8]+([q%8] if q%8 else []))
    for step in range(1,GENERATED):needed.setdefault((b,n+step),set()).add(1)
from fugue.attention import profiler
contexts={}
for b in sorted(set(b for b,n in needed)):
    raw=RAW/f'Fugue-asplos-B{b}';raw.mkdir()
    contexts[b]=profiler(REPO,ROOT,raw,RUNTIME,model,b)
for b,n in sorted(needed):key,lines=contexts[b].make_base(n);contexts[b].base_lines[key]=lines
signatures={}
def profile(b,n,r):
    key,value=contexts[b].simulate(n,r);data=value[0];data['batch']=b;data['profile_heads_per_HBM']=math.ceil(32*b/5)
    return (b,n,r),data
jobs=[(b,n,r) for (b,n),rs in sorted(needed.items()) for r in sorted(rs)]
print('RAMULATOR',len(jobs),'fresh signatures',flush=True)
with ThreadPoolExecutor(max_workers=JOBS) as pool:
    for i,f in enumerate(as_completed([pool.submit(profile,*j) for j in jobs]),1):
        key,value=f.result();signatures[key]=value
        if i%10==0 or i==len(jobs):print('RAMULATOR',i,'/',len(jobs),flush=True)
csvout(OUT/'Fugue-asplos-signatures.csv',[signatures[k] for k in sorted(signatures)])
class NativeTraceReplay:
    def output(self,pim_type,layer,power_constraint=True):
        assert layer.numOp%32==0
        b,n,q=int(layer.numOp//32),int(layer.n),int(layer.m)
        tiles=[8]*(q//8)+([q%8] if q%8 else [])
        elapsed=0.;traffic=[0.]*5
        for resident in tiles:
            d=signatures[b,n,resident];elapsed+=d['scan_us']*1e-6
            counts=json.loads((ROOT/d['path']/'Fugue-asplos-command.json').read_text())['command_counts']
            mac,wr,sb,gb=[counts[k] for k in ['PIM_MAC_AB','PIM_WR_GB','PIM_MV_SB','PIM_MV_GB']]
            one=[wr*32,(wr+sb+gb)*32,(wr+sb+gb)*32,(wr+sb+gb)*32,mac*32*2*2*4*4]
            traffic=[a+x*5 for a,x in zip(traffic,one)]
        return elapsed,traffic
pim.ramulator=NativeTraceReplay()
operator_rows=[];GPU_CACHE={};PIM_CACHE={}
def op(device,layer):
    cache=GPU_CACHE if device is gpu else PIM_CACHE
    key=(layer.name,layer.type.name,layer.m,layer.n,layer.k,layer.numOp)
    if key not in cache:
        t,e=device.get_time_and_energy(layer)
        cache[key]=(t,sum(e)*1e-12)
        operator_rows.append(dict(device='GPU' if device is gpu else 'PIM',name=layer.name,kind=layer.type.name,
            m=layer.m,n=layer.n,k=layer.k,numOp=layer.numOp,time_us=t*1e6,energy_J=sum(e)*1e-12,
            energy_offchip_pJ=e[0],energy_L2_pJ=e[1],energy_L1_pJ=e[2],energy_reg_pJ=e[3],energy_alu_pJ=e[4],energy_link_pJ=e[5]))
    return cache[key]

def transfer(tokens,kind='kv'):
    if tokens==0:return 0.,0.
    return op(gpu,Layer('sum','comm_x2g',LayerType.X2G,False,DataType.W16A16,tokens,8192 if kind=='kv' else 4096,1,1))
def gpu_attention(q,n,b):
    items=[op(gpu,Layer('sum',name,kind,False,DataType.W16A16,m,nn,k,32*b)) for name,kind,m,nn,k in
        [('score',LayerType.MATMUL,q,n,128),('softmax',LayerType.SOFTMAX,q,n,1),('context',LayerType.MATMUL,q,128,n)]]
    return sum(t for t,e in items),sum(e for t,e in items)
def pim_attention(q,n,b):
    scan=op(pim,Layer('sum','score',LayerType.MATMUL,False,DataType.W16A16,q,n,128,32*b))
    sm=op(pim,Layer('sum','softmax',LayerType.SOFTMAX,False,DataType.W16A16,q,n,1,32*b))
    return scan[0]+sm[0],scan[1]+sm[1],scan[0],sm[0]
def common(q,n,b):
    tr=Transformer(model,tensor_parallel=1);tr.build(b,q,GENERATED,False)
    return [(l.name,*op(gpu,l)) for l in tr.sum_decoder if l.type not in (LayerType.MATMUL,LayerType.SOFTMAX,LayerType.X2G)]

# Compare the replayed decode against an independent native wrapper table, not itself.
# Rows come from actual r=1 native traces, with original command-to-byte conversion.
native_table=[]
for (b,n,r),d in sorted(signatures.items()):
    if r!=1:continue
    counts=json.loads((ROOT/d['path']/'Fugue-asplos-command.json').read_text())['command_counts']
    native_table.append(dict(L=n,nhead=math.ceil(32*b/5),dhead=128,dbyte=2,pim_type='BA',power_constraint=True,cycle=d['cycles'],
        mac=counts['PIM_MAC_AB'],softmax=counts['PIM_SFM'],mvgb=counts['PIM_MV_GB'],mvsb=counts['PIM_MV_SB'],wrgb=counts['PIM_WR_GB']))
csvout(AUDIT/'Fugue-asplos-native-ramulator-table.csv',native_table)
native_wrapper=Ramulator(model,str(RUNTIME),str(AUDIT/'Fugue-asplos-native-ramulator-table.csv'),False,5)
for (b,n,r),d in signatures.items():
    if r!=1:continue
    layer=Layer('gen','score',LayerType.MATMUL,False,DataType.W16A16,1,n,128,32*b)
    t,traffic=native_wrapper.output(PIMType.BA,layer,True);tt,xx=pim.ramulator.output(PIMType.BA,layer,True)
    same(t,tt)
    for a,z in zip(traffic,xx):same(a,z)
summary=[];events=[];layers=[];decisions=[];transfers=[];storage=[];warmup=[];sensitivities=[]
for case in cases:
    cid,b,n,q=case['case_id'],case['batch'],case['total_prompt_tokens'],case['selected_query_tokens']
    pool=case['immutable_pool_tokens']*LAYERS*KV_BYTES
    # Unique immutable chunks, independently prefetched as in original EPIC.
    for key in case['immutable_pool_chunk_ids']:
        length=catalog[key]['tokens'];details=common(length,length,1);ga=gpu_attention(length,length,1);wr=transfer(length)
        warmup.append(dict(case_id=cid,chunk_id=key,tokens=length,latency_s=LAYERS*(sum(x[1] for x in details)+ga[0]+wr[0]),
            energy_J=LAYERS*(sum(x[2] for x in details)+ga[1]+wr[1])))
    decodes={}
    for mode in ['GPU','PIM']:
        system=System(gpu_config,model)
        if mode=='PIM':system.hetero_name=DeviceType.PIM;system.devices['Acc']=PIM(pim_config,SCALING_FACTOR,native_wrapper)
        result=[];log=io.StringIO()
        with redirect_stdout(log):system.simulate(b,n,GENERATED,perfs=result,pipe=False,parallel_ff=False,power_constraint=True)
        (AUDIT/f'{cid}-native-{mode}.log').write_text(log.getvalue());save(AUDIT/f'{cid}-native-{mode}.json',result)
        rows=[dict(step=s,time_s=sum(l.exec_time for l in ls)*LAYERS,energy_J=sum(sum(l.energy) for l in ls)*LAYERS*1e-12) for s,ls in enumerate(system.model.gen_decoder,1)]
        same(sum(x['time_s'] for x in rows)/len(rows),result[0][2][7]/1000)
        same(sum(x['energy_J'] for x in rows)/len(rows),result[0][3][0]*1e-9)
        decodes[mode]=rows
    details=common(q,n,b);ct=sum(x[1] for x in details);ce=sum(x[2] for x in details)
    gt,ge=gpu_attention(q,n,b);pt,pe,scan,sm=pim_attention(q,n,b)
    fetch=transfer((n-q)*b);write=transfer(q*b);fullwrite=transfer(n*b);qin=transfer(q*b,'q')
    gc=max(gt+fetch[0],write[0]);pc=qin[0]+pt+write[0]+qin[0]
    eg=ge+fetch[1]+write[1];ep=pe+2*qin[1]+write[1]
    side='PIM' if pc<gc else 'GPU'
    decisions.append(dict(case_id=cid,batch=b,prompt_tokens=n,query_tokens=q,profile_heads_per_HBM=math.ceil(32*b/5),
        GPU_candidate_us=gc*1e6,PIM_candidate_us=pc*1e6,choice=side,GPU_attention_us=gt*1e6,PIM_scan_us=scan*1e6,
        PIM_softmax_us=sm*1e6,Q_input_us=qin[0]*1e6,O_return_us=qin[0]*1e6,cached_KV_read_us=fetch[0]*1e6,
        new_KV_write_us=write[0]*1e6,overlap_window_us=0,GPU_energy_J=eg,PIM_energy_J=ep,common_GPU_us=ct*1e6))
    for variant in ['F1','F2','F3','F4']:
        choice=side if variant=='F4' else 'GPU'
        if variant=='F1':service=gt+fetch[0];en=ge+fetch[1];ft=(n-q)*b;wt=0;qt=0
        elif variant=='F2':service=fetch[0]+fullwrite[0]+gt;en=fetch[1]+fullwrite[1]+ge;ft=(n-q)*b;wt=n*b;qt=0
        elif choice=='GPU':service=gc;en=eg;ft=(n-q)*b;wt=q*b;qt=0
        else:service=pc;en=ep;ft=0;wt=q*b;qt=q*b
        clock=0.;energy=0.;private=0;active=0;peak_remote=pool;peak_gpu=0;peak_combined=pool
        def event(phase,l,name,device,t,e,step=0):
            events.append(dict(case_id=cid,variant=variant,phase=phase,layer=l,event=name,device=device,step=step,
                start_s=clock,end_s=clock+t,duration_s=t,energy_J=e))
        for l in range(LAYERS):
            for name,t,e in details:
                if name!='qkv':continue
                event('prefill_common',l,name,'GPU',t,e);clock+=t;energy+=e
            event('prefill_attention',l,'attention_service',choice,service,en);clock+=service;energy+=en
            for name,t,e in details:
                if name=='qkv':continue
                event('prefill_common',l,name,'GPU',t,e);clock+=t;energy+=e
            layers.append(dict(case_id=cid,variant=variant,layer=l,attention_device=choice,common_GPU_ms=ct*1e3,
                attention_service_ms=service*1e3,GPU_attention_kernel_ms=gt*1e3 if choice=='GPU' else 0,
                PIM_scan_ms=scan*1e3 if choice=='PIM' else 0,layer_latency_ms=(ct+service)*1e3))
            for name,tokens,kind in [('untouched_KV_readback',ft,'kv'),('KV_export',wt,'kv'),('Q_input',qt,'q'),('O_return',qt,'q')]:
                t,e=transfer(tokens,kind);transfers.append(dict(case_id=cid,variant=variant,layer=l,operation=name,
                    bytes=tokens*(KV_BYTES if kind=='kv' else Q_BYTES),standalone_us=t*1e6,energy_J=e))
            if variant=='F1':active+=n*b*KV_BYTES
            else:private+=wt*KV_BYTES
            scratch=(n if choice=='GPU' else q)*b*KV_BYTES
            gpu_live=active if variant=='F1' else scratch
            peak_gpu=max(peak_gpu,gpu_live);peak_remote=max(peak_remote,pool+private);peak_combined=max(peak_combined,pool+private+gpu_live)
            storage.append(dict(case_id=cid,variant=variant,phase='prefill',layer=l,step=0,remote_immutable_bytes=pool,
                remote_private_bytes=private,GPU_live_KV_bytes=gpu_live,combined_live_KV_bytes=pool+private+gpu_live))
        ttft,ttfte=clock,energy
        dec=decodes['GPU' if variant=='F1' else 'PIM']
        for row in dec:
            t,e=row['time_s'],row['energy_J'];event('decode','all','native_transformer','GPU' if variant=='F1' else 'GPU+PIM',t,e,row['step'])
            clock+=t;energy+=e
            if variant=='F1':active+=LAYERS*b*KV_BYTES
            else:private+=LAYERS*b*KV_BYTES
            gpu_live=active if variant=='F1' else b*KV_BYTES
            peak_gpu=max(peak_gpu,gpu_live);peak_remote=max(peak_remote,pool+private);peak_combined=max(peak_combined,pool+private+gpu_live)
            storage.append(dict(case_id=cid,variant=variant,phase='decode',layer='all',step=row['step'],remote_immutable_bytes=pool,
                remote_private_bytes=private,GPU_live_KV_bytes=gpu_live,combined_live_KV_bytes=pool+private+gpu_live))
        weights,nativekv,temp=System(gpu_config,model).get_required_mem_capacity(b,n,GENERATED)
        same(nativekv,LAYERS*(n+GENERATED-1)*b*KV_BYTES)
        warm=[x for x in warmup if x['case_id']==cid]
        row=dict(case_id=cid,experiment=case['experiment'],variant=variant,batch=b,model=MODEL,GPU='A100a',one_way_link_GBps=300,
            prompt_tokens=n,query_tokens=q,generated_tokens=GENERATED,TTFT_ms=ttft*1000,TBT_ms=sum(x['time_s'] for x in dec)/len(dec)*1000,
            E2E_ms=clock*1000,TTFT_energy_mJ=ttfte*1000,TBT_energy_mJ=sum(x['energy_J'] for x in dec)/len(dec)*1000,
            E2E_energy_mJ=energy*1000,E2E_energy_per_request_mJ=energy*1000/b,throughput_requests_per_s=b/clock,
            prefill_attention_service_ms=service*LAYERS*1000,prefill_common_GPU_ms=ct*LAYERS*1000,
            prefill_attention_service_fraction=service/(ct+service),prefill_attention_kernel_ms=(gt if choice=='GPU' else pt)*LAYERS*1000,
            prefill_PIM_scan_ms=scan*LAYERS*1000 if choice=='PIM' else 0,
            remote_peak_KV_MiB=peak_remote/2**20,remote_immutable_KV_MiB=pool/2**20,remote_peak_private_KV_MiB=(peak_remote-pool)/2**20,
            GPU_peak_KV_MiB=peak_gpu/2**20,total_GPU_remote_peak_KV_MiB=peak_combined/2**20,
            GPU_native_weights_MiB=weights/2**20,GPU_native_temp_MiB=temp/2**20,GPU_peak_modeled_total_MiB=(weights+temp+peak_gpu)/2**20,
            prefill_GPU_layers=LAYERS if choice=='GPU' else 0,prefill_PIM_layers=LAYERS if choice=='PIM' else 0,
            cold_warmup_plus_batch_E2E_ms=(clock+sum(x['latency_s'] for x in warm))*1000,
            cold_warmup_plus_batch_E2E_energy_mJ=(energy+sum(x['energy_J'] for x in warm))*1000)
        summary.append(row)
        print('RESULT',cid,variant,'TTFT',row['TTFT_ms'],'TBT',row['TBT_ms'],'E2E',row['E2E_ms'],'attn%',row['prefill_attention_service_fraction']*100,flush=True)
    # Expose F2 scheduling fairness: keep full transfer bytes and energy, allow its write to overlap GPU attention after fetch.
    sensitivities.append(dict(case_id=cid,F2_native_TTFT_ms=LAYERS*(ct+fetch[0]+fullwrite[0]+gt)*1000,
        F2_export_overlap_TTFT_ms=LAYERS*(ct+fetch[0]+max(fullwrite[0],gt))*1000,F3_TTFT_ms=LAYERS*(ct+gc)*1000,
        F2_export_overlap_E2E_ms=(LAYERS*(ct+fetch[0]+max(fullwrite[0],gt))+sum(x['time_s'] for x in decodes['PIM']))*1000))
    items={x['variant']:x for x in summary if x['case_id']==cid}
    for v in ['F2','F3']:same(items[v]['TBT_ms'],items['F4']['TBT_ms']);same(items[v]['TBT_energy_mJ'],items['F4']['TBT_energy_mJ'])
    assert items['F4']['TTFT_ms']<=items['F3']['TTFT_ms']+1e-8
    same(items['F3']['remote_peak_KV_MiB'],items['F4']['remote_peak_KV_MiB'])
    for row in items.values():
        same(row['TTFT_ms']+(GENERATED-1)*row['TBT_ms'],row['E2E_ms'])
        same(row['TTFT_energy_mJ']+(GENERATED-1)*row['TBT_energy_mJ'],row['E2E_energy_mJ'])
        same(row['prefill_attention_service_ms']+row['prefill_common_GPU_ms'],row['TTFT_ms'])
        assert row['GPU_peak_modeled_total_MiB']*2**20<gpu_config['MEM_CAPACITY_PER_DEVICE']
for name,rows in [('summary',summary),('operators',operator_rows),('events',events),('layers',layers),('decisions',decisions),
    ('transfers',transfers),('storage',storage),('warmup',warmup),('F2-export-overlap-sensitivity',sensitivities)]:csvout(OUT/f'Fugue-asplos-{name}.csv',rows)
for name,digest in base_plan['source_sha256'].items():assert sha(REPO/name)==digest,name
checks=dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=len(cases),result_rows=len(summary),fresh_ramulator_signatures=len(jobs),
    execution_entirely_inside_attacc_fugue=True,native_source_unchanged=True,native_wrapper_timing_and_traffic_parity=True,native_decode_metrics_match=True,
    no_request_latency_division=True,no_request_extrapolation=True,F4_never_slower_than_F3=True,all_capacity_checks_fit_GPU=True,
    layer_service_plus_common_equals_TTFT=True,E2E_equals_TTFT_plus_decode=True)
save(AUDIT/'Fugue-asplos-checks.json',checks)
manifest=ROOT/'Fugue-asplos-SHA256SUMS.txt';manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in sorted(ROOT.rglob('*')) if p.is_file() and p!=manifest))
print('COMPLETE',json.dumps(checks),flush=True)
