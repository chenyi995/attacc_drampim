#!/usr/bin/env python3
"""Native AttAcc cost/energy replay of two small CacheBlend inputs; no placement changes."""
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
from contextlib import redirect_stdout
import csv,datetime,hashlib,io,json,math,os
from pathlib import Path
import resource,shutil,subprocess,sys

from fugue.runtime import REPO,RUN,JOBS,setup,check_sources
ROOT=RUN/'experiment3'
RAW,OUT,AUDIT,RUNTIME=setup(ROOT)
from src.config import make_model_config,make_xpu_config,make_pim_config,SCALING_FACTOR,ENERGY_TABLE
from src.devices import xPU,PIM
from src.model import Layer,Transformer
from src.system import System
from src.type import DataType,DeviceType,LayerType,GPUType,PIMType,InterfaceType
from src.ramulator_wrapper import Ramulator

BASE=RUN/'unused'
SOFTWARE=REPO/'artifact/inputs'

MODEL='LLAMA-7B';LAYERS=32;KV_BYTES=16384;Q_BYTES=8192;GENERATED=10;RATIO=.16;B=300e9
TCK=.769;CAP=8;E_COL=ENERGY_TABLE['PIM'][PIMType.BA]['mem']*32
E_OP=16*ENERGY_TABLE['PIM'][PIMType.BA]['alu']+32*ENERGY_TABLE['PIM'][PIMType.BA]['sram']
model=make_model_config(MODEL,DataType.W16A16)
gpu_config=make_xpu_config(GPUType.A100a,num_gpu=1)['GPU']
pim_config=make_pim_config(PIMType.BA,InterfaceType.NVLINK3,num_attacc=1,num_hbm=5,power_constraint=True)
gpu=xPU(DeviceType.GPU,gpu_config,SCALING_FACTOR)
pim=PIM(pim_config,SCALING_FACTOR,None)
cfg={'cached_lengths':[]} # scattered recomputation: do not assume an old-prefix overlap window

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
def same(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
base_plan={'source_sha256':check_sources(),'base_commit':'c60005143a6b492d7ef83231723386478b59a506'}
method=REPO/'fugue/attention.py'
from fugue.attention import profiler
profile=profiler(REPO,ROOT,RAW,RUNTIME,model)
make_base,simulate,base_lines=profile.make_base,profile.simulate,profile.base_lines
work=json.loads((SOFTWARE/'cacheblend.json').read_text())
requests=work['requests'];catalog={r['chunk_id']:r for r in work['cache_catalog']}
for r in requests:shutil.copy2(SOFTWARE/Path(r['source_path']).name,RAW/('Fugue-asplos-source-'+Path(r['source_path']).name))
shutil.copy2(SOFTWARE/'cacheblend-token-ids.json',RAW/'Fugue-asplos-token-ids.json')
shutil.copy2(SOFTWARE/'cacheblend-selection.csv',OUT/'Fugue-asplos-workload-selection.csv')
(ROOT/'Fugue-asplos-workload.json').write_text(json.dumps(work,indent=2)+'\n')
plan=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),driver_sha256=sha(Path(__file__)),
    native_source_sha256=base_plan['source_sha256'],native_commit=base_plan['base_commit'],
    input_sha256=sha(SOFTWARE/'cacheblend.json'),input_provenance=json.loads((SOFTWARE/'cacheblend-provenance.json').read_text()),
    reused_method_sha256=sha(method),cwd=str(Path.cwd()),repo=str(REPO),argv=sys.argv,requests=requests,
    methodology='Frozen CacheBlend tokens; unchanged numerical evaluation from final experiment3. See repository methodology.')
(AUDIT/'Fugue-asplos-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
needed={}
for r in requests:
    n,q=r['total_prompt_tokens'],r['selected_query_tokens']
    needed.setdefault(n,set()).update([1,8]+([q%8] if q%8 else [])+([n%8] if n%8 else []))
    for step in range(1,GENERATED):needed.setdefault(n+step,set()).add(1)
signatures={}
for n in sorted(needed):key,lines=make_base(n);base_lines[key]=lines
jobs=[(n,r) for n,rs in sorted(needed.items()) for r in sorted(rs)]
print('WORKLOAD',[(r['source_input'],r['total_prompt_tokens'],r['selected_query_tokens']) for r in requests],flush=True)
print('RAMULATOR signatures',len(jobs),flush=True)
with ThreadPoolExecutor(max_workers=JOBS) as pool:
    for i,future in enumerate(as_completed([pool.submit(simulate,n,r) for n,r in jobs]),1):
        key,value=future.result();signatures[key]=value[0]
        if i%10==0 or i==len(jobs):print('RAMULATOR',i,'/',len(jobs),flush=True)

class NativeTraceReplay:
    def output(self,pim_type,layer,power_constraint=True):
        n,q=int(layer.n),int(layer.m);tiles=[8]*(q//8)+([q%8] if q%8 else [])
        # Decode q=1 is exactly the native single-query signature.
        elapsed=0.;traffic=[0.]*5
        for resident in tiles:
            d=signatures[n,resident];elapsed+=d['scan_us']*1e-6
            path=ROOT/d['path'];counts=json.loads((path/'Fugue-asplos-command.json').read_text())['command_counts']
            mac=counts['PIM_MAC_AB'];wr=counts['PIM_WR_GB'];sb=counts['PIM_MV_SB'];gb=counts['PIM_MV_GB']
            # Exact native Ramulator.output bank-PIM byte conversion, including 5 HBM.
            one=[wr*32,(wr+sb+gb)*32,(wr+sb+gb)*32,(wr+sb+gb)*32,mac*32*2*2*4*4]
            traffic=[a+b*5 for a,b in zip(traffic,one)]
        return elapsed,traffic
pim.ramulator=NativeTraceReplay()
operator_rows=[];event_rows=[];decision_rows=[];transfer_rows=[];storage_rows=[];layer_rows=[]
GPU_CACHE={};PIM_CACHE={}

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
    hdim=8192 if kind=='kv' else 4096
    return op(gpu,Layer('sum','comm_x2g',LayerType.X2G,False,DataType.W16A16,tokens,hdim,1,1))

def gpu_attention(q,n):
    spec=[('score',LayerType.MATMUL,q,n,128),('softmax',LayerType.SOFTMAX,q,n,1),('context',LayerType.MATMUL,q,128,n)]
    items=[op(gpu,Layer('sum',name,kind,False,DataType.W16A16,m,nn,k,32)) for name,kind,m,nn,k in spec]
    return sum(t for t,e in items),sum(e for t,e in items)

def pim_attention(q,n):
    scan=op(pim,Layer('sum','score',LayerType.MATMUL,False,DataType.W16A16,q,n,128,32))
    sm=op(pim,Layer('sum','softmax',LayerType.SOFTMAX,False,DataType.W16A16,q,n,1,32))
    return scan[0]+sm[0],scan[1]+sm[1],scan[0],sm[0]

def common(qkv_tokens,post_tokens,n):
    tr=Transformer(model,tensor_parallel=1);tr.build(1,post_tokens,GENERATED,False)
    total=[0.,0.];details=[]
    for layer in tr.sum_decoder:
        if layer.type in (LayerType.MATMUL,LayerType.SOFTMAX,LayerType.X2G):continue
        if layer.name=='qkv':layer.m=qkv_tokens
        t,e=op(gpu,layer);total[0]+=t;total[1]+=e;details.append((layer.name,t,e))
    return total,details

# Native GPU and PIM top-level decode check: collect per-token post-pipeline values.
decode_results={}
for r in requests:
    rid,n=r['request_id'],r['total_prompt_tokens']
    for mode in ('GPU','PIM'):
        system=System(gpu_config,model)
        if mode=='PIM':
            system.hetero_name=DeviceType.PIM
            system.devices['Acc']=PIM(pim_config,SCALING_FACTOR,NativeTraceReplay())
        output=[];log=io.StringIO()
        with redirect_stdout(log):system.simulate(1,n,GENERATED,perfs=output,pipe=False,parallel_ff=False,power_constraint=True)
        (AUDIT/f'Fugue-asplos-native-decode-{rid}-{mode}.log').write_text(log.getvalue())
        (AUDIT/f'Fugue-asplos-native-decode-{rid}-{mode}.json').write_text(json.dumps(output,default=str,indent=2)+'\n')
        per_token=[]
        for step,layers in enumerate(system.model.gen_decoder,1):
            t=sum(x.exec_time for x in layers)*LAYERS;e=sum(sum(x.energy) for x in layers)*LAYERS*1e-12
            per_token.append(dict(step=step,time_s=t,energy_J=e))
        same(sum(x['time_s'] for x in per_token)/(GENERATED-1),output[0][2][7]/1000)
        same(sum(x['energy_J'] for x in per_token)/(GENERATED-1),output[0][3][0]*1e-9)
        decode_results[rid,mode]=per_token

pool_tokens=sum(chunk['tokens'] for chunk in catalog.values());pool_bytes=pool_tokens*KV_BYTES*LAYERS
# Reproduce original standalone warm-cache construction costs separately.
warm_rows=[]
for chunk in catalog.values():
    n=chunk['tokens'];base,details=common(n,n,n);ga=gpu_attention(n,n);write=transfer(n)
    warm_rows.append(dict(chunk_id=chunk['chunk_id'],tokens=n,latency_s=(base[0]+ga[0]+write[0])*LAYERS,
                          energy_J=(base[1]+ga[1]+write[1])*LAYERS,remote_KV_bytes=n*KV_BYTES*LAYERS))
csvout(OUT/'Fugue-asplos-warmup.csv',warm_rows)
summary=[]
for r in requests:
    rid,n,q=r['request_id'],r['total_prompt_tokens'],r['selected_query_tokens']
    per_layer=[]
    for l in range(LAYERS):
        qkv_n=n if l<2 else q;aq=n if l==0 else q;post=n if l==0 else q;updated=n if l<2 else q
        cached=n-updated
        c,details=common(qkv_n,post,n);gt,ge=gpu_attention(aq,n);pt,pe,scan,sm=pim_attention(aq,n)
        fetch=transfer(cached);write=transfer(updated);qin=transfer(aq,'q');out=qin
        # At check layer the software compares old V. Charge that read equally;
        # top-k CPU/GPU selection itself is outside the native AttAcc operator set.
        selector=transfer(r['cached_input_tokens'],'q') if l==1 else (0.,0.)
        gpu_cost=max(gt+fetch[0],write[0]);pim_cost=qin[0]+pt+write[0]+out[0]
        gpu_energy=ge+fetch[1]+write[1];pim_energy=pe+qin[1]+write[1]+out[1]
        decision='PIM' if pim_cost<gpu_cost else 'GPU'
        decision_rows.append(dict(request_id=rid,layer=l,qkv_tokens=qkv_n,attention_queries=aq,total_KV=n,
            untouched_cached_KV=cached,updated_KV_tokens=updated,GPU_candidate_us=gpu_cost*1e6,PIM_candidate_us=pim_cost*1e6,
            GPU_candidate_energy_J=gpu_energy,PIM_candidate_energy_J=pim_energy,choice=decision,
            selection_objective='minimum full attention service latency, not energy',overlap_window_us=0,
            query_input_us=qin[0]*1e6,output_us=out[0]*1e6,cached_KV_readback_us=fetch[0]*1e6,new_KV_write_us=write[0]*1e6,
            GPU_attention_us=gt*1e6,PIM_scan_us=scan*1e6,PIM_softmax_us=sm*1e6,
            applicability='Actual per-layer Q/N/residency; cost equations of experiment1/2, with W=0 for scattered updates.'))
        per_layer.append(dict(l=l,aq=aq,updated=updated,cached=cached,common=c,details=details,selector=selector,
            gt=gt,ge=ge,pt=pt,pe=pe,fetch=fetch,write=write,qin=qin,out=out,gpu_cost=gpu_cost,pim_cost=pim_cost,
            gpu_energy=gpu_energy,pim_energy=pim_energy,decision=decision))
    for variant in ('F1','F2','F3','F4'):
        clock=0.;energy=0.;private_remote=0;peak_remote=pool_bytes;peak_gpu_kv=0;active_gpu=0;peak_combined_KV=pool_bytes
        for d in per_layer:
            l=d['l'];phase_start=clock
            # Common native GPU work. Record native operator costs individually;
            # summary grouping does not claim intra-layer head-level scheduling.
            for name,t,e in d['details']:
                if name!='qkv':continue
                event_rows.append(dict(request_id=rid,variant=variant,phase='prefill_common',layer=l,token_step=0,
                    event=name,device='GPU',start_s=clock,end_s=clock+t,duration_s=t,energy_J=e))
                clock+=t;energy+=e
            st,se=d['selector']
            if st:
                event_rows.append(dict(request_id=rid,variant=variant,phase='prefill_selection_input',layer=l,token_step=0,
                    event='old_V_transfer_for_software_selection',device='link',start_s=clock,end_s=clock+st,duration_s=st,energy_J=se))
            clock+=st;energy+=se
            choice=d['decision'] if variant=='F4' else 'GPU'
            if variant=='F1':
                service=d['gt']+d['fetch'][0];service_energy=d['ge']+d['fetch'][1]
                fetch_tokens=d['cached'];write_tokens=0;qin_tokens=0
                active_gpu+=n*KV_BYTES
            elif variant=='F2':
                # Fetch untouched chunks, then assemble/export N current K/V before
                # GPU attention, matching the native prefill qkv -> X2G -> attention order.
                wt,we=transfer(n);service=d['fetch'][0]+wt+d['gt'];service_energy=d['fetch'][1]+we+d['ge']
                fetch_tokens=d['cached'];write_tokens=n;qin_tokens=0;private_remote+=n*KV_BYTES
            elif choice=='GPU':
                service=d['gpu_cost'];service_energy=d['gpu_energy']
                fetch_tokens=d['cached'];write_tokens=d['updated'];qin_tokens=0;private_remote+=d['updated']*KV_BYTES
            else:
                service=d['pim_cost'];service_energy=d['pim_energy']
                fetch_tokens=0;write_tokens=d['updated'];qin_tokens=d['aq'];private_remote+=d['updated']*KV_BYTES
            event_rows.append(dict(request_id=rid,variant=variant,phase='prefill_attention',layer=l,token_step=0,
                event='attention_service',device=choice,start_s=clock,end_s=clock+service,duration_s=service,energy_J=service_energy))
            clock+=service;energy+=service_energy
            for name,t,e in d['details']:
                if name=='qkv':continue
                event_rows.append(dict(request_id=rid,variant=variant,phase='prefill_common',layer=l,token_step=0,
                    event=name,device='GPU',start_s=clock,end_s=clock+t,duration_s=t,energy_J=e))
                clock+=t;energy+=e
            layer_rows.append(dict(request_id=rid,variant=variant,layer=l,attention_device=choice,query_tokens=d['aq'],
                elapsed_s=clock-phase_start,energy_J=d['common'][1]+se+service_energy,attention_service_s=service,
                common_GPU_s=d['common'][0],selector_V_transfer_s=st,KV_stored_private_tokens=0 if variant=='F1' else write_tokens))
            for name,tokens,kind in [('untouched_KV_readback',fetch_tokens,'kv'),('new_or_materialized_KV_write',write_tokens,'kv'),
                                     ('Q_input',qin_tokens,'q'),('O_return',qin_tokens,'q'),('selector_old_V_read',r['cached_input_tokens'] if l==1 else 0,'q')]:
                t,e=transfer(tokens,kind)
                transfer_rows.append(dict(request_id=rid,variant=variant,phase='prefill',layer=l,operation=name,tokens=tokens,
                    bytes=tokens*(KV_BYTES if kind=='kv' else Q_BYTES),standalone_s=t,energy_J=e,
                    time_accounting='Included in service schedule; overlap affects time only, energy charged for every transaction.'))
            scratch=n*KV_BYTES if choice=='GPU' and variant!='F1' else d['updated']*KV_BYTES
            peak_gpu_kv=max(peak_gpu_kv,active_gpu if variant=='F1' else scratch)
            peak_remote=max(peak_remote,pool_bytes+private_remote)
            peak_combined_KV=max(peak_combined_KV,pool_bytes+private_remote+(active_gpu if variant=='F1' else scratch))
            storage_rows.append(dict(request_id=rid,variant=variant,phase='prefill',layer=l,decode_step=0,
                immutable_remote_KV_bytes=pool_bytes,private_remote_KV_bytes=private_remote,active_gpu_KV_bytes=active_gpu,
                layer_GPU_KV_buffer_bytes=scratch,remote_total_KV_bytes=pool_bytes+private_remote))
        ttft,ttft_e=clock,energy
        dec=decode_results[rid,'GPU' if variant=='F1' else 'PIM']
        for entry in dec:
            t,e=entry['time_s'],entry['energy_J'];step=entry['step']
            event_rows.append(dict(request_id=rid,variant=variant,phase='decode',layer='all',token_step=step,
                event='native_full_transformer_decode',device='GPU' if variant=='F1' else 'GPU+PIM',start_s=clock,end_s=clock+t,duration_s=t,energy_J=e))
            clock+=t;energy+=e
            if variant=='F1':active_gpu+=LAYERS*KV_BYTES;peak_gpu_kv=max(peak_gpu_kv,active_gpu)
            else:private_remote+=LAYERS*KV_BYTES;peak_remote=max(peak_remote,pool_bytes+private_remote)
            peak_combined_KV=max(peak_combined_KV,pool_bytes+private_remote+(active_gpu if variant=='F1' else KV_BYTES))
            storage_rows.append(dict(request_id=rid,variant=variant,phase='decode',layer='all',decode_step=step,
                immutable_remote_KV_bytes=pool_bytes,private_remote_KV_bytes=private_remote,active_gpu_KV_bytes=active_gpu,
                layer_GPU_KV_buffer_bytes=KV_BYTES,remote_total_KV_bytes=pool_bytes+private_remote))
        original_system=System(gpu_config,model)
        weights,original_kv,temp=original_system.get_required_mem_capacity(1,n,GENERATED)
        native_kv=LAYERS*(n+GENERATED-1)*KV_BYTES;same(original_kv,native_kv)
        summary.append(dict(request_id=rid,source_input=r['source_input'],variant=variant,prompt_tokens=n,selected_query_tokens=q,
            generated_tokens=GENERATED,batch=1,model=MODEL,GPU='A100a',one_way_link_GBps=300,
            TTFT_ms=ttft*1000,TBT_ms=sum(d['time_s'] for d in dec)/len(dec)*1000,E2E_ms=clock*1000,
            TTFT_energy_mJ=ttft_e*1000,TBT_energy_mJ=sum(d['energy_J'] for d in dec)/len(dec)*1000,E2E_energy_mJ=energy*1000,
            remote_peak_KV_MiB=peak_remote/2**20,remote_immutable_KV_MiB=pool_bytes/2**20,
            remote_peak_private_KV_MiB=(peak_remote-pool_bytes)/2**20,GPU_peak_KV_MiB=peak_gpu_kv/2**20,
            GPU_native_weights_MiB=weights/2**20,GPU_native_temp_MiB=temp/2**20,
            GPU_peak_modeled_total_MiB=(weights+temp+peak_gpu_kv)/2**20,
            total_GPU_remote_peak_KV_MiB=peak_combined_KV/2**20,
            prefill_GPU_layers=sum(1 for d in per_layer if variant!='F4' or d['decision']=='GPU'),
            prefill_PIM_layers=sum(1 for d in per_layer if variant=='F4' and d['decision']=='PIM')))
        same(clock,ttft+sum(d['time_s'] for d in dec));same(energy,ttft_e+sum(d['energy_J'] for d in dec))
# Mandatory invariants: decode unchanged by sharing, dynamic never worse than F3
# on the exact same service candidates, capacity references count old tensors once.
for r in requests:
    items={x['variant']:x for x in summary if x['request_id']==r['request_id']}
    for metric in ('TBT_ms','TBT_energy_mJ'):
        same(items['F2'][metric],items['F3'][metric]);same(items['F3'][metric],items['F4'][metric])
    assert items['F4']['TTFT_ms']<=items['F3']['TTFT_ms']+1e-8
    same(items['F3']['remote_peak_KV_MiB'],items['F4']['remote_peak_KV_MiB'])
    assert items['F3']['remote_peak_KV_MiB']<items['F2']['remote_peak_KV_MiB']
    for item in items.values():
        same(item['E2E_ms'],item['TTFT_ms']+(GENERATED-1)*item['TBT_ms'])
        same(item['E2E_energy_mJ'],item['TTFT_energy_mJ']+(GENERATED-1)*item['TBT_energy_mJ'])
for filename,records in [('summary',summary),('operators',operator_rows),('events',event_rows),('decisions',decision_rows),
                         ('transfers',transfer_rows),('storage',storage_rows),('layers',layer_rows)]:
    csvout(OUT/f'Fugue-asplos-{filename}.csv',records)
for name,digest in base_plan['source_sha256'].items():assert sha(REPO/name)==digest
checks=dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),requests=len(requests),variants=4,
    fresh_ramulator_signatures=len(jobs),source_unchanged=True,no_placement_changes=True,no_request_extrapolation=True,
    native_decode_time_and_energy_match=True,F2_F3_F4_decode_identical=True,F4_latency_not_above_F3=True,
    E2E_equals_TTFT_plus_decode_latency_and_energy=True,F3_F4_share_storage=True,
    warmup_latency_s=sum(x['latency_s'] for x in warm_rows),warmup_energy_J=sum(x['energy_J'] for x in warm_rows),
    immutable_pool_tokens=pool_tokens,unique_cache_chunks=len(catalog),result_rows=len(summary),
    energy_units='Native pJ / 1e9 = mJ; dynamic modeled energy only',
    omissions=['No numerical LLM inference/accuracy validation','No top-k selection runtime, RoPE, embedding, final logits or host scheduling model in native AttAcc',
               'Original X2G energy counts link traffic but not separate DMA endpoint energy or static power',
               'Capacity is tensor payload plus original weight/temp estimate; allocator metadata and fragmentation are not modeled'])
(AUDIT/'Fugue-asplos-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
manifest=ROOT/'Fugue-asplos-SHA256SUMS.txt'
manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in sorted(ROOT.rglob('*')) if p.is_file() and p!=manifest))
print('COMPLETE',json.dumps(checks),flush=True)
