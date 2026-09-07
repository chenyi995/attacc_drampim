"""Directly simulate the union of the final experiment 1/2 shapes once."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime,json,math
from pathlib import Path
from fugue.runtime import REPO,RUN,JOBS,setup,check_sources,sha,load,save,csvout
from fugue.attention import profiler,CAP
from src.config import make_model_config,make_xpu_config,make_pim_config,SCALING_FACTOR
from src.devices import xPU,PIM
from src.model import Layer
from src.type import *
ROOT=RUN/'experiment12';RAW,OUT,AUDIT,RUNTIME=setup(ROOT)
lock=check_sources();config=load(REPO/'artifact/inputs/sweep.json')
points=sorted(set(tuple(p) for group in ['experiment1','experiment2'] for p in config[group]),key=lambda p:(p[1],p[0]))
model=make_model_config('LLAMA-7B',DataType.W16A16)
gpu=xPU(DeviceType.GPU,make_xpu_config(GPUType.A100a,num_gpu=1)['GPU'],SCALING_FACTOR)
pim=PIM(make_pim_config(PIMType.BA,InterfaceType.NVLINK3,num_attacc=1,num_hbm=5,power_constraint=True),SCALING_FACTOR,None)
profile=profiler(REPO,ROOT,RAW,RUNTIME,model,cached_lengths=sorted({c for q,c in points}))
needed={}
for q,c in points:
    needed.setdefault(q+c,set()).update([1,min(q,CAP)]+([q%CAP] if q%CAP else []))
# These signatures back the published native-command discontinuity evidence table.
for n in [1536,1537,1568,1569]:needed.setdefault(n,set()).update([1,8])
save(AUDIT/'Fugue-asplos-plan.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repo=str(REPO),
    driver_sha256=sha(Path(__file__)),native_source_sha256=lock,input_sha256=sha(REPO/'artifact/inputs/sweep.json'),
    points=points,jobs=JOBS,note='Fresh native signatures; no dependency on publication references or old output.'))
for n in sorted(needed):key,lines=profile.make_base(n);profile.base_lines[key]=lines
jobs=[(n,r) for n,rs in sorted(needed.items()) for r in sorted(rs)]
print('SWEEP',len(points),'shapes',len(jobs),'fresh signatures',flush=True)
signatures={}
with ThreadPoolExecutor(max_workers=JOBS) as pool:
    for i,f in enumerate(as_completed([pool.submit(profile.simulate,*j) for j in jobs]),1):
        key,value=f.result();signatures[key]=value
        if i%25==0 or i==len(jobs):print('RAMULATOR',i,'/',len(jobs),flush=True)
operator_rows=[]
def gpu_op(q,n,name):
    kind=LayerType.SOFTMAX if name=='softmax' else LayerType.MATMUL
    dims=(q,128,n) if name=='context' else (q,n,1 if name=='softmax' else 128)
    layer=Layer('sum',name,kind,False,DataType.W16A16,*dims,32)
    t,_=gpu.get_time_and_energy(layer)
    operator_rows.append(dict(q=q,n=n,operator=name,time_us=t*1e6,compute_us=gpu._compute_time(layer)*1e6,
        memory_us=max(gpu._mem_time(layer))*1e6,offchip_bytes=layer.off_traffic))
    return t*1e6
rows=[];BW=300e9
for q,c in points:
    n=q+c;qk,sm,pv=[gpu_op(q,n,name) for name in ['score','softmax','context']]
    psm=pim.get_time_and_energy(Layer('sum','softmax',LayerType.SOFTMAX,False,DataType.W16A16,q,n,1,32))[0]*1e6
    qi=q*8192/BW*1e6;kv=2*qi;fetch=c*16384/BW*1e6;gb=qk+sm+pv+fetch
    row=dict(q=q,cached=c,total_kv=n,gpu_qk_us=qk,gpu_softmax_us=sm,gpu_pv_us=pv,gpu_attention_us=qk+sm+pv,
        pim_softmax_us=psm,q_input_bytes=q*8192,q_input_us=qi,output_us=qi,new_kv_bytes=q*16384,
        new_kv_transfer_us=kv,cached_kv_readback_us=fetch,gpu_new_kv_hidden_us=min(kv,gb),
        gpu_new_kv_exposed_us=max(0,kv-gb),gpu_service_us=max(gb,kv),data_source='portable fresh simulation')
    for mode in ['plain','mq']:
        tiles=[1]*q if mode=='plain' else [CAP]*(q//CAP)+([q%CAP] if q%CAP else [])
        scan=sum(signatures[n,r][0]['scan_us'] for r in tiles)
        w=0. if c==0 else signatures[n,tiles[0]][1][c]
        exposed=max(0,kv-w);total=2*qi+scan+psm+exposed
        row.update({f'{mode}_scan_us':scan,f'{mode}_first_tile_queries':tiles[0],f'{mode}_tiles':len(tiles),
            f'{mode}_kv_overlap_window_us':w,f'{mode}_new_kv_hidden_us':min(kv,w),f'{mode}_new_kv_exposed_us':exposed,
            f'{mode}_service_us':total,f'{mode}_speedup_over_gpu':row['gpu_service_us']/total,
            f'{mode}_winner':'PIM' if total<row['gpu_service_us'] else 'GPU'})
    rows.append(row)
for group in ['experiment1','experiment2']:
    selected={tuple(p) for p in config[group]};csvout(OUT/f'Fugue-asplos-{group}.csv',[r for r in rows if (r['q'],r['cached']) in selected])
csvout(OUT/'Fugue-asplos-gpu-operators.csv',operator_rows)
csvout(OUT/'Fugue-asplos-signatures.csv',[signatures[k][0] for k in sorted(signatures)])
check_sources()
save(AUDIT/'Fugue-asplos-checks.json',dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    shapes=len(rows),fresh_ramulator_signatures=len(jobs),source_unchanged=True,all_profiles_succeeded=True))
print('SWEEP COMPLETE',len(rows),flush=True)
