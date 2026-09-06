#!/usr/bin/env python3
"""Address/plan hand checks and latency budgets. NEVER runs Ramulator or DAG."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from src.workload import load_workload,build_reuse_plan
from src.config import make_model_config,make_xpu_config,SCALING_FACTOR
from src.devices import xPU
from src.model import Transformer,Layer
from src.type import DataType,DeviceType,GPUType,LayerType

REV="958dd24e6ae5e435d5b05a7d376877f161300f79"


def sha(raw):return hashlib.sha256(raw).hexdigest()


def runner_from_source(source,name):
    m=types.ModuleType(name);m.__file__=str(ROOT/'src/workload_runner.py')
    m.__package__="src";sys.modules[name]=m
    exec(compile(source,m.__file__,"exec"),m.__dict__)
    return m


def shape(groups):
    """Independent transcription of dhead128 QK column addresses, not timing."""
    lanes={}
    for channel,_n,extents in groups:
        if channel>=8:continue # one of the two identical head stripes
        commands=[];physical=set()
        for key,_value,count in extents:
            physical.update(range(key//1024,(key+count*4-1)//1024+1))
            commands += [key+i*32 for i in range(2*math.ceil(count/16))]
        rows=[a//1024 for a in commands]
        lanes[channel]=dict(token_rows=sum(n for _,_,n in extents),extent_count=len(extents),
                            physical_rows=len(physical),qk_mac_requests=len(commands),
                            qk_address_rows=len(set(rows)),
                            qk_row_runs=sum(i==0 or rows[i]!=rows[i-1] for i in range(len(rows))))
    return lanes


def describe(m,wl,plan,rung,targets):
    classes={"A3b":m.NaiveKVLayout,"A4c":m.LocalDiffKVLayout,"A4e":m.TableLocalDiffKVLayout}
    tlb=classes[rung](256,"slice")
    m._prepare_cacheblend_tlb(wl,plan,1,tlb,m._parent_output_fingerprints(wl))
    ledger=tlb.physical_ledger(tlb.layout_policy,2)
    result={}
    for req in targets:
        b=m._cacheblend_tlb_rows(wl,plan,0,req,tlb)
        logical=[x[3] for x in b]
        reads,_,_=m._pool_reads(tlb,logical)
        diff=[x for x in logical if x.kind=="diff"]
        master=[x for x in reads if x.kind!="diff"]
        doc_fingerprints={s.fingerprint for s in req.segments if s.role=="doc"}
        doc_master=[x for x in master if x.fingerprint in doc_fingerprints]
        owners=sorted(set(x.owner for x in diff))
        result[req.request_id]={
            "tier":req.tier,"prompt_tokens":req.total_length,"diff_tokens":len(diff),
            "inherited_diff_tokens":sum(x.owner!=req.request_id for x in diff),
            "new_diff_tokens":sum(x.owner==req.request_id for x in diff),
            "compute_tokens":sum(not reused or (corrected and loc.owner==req.request_id) for _,reused,corrected,loc in b),
            "resident_tokens":sum(reused and (not corrected or loc.owner!=req.request_id) for _,reused,corrected,loc in b),
            "physical_scan_tokens":len(reads),"diff_writers":owners,
            "diff":shape(ledger.extent_groups(diff)) if diff else {},
            "master":shape(ledger.extent_groups(master)),
            "document_master":shape(ledger.extent_groups(doc_master)) if doc_master else {},
            "full":shape(ledger.extent_groups(reads)),
            "extents":ledger.extent_groups(reads),
        }
        # The same old diff is still an object of its original request.
        for loc in diff:
            assert (0,loc.owner,loc.fingerprint,"diff") in ledger.index
        tlb.entries.clear()
    return result


def gpu_budgets():
    out={}
    for name in ("CACHEBLEND-TINY","LLAMA3-8B"):
        config=make_xpu_config(GPUType.A100a,num_gpu=1,gpu_model="flash")["GPU"]
        gpu=xPU(DeviceType.GPU,config,SCALING_FACTOR)
        model=Transformer(make_model_config(name,DataType.W16A16),1)
        out[name]={"layers":model.ndec,"gqa":model.gqa_size,"prefill_query_cap":8//model.gqa_size,"batches":{}}
        for b in (1,3,8,32):
            model.build(b,1,2,attn_on_hetero=True)
            layers=model.gen_decoder[0] if isinstance(model.gen_decoder[0],list) else model.gen_decoder
            times={}
            for layer in layers:
                if layer.type==LayerType.FC or layer.type in (LayerType.ACT,LayerType.NORM):
                    times[layer.name]=gpu.get_time_and_energy(layer)[0]*1e6
            total=sum(times.values())
            out[name]["batches"][b]={"gpu_linear_norm_activation_us_per_layer":times,
                                       "gpu_work_us_per_layer":total,
                                       "required_exposed_scan_us_for_5pct_if_scan_gain_15pct":total*.5,
                                       "required_exposed_scan_us_for_10pct_if_scan_gain_15pct":total*2}
    return out


def prefill_budget(m,n,name,link_layer):
    config=make_xpu_config(GPUType.A100a,num_gpu=1,gpu_model="flash")["GPU"]
    gpu=xPU(DeviceType.GPU,config,SCALING_FACTOR)
    model=Transformer(make_model_config(name,DataType.W16A16),1)
    width=model.hdim//model.gqa_size
    resident=n-m
    readback_bytes=resident*2*width*2
    link=Layer("sum","kv_pim_to_gpu",LayerType.X2G,False,DataType.W16A16,1,readback_bytes//2,1,1)
    link=link_layer(link,"kv_pim_to_gpu",readback_bytes)
    tx=gpu.get_time_and_energy(link)[0] if resident else 0
    for label,typ,n2,k in (("score",LayerType.MATMUL,n,128),("softmax",LayerType.SOFTMAX,n,1),("context",LayerType.MATMUL,128,n)):
        op=Layer("sum",label,typ,False,DataType.W16A16,m,n2,k,model.num_heads)
        tx+=gpu.get_time_and_energy(op)[0]
    ctx=Layer("sum","ctx_pim_to_gpu",LayerType.X2G,False,DataType.W16A16,m,model.hdim,1,1)
    ctx=link_layer(ctx,"ctx_pim_to_gpu",m*model.hdim*2)
    tc=gpu.get_time_and_energy(ctx)[0]
    batch_limit=8
    cap=min(batch_limit,8//model.gqa_size);sweeps=math.ceil(m/cap)
    return dict(compute=m,logical_context=n,resident=resident,readback_bytes=readback_bytes,
                query_capacity=cap,batch_limit=batch_limit,sweeps=sweeps,gpu_price_us_per_layer=tx*1e6,
                context_return_us_per_layer=tc*1e6,
                pim_average_sweep_must_be_below_us=(tx-tc)*1e6/sweeps,
                meaning="Break-even budget, not a PIM measurement or predicted side")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir",type=Path,default=HERE/"checks")
    ap.add_argument("--only",default="")
    args=ap.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(["git","show",REV+":src/workload_runner.py"],cwd=ROOT)
    work=(ROOT/'src/workload_runner.py').read_bytes()
    (args.outdir/'head_workload_runner.py.txt').write_bytes(head)
    (args.outdir/'working_workload_runner.py.txt').write_bytes(work)
    variants={"HEAD_partners":runner_from_source(head,"src.targeted_head"),
              "working_table":runner_from_source(work,"src.targeted_working")}
    evidence={"scope":"Static reuse plan / physical ledger and GPU analytical budget; no Ramulator or DAG execution",
              "revision":REV,"head_runner_sha256":sha(head),"working_runner_sha256":sha(work),
              "working_heuristic":getattr(variants["working_table"],"_TABLE_HEURISTIC","partners"),
              "inputs":{}}
    for p in sorted((HERE/'inputs').glob('*.json')):
        if p.name=='manifest.json' or (args.only and args.only not in p.stem):continue
        wl=load_workload(str(p));plan=build_reuse_plan(wl,"recompute",epic_prefix_recompute_tokens=8)
        requests=list(wl.requests)
        if p.stem.startswith('D_'):
            last_tier=max(x.tier for x in requests)
            targets=[r for r in requests if ('_s_' in r.request_id and r.tier==last_tier) or
                     (r.request_id.startswith('g00_s_') and r.tier in (0,15,31,63))]
        else:
            targets=[r for r in requests if r.request_id.startswith('s00_')]
        result={"input_sha256":sha(p.read_bytes()),"meta":wl.raw['meta'] if hasattr(wl,'raw') else json.loads(p.read_text())['meta'],
                "request_count":len(requests),"variants":{}}
        for label,m in variants.items():
            rungs=("A3b","A4c","A4e") if label=='HEAD_partners' else ("A4e",)
            result['variants'][label]={rung:describe(m,wl,plan,rung,targets) for rung in rungs}
        base=result['variants']['HEAD_partners']
        for req in targets:
            for r in ('A4c','A4e'):
                for k in ('prompt_tokens','diff_tokens','new_diff_tokens','inherited_diff_tokens','physical_scan_tokens','compute_tokens','resident_tokens'):
                    assert base[r][req.request_id][k]==base['A3b'][req.request_id][k],(p.stem,k)
        if p.stem.startswith('P_'):
            result['prefill_budgets']={rid:{name:prefill_budget(v['compute_tokens'],v['prompt_tokens'],name,variants['HEAD_partners']._link_layer)
                                          for name in ('CACHEBLEND-TINY','LLAMA3-8B')}
                                       for rid,v in base['A4e'].items()}
            if result['meta']['fresh']:
                result['fresh_budgets']={str(n):prefill_budget(n,n,'LLAMA3-8B',variants['HEAD_partners']._link_layer) for n in (2048,4096,8192)}
        evidence['inputs'][p.stem]=result
        (args.outdir/(p.stem+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        last=targets[-1].request_id
        compact={}
        for rung,values in base.items():
            v=values[last]
            compact[rung]={'diff_lanes':{c:d['qk_address_rows'] for c,d in v['diff'].items()},
                           'master_lanes':{c:d['token_rows'] for c,d in v['master'].items()},
                           'qk_mac_max':max(x['qk_mac_requests'] for x in v['full'].values())}
        print(p.stem,last,json.dumps(compact),flush=True)
    evidence['gpu_budgets']=gpu_budgets()
    (args.outdir/'all.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Static hand checks complete; no simulated performance.',flush=True)


if __name__=='__main__':main()
