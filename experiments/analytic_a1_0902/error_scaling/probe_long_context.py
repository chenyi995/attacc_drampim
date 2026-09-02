"""Price genuinely LONGER runs with Ramulator and test the model past its domain."""
import json, sys, os, time
from pathlib import Path
import numpy as np
sys.path.insert(0,"/home/xw338/attacc/attacc_drampim")
os.chdir("/home/xw338/attacc/attacc_drampim")
from src.analytic_pim import estimate
from src.ramulator_wrapper import Ramulator
from src.config import make_model_config
from src.model import Layer
from src.type import DataType, PIMType, LayerType

D="/tmp/claude-1614106/-zpool-00-home-xw338/09c77f20-3fa4-470c-9e66-492ac633afa2/scratchpad/extrap_ram"
os.makedirs(D, exist_ok=True)
for name in ("ramulator2","trace_gen"):
    dst=os.path.join(D,name)
    if not os.path.exists(dst): os.symlink(f"/home/xw338/attacc/attacc_drampim/ramulator2/{name}", dst)
open(os.path.join(D,"signature_cache.jsonl"),"w").close()

info=make_model_config("LLAMA-7B", DataType.W16A16)
ram=Ramulator(info, D, output_log="", workers=1, num_hbm=5, signature_cache=False)
models=json.load(open("experiments/analytic_a1_0902/timing_models.json"))
dom=models["regimes"]["chunkstripe1|replicate"]["domain"]["run_length"]
print(f"calibrated run_length domain: {dom}")

# A1's two real shapes, pushed well past the domain edge.
CASES=[]
for L in (8192, 16384, 32768, 39168, 49152, 65536, 98304, 131072, 196608, 262144):
    CASES.append(("decode ch=1 q=1", L, 1, 1, 1, 0))       # heads, channels, q, base
    CASES.append(("prefill ch=16 q=8", L, 1, 16, 8, 0))
print(f"\n{'shape':>18} {'run_length':>10} {'ramulator':>12} {'analytic':>12} {'ratio':>8}  {'in domain':>10}")
rows=[]
for shape, L, heads, ch, q, base in CASES:
    layer=Layer("x","score",LayerType.MATMUL,False,DataType.W16A16,q,L,128,heads*5)
    key=base*(1<<30); val=key+(1<<23)
    layer.pim_kv_runs=((key,val,L,base,ch),)
    layer.pim_shared_kv = q>1
    layer.pim_shared_queries=q
    t=time.perf_counter()
    seconds,_=ram.run(PIMType.BA, layer, True)
    wall=time.perf_counter()-t
    truth=round(seconds*1e9/ram.tCK)
    diag={}
    pred=estimate(pim_type="BA",run_length=L,num_ops_per_hbm=heads,dbyte=2,power_constraint=True,
                  dhead=128,num_hbm=5,channel_count=ch,shared_kv=q>1,shared_queries=q,
                  channel_base=base,mq_command=False,key_addr=key,value_addr=val,
                  phase="full",trace_revision="chunkstripe1",timing_models=models,
                  diagnostics=diag)[0]
    inside = "yes" if dom[0]<=L<=dom[1] else "NO (extrap)"
    print(f"{shape:>18} {L:10d} {truth:12d} {pred:12d} {pred/truth:8.4f}  {inside:>10}   ramulator {wall:.1f}s", flush=True)
    rows.append(dict(shape=shape,L=L,truth=truth,pred=pred,ratio=pred/truth,inside=inside))
json.dump(rows, open("/tmp/claude-1614106/-zpool-00-home-xw338/09c77f20-3fa4-470c-9e66-492ac633afa2/scratchpad/extrap.json","w"), indent=1)
