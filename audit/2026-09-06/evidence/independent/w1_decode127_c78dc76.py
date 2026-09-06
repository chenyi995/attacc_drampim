from pathlib import Path
import importlib.util,json,sys
ROOT=Path('/data2/chenyi9/KV-PIM/attacc_drampim_822');sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.workload import load_workload,build_reuse_plan
from src.workload_runner import TableLocalDiffKVLayout,_prepare_cacheblend_tlb,_parent_output_fingerprints,_cacheblend_tlb_rows,_pool_reads
wl=load_workload(ROOT/'workload/probe/sweep/W1_turns.json');p=build_reuse_plan(wl,'recompute',epic_prefix_recompute_tokens=8)
t=TableLocalDiffKVLayout(256,'slice');outputs=_parent_output_fingerprints(wl);_prepare_cacheblend_tlb(wl,p,1,t,outputs)
policies={'A3b':'slice-append','A4c':'master-diff-local-append','A4e':'master-diff-table-local-append'}
results=[]
for req in wl.requests:
 if not req.request_id.endswith('_m_t023'):continue
 bindings=_cacheblend_tlb_rows(wl,p,0,req,t)
 for output_row in [0,127]:
  old=[loc for _,_,_,loc in bindings]
  old += [t.locate(0,req.request_id,outputs.get(req.request_id,req.request_id+'::output'),r,'master') for r in range(output_row)]
  reads,_,_=_pool_reads(t,old)
  row={'request':req.request_id,'output_row':output_row,'previous_output_tokens':output_row,'physical_tokens':len(reads),'layouts':{}}
  for label,policy in policies.items():
   channels=[]
   ledger=t.physical_ledger(policy,2)
   for ch,n,exts in ledger.extent_groups(reads):
    if ch>=8:continue
    local=[(k-ch*(1<<30),v-ch*(1<<30),n) for k,v,n in exts]
    spec=importlib.util.spec_from_file_location('decode127_bank',ROOT/'pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    g.n_channel=1;g.head_hbm_stripe=True;g.n_head_per_hbm=1
    g.Attention(sum(n for k,v,n in local),local[0][0],local[0][1],0,valid_channel=1,extents=local)
    addrs=[int(s.split()[1],16) for b in g.cmd_score_mac[0] for s in b]
    nominal={r for k,v,n in local for r in range(k//1024,(k+n*4-1)//1024+1)}
    channels.append({'channel':ch,'physical_tokens':sum(n for k,v,n in local),'nominal_rows':len(nominal),'score_rows':len({a//1024 for a in addrs}),'score_MAC_AB':len(addrs)})
   row['layouts'][label]={'channels':channels,'score_MAC_AB':sum(c['score_MAC_AB'] for c in channels),
     'nominal_rows':sum(c['nominal_rows'] for c in channels),'score_rows':sum(c['score_rows'] for c in channels),
     'max_channel_nominal_rows':max(c['nominal_rows'] for c in channels),'max_channel_score_rows':max(c['score_rows'] for c in channels),
     'max_channel_MAC_AB':max(c['score_MAC_AB'] for c in channels)}
  results.append(row)
x={'method':'Real W1 k8 layer0 full per-query decode physical read set. At output_row127 add already generated output rows0..126; current token remains the GPU-side local attention item, matching runner previous_output. No batching, Ramulator, device or performance simulation.', 'results':results}
Path('/tmp/w1_decode127_c78dc76_evidence.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps([{**{k:v for k,v in r.items() if k!='layouts'},'layouts':{l:{k:v for k,v in c.items() if k!='channels'} for l,c in r['layouts'].items()}} for r in results],indent=2))
