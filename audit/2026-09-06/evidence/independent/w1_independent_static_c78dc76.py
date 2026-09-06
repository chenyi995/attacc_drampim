from pathlib import Path
from collections import defaultdict
import json,sys
ROOT=Path('/data2/chenyi9/KV-PIM/attacc_drampim_822')
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.workload import load_workload,build_reuse_plan
from src.workload_runner import TableLocalDiffKVLayout,_prepare_cacheblend_tlb,_parent_output_fingerprints,_cacheblend_tlb_rows,_pool_reads,_block_slot_table
wl=load_workload(ROOT/'workload/probe/sweep/W1_turns.json');p=build_reuse_plan(wl,'recompute',epic_prefix_recompute_tokens=8)
t=TableLocalDiffKVLayout(256,'slice');_prepare_cacheblend_tlb(wl,p,1,t,_parent_output_fingerprints(wl));ledger=t.physical_ledger('master-diff-table-local-append',2)
order=list(dict.fromkeys((fp,row//256) for layer,owner,fp,kind in t._reserved_rows if kind!='diff' for row in sorted(t._reserved_rows[(layer,owner,fp,kind)])))
slots=_block_slot_table(order,t.chunk_coread,8,'table');load=defaultdict(lambda:[0]*8)
for root,coread in zip(t.chunk_coread_roots,t.chunk_coread):
 for (fp,block),slot in slots.items():
  if fp in coread:load[root][slot]+=1
rows_cache={};requests={r.request_id:r for r in wl.requests}
def footprints(req):
 if req not in rows_cache:
  binding=_cacheblend_tlb_rows(wl,p,0,requests[req],t);reads,_,_=_pool_reads(t,[l for _,_,_,l in binding]);out=[]
  for isdiff in [False,True]:
   extents=ledger.extent_groups([l for l in reads if (l.kind=='diff')==isdiff]);rr=set()
   for ch,_,ee in extents:
    if ch>=8:continue
    for k,v,n in ee:
     start=k-ch*(1<<30);rr.update((ch,r) for r in range(start//1024,(start+n*4-1)//1024+1))
   out.append(rr)
  rows_cache[req]=out
 return rows_cache[req]
seen=set();rotation=0;events=[]
for key,(rows,index,slot,start) in ledger.objects.items():
 if key[3]!='diff':continue
 identity=(slot,start//1024)
 if identity in seen:continue
 req=key[1];root=t.chain_root[req];weighted=list(load[root]);chosen=min(range(8),key=lambda c:(weighted[c],(c-rotation)%8));assert chosen==slot
 masters,diffs=footprints(req);prior=diffs&seen
 current=[sum(ch==c for ch,r in masters|prior) for c in range(8)]
 events.append({'request':req,'root':root,'new_diff_row':identity,'weighted_score_before_row':weighted,'current_prompt_master_plus_prior_diff_unique_rows':current,'chosen_channel':chosen,'current_minimum_channels':[c for c,v in enumerate(current) if v==min(current)],'chosen_excess_current_rows':current[chosen]-min(current)})
 seen.add(identity);load[root][slot]+=1;rotation+=1
raw=json.loads((ROOT/'workload/probe/sweep/W1_turns.json').read_text());batch=[]
for tier,step in [(0,0),(0,1),(1,0),(2,0),(23,0)]:
 active=[a for a in raw['agents'] if a['tier']==tier and a['lout']>step]
 sets=[set(s['sha'] for s in a['segs']) for a in active]
 batch.append({'tier':tier,'decode_output_row':step,'active_requests':len(active),'whole_group_common_segment_fingerprints':len(set.intersection(*sets))})
examples=[e for e in events if e['chosen_excess_current_rows']>0]
result={'scope':'Real W1 k8, ndec1, heads_per_HBM2; static plan and PhysicalLedger only. No DAG/device/Ramulator timing. Current row vector is unique nominal row footprint at prompt scan, excluding not-yet-allocated diff rows. It is not MAC command/time estimate.','a4e_diff_new_row_count':len(events),'chosen_not_current_minimum_count':len(examples),'examples':examples[:8],'all_row_allocations':events,'batch_default8':batch}
Path('/tmp/w1_independent_static_c78dc76.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='all_row_allocations'},indent=2))
