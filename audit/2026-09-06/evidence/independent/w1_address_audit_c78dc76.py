from pathlib import Path
import collections,hashlib,importlib.util,json,subprocess,sys
ROOT=Path('/data2/chenyi9/KV-PIM/attacc_drampim_822')
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.workload import load_workload,build_reuse_plan
from src.workload_runner import (TableLocalDiffKVLayout,_prepare_cacheblend_tlb,_parent_output_fingerprints,_cacheblend_tlb_rows,_pool_reads)

def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
POLICIES={'A3b':'slice-append','A4c':'master-diff-local-append','A4e':'master-diff-table-local-append'}
CH=1<<30;ROW=1024
wl=load_workload(ROOT/'workload/probe/sweep/W1_turns.json')
plan=build_reuse_plan(wl,'recompute',epic_prefix_recompute_tokens=8)
outputs=_parent_output_fingerprints(wl)
out={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/workload_runner.py','src/workload.py','pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py','workload/probe/gen_main_workers.py','workload/probe/sweep/W1_turns.json']},
 'method':'Real W1 input, recompute k8 reuse plan, full prompt _pool_reads and actual PhysicalLedger, then score-only command generation. Heads/HBM2, one head selected; 1 and 2 layer controls. No device or Ramulator execution.',
 'main_rounds':[],'layer_controls':{}}

for ndec in [1,2]:
 t=TableLocalDiffKVLayout(256,'slice');_prepare_cacheblend_tlb(wl,plan,ndec,t,outputs)
 ledgers={p:t.physical_ledger(policy,2) for p,policy in POLICIES.items()}
 if ndec==1:
  for req in wl.requests:
   if '_m_' not in req.request_id:continue
   b=_cacheblend_tlb_rows(wl,plan,0,req,t)
   diff=[loc for pos,reused,corrected,loc in b if loc.kind=='diff']
   new=[l for l in diff if l.owner==req.request_id]
   inh=[l for l in diff if l.owner!=req.request_id]
   reads,mask,pr=_pool_reads(t,[l for pos,reused,corrected,l in b])
   out['main_rounds'].append({'request':req.request_id,'prompt_tokens':req.total_length,
     'compute_tokens':sum(not reused or (corrected and l.owner==req.request_id) for pos,reused,corrected,l in b),
     'new_diff_tokens':len(new),'inherited_diff_tokens':len(inh),'total_diff_tokens':len(diff),
     'physical_read_tokens':len(reads),'distinct_diff_owners':sorted({l.owner for l in diff}),
     'distinct_diff_fingerprints':len({l.fingerprint for l in diff})})
 case={'layouts':{},'final_mains':[]}
 for label,ledger in ledgers.items():
  intervals=collections.defaultdict(list);diff_rows={};objects=[]
  for key,(rows,index,slot,start) in ledger.objects.items():
   ch=ledger._channel(0,slot)
   intervals[ch].append((start,start+len(rows)*4,key))
   isdiff=(len(key)>3 and (key[3]=='diff' or key[2]=='burst'))
   if isdiff:
    root=t.chain_root.get(key[1],key[1]);row=(ch,start//ROW)
    diff_rows.setdefault(row,set()).add(root)
    objects.append({'key':key,'layer':key[0],'root':root,'channel':ch,'start':start,'token_count':len(rows),'start_col':start%ROW//32,'start_offset':start%32})
  for ch,parts in intervals.items():
   parts.sort();assert all(parts[i][1]<=parts[i+1][0] for i in range(len(parts)-1))
  if label=='A4e':assert all(len(roots)==1 for roots in diff_rows.values())
  case['layouts'][label]={'diff_objects':objects,'allocated_diff_rows':sorted(diff_rows),
    'cross_root_row_count':sum(len(v)>1 for v in diff_rows.values()),
    'roots':{root:sorted(row for row,owners in diff_rows.items() if root in owners) for root in sorted({r for owners in diff_rows.values() for r in owners})}}
 for layer in range(ndec):
  for req in wl.requests:
   if not req.request_id.endswith('_m_t023'):continue
   b=_cacheblend_tlb_rows(wl,plan,layer,req,t)
   reads,mask,pr=_pool_reads(t,[l for pos,reused,corrected,l in b])
   logical_ids=[(l.layer,l.owner,l.fingerprint,l.owner_row,l.kind) for l in reads]
   assert len(logical_ids)==len(set(logical_ids))
   diff=[l for l in reads if l.kind=='diff'];master=[l for l in reads if l.kind!='diff']
   entry={'request':req.request_id,'layer':layer,'prompt_tokens':req.total_length,'physical_tokens':len(reads),'diff_tokens':len(diff),'master_tokens':len(master),'layouts':{}}
   for label,ledger in ledgers.items():
    groups={ch:ext for ch,n,ext in ledger.extent_groups(reads) if ch<8}
    diff_groups={ch:ext for ch,n,ext in ledger.extent_groups(diff) if ch<8}
    master_groups={ch:ext for ch,n,ext in ledger.extent_groups(master) if ch<8}
    assert sum(n for ext in groups.values() for k,v,n in ext)==len(reads)
    channels=[];boundary=[]
    allocated_diff={tuple(x) for x in case['layouts'][label]['allocated_diff_rows']}
    for ch,exts in groups.items():
     g=module('w1_qk_%s_%s_%d_%d_%d'%(label,req.request_id,ndec,layer,ch),ROOT/'pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py')
     g.n_channel=1;g.head_hbm_stripe=True;g.n_head_per_hbm=1;g.dhead=128;g.n_mac=16
     local=[(k-ch*CH,v-ch*CH,n) for k,v,n in exts]
     g.Attention(sum(n for k,v,n in local),local[0][0],local[0][1],0,valid_channel=1,extents=local)
     addrs=[int(s.split()[1],16) for bkt in g.cmd_score_mac[0] for s in bkt]
     assert all(a<CH for a in addrs)
     nominal_rows=sorted({r for k,v,n in local for r in range(k//ROW,(k+n*4-1)//ROW+1)})
     macrows=sorted({a//ROW for a in addrs})
     drows=sorted({r for k,v,n in diff_groups.get(ch,[]) for r in range((k-ch*CH)//ROW,(k-ch*CH+n*4-1)//ROW+1)})
     mrows=sorted({r for k,v,n in master_groups.get(ch,[]) for r in range((k-ch*CH)//ROW,(k-ch*CH+n*4-1)//ROW+1)})
     channels.append({'channel':ch,'physical_tokens':sum(n for k,v,n in local),'master_tokens':sum(n for k,v,n in master_groups.get(ch,[])),
       'diff_tokens':sum(n for k,v,n in diff_groups.get(ch,[])),'nominal_total_rows':nominal_rows,'nominal_diff_rows':drows,'nominal_master_rows':mrows,
       'score_MAC_AB':len(addrs),'score_rows':macrows,'extra_score_rows':sorted(set(macrows)-set(nominal_rows))})
    # Individual actual diff extent -> exact score MAC address footprint.
    for ch,exts in diff_groups.items():
     for k,v,n in exts:
      local=k-ch*CH
      commands=[local+i*32 for i in range(2*((n+15)//16))]
      intended=set(range(local//ROW,(local+n*4-1)//ROW+1))
      touched={a//ROW for a in commands}
      if touched-intended:
       boundary.append({'channel':ch,'key_address':k,'start_local':local,'tokens':n,'nominal_rows':sorted(intended),
                        'actual_score_rows':sorted(touched),'raw_MAC_addresses':[a+ch*CH for a in commands],
                        'extra_score_rows_unallocated_to_any_diff':sorted(r for r in touched-intended if (ch,r) not in allocated_diff)})
    entry['layouts'][label]={'channels':channels,'physical_tokens':sum(c['physical_tokens'] for c in channels),
      'score_MAC_AB':sum(c['score_MAC_AB'] for c in channels),
      'nominal_diff_row_count':sum(len(c['nominal_diff_rows']) for c in channels),
      'nominal_total_row_count':sum(len(c['nominal_total_rows']) for c in channels),
      'max_channel_nominal_rows':max(len(c['nominal_total_rows']) for c in channels),
      'max_channel_MAC_AB':max(c['score_MAC_AB'] for c in channels),
      'score_extra_row_count':sum(len(c['extra_score_rows']) for c in channels),
      'diff_boundary_extents':boundary}
   case['final_mains'].append(entry)
 out['layer_controls'][str(ndec)]=case
 Path('/tmp/w1_address_audit_c78dc76_evidence.json').write_text(json.dumps(out,indent=2)+'\n')
 print('completed layers',ndec,flush=True)
brief={'final_round_counts':[r for r in out['main_rounds'] if r['request'].endswith('_m_t023')],
       'main_g00_first_rounds':[r for r in out['main_rounds'] if r['request'].startswith('g00')][:4],
       'final_mains':{nd:[{'request':r['request'],'layer':r['layer'],'layout':{lab:{k:v for k,v in data.items() if k!='channels'} for lab,data in r['layouts'].items()}} for r in c['final_mains']] for nd,c in out['layer_controls'].items()}}
print(json.dumps(brief,indent=2))
