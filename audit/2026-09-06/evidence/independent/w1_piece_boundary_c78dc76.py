from pathlib import Path
import collections,importlib.util,json,sys
ROOT=Path('/data2/chenyi9/KV-PIM/attacc_drampim_822');sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.workload_runner import TableLocalDiffKVLayout

def score(ext,ch):
 s=importlib.util.spec_from_file_location('w1_piece_generator',ROOT/'pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
 g.n_channel=1;g.head_hbm_stripe=True;g.n_head_per_hbm=1
 local=[(k-ch*(1<<30),v-ch*(1<<30),n) for k,v,n in ext]
 g.Attention(sum(n for k,v,n in local),local[0][0],local[0][1],0,valid_channel=1,extents=local)
 a=[int(s.split()[1],16) for b in g.cmd_score_mac[0] for s in b]
 return [{'channel':ch,'row':v//1024,'col':v%1024//32,'raw_local':v} for v in a]

control={}
for policy in ['master-diff-local-append','master-diff-table-local-append']:
 t=TableLocalDiffKVLayout(256,'slice');t.chain_root={'earlier':'agent','later':'agent'}
 for row in range(248):t.reserve(0,'earlier','prefix',row,'diff')
 for row in range(16):t.reserve(0,'later','target',row,'diff')
 t.finalize();ledger=t.physical_ledger(policy,2)
 target=[t.locate(0,'later','target',r,'diff') for r in range(16)]
 groups=[(ch,ex) for ch,n,ex in ledger.extent_groups(target) if ch<8]
 mapping=[]
 for r in range(16):
  key=(0,'later','target','diff');ok=ledger.index[key][r];rows,ordinal,slot,start=ledger.objects[ok]
  address=ledger._channel(0,slot)*(1<<30)+start+ordinal[r]*4
  mapping.append({'owner_row':r,'channel':address//(1<<30),'local_address':address%(1<<30),'piece':ok[-1]})
 assert len({(m['channel'],m['local_address']) for m in mapping})==16
 assert sum(n for ch,e in groups for k,v,n in e)==16
 control[policy]={'target_tokens':mapping,'target_extent_groups':groups,'generated_target_MACs':[c for ch,ex in groups for c in score(ex,ch)]}

w1=json.loads(Path('/tmp/w1_address_audit_c78dc76_evidence.json').read_text());real={}
for ndec,c in w1['layer_controls'].items():
 layoutinfo={}
 for lab,lay in c['layouts'].items():
  rows=collections.defaultdict(set)
  for obj in lay['diff_objects']:rows[(obj['channel'],obj['start']//1024)].add(obj['layer'])
  layoutinfo[lab]={'rows_shared_across_layers':[{'channel':ch,'row':row,'layers':sorted(ls)} for (ch,row),ls in rows.items() if len(ls)>1]}
 boundaries=[]
 for req in c['final_mains']:
  for b in req['layouts']['A4e']['diff_boundary_extents']:
   obj=next(o for o in c['layouts']['A4e']['diff_objects'] if o['layer']==req['layer'] and o['channel']==b['channel'] and o['start']==b['start_local'])
   later=[o for o in c['layouts']['A4e']['diff_objects'] if o['root']==obj['root'] and o['layer']==obj['layer']]
   ix=next(i for i,o in enumerate(later) if o['key']==obj['key'])
   boundaries.append({'request':req['request'],'layer':req['layer'],'boundary':b,'object':obj,'next_agent_diff_object':later[ix+1] if ix+1<len(later) else None})
 real[ndec]={'layout_info':layoutinfo,'A4e_actual_boundaries':boundaries}
result={'method':'Real ledger target-row mapping and actual generator Attention MAC extraction. No Ramulator or device calls; all boundaries are command/address evidence, not numerical attention or timing measurements.',
        'control_prefix248_target16':control,'W1_real_boundary_objects':real}
Path('/tmp/w1_piece_boundary_c78dc76_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'control':control,'real':real},indent=2))
