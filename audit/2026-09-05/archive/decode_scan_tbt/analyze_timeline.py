#!/usr/bin/env python3
"""Audit retained event timestamps without changing or rerunning any model."""
import bisect,collections,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
finish={};totals=collections.defaultdict(lambda:[0,0.0,0.0]);gpu=[];pim=[];links=[];samples=[];windows={}
metadata=json.loads((HERE/'diag_A6_metadata.json').read_text())
with Path(metadata['slim_path']).open() as f:
    for index,line in enumerate(f):
        e=json.loads(line);e['index']=index
        e['ready_s']=max((finish[d] for d in e['depends_on']),default=0.0)
        finish[e['id']]=e['end_s']
        if not e['name'].startswith('decode_'):continue
        slot=totals[e['name']];slot[0]+=1;slot[1]+=e['time_s'];slot[2]=max(slot[2],e['time_s'])
        w=windows.setdefault(e['tier'],[e['start_s'],e['end_s']]);w[0]=min(w[0],e['start_s']);w[1]=max(w[1],e['end_s'])
        if e['time_s']>0:
            if e['device']=='GPU':gpu.append(e)
            elif e['device'].startswith('PIM:pool'):pim.append(e)
            elif e['device']=='LINK':links.append(e)
# Stored schedules are monotone on each physical device. Find a concrete
# ready event that could fit a preceding GPU-idle interval, without moving
# any dependency or any already executed operation.
gpu.sort(key=lambda e:(e['start_s'],e['index']))
gaps=[]
for previous,current in zip(gpu,gpu[1:]):
    if previous['tier']==current['tier'] and current['start_s']-previous['end_s']>1e-12:
        gaps.append({'start':previous['end_s'],'end':current['start_s'],'before':previous['id'],'after':current['id'],
                     'after_index':current['index'],'tier':current['tier']})
gap_ends=[g['end'] for g in gaps]
witnesses=[]
for event in gpu:
    if not event['name'].startswith('decode_batch_gpu_local'):continue
    i=bisect.bisect_right(gap_ends,event['ready_s']+event['time_s'])
    for g in gaps[i:]:
        if g['start']>=event['start_s']:break
        if g['tier']!=event['tier'] or event['index']<=g['after_index']:continue
        possible=max(g['start'],event['ready_s'])
        if possible+event['time_s']<=g['end']+1e-15:
            witnesses.append({'gap':g,'event':event,'fits_at_s':possible,'advance_us':(event['start_s']-possible)*1e6})
            break
    if len(witnesses)>=6:break

def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    return merged

def overlap(a,b):
    i=j=0;result=0.0
    while i<len(a) and j<len(b):
        result+=max(0.0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return result
stats=[]
for tier,(start,end) in sorted(windows.items()):
    g=union((e['start_s'],e['end_s']) for e in gpu if e['tier']==tier)
    p=union((e['start_s'],e['end_s']) for e in pim if e['tier']==tier)
    l=union((e['start_s'],e['end_s']) for e in links if e['tier']==tier)
    gb=sum(b-a for a,b in g);pb=sum(b-a for a,b in p)
    stats.append({'tier':tier,'decode_span_us':(end-start)*1e6,'gpu_busy_us':gb*1e6,
                  'pim_any_lane_busy_us':pb*1e6,'gpu_pim_overlap_us':overlap(g,p)*1e6,
                  'gpu_busy_fraction':gb/(end-start),'pim_any_lane_busy_fraction':pb/(end-start),
                  'link_busy_fraction':sum(b-a for a,b in l)/(end-start)})
ops=[{'name':name,'count':v[0],'total_us':v[1]*1e6,'mean_us':v[1]/v[0]*1e6,'max_us':v[2]*1e6} for name,v in sorted(totals.items())]
gpu_total=sum(e['time_s'] for e in gpu)
normact_total=sum(e['time_s'] for e in gpu if any(s in e['name'] for s in ('norm','act','gelu','silu','relu')))
result={'scope':'Historical C1v1 A6 diagnostic, not paired with current C1v2 metrics; timestamps reused verbatim, no repricing or rescheduling.',
        'ready_work_in_gpu_idle_gaps':witnesses,'tiers':stats,'ops':ops,
        'decode_gpu_time_us':gpu_total*1e6,'decode_norm_act_time_us':normact_total*1e6,'norm_act_fraction_gpu_time':normact_total/gpu_total}
(HERE/'timeline_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'witnesses':witnesses[:2],'tiers':stats,'norm_act_fraction_gpu_time':normact_total/gpu_total,
                  'gpu_ops':[o for o in ops if 'gpu_' in o['name'] or 'qkv' in o['name']]},indent=2))
