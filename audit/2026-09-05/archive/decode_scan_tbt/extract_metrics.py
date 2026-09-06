#!/usr/bin/env python3
"""Derive scan/TBT percentage reductions from existing, paired reports only."""
import csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=Path('/data2/chenyi9/KV-PIM/scratch_0905/out_C1v2_CACHEBLEND-TINY_hbm4_k8')
rows=[];provenance=[]
for tier in ('A3b','A4c','A4e','A5','A6'):
    p=source/f'dag_{tier}.json';r=json.loads(p.read_text());config=r['run_config']
    workload_path=Path(config['workload']);workload=json.loads(workload_path.read_text())
    whash=hashlib.sha256(workload_path.read_bytes()).hexdigest()
    assert whash==config['workload_sha256'],('Workload changed; need original lout',whash,config['workload_sha256'])
    lout={a['id']:a['lout'] for a in workload['agents']}
    numerator=0;denominator=0
    for rid,q in r['summary']['requests'].items():
        if lout[rid]>1:numerator+=q['end_s']-q['first_token_s'];denominator+=lout[rid]-1
    scans=r['summary']['decode_scans']
    row={'tier':tier,'tbt_weighted_us':numerator/denominator*1e6,'interval_count':denominator,
         'private_service_us':scans['private_service']['mean_us'],
         'shared_service_us':scans['shared_service']['mean_us'],
         'scan_step_elapsed_us':scans['per_step_elapsed']['mean_us'],
         'e2e_s':r['makespan_s'],'gpu_busy_fraction_e2e':r['gpu_time_s_unoverlapped']/r['makespan_s']}
    rows.append(row);provenance.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'run_config':config,'corrected_rows_sha':r['corrected_rows_sha'],'scans':scans})
assert len({p['corrected_rows_sha'] for p in provenance})==1
assert len({json.dumps(p['run_config'],sort_keys=True) for p in provenance})==1
comparisons=[]
for a,b in [('A3b','A4c'),('A4c','A4e'),('A3b','A4e'),('A4e','A5'),('A5','A6')]:
    old=next(r for r in rows if r['tier']==a);new=next(r for r in rows if r['tier']==b)
    c={'comparison':a+'->'+b}
    for key in ['private_service_us','shared_service_us','scan_step_elapsed_us','tbt_weighted_us']:
        c[key+'_saved_pct']=(1-new[key]/old[key])*100
    tbt=c['tbt_weighted_us_saved_pct']
    for key in ['private_service_us','scan_step_elapsed_us']:
        s=c[key+'_saved_pct'];c['percentage_retention_vs_'+key]=100*tbt/s if abs(s)>1e-6 else None
    comparisons.append(c)
# This is the Fig. 3b K-only controlled case, NOT the C1 full decode scan.
fig=Path('/data2/chenyi9/KV-PIM/KVPIM-1Fugue-ASPLOS2027/fig/plots/motiv/experiments/03_multiagent_reuse/result.csv')
with fig.open() as fh:raw=list(csv.DictReader(fh))
figrows=[]
for old in (r for r in raw if r['tier']=='A3b'):
    new=next(r for r in raw if r['tier']=='A4e' and r['rounds']==old['rounds'])
    figrows.append({'rounds':int(old['rounds']),'a3b_cycles':int(old['cycles']),'a4e_cycles':int(new['cycles']),
                    'latency_saved_pct':100*(1-int(new['cycles'])/int(old['cycles']))})
result={'existing_reports':provenance,'rows':rows,'comparisons':comparisons,'fig3_k_only':figrows,
        'limitation':'Percentage retention is the ratio of two relative reductions with different denominators, not a fraction of saved microseconds. Summary scan means include first-token scans; TBT excludes first-token interval. TINY diagnostic, not LLAMA3 performance conclusion.'}
(HERE/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
text='| 比较 | 私有 scan 减少 | 共读 scan 减少 | 每步扫描跨度减少 | 加权 TBT 减少 | 百分比保留：TBT / 每步扫描 |\n|---|---:|---:|---:|---:|---:|\n'
for r in comparisons:
    retention=r['percentage_retention_vs_scan_step_elapsed_us']
    text+='| '+r['comparison']+' | '+' | '.join(f"{r[k]:.3f}%" for k in ['private_service_us_saved_pct','shared_service_us_saved_pct','scan_step_elapsed_us_saved_pct','tbt_weighted_us_saved_pct'])+' | '+(f'{retention:.2f}%' if retention is not None else 'N/A')+' |\n'
(HERE/'metrics_table.md').write_text(text)
print(text);print(json.dumps({'absolute':rows,'fig3':figrows},indent=2))
