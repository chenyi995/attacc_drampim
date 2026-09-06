#!/usr/bin/env python3
"""Read compact completed reports; retain full-run and summarizer cohort TBT."""
import argparse,csv,hashlib,json,re
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reports',type=Path)
    ap.add_argument('workload',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--summary-pattern',default=r'(_s_t|^s\d+_t)',
                    help='Request-ID regex selecting the continuing summary chains')
    a=ap.parse_args()
    raw=json.loads(a.workload.read_text())
    reqs={r['id']:r for r in raw['agents']}
    summarizers=[r for r in reqs if re.search(a.summary_pattern,r)]
    if not summarizers:raise ValueError('No requests matched --summary-pattern')
    last=max(reqs[r]['tier'] for r in summarizers)
    groups={'all_requests':list(reqs),'all_summary_turns':summarizers,
            'last_quarter_summary_turns':[r for r in summarizers if reqs[r]['tier']>=3*(last+1)//4],
            'last_summary_turn':[r for r in summarizers if reqs[r]['tier']==last]}
    rows=[]
    for p in sorted(a.reports.glob('dag_A*.json')):
        report=json.loads(p.read_text())
        claimed=report.get('run_config',{}).get('workload_sha256')
        if claimed and claimed!=hashlib.sha256(a.workload.read_bytes()).hexdigest():
            raise ValueError((str(p),'workload SHA256 mismatch'))
        summary=report['summary']['requests']
        missing=set(reqs)-set(summary)
        if missing:raise ValueError((str(p),'report/workload mismatch',sorted(missing)))
        rung=p.stem.removeprefix('dag_')
        for name,ids in groups.items():
            tbt_n=sum(reqs[r]['lout']-1 for r in ids if reqs[r]['lout']>1)
            tbt_s=sum(summary[r]['end_s']-summary[r]['first_token_s'] for r in ids if reqs[r]['lout']>1)
            ttft=[summary[r]['ttft_s'] for r in ids if summary[r].get('ttft_s') is not None]
            rows.append(dict(rung=rung,cohort=name,requests=len(ids),tbt_intervals=tbt_n,
                             tbt_weighted_us=tbt_s/tbt_n*1e6 if tbt_n else None,
                             ttft_mean_us=sum(ttft)/len(ttft)*1e6 if ttft else None,
                             full_workload_e2e_s=report['makespan_s'],
                             corrected_rows_sha=report.get('corrected_rows_sha'),
                             run_git_rev=report.get('run_config',{}).get('git_rev'),
                             run_git_dirty=report.get('run_config',{}).get('git_dirty')))
    if not rows:raise ValueError('No completed dag_A*.json reports')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(a.output)


if __name__=='__main__':main()
