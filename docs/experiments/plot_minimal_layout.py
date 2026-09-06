#!/usr/bin/env python3
"""Draw the documented small examples; never launch Ramulator or a DAG run.

Static panels read current physical ledgers. Performance panels require actual
reports/metrics; missing performance inputs produce no performance figures.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import src.workload_runner as runner
from src.workload import load_workload, build_reuse_plan
from src.config import make_model_config
from src.type import DataType
from src.ablation import resolve_config

COLORS = ['#4477AA', '#EE7733', '#228833', '#AA3377']
RUNG_COLORS = {'A3b':'#999999', 'A4c':'#4477AA', 'A4e':'#228833', 'A5':'#EE7733', 'A6':'#AA3377'}


def save(fig, out, name):
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(out/(name+'.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def paths(root, out):
    path = root/'inputs/D4_diff.json'
    wl = load_workload(str(path))
    plan = build_reuse_plan(wl, 'recompute', epic_prefix_recompute_tokens=8)
    targets = [q for q in wl.requests if '_s_t' in q.request_id]
    assert len(targets) == 4
    data = []
    for rung, cls in [('A3b', runner.NaiveKVLayout), ('A4c', runner.LocalDiffKVLayout)]:
        tlb = cls(256, 'slice')
        runner._prepare_cacheblend_tlb(wl, plan, 1, tlb, runner._parent_output_fingerprints(wl))
        ledger = tlb.physical_ledger(tlb.layout_policy, 2)
        for i, q in enumerate(targets):
            bindings = runner._cacheblend_tlb_rows(wl, plan, 0, q, tlb)
            diffs = [b[3] for b in bindings if b[3].kind == 'diff']
            for ch, _, extents in ledger.extent_groups(diffs):
                if ch >= ledger.stripe:
                    continue
                for key, value, n in extents:
                    start = key - ch*runner._HBM_CHANNEL_BYTES
                    end = start + n*4
                    for row in range(start//1024, (end-1)//1024+1):
                        data.append(dict(rung=rung, scan=i+1, request=q.request_id, channel=ch,
                                         K_row=row, tokens=(min(end,(row+1)*1024)-max(start,row*1024))//4,
                                         key_extent_start=key, value_extent_start=value))
    write_csv(out/'scan_paths.csv', data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, rung in zip(axes, ('A3b', 'A4c')):
        items = [x for x in data if x['rung']==rung]
        rows = sorted({x['K_row'] for x in items})
        index = {r:i for i,r in enumerate(rows)}
        cells = {}
        for x in items:
            cells.setdefault((x['channel'],x['K_row']),set()).add(x['scan'])
        for row in rows:
            y = index[row]
            for ch in range(8):
                ax.add_patch(Rectangle((ch+.08, y+.1), .84, .64, facecolor='#F2F2F2', edgecolor='#BDBDBD', lw=.6))
                for scan in sorted(cells.get((ch,row),set())):
                    # Fixed stripes encode membership in different scans,
                    # not occupied bytes/columns within the physical row.
                    ax.add_patch(Rectangle((ch+.10+(scan-1)*.20,y+.12),.19,.60,facecolor=COLORS[scan-1],edgecolor='none'))
        for scan in range(1,5):
            for ch in range(8):
                touched = sorted({x['K_row'] for x in items if x['scan']==scan and x['channel']==ch})
                for a,b in zip(touched,touched[1:]):
                    x = ch+.10+(scan-1)*.20+.095
                    ax.annotate('',xy=(x,index[b]+.08),xytext=(x,index[a]+.76),
                                arrowprops=dict(arrowstyle='->',lw=1,color=COLORS[scan-1]))
        ax.set_xlim(0,8); ax.set_ylim(4.15,-.15)
        ax.set_xticks([i+.5 for i in range(8)],['ch%d'%i for i in range(8)])
        ax.xaxis.tick_top()
        ax.set_yticks([i+.42 for i in range(len(rows))],['row %d'%r for r in rows])
        ax.tick_params(length=0, labelsize=9)
        ax.set_title(rung+': %d correction row%s'%(len(rows),'' if len(rows)==1 else 's'),pad=28)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.legend([Rectangle((0,0),1,1,color=c) for c in COLORS], ['Round %d scan'%i for i in range(1,5)],
               loc='lower center',ncol=4,bbox_to_anchor=(.5,.045),frameon=False)
    fig.suptitle('D4: the same head, eight channels, four continuing scans',y=1.03)
    fig.text(.5,0,'Static physical K addresses; correction subset only. Arrows show ascending row addresses, not cycle timing.\n'
                 'A colored stripe means the row is touched by that scan; it does not encode column occupancy. Other rows are omitted.',
             ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.24,wspace=.36)
    save(fig,out,'scan_paths')


def hotspots(static, out):
    rows = []
    for case in ('E2_hot','E2_balanced'):
        for rung in ('A4c','A4e'):
            row = next(x for x in static['compact'] if x['case']==case and x['rung']==rung)
            rows.append((case,rung,row['full_QK_by_ch']))
    raw = [dict(case=c,rung=r,channel=ch,QK_column_requests=n) for c,r,ns in rows for ch,n in enumerate(ns)]
    write_csv(out/'channel_hotspots.csv',raw)
    fig,ax = plt.subplots(figsize=(11,3.3))
    norm=Normalize(0,max(x['QK_column_requests'] for x in raw))
    cmap=plt.get_cmap('YlOrRd')
    for i,(case,rung,ns) in enumerate(rows):
        for ch,n in enumerate(ns):
            ax.add_patch(Rectangle((ch,i),1,.8,facecolor=cmap(norm(n)),edgecolor='white',lw=1))
            ax.text(ch+.5,i+.4,str(n),ha='center',va='center',color='white' if norm(n)>.65 else '#222222')
    ax.set_xlim(0,8);ax.set_ylim(4,-.65)
    ax.set_xticks([i+.5 for i in range(8)],['ch%d'%i for i in range(8)])
    ax.xaxis.tick_top()
    ax.set_yticks([i+.4 for i in range(4)],[c+' / '+r for c,r,_ in rows])
    ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,pad=.02,label='QK column requests (static)')
    fig.suptitle('Last-request channel load: two co-read documents',y=1.04)
    fig.text(.5,-.01,'Each tile is one channel; a shared color scale preserves the load difference.\n'
                     'The two large documents spread across two channels, not uniformly across all eight. This is not a latency heatmap.',ha='center',fontsize=9)
    save(fig,out,'channel_hotspots')


def decode_fig(root,out):
    path=root/'metrics.csv'
    if not path.exists(): return
    records=list(csv.DictReader(path.open()))
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for col,case in enumerate(('D4_diff','E2_hot','E2_balanced')):
        subset=[r for r in records if r['case']==case]
        for row,(key,label) in enumerate((('last_scan_service_us_per_scan','Scan service (us)'),('last_tbt_us','TBT (us)'))):
            ax=axes[row,col]
            ax.bar(range(len(subset)),[float(x[key]) for x in subset],color=[RUNG_COLORS[x['rung']] for x in subset])
            ax.set_xticks(range(len(subset)),[x['gpu']+'\n'+x['rung'] for x in subset],fontsize=8)
            ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.set_title(case)
            for i,x in enumerate(subset):ax.annotate('%.4g'%float(x[key]),(i,float(x[key])),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
            high=max(float(x[key]) for x in subset)
            for gpu in dict.fromkeys(x['gpu'] for x in subset):
                pair=[i for i,x in enumerate(subset) if x['gpu']==gpu]
                if len(pair)==2:
                    a,b=pair
                    old,new=float(subset[a][key]),float(subset[b][key])
                    if old:
                        ax.text((a+b)/2,high*1.18,'gain {:+.1%}'.format(1-new/old),ha='center',fontsize=8)
            ax.set_ylim(0,high*1.38 if high else 1)
    fig.suptitle('Matched last-request decode positions; simulated performance')
    fig.tight_layout();save(fig,out,'scan_and_tbt')


def bank_measure(events,rid,decode,heads,dhead):
    selected=[]
    for e in events:
        members=e.get('batch_members') or [e['request']]
        if rid not in members or 'pim_kv_scan' not in e['name'] or e['name'].startswith('decode_')!=decode:
            continue
        positions=tuple(e['query_positions'])
        # Shared scans carry a batch label in request, and one query
        # position per member. Count this request's useful work only.
        # Unbatched prefill sweeps may contain several queries of rid.
        if e.get('batch_members') and len(positions)==len(members):
            positions=(positions[members.index(rid)],)
        selected.append((e,positions))
    if not selected: return None,None
    if decode:
        first=min(p for e,positions in selected for p in positions)
        selected=[(e,tuple(p for p in positions if p>first)) for e,positions in selected
                  if any(p>first for p in positions)]
    sweeps={};queries=set()
    for e,positions in selected:
        key=(e['transformer_layer'],e['name'],tuple(e['query_positions']))
        sweeps[key]=max(sweeps.get(key,0),e['time_s'])
        queries.update((e['transformer_layer'],p) for p in positions)
    seconds=sum(sweeps.values())
    # Useful QK+PV MACs: one query attends to positions 0..p.  Two phases,
    # one multiply+add per element; H_Q already includes the GQA Q heads.
    work=4*heads*dhead*sum(p+1 for _,p in queries)
    return (work/seconds/1e9 if seconds else None),seconds


def prefill_fig(root,out):
    paths=[root/'prefill_runs/B200/P4_reuse'/('dag_'+r+'.json') for r in ('A4e','A5','A6')]
    if not any(p.exists() for p in paths):return
    assert all(p.exists() for p in paths),'Complete all three P4 reports before plotting'
    inp=root/'inputs/P4_reuse.json'
    raw=inp.read_bytes();wl=json.loads(raw)
    model=make_model_config('LLAMA3-8B',DataType.W16A16)
    heads=model['num_heads'];dhead=model['hdim']//heads
    records=[];corrections=set()
    for rung,path in zip(('A4e','A5','A6'),paths):
        report=json.loads(path.read_text());cfg=report['run_config']
        for key,val in dict(gpu='B200',model='LLAMA3-8B',gpu_model='flash',ngpu=1,num_hbm=5,
                            pim_link='nvlink3',pipeopt=True,powerlimit=True,word=2,engine='dag',
                            reuse='recompute',epic_prefix_recompute_tokens=8,cacheblend_batch_size=8).items():
            assert cfg[key]==val,(rung,key,cfg[key],val)
        assert cfg['workload_sha256']==hashlib.sha256(raw).hexdigest()
        preset=resolve_config(rung,None,None,None,policy='recompute')
        for key,val in dict(pim_batch_command=preset.pim_batch_command,
                            pim_prefill_mode=preset.prefill_attn,
                            pim_pe_freq_ghz=preset.pim_pe_freq_ghz,
                            gemv_buffer_bytes=preset.gemv_buffer_bytes).items():
            assert report[key]==val,(rung,key,report[key],val)
        corrections.add(report['corrected_rows_sha'])
        assert report['events'],'P4 requires full events'
        for req in wl['agents']:
            rid=req['id'];summary=report['summary']['requests'][rid]
            pre,seconds=bank_measure(report['events'],rid,False,heads,dhead)
            dec,_=bank_measure(report['events'],rid,True,heads,dhead) if req['lout']>1 else (None,None)
            records.append(dict(rung=rung,request=rid,ttft_us=summary['ttft_s']*1e6,
                                prefill_side='pim' if pre is not None else 'gpu',
                                bank_prefill_effective_gflops=pre,bank_prefill_service_s=seconds,
                                bank_decode_effective_gflops=dec,pe_freq_ghz=report.get('pim_pe_freq_ghz'),
                                full_e2e_s=report['makespan_s']))
    assert len(corrections)==1,'P4 correction work differs across rungs'
    for rec in records:
        baseline=next(x for x in records if x['rung']=='A4e' and x['request']==rec['request'])
        old=baseline['bank_decode_effective_gflops'];new=rec['bank_decode_effective_gflops']
        rec['bank_decode_speedup_vs_A4e']=new/old if old and new is not None else None
        rec['ttft_reduction_vs_A4e']=1-rec['ttft_us']/baseline['ttft_us'] if baseline['ttft_us'] else None
    write_csv(out/'prefill_bank.csv',records)
    fig,axes=plt.subplots(1,4,figsize=(13,4))
    ids=[q['id'] for q in wl['agents']]
    for ax,rid,title in zip(axes[:3],ids,('Cold import','First reuse','Next: q4')):
        vals=[next(x for x in records if x['request']==rid and x['rung']==r) for r in ('A4e','A5','A6')]
        ax.bar(range(3),[x['ttft_us'] for x in vals],color=[RUNG_COLORS[x['rung']] for x in vals])
        ax.set_xticks(range(3),[x['rung'] for x in vals]);ax.set_ylabel('TTFT (us)');ax.set_title(title)
        ax.set_ylim(0,max(x['ttft_us'] for x in vals)*1.22)
        for i,x in enumerate(vals):ax.annotate('%.4g'%x['ttft_us'],(i,x['ttft_us']),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    target=ids[-1]
    vals=[next(x for x in records if x['request']==target and x['rung']==r) for r in ('A4e','A5','A6')]
    axes[3].bar(range(3),[x['bank_decode_effective_gflops'] for x in vals],color=[RUNG_COLORS[x['rung']] for x in vals])
    axes[3].set_xticks(range(3),[x['rung'] for x in vals])
    axes[3].set_ylim(0,max(x['bank_decode_effective_gflops'] for x in vals)*1.22)
    axes[3].set_ylabel('Bank-side effective QK+PV GFLOP/s')
    axes[3].set_title('Same final decode work')
    for i,x in enumerate(vals):axes[3].annotate('{:.2f}x'.format(x['bank_decode_speedup_vs_A4e']),
            (i,x['bank_decode_effective_gflops']),xytext=(0,3),textcoords='offset points',ha='center',fontsize=9)
    fig.suptitle('P4 / B200: prefill placement and achieved bank throughput')
    fig.text(.5,-.03,'Throughput covers all PIM stacks of one GPU; service includes the attention scan and softmax.\n'
                      'A4e prefill executes on GPU (bank-prefill throughput: N/A). A5 and A6 have the same bank hardware.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,out,'ttft_and_bank_throughput')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    static=json.loads((args.root/'static.json').read_text())
    for name,digest in static['source_sha256'].items():
        assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==digest,('Source changed',name)
    provenance={'scope':'Static panels are address/command counts; performance panels exist only when reports exist',
                'source_sha256':static['source_sha256'],
                'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((args.root/'inputs').glob('*.json'))}}
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    paths(args.root,args.out);hotspots(static,args.out)
    decode_fig(args.root,args.out);prefill_fig(args.root,args.out)
    print('Figures and source data:',args.out)


if __name__=='__main__':main()
