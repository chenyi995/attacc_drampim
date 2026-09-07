"""Regenerate final figures/tables from fresh runs or the checked-in final tables."""
import csv,json,math,os,shutil
from pathlib import Path
from collections import Counter
from fugue.runtime import REPO,RUN,load,save,sha,csvout
os.environ['MPLBACKEND']='Agg';os.environ['MPLCONFIGDIR']=str(RUN/'matplotlib-cache')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.ticker import FixedLocator,FuncFormatter,LogLocator,NullFormatter
PAPER=REPO/'Fugue-paper';DEST=RUN/'paper'
FROM_PAPER=os.environ.get('FUGUE_PLOT_FROM_PAPER')=='1'
WANTED={int(v) for v in os.environ.get('FUGUE_EXPERIMENTS','1,2,3,4,5').split(',')}
def loadcsv(path):return list(csv.DictReader(path.open()))
def f(row,key):return float(row[key])
num=f
read=loadcsv
write=save
def same(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def folder(i):return next(PAPER.glob(f'Fugue-asplos-experiment-{i}-*'))
def target(i):
    root=DEST/folder(i).name
    for sub in ['figures','tables','provenance','workload']:(root/sub).mkdir(parents=True,exist_ok=True)
    return root
def inputs(i):
    return folder(i)/'tables' if FROM_PAPER else RUN/('experiment12' if i<3 else 'experiment3' if i==3 else 'experiment45')/'Fugue-asplos-results'
def read_tables(i):
    dest=target(i);src=inputs(i)
    for p in src.glob('*.csv'):shutil.copy2(p,dest/'tables'/p.name)
    (dest/'README.md').write_text('# Reproduced experiment '+str(i)+'\n\nFigures and tables in this directory were generated from '+('checked-in final tables.' if FROM_PAPER else 'fresh simulator results.')+'\n\n[Workload, metrics and interpretation]('+os.path.relpath(folder(i)/'README.md',dest)+')\n')
    if (folder(i)/'workload').exists():shutil.copytree(folder(i)/'workload',dest/'workload',dirs_exist_ok=True)
    return dest
def price(row,b,mode='mq'):
    q,c=int(row['q']),int(row['cached']);qi=q*8192/b/1000;kv=2*qi;fetch=c*16384/b/1000
    gb=f(row,'gpu_attention_us')+fetch;g=max(gb,kv)
    w=f(row,mode+'_kv_overlap_window_us');exposed=max(0,kv-w)
    p=2*qi+f(row,mode+'_scan_us')+f(row,'pim_softmax_us')+exposed
    record=dict(q=q,cached=c,total_kv=q+c,link_GBps_one_way=b,pim_mode=mode,
        gpu_qk_us=f(row,'gpu_qk_us'),gpu_softmax_us=f(row,'gpu_softmax_us'),gpu_pv_us=f(row,'gpu_pv_us'),
        gpu_attention_us=f(row,'gpu_attention_us'),pim_scan_us=f(row,mode+'_scan_us'),
        pim_softmax_us=f(row,'pim_softmax_us'),q_input_us=qi,output_us=qi,new_kv_transfer_us=kv,
        cached_kv_readback_us=fetch,pim_kv_overlap_window_us=w,pim_new_kv_hidden_us=min(kv,w),
        pim_new_kv_exposed_us=exposed,gpu_new_kv_hidden_us=min(kv,gb),gpu_new_kv_exposed_us=max(0,kv-gb),
        gpu_service_us=g,pim_service_us=p,gpu_over_pim_latency_ratio=g/p,
        signed_faster_side_latency_reduction=(g-p)/max(g,p),winner='PIM' if p<g else 'GPU',
        shape_data_source=row.get('data_source','original sweep'),
        link_pricing='Analytical transfer and overlap terms; native scan and GPU operator costs unchanged')
    same(record['pim_new_kv_hidden_us']+exposed,kv)
    same(record['gpu_new_kv_hidden_us']+record['gpu_new_kv_exposed_us'],kv)
    same(record['gpu_attention_us'],sum(record[k] for k in ('gpu_qk_us','gpu_softmax_us','gpu_pv_us')))
    if b==300:
        same(g,f(row,'gpu_service_us'));same(p,f(row,mode+'_service_us'))
    return record

def edges(values):
    x=np.log2(np.asarray(values,dtype=float));mid=(x[1:]+x[:-1])/2
    return 2**np.concatenate(([x[0]-(mid[0]-x[0])],mid,[x[-1]+(x[-1]-mid[-1])]))

def sweep_plots():
    E1,E2=read_tables(1),read_tables(2)
    cfg=load(REPO/'artifact/inputs/sweep.json')
    if FROM_PAPER:rows=loadcsv(inputs(1)/'Fugue-asplos-experiment1-latency.csv');heat=loadcsv(inputs(2)/'Fugue-asplos-experiment2-shapes.csv')
    else:rows=loadcsv(inputs(1)/'Fugue-asplos-experiment1.csv');heat=loadcsv(inputs(2)/'Fugue-asplos-experiment2.csv')
    groups={c:[r for r in rows if int(r['cached'])==c] for c in [0,1024]}
    csvout(E1/'tables/Fugue-asplos-experiment1-latency.csv',rows)
    csvout(E1/'tables/Fugue-asplos-experiment1-bandwidth.csv',[price(r,b) for b in [32,64,300,450] for r in groups[1024]])
    panels=[dict(panel='a',c=0,b=300,mode='plain'),dict(panel='b',c=1024,b=300,mode='plain'),dict(panel='c',c=1024,b=450,mode='mq'),
            dict(panel='d',c=0,b=300,mode='mq'),dict(panel='e',c=1024,b=300,mode='mq'),dict(panel='f',c=1024,b=32,mode='mq')]
    crosses=[]
    for p in panels:
        data=[price(r,p['b'],p['mode']) for r in groups[p['c']]]
        for before,after in zip(data,data[1:]):
            if before['winner']==after['winner']:continue
            assert after['q']==before['q']+1
            crosses.append(dict(panel=p['panel'],cached=p['c'],link_GBps_one_way=p['b'],pim_mode=p['mode'],direction=before['winner']+'_to_'+after['winner'],
                integer_switch_q=after['q'],q_lower=before['q'],q_upper=after['q'],lower_pim_us=before['pim_service_us'],lower_gpu_us=before['gpu_service_us'],
                upper_pim_us=after['pim_service_us'],upper_gpu_us=after['gpu_service_us']))
    csvout(E1/'tables/Fugue-asplos-experiment1-crossings.csv',crosses)
    index={(int(r['cached']),int(r['q'])):r for r in rows}
    csvout(E1/'tables/Fugue-asplos-experiment1-selected-latency.csv',[index[c,q] for c,q in [(0,51),(1024,8),(1024,54),(1024,55),(1024,200),(1024,512),(1024,513),(1024,544),(1024,545),(1024,1024),(1024,2048)]])
    csvout(E1/'tables/Fugue-asplos-experiment1-components.csv',[price(index[c,q],300,mode) for c,q in [(0,51),(1024,8),(1024,200),(1024,512),(1024,513),(1024,544),(1024,545)] for mode in ['plain','mq']])
    qgrid=sorted({int(r['q']) for r in heat});cgrid=sorted({int(r['cached']) for r in heat})
    hindex={(int(r['cached']),int(r['q'])):r for r in heat}
    csvout(E2/'tables/Fugue-asplos-experiment2-shapes.csv',heat)
    csvout(E2/'tables/Fugue-asplos-experiment2-link-Q.csv',[dict(price(hindex[1024,q],b),heatmap='link_Q') for b in cfg['link_GBps'] for q in qgrid])
    csvout(E2/'tables/Fugue-asplos-experiment2-cache-Q.csv',[dict(price(hindex[c,q],300),heatmap='cache_Q') for c in cgrid for q in qgrid])

    # Rebuild the command-count evidence from fresh native recorder files.
    if not FROM_PAPER:
        evidence=[]
        for n in [1536,1537,1568,1569]:
            for resident in [1,8]:
                timing=RUN/'experiment12/Fugue-asplos-raw'/f'Fugue-asplos-N{n:05d}-Q{resident}'/'Fugue-asplos-timing.json'
                d=load(timing)
                counts=Counter(line.split(',')[1].strip() for p in timing.parent.glob('Fugue-asplos-commands.ch*') for line in p.read_text().splitlines())
                evidence.append(dict(total_kv=n,resident_queries=resident,scan_us=d['scan_us'],cycles=d['cycles'],input_commands=d['trace_commands'],
                    actual_MACAB=counts['MACAB'],actual_ACTAB=counts['ACTAB'],actual_MVSB=counts['MVSB'],actual_MVGB=counts['MVGB'],actual_REFab=counts['REFab'],
                    timing_path=str(timing),timing_sha256=sha(timing)))
        csvout(E1/'tables/Fugue-asplos-experiment1-transition-evidence.csv',evidence)
    def crossing_label(panel):
        group=[r for r in crosses if r['panel']==panel]
        if not group:return 'No crossing in sampled Q'
        if len(group)==1:return str(group[0]['integer_switch_q'])
        return str(group[0]['q_lower'])+'--'+str(group[-1]['q_upper'])
    def mdtable(headers,values):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in values])
    (E1/'tables/Fugue-asplos-experiment1-summary.md').write_text('# Final crossover samples\n\n'+mdtable(['Panel','Mode','C','GB/s one way','Crossing Q'],
        [[p['panel'],p['mode'],p['c'],p['b'],crossing_label(p['panel'])] for p in panels])+'\n')
    bs=chr(92)
    (E1/'tables/Fugue-asplos-experiment1-bandwidth-table.tex').write_text(bs+'begin{tabular}{llr}\n'+bs+'toprule\nMode & One-way link & Crossover $Q$ '+bs*2+'\n'+bs+'midrule\n'+
        ''.join(f'{mode} & {link} & {crossing_label(panel)} '+bs*2+'\n' for mode,link,panel in [('PIM','NVLink 3, 300 GB/s','b'),('MQ PIM','NVLink 3, 300 GB/s','e'),('MQ PIM','NVLink 4, 450 GB/s','c'),('MQ PIM','PCIe 4, 32 GB/s','f')])+bs+'bottomrule\n'+bs+'end{tabular}\n')
    intervals=[]
    for c in cgrid:
        values=[price(hindex[c,q],300) for q in qgrid]
        changes=[str(a['q'])+'--'+str(b['q']) for a,b in zip(values,values[1:]) if a['winner']!=b['winner']]
        intervals.append([c,', '.join(changes) if changes else values[0]['winner']+' faster over sampled Q'])
    (E2/'tables/Fugue-asplos-experiment2-summary.md').write_text('# Final coarse-grid crossover brackets\n\n'+mdtable(['Cached tokens','MQ/GPU switch bracket'],intervals)+'\n')

    panels=[dict(panel='a',c=0,b=300,mode='plain'),dict(panel='b',c=1024,b=300,mode='plain'),
            dict(panel='c',c=1024,b=450,mode='mq'),dict(panel='d',c=0,b=300,mode='mq'),
            dict(panel='e',c=1024,b=300,mode='mq'),dict(panel='f',c=1024,b=32,mode='mq')]
    for panel in panels:
        panel['data']=[price(row,panel['b'],panel['mode']) for row in groups[panel['c']]]
        panel['crossings']=[{k:(int(v) if k in ('q_lower','q_upper','integer_switch_q') else float(v) if k.endswith('_us') else v) for k,v in row.items()}
                            for row in crosses if row['panel']==panel['panel']]
        panel['q_window']=(.82,2500) if panel['c']==0 else (16,192) if panel['mode']=='plain' else (192,1536)
    FIG=E1/'figures'
    GPU,PIM='#C15B33','#236892'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
        'axes.titlesize':11,'legend.fontsize':9.5,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'xtick.labelsize':9,'ytick.labelsize':9,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,3,figsize=(11.8,6.7),sharex=False)
    fig.subplots_adjust(left=.076,right=.985,bottom=.145,top=.79,wspace=.30,hspace=.58)
    marker_q={1,2,4,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048}
    for ax,panel in zip(axes.flat,panels):
        data=panel['data'];q=np.array([r['q'] for r in data]);g=np.array([r['gpu_service_us'] for r in data]);p=np.array([r['pim_service_us'] for r in data])
        ax.set_xscale('log',base=2);ax.set_yscale('log')
        ax.fill_between(q,g,p,where=p<g,color=PIM,alpha=.10,interpolate=True)
        ax.fill_between(q,g,p,where=g<=p,color=GPU,alpha=.10,interpolate=True)
        marks=[i for i,val in enumerate(q) if val in marker_q]
        for values,color,style,marker in [(g,GPU,'--','s'),(p,PIM,'-','o')]:
            ax.plot(q,values,color=color,lw=1.6,ls=style,marker=marker,markersize=3,markerfacecolor='white',markeredgewidth=.85,markevery=marks)
        lo,hi=panel['q_window'];ax.set_xlim(lo,hi)
        visible=(q>=lo)&(q<=hi)
        ax.set_ylim(min(min(g[visible]),min(p[visible]))*(.67 if panel['c']==0 else .78),max(max(g[visible]),max(p[visible]))*(1.65 if panel['c']==0 else 1.5))
        # Light full-height shading shows which side wins, not fabricated cost magnitude.
        if panel['crossings']:
            first,last=panel['crossings'][0],panel['crossings'][-1]
            ax.axvspan(lo,first['q_lower'],color=PIM,alpha=.055,zorder=0)
            ax.axvspan(last['q_upper'],hi,color=GPU,alpha=.055,zorder=0)
        ticks=[1,8,64,512,2048] if panel['c']==0 else [16,32,64,128,192] if panel['mode']=='plain' else [192,384,768,1536]
        ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:str(int(v))))
        ax.xaxis.set_minor_locator(FixedLocator([]));ax.yaxis.set_major_locator(LogLocator(base=10,numticks=6));ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis='y',color='#dddddd',lw=.55);ax.set_axisbelow(True)
        mode='Without MQ' if panel['mode']=='plain' else 'With MQ'
        ax.set_title(f"({panel['panel']}) {mode} · Cache: {panel['c']:,}",loc='left',pad=23,fontsize=10.5)
        interface={300:'NVLink 3',450:'NVLink 4 (maximum)',32:'PCIe 4 (minimum)'}[panel['b']]
        ax.text(0,1.027,f"{interface} · {panel['b']} GB/s",transform=ax.transAxes,fontsize=8.8,color='#555555',va='bottom')
        cross=panel['crossings']
        if not cross:
            ax.text(.025,.94,'GPU faster over sampled Q',transform=ax.transAxes,fontsize=8,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1.2),va='top')
        elif len(cross)==1:
            t=cross[0];db=math.log(t['lower_pim_us']/t['lower_gpu_us']);da=math.log(t['upper_pim_us']/t['upper_gpu_us'])
            fraction=-db/(da-db);x=t['q_lower']*(t['q_upper']/t['q_lower'])**fraction
            y=t['lower_gpu_us']*(t['upper_gpu_us']/t['lower_gpu_us'])**fraction
            ax.plot(x,y,'o',color='#333333',markersize=3,zorder=5)
            ax.annotate(str(t['integer_switch_q']),xy=(x,y),xytext=(-10,20),textcoords='offset points',ha='center',fontsize=10,
                        arrowprops=dict(arrowstyle='-',lw=.7,color='#444444'),bbox=dict(facecolor='white',edgecolor='none',alpha=.95,pad=1.4))
        else:
            lo,hi=cross[0]['q_lower'],cross[-1]['q_upper'];x=math.sqrt(lo*hi)
            y=np.interp(x,q,g)
            ax.axvspan(lo,hi,color='#777777',alpha=.12,lw=0,zorder=1)
            ax.annotate(f'{lo}–{hi}',xy=(x,y),xytext=(-14,24),textcoords='offset points',ha='center',fontsize=9.5,
                        arrowprops=dict(arrowstyle='-',lw=.7,color='#444444'),bbox=dict(facecolor='white',edgecolor='none',alpha=.95,pad=1.4))
    fig.suptitle('Prefill attention crossover',fontsize=14,y=.976)
    fig.legend(handles=[Line2D([],[],color=GPU,lw=1.6,ls='--',marker='s',markersize=4,markerfacecolor='white',label='GPU'),
        Line2D([],[],color=PIM,lw=1.6,marker='o',markersize=4,markerfacecolor='white',label='PIM'),
        Patch(facecolor=PIM,alpha=.13,label='PIM lower cost'),Patch(facecolor=GPU,alpha=.13,label='GPU lower cost')],
        loc='upper center',bbox_to_anchor=(.5,.939),ncol=4,frameon=False,columnspacing=1.8)
    fig.supxlabel('New query tokens, Q',y=.069,fontsize=11)
    fig.supylabel('Attention service time (µs / layer)',x=.014,fontsize=11)
    fig.text(.5,.023,'A100a · batch 1 · cache initially on PIM · uncovered KV transfer counted · one-way link bandwidth',ha='center',fontsize=8.5,color='#484848')
    stem1='Fugue-asplos-experiment1-prefill-crossover'
    fig.savefig(FIG/(stem1+'.pdf'));fig.savefig(FIG/(stem1+'.png'),dpi=220);plt.close(fig)

    # Experiment 2: focus the plotted Q windows; keep all original points in full tables/appendix.
    all_link=loadcsv(E2/'tables/Fugue-asplos-experiment2-link-Q.csv')
    all_cache=loadcsv(E2/'tables/Fugue-asplos-experiment2-cache-Q.csv')
    link_heat=[r for r in all_link if int(r['q'])>=128]
    cache_heat=[r for r in all_cache if int(r['q'])>=64]
    window_audit=[]
    cmap=LinearSegmentedColormap.from_list('Fugue_GPU_equal_MQ',[GPU,'#fdfdfd',PIM],N=257)
    norm=Normalize(-1,1)
    fig,axes=plt.subplots(1,2,figsize=(11.0,4.0))
    fig.subplots_adjust(left=.075,right=.985,top=.83,bottom=.32,wspace=.29)
    for ax,records,kind in [(axes[0],link_heat,'link'),(axes[1],cache_heat,'cache')]:
        qgrid=sorted({int(r['q']) for r in records});yk='link_GBps_one_way' if kind=='link' else 'cached'
        ygrid=sorted({float(r[yk]) for r in records})
        by={(float(r[yk]),int(r['q'])):r for r in records}
        values=np.array([[f(by[y,q],'signed_faster_side_latency_reduction') for q in qgrid] for y in ygrid])
        xe=edges(qgrid);ye=edges(ygrid) if kind=='link' else np.arange(len(ygrid)+1)-.5
        im=ax.pcolormesh(xe,ye,values,cmap=cmap,norm=norm,shading='flat',edgecolors=(1,1,1,.18),linewidth=.25)
        ax.set_xscale('log',base=2);ax.set_xlim(xe[0],xe[-1]);ax.set_xticks([128,256,512,1024,2048] if kind=='link' else [64,128,256,512,1024,2048])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:str(int(v))));ax.xaxis.set_minor_locator(FixedLocator([]));ax.set_xlabel('New query tokens, Q')
        if kind=='link':
            ax.set_yscale('log',base=2);ax.set_yticks([32,64,150,300,450]);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:str(int(v))))
            ax.yaxis.set_minor_locator(FixedLocator([]));ax.set_ylabel('One-way link bandwidth (GB/s)');ax.set_title('(a) Link bandwidth × Q',loc='left',pad=21)
            ax.text(0,1.025,'Cache: 1,024 tokens',transform=ax.transAxes,fontsize=9,color='#555555',va='bottom')
        else:
            ax.set_yticks(np.arange(len(ygrid)));ax.set_yticklabels([f'{int(v):,}' for v in ygrid]);ax.set_ylabel('Cached tokens');ax.set_title('(b) Cache size × Q',loc='left',pad=21)
            ax.text(0,1.025,'NVLink 3 · 300 GB/s one way',transform=ax.transAxes,fontsize=9,color='#555555',va='bottom')
        ax.tick_params(labelsize=8.5,length=2)
        area=np.outer(np.diff(np.log2(ye)) if kind=='link' else np.diff(ye),np.diff(np.log2(xe)))
        fraction=float(area[values>0].sum()/area.sum())
        assert .35<fraction<.65,(kind,fraction)
        window_audit.append(dict(panel=kind,Q_min=min(qgrid),Q_max=max(qgrid),cells=len(records),PIM_winning_display_area_fraction=fraction,
                                 GPU_winning_display_area_fraction=1-fraction,no_repriced_or_removed_source_data=True))
        csvout((E2/'tables')/f'Fugue-asplos-experiment2-{kind}-Q-focused.csv',records)
    cax=fig.add_axes([.275,.125,.45,.035]);bar=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-1,-.5,0,.5,1])
    bar.ax.set_xticklabels(['100%','50%','0','50%','100%']);bar.ax.tick_params(labelsize=8,length=2);bar.outline.set_linewidth(.5)
    fig.text(.25,.139,'GPU faster',ha='right',va='center',fontsize=9,color=GPU);fig.text(.75,.139,'MQ PIM faster',ha='left',va='center',fontsize=9,color=PIM)
    fig.text(.5,.038,'Latency reduction on the faster device relative to the slower device',ha='center',fontsize=9)
    fig.suptitle('MQ PIM versus GPU: latency advantage',fontsize=13,y=.987)
    stem2='Fugue-asplos-experiment2-mq-sensitivity'
    fig.savefig(E2/'figures'/(stem2+'.pdf'));fig.savefig(E2/'figures'/(stem2+'.png'),dpi=250);plt.close(fig)


def cacheblend_plots():
    E3=read_tables(3)
    requests=load(REPO/'artifact/inputs/cacheblend.json')['requests']
    if FROM_PAPER:
        w=loadcsv(E3/'tables/Fugue-asplos-warmup.csv')
        checks={'warmup_latency_s':sum(f(r,'latency_s') for r in w),'warmup_energy_J':sum(f(r,'energy_J') for r in w)}
    else:checks=load(RUN/'experiment3/Fugue-asplos-audit/Fugue-asplos-checks.json')
    summary=loadcsv(E3/'tables/Fugue-asplos-summary.csv')
    metrics=['TTFT_ms','TBT_ms','E2E_ms','TTFT_energy_mJ','TBT_energy_mJ','E2E_energy_mJ']
    capacities=['remote_peak_KV_MiB','remote_immutable_KV_MiB','remote_peak_private_KV_MiB','GPU_peak_KV_MiB','GPU_native_weights_MiB','GPU_native_temp_MiB','GPU_peak_modeled_total_MiB','total_GPU_remote_peak_KV_MiB']
    aggregate=[]
    for variant in ('F1','F2','F3','F4'):
        group=[r for r in summary if r['variant']==variant]
        aggregate.append(dict(variant=variant,requests=len(group),**{key:sum(f(r,key) for r in group)/len(group) for key in metrics},
            **{key:max(f(r,key) for r in group) for key in capacities},
            total_online_E2E_ms=sum(f(r,'E2E_ms') for r in group),total_online_E2E_energy_mJ=sum(f(r,'E2E_energy_mJ') for r in group),
            warmup_plus_two_requests_ms=checks['warmup_latency_s']*1000+sum(f(r,'E2E_ms') for r in group),
            warmup_plus_two_requests_energy_J=checks['warmup_energy_J']+sum(f(r,'E2E_energy_mJ') for r in group)/1000))
    csvout(E3/'tables/Fugue-asplos-aggregate.csv',aggregate)
    agg={r['variant']:r for r in aggregate}
    comparisons=[]
    for before,after in [('F1','F2'),('F2','F3'),('F3','F4'),('F1','F4')]:
        for metric in metrics+['remote_peak_KV_MiB','remote_peak_private_KV_MiB','total_GPU_remote_peak_KV_MiB']:
            a,b=agg[before][metric],agg[after][metric]
            comparisons.append(dict(baseline=before,variant=after,metric=metric,baseline_value=a,variant_value=b,reduction_percent=100*(1-b/a) if a else 'undefined'))
    csvout(E3/'tables/Fugue-asplos-comparisons.csv',comparisons)
    # Check event accounting independently of the simulation driver.
    events=loadcsv(E3/'tables/Fugue-asplos-events.csv')
    for row in summary:
        group=[r for r in events if r['request_id']==row['request_id'] and r['variant']==row['variant']]
        for before,after in zip(group,group[1:]):same(f(before,'end_s'),f(after,'start_s'))
        same(sum(f(r,'duration_s') for r in group)*1000,f(row,'E2E_ms'));same(sum(f(r,'energy_J') for r in group)*1000,f(row,'E2E_energy_mJ'))
        for layer in range(32):
            names=[r['event'] for r in group if r['layer']==str(layer) and r['phase'].startswith('prefill')]
            assert names.index('qkv')<names.index('attention_service')<names.index('proj')<names.index('ff1')
    # Sensitivity, not a fifth baseline: how much of F2→F3 TTFT is native serial export?
    decisions=loadcsv(E3/'tables/Fugue-asplos-decisions.csv');layers=loadcsv(E3/'tables/Fugue-asplos-layers.csv')
    di={(r['request_id'],r['layer']):r for r in decisions}
    sensitivity=[]
    for request in requests:
        rid=request['request_id'];parts=[r for r in layers if r['request_id']==rid and r['variant']=='F2']
        overlap_time=0.
        for row in parts:
            d=di[rid,row['layer']];write=int(d['total_KV'])*16384/300e9
            overlap_time+=f(row,'common_GPU_s')+f(row,'selector_V_transfer_s')+f(d,'cached_KV_readback_us')/1e6+max(f(d,'GPU_attention_us')/1e6,write)
        baseline=next(r for r in summary if r['request_id']==rid and r['variant']=='F2')
        sharing=next(r for r in summary if r['request_id']==rid and r['variant']=='F3')
        same(overlap_time*1000,f(sharing,'TTFT_ms'))
        sensitivity.append(dict(request_id=rid,F2_native_serial_export_TTFT_ms=f(baseline,'TTFT_ms'),
            F2_export_overlap_sensitivity_TTFT_ms=overlap_time*1000,F3_TTFT_ms=f(sharing,'TTFT_ms'),
            latency_saved_if_F2_export_overlaps_attention_ms=f(baseline,'TTFT_ms')-overlap_time*1000,
            interpretation='Diagnostic only; F2 main result preserves native serial prefill export. Export bytes, energy and capacity unchanged in the sensitivity case.'))
    csvout(E3/'tables/Fugue-asplos-F2-export-overlap-sensitivity.csv',sensitivity)

    colors=['#777777','#C15B33','#236892','#41836A']
    for category,unit,factor in [('latency','ms',1),('energy','J',.001)]:
        fig,axes=plt.subplots(1,3,figsize=(9.4,2.7));fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.80,wspace=.36)
        for ax,metric in zip(axes,('TTFT','TBT','E2E')):
            key=metric+'_ms' if category=='latency' else metric+'_energy_mJ'
            values=[r[key]*factor for r in aggregate]
            ax.bar(range(4),values,width=.63,color=colors,edgecolor='white',linewidth=.5)
            ax.set_xticks(range(4));ax.set_xticklabels(['F1','F2','F3','F4']);ax.set_ylim(0,max(values)*1.22)
            ax.set_title(metric,loc='left',pad=5);ax.set_ylabel(unit);ax.grid(axis='y',color='#dddddd',lw=.5);ax.set_axisbelow(True)
            for x,y in enumerate(values):ax.text(x,y+max(values)*.025,f'{y:.2f}' if y>=1 else f'{y:.3f}',ha='center',va='bottom',fontsize=8)
        fig.suptitle('CacheBlend reuse: '+('request latency' if category=='latency' else 'modeled dynamic energy'),fontsize=12,y=.98)
        fig.text(.5,.015,'Mean of two complete input cases · batch 1 · 10 output tokens · A100a / NVLink 3',ha='center',fontsize=8,color='#555555')
        stem=f'Fugue-asplos-experiment3-{category}'
        fig.savefig(E3/'figures'/(stem+'.pdf'));fig.savefig(E3/'figures'/(stem+'.png'),dpi=250);plt.close(fig)
    storage=loadcsv(E3/'tables/Fugue-asplos-storage.csv')
    peaks=[]
    for variant in ('F1','F2','F3','F4'):
        records=[]
        for r in storage:
            if r['variant']!=variant:continue
            gpu_bytes=f(r,'active_gpu_KV_bytes') if variant=='F1' else f(r,'layer_GPU_KV_buffer_bytes')
            records.append(dict(variant=variant,request_id=r['request_id'],phase=r['phase'],layer=r['layer'],decode_step=r['decode_step'],
                shared_remote_GiB=f(r,'immutable_remote_KV_bytes')/2**30,private_remote_GiB=f(r,'private_remote_KV_bytes')/2**30,GPU_KV_GiB=gpu_bytes/2**30,
                simultaneous_total_GiB=(f(r,'remote_total_KV_bytes')+gpu_bytes)/2**30))
        peak=max(records,key=lambda r:r['simultaneous_total_GiB']);peaks.append(peak)
        same(peak['simultaneous_total_GiB'],agg[variant]['total_GPU_remote_peak_KV_MiB']/1024)
    csvout(E3/'tables/Fugue-asplos-capacity-peaks.csv',peaks)
    fig,ax=plt.subplots(figsize=(5.6,3.2));fig.subplots_adjust(left=.13,right=.98,bottom=.22,top=.80)
    bottom=np.zeros(4)
    for key,color,label in [('shared_remote_GiB','#b9bec4','Shared remote cache'),('private_remote_GiB','#C15B33','Private remote KV'),('GPU_KV_GiB','#236892','GPU KV')]:
        values=np.array([r[key] for r in peaks]);ax.bar(range(4),values,bottom=bottom,color=color,width=.60,edgecolor='white',linewidth=.5,label=label);bottom+=values
    for i,value in enumerate(bottom):ax.text(i,value+.05,f'{value:.2f}',ha='center',va='bottom',fontsize=9)
    ax.set_ylim(0,max(bottom)*1.18);ax.set_xticks(range(4));ax.set_xticklabels(['F1','F2','F3','F4']);ax.set_ylabel('Peak simultaneous KV (GiB)')
    ax.grid(axis='y',color='#dddddd',lw=.5);ax.set_axisbelow(True)
    fig.suptitle('CacheBlend reuse: KV capacity',fontsize=12,y=.98)
    fig.legend(loc='upper center',bbox_to_anchor=(.53,.90),ncol=3,frameon=False,fontsize=7.8,handlelength=1,columnspacing=1.1)
    fig.text(.5,.055,'Shared cache included · GPU weights and non-KV workspace excluded',ha='center',fontsize=8,color='#555555')
    fig.savefig(E3/'figures/Fugue-asplos-experiment3-capacity.pdf');fig.savefig(E3/'figures/Fugue-asplos-experiment3-capacity.png',dpi=250);plt.close(fig)

    # Experiment-local READMEs and writing material.
def epic_plots():
    global rows,choices,signatures,OUT,work,storage
    for i in WANTED & {4,5}:read_tables(i)
    work=load(REPO/'artifact/inputs/epic.json')
    OUT=inputs(4 if 4 in WANTED else 5)
    if FROM_PAPER:
        rows=sum([loadcsv(inputs(i)/'Fugue-asplos-summary.csv') for i in [4,5]],[])
        choices=sum([loadcsv(inputs(i)/'Fugue-asplos-decisions.csv') for i in [4,5]],[])
        storage=sum([loadcsv(inputs(i)/'Fugue-asplos-storage.csv') for i in [4,5]],[])
    else:
        rows=loadcsv(OUT/'Fugue-asplos-summary.csv');choices=loadcsv(OUT/'Fugue-asplos-decisions.csv');storage=loadcsv(OUT/'Fugue-asplos-storage.csv')
    signatures=loadcsv(OUT/'Fugue-asplos-signatures.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'legend.fontsize':8.4,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65,'xtick.labelsize':8,'ytick.labelsize':8,'savefig.facecolor':'white'})
    colors={'F1':'#B06442','F2':'#9A805A','F3':'#27759D','F4':'#3D907C'}
    styles={'F1':(':','s'),'F2':('--','^'),'F3':('-','o'),'F4':('-','D')}
    figures=[]
    def finish(fig,dest):
        for ext in ['pdf','png']:fig.savefig(dest.with_suffix('.'+ext),dpi=220,bbox_inches='tight')
        figures.append(dest.with_suffix('.pdf'));plt.close(fig)
    folder_names={4:'Fugue-asplos-experiment-4-epic-long-prefill',5:'Fugue-asplos-experiment-5-epic-shared-readers'}
    # Additive publication: leave validated experiment 1/2/3 payloads untouched.
    for exp in sorted(WANTED & {4,5}):
        folder=DEST/folder_names[exp]
        for sub in ['figures','tables','provenance','scripts','workload']:(folder/sub).mkdir(parents=True,exist_ok=True)
        cases=[c for c in work['cases'] if c['experiment']==exp];ids={c['case_id'] for c in cases};rs=[r for r in rows if r['case_id'] in ids]
        cs=[c for c in choices if c['case_id'] in ids];lookup={(r['case_id'],r['variant']):r for r in rs}
        x=[c['target_context_tokens']/1000 if exp==4 else c['batch'] for c in cases]
        xlabel='Source context (K tokens)' if exp==4 else 'Concurrent readers'
        for source in (inputs(exp) if FROM_PAPER else OUT).glob('*.csv'):
            data=read(source)
            filtered=[r for r in data if r.get('case_id') in ids] if data and 'case_id' in data[0] else data
            csvout(folder/'tables'/source.name,filtered)
        comparisons=[];scan_rows=[];peaks=[]
        for case in cases:
            cid=case['case_id'];baseline=lookup[cid,'F3'];new=lookup[cid,'F4'];full=lookup[cid,'F2']
            comparisons.append(dict(case_id=cid,**{k+'_F4_reduction_vs_F3_percent':100*(1-num(new,k)/num(baseline,k)) for k in
                ['TTFT_ms','TBT_ms','E2E_ms','TTFT_energy_mJ','TBT_energy_mJ','E2E_energy_mJ']},
                F3_remote_capacity_reduction_vs_F2_percent=100*(1-num(baseline,'remote_peak_KV_MiB')/num(full,'remote_peak_KV_MiB'))))
            dec=next(c for c in cs if c['case_id']==cid)
            single=next(s for s in signatures if int(s['batch'])==case['batch'] and int(s['n'])==case['total_prompt_tokens'] and int(s['resident_queries'])==1)
            scan_rows.append(dict(case_id=cid,Q=case['selected_query_tokens'],N=case['total_prompt_tokens'],batch=case['batch'],
                native_plain_PIM_scan_us=num(single,'scan_us')*case['selected_query_tokens'],MQ_PIM_scan_us=num(dec,'PIM_scan_us'),
                scan_speedup=num(single,'scan_us')*case['selected_query_tokens']/num(dec,'PIM_scan_us'),
                note='Ordinary PIM repeats Q native single-query scans; MQ sums ceil(Q/8) profiled tiles, not request extrapolation.'))
            for variant in ['F1','F2','F3','F4']:
                pts=[p for p in storage if p['case_id']==cid and p['variant']==variant]
                peak=max(pts,key=lambda p:num(p,'combined_live_KV_bytes'))
                peaks.append({k:peak[k] for k in ['case_id','variant','phase','layer','step','remote_immutable_bytes','remote_private_bytes','GPU_live_KV_bytes','combined_live_KV_bytes']})
        csvout(folder/'tables/Fugue-asplos-comparisons.csv',comparisons);csvout(folder/'tables/Fugue-asplos-scan.csv',scan_rows)
        csvout(folder/'tables/Fugue-asplos-capacity-peaks.csv',peaks)
        # Three clean line panels for latency, and a parallel figure for dynamic energy.
        for family,keys,unit in [('latency',['TTFT_ms','TBT_ms','E2E_ms'],'Latency (ms)'),
                                 ('energy',['TTFT_energy_mJ','TBT_energy_mJ','E2E_energy_mJ'],'Batch energy (J)')]:
            fig,axes=plt.subplots(1,3,figsize=(10.1,2.9));fig.subplots_adjust(wspace=.35,top=.76,bottom=.22)
            for ax,key,title in zip(axes,keys,['TTFT','TBT','E2E']):
                for v in ['F1','F2','F3','F4']:
                    values=[num(lookup[c['case_id'],v],key)/(1000 if family=='energy' else 1) for c in cases]
                    style,marker=styles[v];ax.plot(x,values,style,marker=marker,color=colors[v],lw=1.5,ms=4,mfc='white',label=v,zorder=5 if v=='F1' else 3)
                ax.set_title(title,loc='left');ax.set_xticks(x);ax.set_xlabel(xlabel);ax.set_ylim(bottom=0);ax.grid(axis='y',color='#dedede',lw=.6);ax.set_axisbelow(True)
                if title=='TBT':ax.text(.04,.90,'F2 = F3 = F4',transform=ax.transAxes,fontsize=8,color='#555555')
                if family=='latency' and title=='TTFT':ax.text(.04,.90,'F1 = F3',transform=ax.transAxes,fontsize=8,color='#555555')
            axes[0].set_ylabel(unit);fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False)
            finish(fig,folder/'figures'/f'Fugue-asplos-experiment{exp}-{family}')
        # Capacity components are taken from the same peak snapshot, not summed independent peaks.
        fig,axes=plt.subplots(1,3,figsize=(10.1,3.0),sharey=True);fig.subplots_adjust(wspace=.18,top=.73,bottom=.21)
        components=[('remote_immutable_bytes','Shared cache','#C6D5DE'),('remote_private_bytes','Private remote KV','#477C98'),('GPU_live_KV_bytes','GPU KV','#BC8767')]
        for ax,case,val in zip(axes,cases,x):
            bottom=np.zeros(4)
            for key,label,color in components:
                y=np.array([num(next(p for p in peaks if p['case_id']==case['case_id'] and p['variant']==v),key)/2**30 for v in ['F1','F2','F3','F4']])
                ax.bar(np.arange(4),y,bottom=bottom,color=color,width=.62,label=label);bottom+=y
            ax.set_xticks(range(4),['F1','F2','F3','F4']);ax.set_title(f'{val:g}K context' if exp==4 else f'{val} reader'+('s' if val>1 else ''),loc='left');ax.grid(axis='y',color='#e4e4e4',lw=.6);ax.set_axisbelow(True)
        axes[0].set_ylabel('Peak total KV (GiB)');fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
        finish(fig,folder/'figures'/f'Fugue-asplos-experiment{exp}-capacity')
        # Attention kernel versus communication must be visibly separable.
        fig,ax=plt.subplots(figsize=(7.3,3.1));fig.subplots_adjust(top=.73,bottom=.23)
        labels=[];vals=[]
        for c in cases:
            for v in ['F3','F4']:
                r=lookup[c['case_id'],v];labels.append(v);vals.append([num(r,'prefill_common_GPU_ms'),num(r,'prefill_attention_kernel_ms'),num(r,'prefill_attention_service_ms')-num(r,'prefill_attention_kernel_ms')])
        pos=np.array([0,1,3,4,6,7]);bottom=np.zeros(6)
        for j,(label,col) in enumerate([('Other GPU work','#C6D5DE'),('Attention kernel','#397E9F'),('Exposed transfer','#B57751')]):
            values=np.array(vals)[:,j];assert min(values)>-1e-8;ax.bar(pos,values,bottom=bottom,width=.65,color=col,label=label);bottom+=values
        ax.set_xticks(pos,labels);ax.set_ylabel('TTFT (ms)');ax.set_ylim(bottom=0);ax.grid(axis='y',color='#e4e4e4',lw=.6);ax.set_axisbelow(True)
        for i,value in enumerate(x):ax.text((3*i+.5),-.18,f'{value:g}K context' if exp==4 else f'B = {value}',ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
        finish(fig,folder/'figures'/f'Fugue-asplos-experiment{exp}-prefill-breakdown')

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    if WANTED & {1,2}:sweep_plots()
    if 3 in WANTED:cacheblend_plots()
    if WANTED & {4,5}:epic_plots()
    files=list(DEST.rglob('*.pdf'))
    save(DEST/'Fugue-asplos-plot-checks.json',dict(plots=len(files),source='checked-in final tables' if FROM_PAPER else 'fresh simulation',
        figures=[str(p.relative_to(DEST)) for p in files],script_sha256=sha(Path(__file__))))
    print('PLOTS',len(files),DEST,flush=True)
if __name__=='__main__':main()
