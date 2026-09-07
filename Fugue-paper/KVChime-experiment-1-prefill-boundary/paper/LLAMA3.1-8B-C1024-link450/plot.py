#!/usr/bin/env python3
"""Standalone figure script: python3 plot.py [--output-dir DIR].

Inputs are raw-data.csv and plot-config.json beside this file.
Only Python, numpy and matplotlib are required; no simulator imports.
"""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter

COLORS=['#777777','#C9754E','#B4A373','#26749A','#36886B']
SERIES_COLORS=dict(zip(['F0','F1','F2','F3','F4'],COLORS))
SERIES_COLORS.update(single_query='#26749A',MQ='#36886B',GPU='#C9754E',PIM='#36886B',Algorithm='#7966A0',Oracle='#777777')
def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def draw(root,dest):
    cfg=json.loads((root/'plot-config.json').read_text());rows=read(root/'raw-data.csv')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    if cfg['kind']=='line':
        fig,ax=plt.subplots(figsize=(5.2,2.8),layout='constrained')
        rows.sort(key=lambda r:float(r[cfg['x_key']]))
        for series in cfg['series']:
            ax.plot([float(r[cfg['x_key']]) for r in rows],[float(r[series['key']]) for r in rows],
                    series.get('style','-o'),color=series['color'],mfc='white',ms=3.5,lw=1.4,label=series['name'])
        ax.set_xlabel(cfg['xlabel']);ax.set_ylabel(cfg['ylabel']);ax.set_title(cfg['title'],loc='left',fontsize=10)
        if cfg.get('logx'):
            ax.set_xscale('log',base=2)
            ticks=[q for q in [1,8,64,512,2048] if min(float(r[cfg['x_key']]) for r in rows)<=q<=max(float(r[cfg['x_key']]) for r in rows)]
            ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:str(int(x))))
        if cfg.get('logy'):ax.set_yscale('log')
        for i,c in enumerate(cfg.get('crossings',[])):
            x=(c['q_low']*c['q_high'])**.5
            xx=np.array([float(r[cfg['x_key']]) for r in rows])
            yy=np.array([float(r['gpu_service_us']) for r in rows])
            y=np.exp(np.interp(np.log(x),np.log(xx),np.log(yy)))
            ax.axvspan(c['q_low'],c['q_high'],color=c['color'],alpha=.055,zorder=0)
            ax.annotate(f"{c['q_low']}–{c['q_high']}",xy=(x,y),xytext=(0,20+10*(i%2)),
                textcoords='offset points',ha='center',fontsize=7,color=c['color'],
                bbox=dict(facecolor='white',edgecolor='none',pad=1,alpha=.9))
        ax.legend(frameon=False,fontsize=8,ncol=len(cfg['series']));ax.grid(axis='y',alpha=.22)
    else:
        # One response metric. Cases vary along x and models form outer groups.
        if cfg.get('wide_series'):
            rows=[dict(r,series=s['name'],value=r[s['key']]) for r in rows for s in cfg['wide_series']]
            series_key,value_key='series','value'
        else:series_key,value_key=cfg['series_key'],cfg['value_key']
        casekeys=cfg['case_keys'];modelkey=cfg.get('model_key','model')
        model_order=cfg.get('models') or list(dict.fromkeys(r.get(modelkey,'') for r in rows))
        groups=[]
        for model in model_order:
            groups.extend((model,k) for k in dict.fromkeys(tuple(r[c] for c in casekeys) for r in rows if r.get(modelkey,'')==model))
        series=cfg['series_order'];width=.8/len(series)
        fig,ax=plt.subplots(figsize=(max(6.2,len(groups)*.60),3.1))
        lookup={(r.get(modelkey,''),tuple(r[c] for c in casekeys),r[series_key]):float(r[value_key]) for r in rows}
        for j,v in enumerate(series):
            yy=[]
            for model,k in groups:
                val=lookup[model,k,v]
                if cfg.get('normalize_to'):
                    baseline=lookup[model,k,cfg['normalize_to']]
                    val=baseline/val if cfg.get('inverse_normalization') else val/baseline
                yy.append(val)
            ax.bar(np.arange(len(groups))+(j-(len(series)-1)/2)*width,yy,width,color=cfg.get('series_colors',{}).get(v,SERIES_COLORS.get(v,COLORS[j%len(COLORS)])),edgecolor='white',linewidth=.2,label=cfg.get('series_labels',{}).get(v,v))
        labels=[]
        for model,k in groups:
            words=[]
            for key,value in zip(casekeys,k):
                val=value.replace('KVChime-','').replace('EPIC-','').replace('shared-readers-','Readers-')
                words.append(cfg.get('case_prefix',{}).get(key,'')+val)
            labels.append('\n'.join(words))
        ax.set_xticks(np.arange(len(groups)),labels,rotation=cfg.get('rotation',60),ha='right',fontsize=7)
        for model in model_order:
            indices=[i for i,g in enumerate(groups) if g[0]==model]
            if not indices:continue
            ax.text((indices[0]+indices[-1])/2,-.59,model,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=9)
            if indices[0]:ax.axvline(indices[0]-.5,color='#AAAAAA',lw=.6)
        if cfg.get('normalize_to'):ax.axhline(1,color='#777777',lw=.6,ls=':')
        if cfg.get('logy'):ax.set_yscale('log')
        else:ax.set_ylim(bottom=0)
        ax.set_ylabel(cfg['ylabel']);ax.set_title(cfg['title'],loc='left',fontsize=10)
        ax.legend(ncol=len(series),frameon=False,loc='upper center',bbox_to_anchor=(.5,1.23),fontsize=8)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.subplots_adjust(bottom=.42,top=.80,left=.06,right=.995)
    assert len(fig.axes)==1,'Every figure must have one response axis'
    dest.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(dest/('figure.'+ext),dpi=200,bbox_inches='tight')
    (dest/'render-check.json').write_text(json.dumps(dict(axes=len(fig.axes),response_metric=cfg['ylabel'],input_rows=len(rows)),indent=2)+'\n')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path);args=p.parse_args()
    root=Path(__file__).resolve().parent;draw(root,args.output_dir or root)
if __name__=='__main__':main()
