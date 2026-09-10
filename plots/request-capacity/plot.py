#!/usr/bin/env python3
"""Redraw this package using only its local CSV, template and style."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_bars():
    with (HERE / 'data.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ('displayed', 'eligible'):
            row[key] = json.loads(row[key])
        row['plotted_value'] = float(row['plotted_value']) if row['plotted_value'] else None
        row['slot_marker'] = row['slot_marker'] or None
    return rows


def draw(spec,bars,path,preview=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator,FuncFormatter
    from matplotlib.text import Text
    from matplotlib.patches import Patch
    from figure_style import configure,COLOR
    configure(spec['fonts_pt'])
    fig,ax=plt.subplots(figsize=(spec['width_inches'],spec['height_inches']))
    fig.subplots_adjust(**spec['margins'])
    models=spec['model_order'];groups=spec['group_order'];positions=np.arange(len(models))
    compact=spec.get('display_layout')=='packed-benefit-cases'
    if compact:
        models=spec['display_model_order'];positions=[];right=0.
        for model in models:
            entries=[b for b in bars if b['model']==model and b['displayed']]
            entry_positions=np.arange(right,right+len(entries))
            positions.append(float(np.mean(entry_positions)) if entries else right)
            for x,bar in zip(entry_positions,entries):
                i=groups.index(bar['group'])
                ax.bar(x,bar['plotted_value'],width=spec['bar_width_group_units'],color=spec['bar_colors'][i],hatch=spec['bar_hatches'][i],edgecolor=COLOR['ink'],linewidth=.25)
            right+=max(len(entries),1)+spec['compact_model_gap']
        values=[b['plotted_value'] for b in bars if b['displayed']]
        ax.set_ylim(0,max(1.1,max(values,default=1)*1.13))
        ax.set_xlim(-.7,max(.7,right-spec['compact_model_gap']-.3))
        if preview:ax.text(.5,.45,'PIM-benefit cases',transform=ax.transAxes,ha='center',va='center',fontsize=spec['fonts_pt']['note'])
    elif not preview:
        lookup={(b['model'],b['group']):b for b in bars}
        for i,group in enumerate(groups):
            offset=(i-(len(groups)-1)/2)*spec['bar_width_group_units']
            valid=[(j,lookup[model,group]['plotted_value']) for j,model in enumerate(models) if lookup[model,group]['displayed']]
            ax.bar([j+offset for j,value in valid],[value for j,value in valid],width=spec['bar_width_group_units'],color=spec['bar_colors'][i],hatch=spec['bar_hatches'][i],edgecolor=COLOR['ink'],linewidth=.25,label=spec['series'][i])
            for j,model in enumerate(models):
                bar=lookup[model,group]
                if bar['slot_marker']=='GPU':ax.plot(j+offset,spec['gpu_marker_y_axes'],marker=spec['gpu_marker'],markersize=2.8,markeredgewidth=.6,color=COLOR['ink'],transform=ax.get_xaxis_transform(),clip_on=False)
                elif not bar['eligible']:ax.text(j+offset,.02,'N/A',rotation=90,ha='center',va='bottom',fontsize=spec['fonts_pt']['note'],transform=ax.get_xaxis_transform())
        values=[b['plotted_value'] for b in bars if b['displayed']]
        ax.set_ylim(0,max(1.1,max(values,default=1)*1.13))
        if any(b['slot_marker']=='GPU' for b in bars):ax.text(.995,1.04,'× GPU choice',transform=ax.transAxes,ha='right',va='bottom',fontsize=spec['fonts_pt']['note'])
    else:
        ax.set_ylim(0,1.2)
        ax.text(.5,.45,'Individual source cases',transform=ax.transAxes,ha='center',va='center',fontsize=spec['fonts_pt']['note'])
    if not compact:ax.set_xlim(*spec['x_limits'])
    rotation=spec['model_label_rotation_degrees']
    labels=[model.replace('LLAMA-','Llama-') for model in models]
    if spec.get('model_label_format')=='family-size-two-lines':
        labels=[label.replace('-', '\n') for label in labels]
    ax.set_xticks(positions,labels,rotation=rotation,ha='center',va='top')
    ax.tick_params(axis='x',length=0,pad=2)
    ax.tick_params(axis='y',length=2,pad=2)
    ax.set_ylabel(spec['metric'],labelpad=7 if compact else 2)
    ax.set_xlabel(spec['xlabel'],labelpad=2)
    ax.set_ylim(0,spec['ylim'])
    ax.set_axisbelow(True);ax.grid(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ticks=[float(v) for v in ax.get_yticks() if 0<=v<=ax.get_ylim()[1]]
    if compact:ticks=[0.,1.,2.]
    ax.set_yticks(spec['yticks'])
    left,right=spec['horizontal_line_bounds']
    grid=ax.hlines([v for v in spec['yticks'] if v != 0],left,right,
                   color='#DDDDDD',linewidth=.5,zorder=0)
    grid.set_capstyle('butt')
    grid.set_gid('horizontal-grid')
    baseline,=ax.plot([left,right],[1,1],color=COLOR['ink'],linewidth=.65,
                     solid_capstyle='butt',zorder=1)
    baseline.set_gid('unit-baseline')
    ax.spines['bottom'].set_bounds(left,right)
    ax.spines['bottom'].set_capstyle('butt')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}×' if compact and v else f'{v:g}'))
    handles=[Patch(facecolor=spec['bar_colors'][i],hatch=spec['bar_hatches'][i],edgecolor=COLOR['ink'],linewidth=.25,label=spec['series'][i]) for i in range(len(groups)) if groups[i] in spec.get('legend_workload_order',groups)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=spec['legend_anchor'],ncol=spec['legend_columns'],frameon=False,borderaxespad=0,columnspacing=.6,handletextpad=.3,handlelength=.8,labelspacing=.35)
    if spec['title']:
        fig.text(*spec['title_anchor'],spec['title'],ha='center',va='top',fontsize=spec['fonts_pt']['title'])
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    for t in texts:
        box=t.get_window_extent(renderer)
        assert t.get_fontsize() in spec['fonts_pt'].values(),(t.get_text(),t.get_fontsize())
        assert box.x0>=bounds.x0-1 and box.x1<=bounds.x1+1 and box.y0>=bounds.y0-1 and box.y1<=bounds.y1+1,(t.get_text(),box,bounds)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,metadata={'CreationDate':None,'ModDate':None,'Creator':'KVChime representative workload figure'})
    fig.savefig(path.with_suffix('.png'),dpi=180);plt.close(fig)
    return dict(width_inches=spec['width_inches'],height_inches=spec['height_inches'],font_sizes_pt=sorted({t.get_fontsize() for t in texts}),text=[dict(text=t.get_text(),font_pt=t.get_fontsize()) for t in texts])

if __name__ == '__main__':
    specification = json.loads((HERE / 'template.json').read_text())
    draw(specification, load_bars(), HERE / 'figure.pdf')
    print('Wrote figure.pdf and figure.png from local data.csv')
