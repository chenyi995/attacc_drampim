#!/usr/bin/env python3
"""Two native attention paths, redrawn from local CSVs without a simulator."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, NullLocator
from figure_style import configure

HERE = Path(__file__).resolve().parent


def load(name):
    with (HERE/name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def draw(output):
    spec = json.loads((HERE/'template.json').read_text())
    configure(spec['fonts_pt'])
    rows, panels, common = load('panel_data.csv'), load('panels.csv'), load('gpu_data.csv')
    crossings = load('crossings.csv')
    gpu_q = np.array([int(r['q']) for r in common])
    gpu_y = np.array([float(r['gpu_throughput_tokens_per_second']) for r in common])
    assert all(y == int(r['q'])/float(r['gpu_service_us'])*1e6 for r,y in zip(common,gpu_y))
    grid = spec.get('panel_grid')
    if grid and grid['rows'] == 2:
        fig = plt.figure(figsize=(spec['width_inches'], spec['height_inches']))
        m = spec['margins']
        axes = [fig.add_axes([m['left'], bottom, m['right']-m['left'], top-bottom])
                for top, bottom in zip(grid['row_top'], grid['row_bottom'])]
    else:
        fig, axes = plt.subplots(1,2,figsize=(spec['width_inches'],spec['height_inches']))
        fig.subplots_adjust(**spec['margins'])
    meta = []
    for panel_index, (ax, panel) in enumerate(zip(axes, panels)):
        data = sorted((r for r in rows if r['panel']==panel['panel']),key=lambda r:int(r['q']))
        q = np.array([int(r['q']) for r in data])
        y = np.array([float(r['pim_throughput_tokens_per_second']) for r in data])
        assert all(v == int(r['q'])/float(r['pim_service_us'])*1e6 for r,v in zip(data,y))
        gaps = [float(r['gpu_service_us'])-float(r['pim_service_us']) for r in data]
        found = [[int(q[i]),int(q[i+1])] for i in range(len(q)-1) if gaps[i]*gaps[i+1]<0]
        expected = [[int(c['q_low']),int(c['q_high'])] for c in crossings if c['panel']==panel['panel']]
        assert found==expected
        low,high=found[0][0],found[-1][1]
        assert all(g > 0 for x,g in zip(q,gaps) if spec['x_limits'][0] <= x <= low)
        assert all(g < 0 for x,g in zip(q,gaps) if high <= x <= spec['x_limits'][1])
        region=spec['favorable_regions']
        ax.axvspan(spec['x_limits'][0],low,color=region['pim_color'],alpha=region['alpha'],lw=0,zorder=0)
        ax.axvspan(high,spec['x_limits'][1],color=region['gpu_color'],alpha=region['alpha'],lw=0,zorder=0)
        ax.axvspan(low,high,color='#879DA9',alpha=.18,lw=0,zorder=0)
        ax.plot(gpu_q,gpu_y,color=spec['colors']['base'],ls='--',lw=1.25,label='GPU')
        markers=[i for i,r in enumerate(data) if json.loads(r['original_marker'])]
        ax.plot(q,y,color=spec['colors']['improved'],lw=1.25,marker='o',markersize=2.5,
                markevery=markers,markerfacecolor='white',markeredgewidth=.65,label=panel['pim_label'])
        ax.set_xscale('log',base=2)
        ax.set_xlim(spec['x_limits']); ax.set_ylim(spec['y_limits_tokens_per_second'])
        ax.set_xticks(spec['x_ticks']);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks(spec['y_ticks_tokens_per_second'])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,p:f'{y/1e6:g}'))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.text(0,1.018,'×10⁶',transform=ax.transAxes,ha='left',va='bottom',fontsize=spec['fonts_pt']['note'])
        ax.set_xlabel(spec['x_label'],labelpad=2)
        ax.set_ylabel('Throughput (tokens/s)',labelpad=2)
        ax.tick_params(axis='both',length=2,pad=2)
        ax.grid(axis='y',color='#D6D6D6',ls=':',lw=.5);ax.set_axisbelow(True)
        ax.legend(loc='upper right',bbox_to_anchor=spec['legend_anchor'],ncol=spec['legend_columns'],frameon=False,fontsize=spec['fonts_pt']['legend'],
                  handlelength=1.7,handletextpad=.45,borderaxespad=.3)
        ax.text(math.sqrt(spec['x_limits'][0]*low),.88,region['pim_label'],
                transform=ax.get_xaxis_transform(),ha='center',va='center',
                color=region['pim_color'],fontsize=spec['fonts_pt']['note'])
        ax.text(math.sqrt(high*spec['x_limits'][1]),.38 if panel['panel']=='A' else .18,region['gpu_label'],
                transform=ax.get_xaxis_transform(),ha='center',va='center',
                color=region['gpu_color'],fontsize=spec['fonts_pt']['note'])
        ax.text(.035,.08,f'{low}–{high}',transform=ax.transAxes,ha='left',va='bottom',
                fontsize=spec['fonts_pt']['note'],bbox=dict(facecolor='white',edgecolor='none',pad=.6,alpha=.9))
        fig.text((ax.get_position().x0+ax.get_position().x1)/2,grid['row_label_y'][panel_index] if grid else spec['panel_label_y'],f"({panel['panel'].lower()})",ha='center',va='bottom',fontsize=spec['fonts_pt']['note'])
        meta.append(dict(panel=panel['panel'],source_panel=panel['source_panel'],mode=panel['mode'],
            cached_tokens=int(panel['cached_tokens']),link_GBps=float(panel['link_GBps']),
            q_window=spec['x_limits'],query_tokens=q.tolist(),pim_service_us=[float(r['pim_service_us']) for r in data],
            gpu_service_us=[float(r['gpu_service_us']) for r in data],
            gpu_throughput_tokens_per_us=[int(r['q'])/float(r['gpu_service_us']) for r in data],
            pim_throughput_tokens_per_us=[int(r['q'])/float(r['pim_service_us']) for r in data],
            plotted_GPU_query_tokens=gpu_q.tolist(),plotted_GPU_tokens_per_second=gpu_y.tolist(),
            plotted_PIM_tokens_per_second=y.tolist(),sampled_rows=len(data),crossing_brackets=found,
            favorable_regions=dict(pim=[spec['x_limits'][0],low],neutral=[low,high],gpu=[high,spec['x_limits'][1]]),
            x_limits=list(ax.get_xlim()),y_limits=list(ax.get_ylim())))
    assert np.array_equal(axes[0].lines[0].get_xdata(),axes[1].lines[0].get_xdata())
    assert np.array_equal(axes[0].lines[0].get_ydata(),axes[1].lines[0].get_ydata())
    assert axes[0].get_xlim()==axes[1].get_xlim() and axes[0].get_ylim()==axes[1].get_ylim()
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    text=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():continue
        box=t.get_window_extent(renderer)
        assert t.get_fontsize() in spec['fonts_pt'].values(),(t.get_text(),t.get_fontsize())
        assert box.x0>=bounds.x0-1 and box.x1<=bounds.x1+1 and box.y0>=bounds.y0-1 and box.y1<=bounds.y1+1,(t.get_text(),box,bounds)
        assert 'cache' not in t.get_text().lower() and 'link' not in t.get_text().lower()
        text.append(dict(text=t.get_text(),font_pt=t.get_fontsize()))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,metadata={'CreationDate':None,'ModDate':None,'Creator':'KVChime paired native attention-service throughput'})
    fig.savefig(output.with_suffix('.png'),dpi=180);plt.close(fig)
    return dict(width_inches=spec['width_inches'],height_inches=spec['height_inches'],font_sizes_pt=sorted({t['font_pt'] for t in text}),
        text=text,panels=meta,identical_gpu_curve=True,shared_axis_limits=True,
        gpu_unique_samples=len(common),plotted_rows=len(rows),
        throughput_unit='attention query tokens/s',unit_multiplier_from_tokens_per_us=1000000,
        no_new_simulation=True,no_interpolated_source_samples=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'figure.pdf')
    parser.add_argument('--check-json',type=Path)
    args=parser.parse_args();result=draw(args.output)
    if args.check_json:args.check_json.write_text(json.dumps(result,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
