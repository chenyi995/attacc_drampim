#!/usr/bin/env python3
"""Redraw the saved measurements in this package; no experiment execution."""
import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_rows():
    with (HERE / 'data.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['valid'] = json.loads(row['valid'].lower())
        if 'displayed' in row:
            row['displayed'] = json.loads(row['displayed'].lower())
        for field in ('TTFT_ms', 'TBT_ms', 'decode_total_ms', 'E2E_ms',
                      'F0_E2E_ms', 'normalized_latency', 'reference_ms',
                      'method_ms', 'speedup'):
            if field in row:
                row[field] = float(row[field])
        for field in ('batch', 'output_tokens', 'decode_steps'):
            if field in row:
                row[field] = int(row[field])
    return rows


def model_name(name, spec=None):
    label=name.replace('LLAMA3.1-', 'Llama3.1-').replace('LLAMA-', 'Llama-')
    return label.replace('-', '\n') if spec and spec.get('model_label_format')=='family-size-two-lines' else label


def check_bounds(fig):
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= bounds.x0 - 1 and box.x1 <= bounds.x1 + 1, (text.get_text(), box)
        assert box.y0 >= bounds.y0 - 1 and box.y1 <= bounds.y1 + 1, (text.get_text(), box)


def interrupted_bar(ax, bar, value, cap, font):
    """Mark only the displayed cap; the stored ratio remains unchanged."""
    if cap is None or value <= cap:
        return
    x=bar.get_x();w=bar.get_width();y=cap*.935;d=cap*.016
    xs=[x,x+w*.25,x+w*.5,x+w*.75,x+w]
    ys=[y-d,y+d,y-d,y+d,y-d]
    ax.plot(xs,ys,color='white',linewidth=2.2,zorder=6,clip_on=False)
    ax.plot(xs,ys,color='#333333',linewidth=.65,zorder=7,clip_on=False)
    ax.text(x+w/2,cap*1.025,f'{value:.2f}×',ha='center',va='bottom',
            fontsize=font,zorder=8,clip_on=False)


def axis_common(ax,spec,models):
    from matplotlib.ticker import FuncFormatter
    ax.set_xlim(*spec['x_limits'])
    ax.set_xticks(range(len(models)),[model_name(m,spec) for m in models],
                  rotation=spec.get('model_label_rotation_degrees',90),ha='center')
    ax.set_xlabel(spec['xlabel'],labelpad=2)
    ax.tick_params(axis='x',length=0,pad=2)
    ax.tick_params(axis='y',pad=1.5,length=2)
    ax.set_axisbelow(True)
    # Native Excel uses five equal category slots. Terminate every horizontal
    # rule at the first and last bar edges, without adding empty categories.
    left,right=spec['horizontal_line_bounds']
    ax.grid(False)
    grid=ax.hlines([v for v in ax.get_yticks() if v != 0],left,right,
                   color='#DDDDDD',linewidth=.45,zorder=0)
    grid.set_capstyle('butt')
    grid.set_gid('horizontal-grid')
    baseline,=ax.plot([left,right],[1,1],color='#333333',linewidth=.55,
                     solid_capstyle='butt',zorder=1)
    baseline.set_gid('unit-baseline')
    ax.spines['bottom'].set_bounds(left,right)
    ax.spines['bottom'].set_capstyle('butt')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))

def draw_e2e(spec, rows):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    models=spec['model_order'];workloads=spec['workload_order']
    groups=spec['model_groups']
    assert [model for group in groups for model in group] == models
    variants=spec['display_variant_order'];source_variants=spec['variant_order']
    lookup={(r['model'],r['workload_id'],r['variant']):r for r in rows}
    assert len(lookup)==len(models)*len(workloads)*len(source_variants)==len(rows)
    for row in rows:
        assert row['decode_steps']==row['output_tokens']-1
        assert math.isclose(row['decode_total_ms'],row['TBT_ms']*row['decode_steps'],rel_tol=1e-12)
        assert math.isclose(row['E2E_ms'],row['TTFT_ms']+row['decode_total_ms'],rel_tol=1e-12)
    fig=plt.figure(figsize=(spec['width_inches'],spec['height_inches']))
    grid=spec['panel_grid'];columns=grid['columns'];m=spec['margins']
    gap=grid['column_gap']
    panel_width=(m['right']-m['left']-(columns-1)*gap)/columns
    width=spec['bar_width_group_units'];plotted=[]
    assert columns == len(workloads)
    for row_index, group in enumerate(groups):
        x=np.arange(len(group))
        for column, workload in enumerate(workloads):
            panel=row_index*columns+column
            bottom=grid['row_bottom'][row_index];top=grid['row_top'][row_index]
            left=m['left']+column*(panel_width+gap)
            ax=fig.add_axes([left,bottom,panel_width,top-bottom])
            for series,variant in enumerate(variants):
                style_index=source_variants.index(variant)
                selected=[lookup[model,workload,variant] for model in group]
                assert all(r['valid'] and r['displayed'] for r in selected)
                values=[r['F0_E2E_ms']/r['E2E_ms'] for r in selected]
                bars=ax.bar(x+(series-(len(variants)-1)/2)*width,values,width=width,
                    color=spec['bar_colors'][style_index],hatch=spec['bar_hatches'][style_index],edgecolor='#333333',linewidth=.2)
                for bar,row,value in zip(bars,selected,values):
                    baseline=lookup[row['model'],workload,spec['baseline_variant']]
                    assert math.isclose(row['F0_E2E_ms'],baseline['E2E_ms'],rel_tol=1e-12)
                    assert math.isclose(bar.get_height(),value,rel_tol=1e-12)
                    plotted.append(dict(model=row['model'],workload_id=workload,variant=variant,value=value,displayed_value=value,valid=True))
            ax.set_ylim(0,spec['ylim'][column]);ax.set_yticks(spec['yticks'][column]);axis_common(ax,spec,group)
            # Equal panel gaps give each vertical scale room without empty model slots.
            ax.set_xlim(*spec['x_limits'])
            for label in ax.get_yticklabels():
                label.set_bbox(dict(facecolor='white',edgecolor='none',pad=.05))
            fig.text(left+panel_width/2,grid['row_label_y'][row_index],
                spec['panel_labels'][panel]+' '+spec['panel_names'][column],
                ha='center',va='bottom',fontsize=spec['fonts_pt']['note'])
    fig.text(spec['shared_y_label_x'],spec['shared_y_label_y'],spec['ylabel'],
        rotation=90,ha='center',va='center',fontsize=spec['fonts_pt']['label'])
    handles=[Patch(facecolor=spec['bar_colors'][source_variants.index(v)],
        hatch=spec['bar_hatches'][source_variants.index(v)],edgecolor='#333333',linewidth=.25,
        label=spec['variant_names'][source_variants.index(v)]) for v in variants]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=spec['legend_anchor'],ncol=spec['legend_columns'],frameon=False,
        fontsize=spec['fonts_pt']['legend'],borderaxespad=0,columnspacing=1.1,handletextpad=.4,handlelength=1.2)
    assert len(plotted)==len(models)*len(workloads)*len(variants)
    return fig,plotted


def draw_phases(spec, rows):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    models=spec['model_order'];workloads=spec['workload_order']
    lookup={(r['model'],r['workload_id'],r['panel']):r for r in rows}
    assert len(lookup)==len(models)*len(workloads)*2==len(rows)
    fig=plt.figure(figsize=(spec['width_inches'],spec['height_inches']))
    grid=spec['panel_grid'];m=spec['margins'];axes=[]
    for top,bottom in zip(grid['row_top'],grid['row_bottom']):
        axes.append(fig.add_axes([m['left'],bottom,m['right']-m['left'],top-bottom]))
    width=spec['bar_width_group_units'];x=np.arange(len(models));plotted=[]
    for pn,(panel,ax) in enumerate(zip(spec['panels'],axes)):
        cap=panel['display_cap']
        for series,workload in enumerate(workloads):
            selected=[lookup[model,workload,panel['key']] for model in models]
            for row in selected:
                assert row['valid'] and row['reference_variant']==panel['reference_variant'] and row['method_variant']=='F4'
                assert math.isclose(row['speedup'],row['reference_ms']/row['method_ms'],rel_tol=1e-12)
            values=[row['speedup'] for row in selected];display=[min(v,cap) if cap else v for v in values]
            bars=ax.bar(x+(series-(len(workloads)-1)/2)*width,display,width=width,
                color=spec['bar_colors'][series],hatch=spec['bar_hatches'][series],edgecolor='#333333',linewidth=.25)
            for bar,row,value in zip(bars,selected,values):
                interrupted_bar(ax,bar,value,cap,spec['fonts_pt']['note'])
                plotted.append(dict(model=row['model'],workload_id=workload,panel=panel['key'],value=value,
                    displayed_value=bar.get_height(),display_cap=cap,interrupted=bool(cap and value>cap),valid=True))
        ax.set_ylim(0,panel['ylim']);ax.set_yticks(panel['yticks']);axis_common(ax,spec,models)
        ax.set_ylabel(panel['ylabel'],labelpad=3)
        pos=ax.get_position();fig.text(pos.x0+pos.width/2,grid['row_label_y'][pn],spec['panel_labels'][pn],
            ha='center',va='bottom',fontsize=spec['fonts_pt']['note'])
    handles=[Patch(facecolor=spec['bar_colors'][i],hatch=spec['bar_hatches'][i],edgecolor='#333333',linewidth=.25,label=spec.get('legend_names',spec['workload_display'])[i]) for i in range(len(workloads))]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=spec['legend_anchor'],ncol=spec['legend_columns'],frameon=False,
        borderaxespad=0,columnspacing=.9,handletextpad=.4,handlelength=1.2,labelspacing=.3)
    return fig,plotted


if __name__ == '__main__':
    from figure_style import configure, save
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'figure.pdf')
    parser.add_argument('--check-json', type=Path, help='Optional local record of exact plotted CSV values')
    args = parser.parse_args()
    specification = json.loads((HERE/'template.json').read_text())
    configure(specification['fonts_pt'])
    rows = load_rows()
    if specification['label'] == 'request-e2e':
        figure, plotted = draw_e2e(specification,rows)
    else:
        figure, plotted = draw_phases(specification,rows)
    check_bounds(figure)
    saved = save(figure,specification['label'],args.output.resolve())
    if args.check_json:
        args.check_json.write_text(json.dumps(dict(figure=saved,plotted=plotted),indent=2)+'\n')
    print(f"Wrote {args.output} from {len(rows)} saved CSV records")
