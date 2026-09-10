#!/usr/bin/env python3
"""Redraw using only this directory's CSVs, template and local style."""
from pathlib import Path
import csv
import json
import math
from figure_style import FONT, COLOR, canvas, save
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, NullLocator, MaxNLocator

BASE = Path(__file__).resolve().parent
SPEC = json.loads((BASE / 'template.json').read_text())
FONT = SPEC['fonts_pt']
LABEL = SPEC['label']

def table(name):
    path = BASE / name
    if not path.is_file():
        raise SystemExit('Missing package CSV: ' + str(path) + '. Copy the complete package, including CSV files.')
    def value(text):
        if text == '': return None
        if text in ('true', 'false'): return text == 'true'
        try: return int(text)
        except ValueError:
            try: return float(text)
            except ValueError: return text
    with path.open(newline='') as stream:
        return [{key:value(text) for key,text in row.items()} for row in csv.DictReader(stream)]


def main():
    curve=table('curve.csv');balance=table('balance.csv')[0]
    intervals=table('intervals.csv');params=table('parameters.csv')[0]
    assert len(intervals)==len(curve)
    count,tck,floor=params['query_capacity'],params['tck_ns'],params['dram_floor_tck']
    power_min=balance['power_interval_tck']
    xs=[r['frequency_ghz'] for r in intervals]
    compute=[r['compute_interval_tck'] for r in intervals]
    ys=[max(c,power_min,floor) for c in compute]
    for old,row,c,e in zip(curve,intervals,compute,ys):
        assert old['frequency_ghz']==row['frequency_ghz']
        assert e==row['effective_interval_tck']
        assert math.isclose(count/(c*tck),old['compute_query_MAC_per_ns'],rel_tol=1e-12)
        assert math.isclose(count/(e*tck),old['effective_query_MAC_per_ns'],rel_tol=1e-12)
    fig,ax=canvas('frequency');fig.subplots_adjust(**SPEC['margins'])
    ax.plot(xs,compute,color='#777777',ls='--',lw=.8,label=SPEC['curve_labels']['compute'])
    ax.plot(xs,ys,color=COLOR['improved'],lw=1.25,label=SPEC['curve_labels']['effective'])
    ax.axhline(power_min,color=SPEC['power_line_color'],ls=SPEC['power_line_style'],lw=SPEC['power_line_width_pt'],label=SPEC['curve_labels']['power'])
    ax.axvline(balance['frequency_ghz'],color='#777777',ls='--',lw=.6)
    handles,labels=ax.get_legend_handles_labels()
    order=[1,0,2]
    fig.legend([handles[i] for i in order],[labels[i] for i in order],loc='upper center',bbox_to_anchor=(.56,1.0),ncol=3,frameon=False,handlelength=1.0,columnspacing=.65,handletextpad=.3)
    ax.set_xlabel(SPEC['xlabel'],labelpad=2);ax.set_ylabel(SPEC['ylabel'],labelpad=2)
    ax.set_xlim(SPEC['x_limits']);ax.set_xticks(SPEC['x_ticks'])
    ax.set_ylim(SPEC['y_limits']);ax.set_yticks(SPEC['y_ticks'])
    ax.text(balance['frequency_ghz'],SPEC['y_limits'][1]*SPEC['operating_label_y_fraction'],
            f"{balance['frequency_ghz']:.1f} GHz",rotation=90,ha='center',va='center',
            size=FONT['note'],bbox=dict(facecolor='white',edgecolor='none',pad=.7))
    ax.text(SPEC['power_label_x_fraction'],power_min+SPEC['power_label_y_offset'],
            f'{power_min:g} cycles',transform=ax.get_yaxis_transform(),ha='right',va='bottom',
            color=SPEC['power_line_color'],size=FONT['note'])
    ax.grid(axis='y',ls=':',lw=.5,color='#CCCCCC')
    fig.canvas.draw()
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            box=t.get_window_extent(fig.canvas.get_renderer())
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.x1+1 and box.y1<=fig.bbox.y1+1,(t.get_text(),box)
    save(fig, LABEL, BASE/'figure.pdf')

if __name__ == '__main__':
    main()
    print('Saved ' + str(BASE / 'figure.pdf') + ' and figure.png')
