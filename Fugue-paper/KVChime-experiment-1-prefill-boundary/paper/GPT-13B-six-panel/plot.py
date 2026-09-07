#!/usr/bin/env python3
"""Standalone Experiment 1 six-panel plot using adjacent raw-data.csv and config."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

GPU, PIM = '#C15B33', '#236892'


def draw(root, dest):
    config = json.loads((root / 'plot-config.json').read_text())
    with (root / 'raw-data.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert config['kind'] == 'crossover_six_panel'
    assert len(config['panels']) == 6
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
        'axes.titlesize': 11, 'legend.fontsize': 9.5, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'savefig.facecolor': 'white',
    })
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.7), sharex=False)
    fig.subplots_adjust(left=.076, right=.985, bottom=.145, top=.79,
                        wspace=.30, hspace=.58)
    checks = []
    for ax, panel in zip(axes.flat, config['panels']):
        data = sorted((r for r in rows if int(r['cached']) == panel['cached']
                       and float(r['link_GBps']) == panel['link_GBps']),
                      key=lambda r: int(r['q']))
        assert data and all(r['model'] == config['model'] for r in data)
        q = np.array([int(r['q']) for r in data])
        g = np.array([float(r['gpu_service_us']) for r in data])
        p = np.array([float(r[panel['mode'] + '_service_us']) for r in data])
        changes = [i for i in range(len(q) - 1) if (p[i] - g[i]) * (p[i+1] - g[i+1]) < 0]
        brackets = [[int(q[i]), int(q[i+1])] for i in changes]
        assert brackets == panel['crossing_brackets']

        # Keep the agreed focus windows; expand only to retain a model's crossings.
        lo, hi = panel['q_window']
        if changes:
            lo = min(lo, max(.82, q[changes[0]] / 2))
            hi = max(hi, min(2500, q[changes[-1] + 1] * 2))
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        visible = (q >= lo) & (q <= hi)
        ax.set_ylim(min(g[visible].min(), p[visible].min()) * (.67 if panel['cached'] == 0 else .78),
                    max(g[visible].max(), p[visible].max()) * (1.65 if panel['cached'] == 0 else 1.5))
        ax.fill_between(q, g, p, where=p < g, color=PIM, alpha=.10, interpolate=True)
        ax.fill_between(q, g, p, where=g <= p, color=GPU, alpha=.10, interpolate=True)
        for i in range(len(q)-1):
            if (p[i] - g[i]) * (p[i+1] - g[i+1]) > 0:
                ax.axvspan(q[i], q[i+1], color=PIM if p[i] < g[i] else GPU,
                           alpha=.055, lw=0, zorder=0)
        for values, color, style, marker in [(g, GPU, '--', 's'), (p, PIM, '-', 'o')]:
            ax.plot(q, values, color=color, lw=1.6, ls=style, marker=marker,
                    markersize=3, markerfacecolor='white', markeredgewidth=.85)

        ticks = ([1, 8, 64, 512, 2048] if panel['cached'] == 0 else
                 [16, 32, 64, 128, 192] if panel['mode'] == 'plain' else [192, 384, 768, 1536])
        if lo < panel['q_window'][0] or hi > panel['q_window'][1]:
            candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            ticks = [v for v in candidates if lo <= v <= hi]
            if len(ticks) > 6:
                ticks = ticks[::2]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: str(int(v))))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis='y', color='#dddddd', lw=.55)
        ax.set_axisbelow(True)
        mode = 'Without MQ' if panel['mode'] == 'plain' else 'With MQ'
        ax.set_title(f"({panel['panel']}) {mode} · Cache: {panel['cached']:,}",
                     loc='left', pad=23, fontsize=10.5)
        interface = {300: 'NVLink 3', 450: 'NVLink 4 (maximum)', 32: 'PCIe 4 (minimum)'}[panel['link_GBps']]
        ax.text(0, 1.027, f"{interface} · {panel['link_GBps']} GB/s",
                transform=ax.transAxes, fontsize=8.8, color='#555555', va='bottom')
        for lower, upper in brackets:
            ax.axvspan(lower, upper, color='#777777', alpha=.12, lw=0, zorder=1)
        if len(changes) == 1:
            i = changes[0]
            x = np.sqrt(q[i] * q[i+1])
            y = np.sqrt(g[i] * g[i+1])
            ax.annotate(f'{q[i]}–{q[i+1]}', xy=(x, y), xytext=(-10, 22),
                        textcoords='offset points', ha='center', fontsize=9.5,
                        arrowprops=dict(arrowstyle='-', lw=.7, color='#444444'),
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.95, pad=1.4))
        else:
            note = ('Q: ' + '; '.join(f'{a}–{b}' for a, b in brackets) if changes else
                    'PIM faster over sampled Q' if np.all(p < g) else
                    'GPU faster over sampled Q' if np.all(g < p) else 'Sampled costs include ties')
            ax.text(.025, .94, note, transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1.2))
        checks.append(dict(panel=panel['panel'], mode=panel['mode'], cached=panel['cached'],
                           link_GBps=panel['link_GBps'], rows=len(data),
                           q_window=[lo, hi], crossing_brackets=brackets))

    fig.suptitle('Prefill attention crossover · ' + config['model'], fontsize=14, y=.976)
    fig.legend(handles=[
        Line2D([], [], color=GPU, lw=1.6, ls='--', marker='s', markersize=4,
               markerfacecolor='white', label='GPU'),
        Line2D([], [], color=PIM, lw=1.6, marker='o', markersize=4,
               markerfacecolor='white', label='PIM'),
        Patch(facecolor=PIM, alpha=.13, label='PIM lower cost'),
        Patch(facecolor=GPU, alpha=.13, label='GPU lower cost'),
    ], loc='upper center', bbox_to_anchor=(.5, .939), ncol=4, frameon=False, columnspacing=1.8)
    fig.supxlabel('New query tokens, Q', y=.069, fontsize=11)
    fig.supylabel('Attention service time (µs / layer)', x=.014, fontsize=11)
    fig.text(.5, .023, f"A100a · TP{config['tensor_parallel']} · batch 1 · cache initially on PIM · uncovered KV transfer counted · sampled crossing brackets",
             ha='center', fontsize=8.5, color='#484848')
    assert len(fig.axes) == 6
    dest.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(dest / ('figure.' + extension), dpi=220)
    (dest / 'render-check.json').write_text(json.dumps(dict(
        axes=6, layout=[2, 3], response_metric=config['ylabel'], input_rows=len(rows),
        model=config['model'], panels=checks), indent=2) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    draw(root, args.output_dir or root)


if __name__ == '__main__':
    main()
