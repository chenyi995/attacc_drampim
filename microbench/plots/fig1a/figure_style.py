"""Standalone scientific-figure style, inspired by the supplied bar-chart reference."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def configure(spec):
    fonts = spec["fonts_pt"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [spec["font_family"], "DejaVu Sans"],
        "font.size": fonts["note"],
        "axes.titlesize": fonts["title"],
        "axes.labelsize": fonts["label"],
        "legend.fontsize": fonts.get("legend", fonts["label"]),
        "xtick.labelsize": fonts["note"],
        "ytick.labelsize": fonts["note"],
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 0,
        "ytick.major.size": 2.5,
        "hatch.linewidth": 0.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
        "axes.unicode_minus": False,
    })


def style_axis(ax, *, detail=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#CDCDCD", linewidth=0.45, linestyle=(0, (2, 2)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if detail:
        for spine in ax.spines.values():
            spine.set_color("#6E6E6E")
            spine.set_linewidth(0.5)
