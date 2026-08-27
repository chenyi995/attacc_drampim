#!/usr/bin/env python3
"""Four-group motivation chart (chenyi9 2026-08-26).

One group per topology workload, three bars per group:
  Problem 1 (A2: software-only, KV in remote dumb storage),
  Problem 2 (A3: PIM with the naive scattered layout),
  Fugue     (A6: master/diff layout + dynamic placement).

usage: plot_motiv_groups.py <out.pdf-stem> <label=dag_ladder.csv> [...]
e.g.:  plot_motiv_groups.py output/motiv_groups \
           star=output/..star../dag_ladder.csv pipeline=... debate=... mapreduce=...

Bars are end-to-end makespan normalized to each group's Fugue bar (Fugue=1);
the absolute Fugue seconds are printed under each group label.  Numbers are
motivation-grade (theoretical referenced workloads), not matrix evidence.
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNGS = ("A2", "A3", "A6")
COLORS = {"A2": "#c44e52", "A3": "#dd8452", "A6": "#4c72b0"}
LABELS = {"A2": "Problem 1: remote-KV transfer (software-only)",
          "A3": "Problem 2: irregular access (naive layout)",
          "A6": "Fugue"}


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    stem = sys.argv[1]
    groups = []
    for spec in sys.argv[2:]:
        label, path = spec.split("=", 1)
        with open(path) as handle:
            rows = {row["ablation"]: row for row in csv.DictReader(handle)}
        groups.append((label, {r: float(rows[r]["makespan_s"]) for r in RUNGS}))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    width = 0.26
    for bar_index, rung in enumerate(RUNGS):
        xs = [g + (bar_index - 1) * width for g in range(len(groups))]
        ys = [values[rung] / values["A6"] for _, values in groups]
        bars = ax.bar(xs, ys, width, color=COLORS[rung], label=LABELS[rung])
        for bar, y in zip(bars, ys):
            ax.annotate("{:.2f}x".format(y),
                        (bar.get_x() + bar.get_width() / 2, y),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["{}\n(Fugue {:.1f} s)".format(label, values["A6"])
                        for label, values in groups], fontsize=9)
    ax.set_ylabel("end-to-end time, normalized to Fugue")
    ax.axhline(1.0, color="#4c72b0", linewidth=0.8, linestyle=":")
    ax.set_ylim(0, max(values[r] / values["A6"]
                       for _, values in groups for r in RUNGS) * 1.22)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=200)
    print("wrote {0}.pdf / {0}.png".format(stem))
    for label, values in groups:
        print("{}: A2={:.2f}s A3={:.2f}s A6={:.2f}s ({:.2f}x / {:.2f}x)".format(
            label, values["A2"], values["A3"], values["A6"],
            values["A2"] / values["A6"], values["A3"] / values["A6"]))


if __name__ == "__main__":
    main()
