#!/usr/bin/env python3
"""Three-bar motivation chart (simplified fig:motiv; chenyi9 2026-08-26).

Left bar   = Problem 1, prefill/KV transfer is memory-bound: the A2 rung
             (software reuse only; KV in remote dumb storage, every byte
             crosses the GPU<->remote NVLink/PCIe link);
Middle bar = Problem 2, irregular memory access: the A3 rung (PIM serving
             with the naive scattered layout -- same reuse, fragmented
             streams);
Right bar  = Fugue (A6: master/diff layout + dynamic placement).

Reads the dag_ladder.csv a run_dag_ladder.sh run produced and writes
motiv_bars.pdf/.png next to it.  Bars are end-to-end makespan of the SAME
workload under the three configurations; annotations give seconds and the
slowdown vs Fugue.  Numbers are motivation-grade (theoretical star-repair
workload), NOT paper-matrix evidence; nothing here writes into the paper
repository.
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BARS = (
    ("A2", "Problem 1\nmemory-bound KV transfer\n(software-only, remote KV)"),
    ("A3", "Problem 2\nirregular access\n(PIM, naive scattered layout)"),
    ("A6", "Fugue\n(master/diff + dynamic)"),
)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: plot_motiv_bars.py <dag_ladder.csv>")
    csv_path = sys.argv[1]
    with open(csv_path) as handle:
        rows = {row["ablation"]: row for row in csv.DictReader(handle)}
    values = [float(rows[rung]["makespan_s"]) for rung, _ in BARS]
    labels = [label for _, label in BARS]
    fugue = values[-1]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = ["#c44e52", "#dd8452", "#4c72b0"]
    bars = ax.bar(range(len(values)), values, color=colors, width=0.62)
    for index, (bar, value) in enumerate(zip(bars, values)):
        note = "{:.3f} s".format(value)
        if index < len(values) - 1 and fugue > 0:
            note += "\n({:.2f}x)".format(value / fugue)
        ax.annotate(note, (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("end-to-end time of one repair task (s)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title("Star multi-agent repair (1 main + 3 workers x 3 rounds)",
                 fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_stem = csv_path.rsplit("/", 1)[0] + "/motiv_bars"
    fig.savefig(out_stem + ".pdf")
    fig.savefig(out_stem + ".png", dpi=200)
    print("wrote {0}.pdf / {0}.png".format(out_stem))
    for (rung, _), value in zip(BARS, values):
        print("{}: {:.4f} s ({:.2f}x vs A6)".format(rung, value, value / fugue))


if __name__ == "__main__":
    main()
