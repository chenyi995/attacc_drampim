#!/usr/bin/env python3
"""Summarise the A6-only probe-fix sweep.

For every cell that produced a ``dag_A6.json`` this reports the headline
makespan and the per-request prefill-side census (``pim_prefill_sides``) --
the two things the probe fix is supposed to move.
"""
import json
import os
import sys
from collections import Counter

PRIO = {}
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "cells.txt")) as handle:
    for line in handle:
        parts = line.split()
        if len(parts) == 5 and not parts[0].startswith("#"):
            prio, model, name, _wl, k = parts
            PRIO[(model, "{}_k{}".format(name, k))] = prio


def cells(root):
    for model in sorted(os.listdir(root)):
        model_dir = os.path.join(root, model)
        if not os.path.isdir(model_dir):
            continue
        for cell in sorted(os.listdir(model_dir)):
            report = os.path.join(model_dir, cell, "dag_A6.json")
            if os.path.exists(report):
                yield model, cell, report


def main(root):
    print("# A6 probe-fix sweep -- {}\n".format(root))
    print("| prio | model | cell | makespan_s | prefill sides pim/gpu | "
          "energy_nJ |")
    print("|---|---|---|---:|---:|---:|")
    rows = []
    for model, cell, path in cells(root):
        try:
            with open(path) as handle:
                report = json.load(handle)
        except (ValueError, OSError) as error:
            print("| ? | {} | {} | READ FAILED: {} | | |".format(model, cell, error))
            continue
        census = Counter((report.get("pim_prefill_sides") or {}).values())
        rows.append((PRIO.get((model, cell), "?"), model, cell,
                     report.get("makespan_s"), census.get("pim", 0),
                     census.get("gpu", 0), report.get("energy_nj")))
    for prio, model, cell, makespan, pim, gpu, energy in sorted(rows):
        print("| {} | {} | {} | {} | pim={} / gpu={} | {} |".format(
            prio, model, cell,
            "n/a" if makespan is None else "{:.3f}".format(makespan),
            pim, gpu,
            "n/a" if energy is None else "{:.4g}".format(energy)))
    print("\n{} cell(s) complete.".format(len(rows)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
