#!/usr/bin/env python3
"""Fold the six dag_Ax.json reports of run_dag_ladder.sh into one CSV.

Columns (chenyi9 order 2026-08-26):
  * makespan / link bytes / event count / unoverlapped device times;
  * pim_prefill_share -- fraction of requests whose PREFILL ATTENTION ran on
    the PIM: definitional 1.0 for the fixed "pim" menu (A5), 0.0 for the
    fixed "gpu" menu (A1-A4 rungs and the A2 GPU-only path), and the actual
    per-request decision count from ``pim_prefill_sides`` under "dynamic"
    (A6);
  * per-part energy: total plus the GPU / LINK / PIM / DIE / TLB classes
    from ``energy_breakdown_nj.by_class`` (0.0 when a class is absent).
"""
import csv
import json
import os
import sys

RUNGS = ("A1", "A2", "A3", "A4", "A5", "A6")
ENERGY_CLASSES = ("GPU", "LINK", "PIM", "DIE", "TLB")


def pim_prefill_share(report):
    """Row-weighted PIM share + per-request side census, uniform denominator.

    Ruling (chenyi9 2026-08-26): the statistic reflects where prefill
    attention ACTUALLY ran, over EVERY request -- never only the requests
    that reached the dynamic estimator.  Share = PIM attention rows /
    (PIM + GPU attention rows); the census counts requests classed
    pim / gpu / mixed / none (none = fully reused, zero-correction, no
    prefill attention at all).
    """
    rows = report.get("prefill_attention_rows", {})
    pim_rows, gpu_rows = rows.get("pim", 0), rows.get("gpu", 0)
    share = pim_rows / (pim_rows + gpu_rows) if (pim_rows + gpu_rows) else 0.0
    sides = report.get("prefill_attention_sides", {})
    census = {"pim": 0, "gpu": 0, "mixed": 0, "none": 0}
    for side in sides.values():
        census[side] = census.get(side, 0) + 1
    census_str = "pim={pim},gpu={gpu},mixed={mixed},none={none}/{total}".format(
        total=len(sides), **census)
    return share, census_str


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: collect_dag_ladder.py <outdir> <workload.json> <model>")
    outdir, workload_path, model = sys.argv[1:4]
    rows = []
    for rung in RUNGS:
        path = os.path.join(outdir, "dag_{}.json".format(rung))
        with open(path) as handle:
            report = json.load(handle)
        share, share_str = pim_prefill_share(report)
        energy_classes = report.get("energy_breakdown_nj", {}).get("by_class", {})
        row = {
            "workload": os.path.basename(workload_path),
            "model": model,
            "ablation": rung,
            "policy": report.get("policy"),
            "engine": report.get("engine", "dag"),
            "kv_mapping": report.get("kv_mapping"),
            "decode_attn": report.get("decode_attn"),
            "pim_prefill_mode": report.get("pim_prefill_mode"),
            "pim_prefill_share": round(share, 4),
            "prefill_side_census": share_str,
            "pim_attn_rows": report.get("prefill_attention_rows", {}).get("pim", 0),
            "gpu_attn_rows": report.get("prefill_attention_rows", {}).get("gpu", 0),
            "makespan_s": report.get("makespan_s"),
            "link_bytes": report.get("link_bytes"),
            "event_count": report.get("event_count"),
            "gpu_time_s_unoverlapped": report.get("gpu_time_s_unoverlapped"),
            "pim_pool_time_s_unoverlapped": report.get("pim_pool_time_s_unoverlapped"),
            "die_time_s_unoverlapped": report.get("die_time_s_unoverlapped"),
            "energy_total_nj": report.get("energy_nj"),
        }
        for name in ENERGY_CLASSES:
            row["energy_{}_nj".format(name.lower())] = energy_classes.get(name, 0.0)
        rows.append(row)
    csv_path = os.path.join(outdir, "dag_ladder.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(csv_path) as handle:
        sys.stdout.write(handle.read())


if __name__ == "__main__":
    main()
