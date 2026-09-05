#!/usr/bin/env python3
"""TBT / E2E / energy / power table for one ladder output directory.

    python3 summarize_ladder.py <outdir> [ref_rung]

E2E  = makespan_s of the whole workload (chenyi9's definition: total time).
TBT  = per request (end_s - first_token_s) / (lout - 1); mean and max over
       requests.  first_token_s / end_s come from report["summary"]["requests"].
power= energy_nj / makespan_s (average over the run), plus per-class energy.
"""
import json
import os
import sys

RUNGS = ("A1", "A2", "A3b", "A4c", "A4e", "A5", "A6")


def load(outdir, rung):
    path = os.path.join(outdir, "dag_{}.json".format(rung))
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def tbt_stats(report, lout):
    reqs = report["summary"]["requests"]
    values = []
    for rid, rec in reqs.items():
        steps = lout.get(rid)
        if not steps or steps < 2:
            continue
        values.append((rec["end_s"] - rec["first_token_s"]) / (steps - 1))
    if not values:
        return float("nan"), float("nan")
    return sum(values) / len(values), max(values)


def main():
    outdir = sys.argv[1]
    workload = json.load(open(sys.argv[2]))
    lout = {agent["id"]: agent["lout"] for agent in workload["agents"]}
    ref = sys.argv[3] if len(sys.argv) > 3 else "A3b"
    rows = {}
    for rung in RUNGS:
        rep = load(outdir, rung)
        if rep is None:
            continue
        mean_tbt, max_tbt = tbt_stats(rep, lout)
        by_class = rep.get("energy_breakdown_nj", {}).get("by_class", {})
        rows[rung] = {
            "e2e_s": rep["makespan_s"],
            "tbt_mean_us": mean_tbt * 1e6,
            "tbt_max_us": max_tbt * 1e6,
            "energy_j": rep["energy_nj"] * 1e-9,
            "power_w": rep["energy_nj"] * 1e-9 / rep["makespan_s"],
            "e_gpu_j": by_class.get("GPU", 0.0) * 1e-9,
            "e_link_j": by_class.get("LINK", 0.0) * 1e-9,
            "e_pim_j": by_class.get("PIM", 0.0) * 1e-9,
            "prefill_pim_rows": rep.get("prefill_attention_rows", {}).get("pim", 0),
            "prefill_gpu_rows": rep.get("prefill_attention_rows", {}).get("gpu", 0),
        }
    head = ("rung", "E2E_s", "TBT_mean_us", "TBT_max_us", "energy_J", "avg_power_W",
            "E_gpu_J", "E_link_J", "E_pim_J", "prefill_rows_pim/gpu")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for rung, r in rows.items():
        print("| {} | {:.4f} | {:.1f} | {:.1f} | {:.3f} | {:.1f} | {:.3f} | {:.3f} | {:.3f} | {}/{} |".format(
            rung, r["e2e_s"], r["tbt_mean_us"], r["tbt_max_us"], r["energy_j"], r["power_w"],
            r["e_gpu_j"], r["e_link_j"], r["e_pim_j"], r["prefill_pim_rows"], r["prefill_gpu_rows"]))
    if ref in rows:
        print()
        print("relative to {} (ratio = {} / rung; >1 means the rung is better)".format(ref, ref))
        print("| rung | E2E | TBT_mean | energy |")
        print("|---|---|---|---|")
        base = rows[ref]
        for rung, r in rows.items():
            print("| {} | {:.4f} | {:.4f} | {:.4f} |".format(
                rung, base["e2e_s"] / r["e2e_s"], base["tbt_mean_us"] / r["tbt_mean_us"],
                base["energy_j"] / r["energy_j"]))


if __name__ == "__main__":
    main()
