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
    """(mean over requests, max over requests, step-weighted) TBT.

    Per request TBT = (end_s - first_token_s) / (lout - 1).  The paper's
    number is the step-weighted one, sum(decode time) / sum(steps); the
    per-request mean is kept alongside (ruling chenyi9 2026-09-05: report
    both, the paper uses the weighted one).
    """
    reqs = report["summary"]["requests"]
    values, total_time, total_steps = [], 0.0, 0
    for rid, rec in reqs.items():
        steps = lout.get(rid)
        if not steps or steps < 2:
            continue
        span = rec["end_s"] - rec["first_token_s"]
        values.append(span / (steps - 1))
        total_time += span
        total_steps += steps - 1
    if not values:
        return float("nan"), float("nan"), float("nan")
    return sum(values) / len(values), max(values), total_time / total_steps


def ttft_mean(report):
    """Mean TTFT over requests that generated a token: first token completed
    minus the request's release (business dependencies satisfied; queueing
    included).  Falls back to first_token_s for reports written before the
    release field existed (then it is the time from the run start)."""
    values = []
    for rec in report["summary"]["requests"].values():
        if rec.get("ttft_s") is not None:
            values.append(rec["ttft_s"])
        elif "ttft_s" not in rec and rec.get("first_token_s", 0.0) > 0.0:
            values.append(rec["first_token_s"])
    return sum(values) / len(values) if values else float("nan")


def scan_stats(report):
    """(private service mean, shared service mean, per-step elapsed mean) in us;
    NaN when the report has no decode scans (A2) or predates the field."""
    scans = report.get("summary", {}).get("decode_scans") or {}
    pick = lambda key: ((scans.get(key) or {}).get("mean_us"))
    return tuple(float("nan") if v is None else v
                 for v in (pick("private_service"), pick("shared_service"), pick("per_step_elapsed")))


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
        mean_tbt, max_tbt, weighted_tbt = tbt_stats(rep, lout)
        by_class = rep.get("energy_breakdown_nj", {}).get("by_class", {})
        scan_private, scan_shared, scan_step = scan_stats(rep)
        run_config = rep.get("run_config") or {}
        rows[rung] = {
            "e2e_s": rep["makespan_s"],
            "ttft_ms": ttft_mean(rep) * 1e3,
            "scan_private_us": scan_private,
            "scan_shared_us": scan_shared,
            "scan_step_us": scan_step,
            "provenance": "{} {} hbm{}".format((run_config.get("git_rev") or "?")[:7],
                                               run_config.get("gpu_model", "?"),
                                               run_config.get("num_hbm", "?")),
            "tbt_mean_us": mean_tbt * 1e6,
            "tbt_max_us": max_tbt * 1e6,
            "tbt_weighted_us": weighted_tbt * 1e6,
            "energy_j": rep["energy_nj"] * 1e-9,
            "power_w": rep["energy_nj"] * 1e-9 / rep["makespan_s"],
            "e_gpu_j": by_class.get("GPU", 0.0) * 1e-9,
            "e_link_j": by_class.get("LINK", 0.0) * 1e-9,
            "e_pim_j": by_class.get("PIM", 0.0) * 1e-9,
            "prefill_pim_rows": rep.get("prefill_attention_rows", {}).get("pim", 0),
            "prefill_gpu_rows": rep.get("prefill_attention_rows", {}).get("gpu", 0),
        }
    head = ("rung", "E2E_s", "TTFT_ms", "TBT_mean_us", "TBT_wtd_us", "TBT_max_us",
            "scan_private_us", "scan_shared_us", "scan_step_us", "energy_J", "avg_power_W",
            "E_gpu_J", "E_link_J", "E_pim_J", "prefill_rows_pim/gpu", "code/gpu/hbm")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for rung, r in rows.items():
        print("| {} | {:.4f} | {:.3f} | {:.1f} | {:.1f} | {:.1f} | {:.2f} | {:.2f} | {:.2f} | {:.3f} | {:.1f} | {:.3f} | {:.3f} | {:.3f} | {}/{} | {} |".format(
            rung, r["e2e_s"], r["ttft_ms"], r["tbt_mean_us"], r["tbt_weighted_us"], r["tbt_max_us"],
            r["scan_private_us"], r["scan_shared_us"], r["scan_step_us"], r["energy_j"], r["power_w"],
            r["e_gpu_j"], r["e_link_j"], r["e_pim_j"], r["prefill_pim_rows"], r["prefill_gpu_rows"],
            r["provenance"]))
    print()
    print("scan_*: decode PIM scan latency, max over the scan's lanes (service); "
          "scan_step: per (request, layer, step) first start to last end over its scans. "
          "TTFT = first token - release (queueing included). A2 has no PIM scan (nan).")
    if ref in rows:
        print()
        print("relative to {} (ratio = {} / rung; >1 means the rung is better)".format(ref, ref))
        print("| rung | E2E | TTFT | TBT_mean | TBT_wtd | scan_private | scan_step | energy |")
        print("|---|---|---|---|---|---|---|---|")
        base = rows[ref]
        ratio = lambda key: base[key] / r[key] if r[key] else float("nan")
        for rung, r in rows.items():
            print("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                rung, ratio("e2e_s"), ratio("ttft_ms"), ratio("tbt_mean_us"), ratio("tbt_weighted_us"),
                ratio("scan_private_us"), ratio("scan_step_us"), ratio("energy_j")))


if __name__ == "__main__":
    main()
