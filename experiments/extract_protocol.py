#!/usr/bin/env python3
"""Extract the protocol's results into one CSV and one Markdown page.

    python3 experiments/extract_protocol.py <outroot> [--manifest workload/probe/sweep/manifest.csv]
                                            [--ref A3b] [--csv out.csv] [--md out.md]

``<outroot>`` is what ``run_sweep.sh`` wrote: one directory per workload
(``C1_turns/``, ``C1_S3_chatty_share_0p0_turns/`` ...), each holding the
``dag_<combo>.json`` reports of the combos that ran there.  For every
(workload, combo) the script computes the four metrics of the protocol
(docs/README_run_protocol.md §4) plus energy, and writes

  * a CSV with one row per (workload, combo) -- every number the tables use;
  * a Markdown page: the baseline tables (all combos, absolute and relative
    to ``--ref``) and one sweep table per axis (A6 over A3b at every value).

Numbers are only ever copied from the reports by this script (agent.md §3);
nothing is typed in by hand.  Rows whose report is missing are left out, a
metric that does not exist for a combo (A2 has no PIM scan) is left empty.
"""
import argparse
import csv
import json
import os
import sys

COMBOS = ("A1", "A2", "A3b", "A4c", "A4e", "A5", "A6")
METRICS = (
    # key, column title, unit scale, better when smaller
    ("e2e_s", "E2E (s)", 1.0),
    ("ttft_ms", "TTFT (ms)", 1.0),
    ("tbt_weighted_us", "TBT wtd (us)", 1.0),
    ("tbt_mean_us", "TBT mean (us)", 1.0),
    ("scan_private_us", "scan private (us)", 1.0),
    ("scan_shared_us", "scan shared (us)", 1.0),
    ("scan_step_us", "scan step (us)", 1.0),
    ("energy_j", "energy (J)", 1.0),
    ("power_w", "power (W)", 1.0),
)


def load_report(outroot, tag, combo):
    path = os.path.join(outroot, tag, "dag_{}.json".format(combo))
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def tbt(report, lout):
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
        return None, None
    return total_time / total_steps * 1e6, sum(values) / len(values) * 1e6


def ttft(report):
    values = [rec["ttft_s"] for rec in report["summary"]["requests"].values()
              if rec.get("ttft_s") is not None]
    return sum(values) / len(values) * 1e3 if values else None


def scans(report):
    block = report.get("summary", {}).get("decode_scans") or {}
    pick = lambda key: (block.get(key) or {}).get("mean_us")
    return pick("private_service"), pick("shared_service"), pick("per_step_elapsed")


def metrics(report, lout):
    weighted, mean = tbt(report, lout)
    private, shared, step = scans(report)
    by_class = report.get("energy_breakdown_nj", {}).get("by_class", {})
    rows = report.get("prefill_attention_rows", {})
    run_config = report.get("run_config") or {}
    return {
        "e2e_s": report["makespan_s"],
        "ttft_ms": ttft(report),
        "tbt_weighted_us": weighted,
        "tbt_mean_us": mean,
        "scan_private_us": private,
        "scan_shared_us": shared,
        "scan_step_us": step,
        "energy_j": report["energy_nj"] * 1e-9,
        "power_w": report["energy_nj"] * 1e-9 / report["makespan_s"] if report["makespan_s"] else None,
        "energy_gpu_j": by_class.get("GPU", 0.0) * 1e-9,
        "energy_link_j": by_class.get("LINK", 0.0) * 1e-9,
        "energy_pim_j": by_class.get("PIM", 0.0) * 1e-9,
        "prefill_rows_pim": rows.get("pim", 0),
        "prefill_rows_gpu": rows.get("gpu", 0),
        "corrected_rows_sha": report.get("corrected_rows_sha"),
        "git_rev": (run_config.get("git_rev") or "")[:12],
        "gpu_model": run_config.get("gpu_model"),
        "model": run_config.get("model"),
        "ngpu": run_config.get("ngpu"),
        "num_hbm": run_config.get("num_hbm"),
        "epic_k": run_config.get("epic_prefix_recompute_tokens"),
        "batch": run_config.get("cacheblend_batch_size"),
    }


def fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, float):
        return "{:.{}f}".format(value, digits)
    return str(value)


def ratio(base, value):
    if base is None or value is None or not value:
        return ""
    return "{:.3f}".format(base / value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outroot")
    parser.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                           "workload", "probe", "sweep", "manifest.csv"))
    parser.add_argument("--sweep-dir", default=None, help="directory of the workload JSONs (default: the manifest's)")
    parser.add_argument("--ref", default="A3b")
    parser.add_argument("--csv", default=None, help="default <outroot>/protocol.csv")
    parser.add_argument("--md", default=None, help="default <outroot>/protocol.md")
    args = parser.parse_args()
    sweep_dir = args.sweep_dir or os.path.dirname(os.path.abspath(args.manifest))
    with open(args.manifest, newline="") as handle:
        manifest = list(csv.DictReader(handle))
    rows, found = [], {}
    for entry in manifest:
        tag = entry["file"][:-len(".json")]
        if not os.path.isdir(os.path.join(args.outroot, tag)):
            continue
        with open(os.path.join(sweep_dir, entry["file"])) as handle:
            lout = {agent["id"]: agent["lout"] for agent in json.load(handle)["agents"]}
        for combo in COMBOS:
            report = load_report(args.outroot, tag, combo)
            if report is None:
                continue
            row = {"workload": tag, "axis": entry["axis"], "value": entry["value"], "combo": combo}
            row.update(metrics(report, lout))
            rows.append(row)
            found.setdefault(tag, {})[combo] = row
    if not rows:
        raise SystemExit("no dag_<combo>.json reports under {} for the manifest's workloads".format(args.outroot))
    csv_path = args.csv or os.path.join(args.outroot, "protocol.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Protocol results: {}".format(os.path.abspath(args.outroot)), ""]
    # A1 is no-reuse and carries no k; describe the runs from the reuse combos
    models = sorted({(r["model"], r["ngpu"], r["num_hbm"], r["gpu_model"], r["epic_k"], r["batch"])
                     for r in rows if r["combo"] != "A1"} or
                    {(r["model"], r["ngpu"], r["num_hbm"], r["gpu_model"], "-", r["batch"]) for r in rows})
    lines.append("Runs: " + "; ".join("model {} ngpu {} num_hbm {} gpu_model {} k {} batch {}".format(*m) for m in models))
    lines.append("Reference combo for the relative tables: {} (ratio = {} / combo; > 1 is better).".format(args.ref, args.ref))
    lines.append("")
    # baselines
    for tag, combos in found.items():
        if len(combos) < 3:
            continue
        lines.append("## {}".format(tag))
        lines.append("")
        head = ["combo"] + [title for _, title, _ in METRICS] + ["prefill rows PIM/GPU", "code"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for combo in COMBOS:
            r = combos.get(combo)
            if r is None:
                continue
            cells = [combo] + [fmt(r[key], 4 if key == "e2e_s" else (2 if key.startswith("scan") else 3)) for key, _, _ in METRICS]
            cells += ["{}/{}".format(r["prefill_rows_pim"], r["prefill_rows_gpu"]), r["git_rev"][:7]]
            lines.append("| " + " | ".join(cells) + " |")
        base = combos.get(args.ref)
        if base is not None:
            lines.append("")
            head = ["combo"] + ["{} vs {}".format(title.split(" (")[0], args.ref) for _, title, _ in METRICS[:8]]
            lines.append("| " + " | ".join(head) + " |")
            lines.append("|" + "---|" * len(head))
            for combo in COMBOS:
                r = combos.get(combo)
                if r is None or combo == args.ref:
                    continue
                lines.append("| " + " | ".join([combo] + [ratio(base[key], r[key]) for key, _, _ in METRICS[:8]]) + " |")
        # one correction plan for every PIM combo (A2 runs the plan on the
        # GPU and writes no sha; A1 has no plan)
        shas = {r["corrected_rows_sha"] for c, r in combos.items()
                if c not in ("A1", "A2") and r["corrected_rows_sha"]}
        lines.append("")
        lines.append("corrected_rows_sha across A3b..A6: {}".format(
            "identical ({})".format(next(iter(shas))) if len(shas) == 1
            else "DIFFERENT {} -- not one plan".format(sorted(map(str, shas)))))
        lines.append("")
    # sweeps: A6 over A3b per axis
    axes = {}
    for tag, combos in found.items():
        entry = next(e for e in manifest if e["file"][:-5] == tag)
        if entry["axis"] == "baseline":
            continue
        axes.setdefault(entry["axis"], []).append((entry["value"], tag, combos))
    if axes:
        lines.append("## Sweeps: A6 over A3b (A3b / A6; > 1 means A6 is better)")
        lines.append("")
        for axis, points in axes.items():
            lines.append("### {}".format(axis))
            lines.append("")
            head = ["value", "workload", "E2E", "TTFT", "TBT wtd", "scan private", "scan step", "energy", "A6 prefill rows PIM/GPU"]
            lines.append("| " + " | ".join(head) + " |")
            lines.append("|" + "---|" * len(head))
            for value, tag, combos in points:
                a, b = combos.get("A3b"), combos.get("A6")
                if a is None or b is None:
                    lines.append("| {} | {} | (missing {}) |".format(value, tag, "A3b" if a is None else "A6"))
                    continue
                cells = [str(value), tag] + [ratio(a[k], b[k]) for k in ("e2e_s", "ttft_ms", "tbt_weighted_us", "scan_private_us", "scan_step_us", "energy_j")]
                cells.append("{}/{}".format(b["prefill_rows_pim"], b["prefill_rows_gpu"]))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    md_path = args.md or os.path.join(args.outroot, "protocol.md")
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write("CSV: {}\nMD:  {}\n".format(csv_path, md_path))


if __name__ == "__main__":
    main()
