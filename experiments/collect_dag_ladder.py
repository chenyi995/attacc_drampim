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

# The canonical ladder order.  A3b (head slicing) and A4b (global co-read
# table) joined the ladder on 2026-08-29 but never reached this tuple, so every
# dag_ladder.csv written between then and 2026-09-03 silently dropped the two
# rungs even though their dag_A3b/A4b.json were on disk beside it -- including
# all six of the 2026-09-02 baseline.  Rungs whose report is absent are
# SKIPPED rather than an error, because a run may deliberately cover a subset:
# the baseline runs A1/A2/A3b/A4/A4b/A5/A6 and every other sweep point runs
# A3b and A6 alone (ruling chenyi9 2026-09-03).
RUNG_ORDER = ("A1", "A2", "A3b", "A4c", "A4e", "A5", "A6")


def present_rungs(outdir):
    """Ladder-ordered rungs that actually have a report in ``outdir``."""
    found = [rung for rung in RUNG_ORDER
             if os.path.exists(os.path.join(outdir, "dag_{}.json".format(rung)))]
    if not found:
        raise SystemExit("no dag_A*.json reports in {}".format(outdir))
    return found
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
    RUNGS = present_rungs(outdir)
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

    # Per-tier ladder (chenyi9 order 2026-08-27): one e2e number hides the
    # cache-growth axis -- as the shared pool accretes tier by tier, each
    # rung separates on a different tier band, so the ladder is also folded
    # per tier from the decode batch records (batch ids carry the request
    # tier; timestamps are DAG times).  cum_end_s is the step curve
    # ("time until tier t is fully decoded"); span_s = last - first
    # attention timestamp inside the tier (tiers may overlap under
    # pipelining, so spans need not sum to the makespan).
    tier_rows = []
    for rung in RUNGS:
        path = os.path.join(outdir, "dag_{}.json".format(rung))
        with open(path) as handle:
            report = json.load(handle)
        batches = report.get("batches", []) or []
        # completion per tier = the last request END of that tier (re-audit
        # R14, 2026-09-05: the batch stamps below are attention STARTS)
        tier_end = {}
        for record in (report.get("summary", {}).get("requests", {}) or {}).values():
            tier_end[record.get("tier")] = max(tier_end.get(record.get("tier"), 0.0),
                                               float(record.get("end_s", 0.0)))
        by_tier = {}
        for batch in batches:
            tier = batch.get("tier")
            if tier is None:
                continue
            stamps = [batch.get(k) for k in ("q_arrival_s", "attention_start_s")]
            stamps = [s for s in stamps if s is not None]
            if not stamps:
                continue
            first, last = min(stamps), max(stamps)
            slot = by_tier.setdefault(tier, [first, last, 0])
            slot[0] = min(slot[0], first)
            slot[1] = max(slot[1], last)
            slot[2] += 1
        # Three metrics per tier (chenyi9 2026-08-27): a rung only needs to
        # separate on ONE of prefill / decode / e2e at a tier.
        #   prefill_s ~= gap from the previous tier's last decode stamp to
        #                this tier's first Q arrival (tier t's requests can
        #                only prefill after their parents decode);
        #   decode_s  = span of the tier's own decode batches;
        #   e2e: tier_total_s = prefill_s + decode_s, cum_end_s = step curve.
        prev_last = 0.0
        for tier in sorted(by_tier):
            first, last, count = by_tier[tier]
            prefill_s = max(0.0, first - prev_last)
            tier_rows.append({
                "workload": os.path.basename(workload_path),
                "ablation": rung,
                "tier": tier,
                "prefill_s": prefill_s,
                "decode_s": last - first,
                "tier_total_s": prefill_s + (last - first),
                "first_s": first,
                "last_s": last,                      # last attention START
                "cum_end_s": tier_end.get(tier, last),   # last request END
                "decode_batches": count,
            })
            prev_last = last
    if tier_rows:
        tier_csv = os.path.join(outdir, "dag_ladder_tiers.csv")
        with open(tier_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tier_rows[0]))
            writer.writeheader()
            writer.writerows(tier_rows)
        sys.stdout.write("TIER_CSV: {}\n".format(tier_csv))


if __name__ == "__main__":
    main()
