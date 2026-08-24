"""Aggregate the paper-ladder matrix into the three claim dimensions.

Per run: TTFT (time to first token = prefill_s), TBT (time between
tokens = decode-step latency, decode-step weighted across tiers),
compression (KV bytes vs the no-reuse baseline), makespan -- plus, for
A6/dynamic runs, HOW MUCH of prefill actually went to the PIM:

- analytic A6: the time share of PIM-side prefill attention
  (pim_prefill_*) vs GPU-committed dynamic classes (gpu_dynamic_* and
  their readback link) in the summed tier breakdowns;
- physical DAG runs: the per-request side decisions the run reports
  (pim_prefill_sides) plus the event-count share of prefill events on
  each side.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def tbt_s(report):
    steps = 0
    time_s = 0.0
    for tier in report.get("tiers", []):
        n = tier.get("decode_steps") or 0
        steps += n
        time_s += tier.get("decode_per_token_s", 0.0) * n
    return time_s / steps if steps else None


def dynamic_pim_share(report):
    pim = gpu = 0.0
    for tier in report.get("tiers", []):
        for key, value in tier.get("prefill_breakdown_s", {}).items():
            if key.startswith("pim_prefill_"):
                pim += value
            elif key.startswith("gpu_dynamic_") or key == "link_kv_pim_to_gpu":
                gpu += value
    total = pim + gpu
    return pim / total if total else None


def dag_pim_share(report):
    sides = report.get("pim_prefill_sides") or {}
    if not sides:
        return None, None
    request_share = sum(1 for side in sides.values() if side == "pim") / len(sides)
    events = report.get("events") or []
    pim_names = ("pim_kv_scan_score_softmax_pv", "die_score_assembly")
    gpu_names = ("gpu_prefill_score", "gpu_prefill_softmax",
                 "gpu_prefill_context", "kv_pim_to_gpu")
    pim = sum(1 for event in events if event["name"] in pim_names)
    gpu = sum(1 for event in events if event["name"] in gpu_names)
    event_share = pim / (pim + gpu) if (pim + gpu) else None
    return request_share, event_share


def main():
    rows = []
    for name in sorted(os.listdir(RESULTS)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(RESULTS, name)) as handle:
            report = json.load(handle)
        memory = report.get("memory", {})
        row = {
            "run": name[:-5],
            "makespan_s": round(report.get("makespan_s", 0.0), 4),
            "ttft_s": round(report.get("prefill_s", 0.0), 4),
            "tbt_ms": (round(tbt_s(report) * 1e3, 4)
                       if tbt_s(report) is not None else None),
            "kv_vs_no_reuse": (round(memory["kv_bytes_vs_no_reuse"], 4)
                               if memory.get("kv_bytes_vs_no_reuse") is not None
                               else None),
        }
        if report.get("ablation", {}).get("prefill_attn") == "dynamic":
            share = dynamic_pim_share(report)
            row["dyn_pim_time_share"] = (round(share, 4)
                                         if share is not None else None)
        if report.get("pim_prefill_sides") is not None:
            request_share, event_share = dag_pim_share(report)
            row["dyn_pim_request_share"] = (round(request_share, 4)
                                            if request_share is not None else None)
            row["dyn_pim_event_share"] = (round(event_share, 4)
                                          if event_share is not None else None)
        rows.append(row)
    json.dump(rows, sys.stdout, indent=1)
    print()
    # Compact ladder table on stdout.
    keys = ("run", "ttft_s", "tbt_ms", "kv_vs_no_reuse", "makespan_s")
    print("\t".join(keys))
    for row in rows:
        print("\t".join(str(row.get(key, "")) for key in keys))


if __name__ == "__main__":
    main()
