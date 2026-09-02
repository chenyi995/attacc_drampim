#!/usr/bin/env python3
"""Assemble the A1 2x2 table from whatever cells have landed.

Rows are {event DAG, DAG-free enumerator} x {Ramulator, analytic}.  Only
quantities that mean the same thing in all four cells go in one table:

  PIM scan energy   -- the DAG's ``pim_kv_scan_score_softmax_pv`` plus
                       ``decode_pim_kv_scan_score_softmax_pv`` by_event
                       entries, NOT by_class['PIM'] (which also holds
                       bandwidth-priced KV stores the enumerator never models).
  PIM device time   -- the DAG's ``pim_time_s_unoverlapped`` (prefill sweeps,
                       device "PIM") PLUS ``pim_pool_time_s_unoverlapped``
                       (decode channel runs, device "PIM:pool*"), against the
                       enumerator's unordered cycle sum.  Both are sums of
                       per-run durations, not schedules.
  wall clock        -- split into work-list construction and PIM pricing.
                       The DAG's construction figure subtracts pricing_wall_s,
                       because dag_build_s contains the pricing done during it.

``makespan`` is deliberately absent from the comparison: the enumerator has no
scheduler, so quoting a makespan beside an unscheduled sum would invite exactly
the comparison that is not valid.  It is printed for the DAG cells only.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_EVENTS = ("pim_kv_scan_score_softmax_pv",
               "decode_pim_kv_scan_score_softmax_pv")


def read_cell(path):
    report = json.loads(path.read_text())
    if "makespan_s" in report:
        by_event = report["energy_breakdown_nj"]["by_event"]
        cache = report["ramulator_signature_cache"]
        pricing = cache.get("host_pricing_seconds", {})
        priced_wall = pricing.get("pricing_wall_s", 0.0)
        return {
            "engine": "dag",
            "pricer": "analytic" if cache.get("analytic_model") else "ramulator",
            "pim_scan_energy_nj": sum(by_event.get(n, 0.0) for n in SCAN_EVENTS),
            "pim_device_time_s": (report["pim_time_s_unoverlapped"] +
                                  report["pim_pool_time_s_unoverlapped"]),
            "worklist_s": report["dag_build_s"] - priced_wall + report["dag_finalize_s"],
            "pricing_s": priced_wall,
            "ramulator_invocations": cache.get("ramulator_invocations", 0),
            "makespan_s": report["makespan_s"],
            "diagnostics": cache.get("analytic_diagnostics", {}),
        }
    pricing = report.get("host_pricing_seconds") or {}
    return {
        "engine": "enum",
        "pricer": report.get("pim_pricer", "?"),
        "pim_scan_energy_nj": report["pim_scan_energy_nj"],
        "pim_device_time_s": report["pim_time_s_unordered"],
        "worklist_s": report["input_enumeration_s"],
        "pricing_s": pricing.get("pricing_wall_s", report["pim_model_eval_s"]),
        "ramulator_invocations": None,
        "makespan_s": None,
        "diagnostics": report.get("pim_model_diagnostics", {}),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path,
                        default=ROOT / "experiments/analytic_a1_0902/matrix2x2")
    parser.add_argument("--prefix", default="", help="only cells whose tag starts with this")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cells = {}
    for report in sorted(args.matrix.glob("*/report.json")):
        tag = report.parent.name
        if args.prefix and not tag.startswith(args.prefix):
            continue
        cells[tag] = read_cell(report)

    reference = next((c for c in cells.values()
                      if c["engine"] == "dag" and c["pricer"] == "ramulator"), None)

    print("{:20s} {:>16s} {:>14s} {:>12s} {:>12s} {:>12s}".format(
        "cell", "PIM energy nJ", "PIM time s", "worklist s", "pricing s", "wall s"))
    for tag, c in sorted(cells.items()):
        print("{:20s} {:16.6g} {:14.6f} {:12.2f} {:12.3f} {:12.2f}".format(
            tag, c["pim_scan_energy_nj"], c["pim_device_time_s"],
            c["worklist_s"], c["pricing_s"], c["worklist_s"] + c["pricing_s"]))

    if reference:
        print("\nrelative to (dag, ramulator):")
        for tag, c in sorted(cells.items()):
            de = 100 * (c["pim_scan_energy_nj"] / reference["pim_scan_energy_nj"] - 1)
            dt = 100 * (c["pim_device_time_s"] / reference["pim_device_time_s"] - 1)
            speed = ((reference["worklist_s"] + reference["pricing_s"]) /
                     max(1e-9, c["worklist_s"] + c["pricing_s"]))
            print("  {:20s} energy {:+7.3f}%   PIM time {:+7.2f}%   wall {:7.1f}x".format(
                tag, de, dt, speed))
    else:
        print("\n(the (dag, ramulator) reference cell has not landed yet)")

    want = {"{}_{}".format(e, p) for e in ("dag", "enum") for p in ("ram", "ana")}
    have = {"{}_{}".format(c["engine"], c["pricer"][:3]) for c in cells.values()}
    if want - have:
        print("\nincomplete matrix; still missing:", sorted(want - have))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(cells, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
