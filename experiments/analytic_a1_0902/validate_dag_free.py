#!/usr/bin/env python3
"""Validate the DAG-free A1 input enumerator against the DAG it replaces.

The enumerator in ``src/a1_dag_free.py`` is a re-derivation of A1's placement
contract.  Its unit tests only assert numbers computed by hand from the same
mental model, which cannot catch a shared misunderstanding.  This script
compares it against ground truth: the multiset of PIM invocations the real
event DAG actually asked Ramulator to price, recorded by running

    ATTACC_RECORD_PIM_SIGNATURES=<path> python3 main.py ... --ablation A1

and reduced to the fields that determine cost (``Ramulator.PRICING_FIELDS``).

The comparison is layered, like the timing model:

  Layer A1  aggregate PIM WORK   -- total MAC/move commands must match; this
                                    is the quantity a makespan is built from.
  Layer A2  invocation MULTISET  -- every distinct (length, heads, channels,
                                    batch, phase) with its multiplicity.
  Layer A3  per-length PROFILE   -- where a multiset mismatch actually lives.

A1 and A2 are reported separately on purpose: an enumerator can get the total
work right while distributing it over the wrong runs, and only A2 catches that.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.a1_dag_free import enumerate_a1_pim_inputs  # noqa: E402
from src.analytic_pim import command_counts  # noqa: E402
from src.config import make_model_config  # noqa: E402
from src.model import Transformer  # noqa: E402
from src.ramulator_wrapper import Ramulator  # noqa: E402
from src.type import DataType  # noqa: E402
from src.workload import load_workload  # noqa: E402


def enumerator_multiset(workload, model, *, dbyte, num_hbm, power_constraint):
    """Enumerator output in the DAG's own pricing-key vocabulary."""
    counts = collections.Counter()
    for item, multiplicity in enumerate_a1_pim_inputs(
            workload, model, dbyte=dbyte, num_hbm=num_hbm).items():
        key = dict(run_length=item.run_length, num_ops_per_hbm=item.num_ops_per_hbm,
                   dbyte=item.dbyte, power_constraint=bool(power_constraint),
                   channel_count=item.channel_count,
                   shared_kv=item.shared_kv,
                   shared_queries=item.shared_queries,
                   channel_base=item.channel_base, mq_command=False,
                   nccdab_override=None, phase=item.phase,
                   trace_revision="chunkstripe1")
        counts[tuple(key[name] for name in Ramulator.PRICING_FIELDS)] += multiplicity
    return counts


def dag_multiset(path):
    payload = json.loads(Path(path).read_text())
    fields = payload["fields"]
    if list(fields) != list(Ramulator.PRICING_FIELDS):
        raise SystemExit("signature log used different fields: {}".format(fields))
    counts = collections.Counter()
    for row in payload["signatures"]:
        counts[tuple(row["key"][name] for name in fields)] += row["count"]
    return counts


def _work(counts, dhead):
    """Aggregate PIM command work implied by a multiset (Layer A1)."""
    fields = Ramulator.PRICING_FIELDS
    totals = collections.Counter()
    for key, multiplicity in counts.items():
        meta = dict(zip(fields, key))
        mac, sfm, mvgb, mvsb, wrgb = command_counts(
            run_length=meta["run_length"], num_ops_per_hbm=meta["num_ops_per_hbm"],
            dbyte=meta["dbyte"], dhead=dhead, channel_count=meta["channel_count"],
            channel_base=meta["channel_base"], shared_queries=meta["shared_queries"],
            mq_command=meta["mq_command"], phase=meta["phase"],
            trace_revision=meta["trace_revision"])
        for name, value in (("mac", mac), ("sfm", sfm), ("mvgb", mvgb),
                            ("mvsb", mvsb), ("wrgb", wrgb)):
            totals[name] += value * multiplicity
        totals["physical_runs"] += multiplicity
    return totals


def _profile(counts, index):
    profile = collections.Counter()
    for key, multiplicity in counts.items():
        profile[key[index]] += multiplicity
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--signature-log", required=True,
                        help="JSON written by ATTACC_RECORD_PIM_SIGNATURES")
    parser.add_argument("--model", default="LLAMA-7B")
    parser.add_argument("--tensor-parallel", type=int, default=8)
    parser.add_argument("--num-hbm", type=int, default=5)
    parser.add_argument("--dbyte", type=int, default=2)
    parser.add_argument("--power-constraint", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    workload = load_workload(args.workload)
    model = Transformer(make_model_config(args.model, DataType.W16A16),
                        tensor_parallel=args.tensor_parallel)
    mine = enumerator_multiset(workload, model, dbyte=args.dbyte,
                               num_hbm=args.num_hbm,
                               power_constraint=args.power_constraint)
    theirs = dag_multiset(args.signature_log)

    fields = list(Ramulator.PRICING_FIELDS)
    length_index = fields.index("run_length")

    work_mine = _work(mine, model.dhead)
    work_theirs = _work(theirs, model.dhead)
    layer_a1 = {}
    for name in ("mac", "sfm", "mvgb", "mvsb", "wrgb", "physical_runs"):
        truth = work_theirs[name]
        layer_a1[name] = {
            "dag": truth, "enumerator": work_mine[name],
            "relative_error": (abs(work_mine[name] - truth) / truth) if truth else None}

    only_mine = {k: v for k, v in mine.items() if theirs.get(k, 0) != v}
    only_theirs = {k: v for k, v in theirs.items() if mine.get(k, 0) != v}
    matched = sum(min(v, theirs.get(k, 0)) for k, v in mine.items())
    layer_a2 = {
        "dag_unique": len(theirs), "enumerator_unique": len(mine),
        "exact_signature_match": len(set(mine) & set(theirs)),
        "runs_matched": matched,
        "runs_dag": sum(theirs.values()),
        "runs_enumerator": sum(mine.values()),
        "run_recall": matched / sum(theirs.values()) if theirs else None,
        "mismatched_signatures": len(set(only_mine) | set(only_theirs)),
    }

    profile_mine = _profile(mine, length_index)
    profile_theirs = _profile(theirs, length_index)
    lengths = sorted(set(profile_mine) | set(profile_theirs))
    worst = sorted(lengths, key=lambda L: -abs(profile_mine.get(L, 0) -
                                               profile_theirs.get(L, 0)))
    layer_a3 = [{"run_length": L, "dag": profile_theirs.get(L, 0),
                 "enumerator": profile_mine.get(L, 0)} for L in worst[:args.top]]

    report = {"workload": args.workload, "model": args.model,
              "signature_log": args.signature_log,
              "layer_A1_aggregate_work": layer_a1,
              "layer_A2_invocation_multiset": layer_a2,
              "layer_A3_worst_run_lengths": layer_a3,
              "verdict": ("MATCH" if not only_mine and not only_theirs
                          else "MISMATCH")}
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if report["verdict"] == "MATCH" else 2


if __name__ == "__main__":
    raise SystemExit(main())
