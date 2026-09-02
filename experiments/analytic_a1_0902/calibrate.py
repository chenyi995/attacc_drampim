#!/usr/bin/env python3
"""Calibrate and VALIDATE the analytic PIM model, one layer at a time.

  Layer 1  command counts     -- must be 100% exact against Ramulator's own
                                 counters.  No fit, no tolerance.
  Layer 2  trace structure    -- must be 100% exact against regenerated
                                 traces (barrier groups, row openings).
                                 No fit, no tolerance.  Sampled, because it
                                 re-runs the trace generator.
  Layer 3  cycles             -- CROSS-VALIDATED, then refitted on all data.
                                 Two held-out protocols, both split on
                                 quantities the model can actually see:
                                   * unseen model INPUTS (feature vectors)
                                   * unseen CONFIGURATIONS (everything the
                                     features see except run length)
                                 Each is repeated over several seeds and
                                 reported as mean +/- std.

Splitting on ``run_length`` -- what an earlier version of this script did --
LEAKS: ceil() steps collapse hundreds of run lengths onto one feature vector
(11,716 legacy cache rows carry only 49 distinct model inputs) and
``channel_base`` never reaches the features at all, so 98.6% of that
protocol's "held-out" rows were byte-identical to a training row and its
4.03% was training error.  Rows are therefore deduplicated to one sample per
distinct model input before anything is fitted or scored.

The written model file carries the cross-validated metrics and the calibrated
domain, so a consumer can print how far it is being trusted.  The shipped
coefficients are fitted on every input, so they have no held-out set of their
own: the honest claim is about the PROCEDURE.
"""
import argparse
import collections
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analytic_pim import (FEATURE_NAMES, command_counts,  # noqa: E402
                              features_from_meta, fit_timing, regime_key,
                              trace_structure)

GS_COL = 32
GS_ROW = 32 * GS_COL
GS_BA = (1 << 14) * GS_ROW
GS_BG = 4 * GS_BA
GS_RANK = 4 * GS_BG
GS_PCH = 2 * GS_RANK
GS_CH = 2 * GS_PCH

META_FIELDS = ("pim_type", "run_length", "num_ops_per_hbm", "dbyte",
               "power_constraint", "dhead", "num_hbm", "channel_count",
               "shared_kv", "shared_queries", "channel_base", "mq_command",
               "nccdab_override", "key_addr", "value_addr", "phase",
               "trace_revision")
COUNT_FIELDS = ("run_length", "num_ops_per_hbm", "dbyte", "dhead",
                "channel_count", "channel_base", "shared_queries",
                "mq_command", "phase", "trace_revision")
STRUCT_FIELDS = ("run_length", "num_ops_per_hbm", "dbyte", "dhead",
                 "channel_count", "channel_base", "key_addr", "value_addr",
                 "phase", "trace_revision")


def records(path):
    """Yield ``(signature, result, meta)`` for every usable cache line."""
    with path.open() as handle:
        for line in handle:
            try:
                signature, result = json.loads(line)
            except (ValueError, TypeError):
                continue
            if len(signature) < 16 or len(result) != 6:
                continue
            revision = signature[16] if len(signature) > 16 else "legacy"
            meta = dict(zip(META_FIELDS, list(signature[:16]) + [revision]))
            yield signature, result, meta


def load(caches):
    """Later files replace an older duplicate signature, matching Ramulator's
    append-only cache loading semantics while retaining every new shape."""
    by_signature = {}
    for cache in caches:
        if not cache.exists():
            print("warning: missing cache {}".format(cache), file=sys.stderr)
            continue
        for signature, result, meta in records(cache):
            by_signature[json.dumps(signature, separators=(",", ":"))] = (result, meta)
    return list(by_signature.values())


# --------------------------------------------------------------------------
# Layer 1
# --------------------------------------------------------------------------
def check_layer1(data):
    mismatches = []
    ok = []
    for result, meta in data:
        counted = command_counts(**{k: meta[k] for k in COUNT_FIELDS})
        if tuple(result[1:]) != counted:
            mismatches.append((meta, list(result[1:]), list(counted)))
        else:
            ok.append((result, meta))
    return ok, mismatches


# --------------------------------------------------------------------------
# Layer 2
# --------------------------------------------------------------------------
def _absolute(mapping, row):
    """Rebuild a byte address from the wrapper's address-mapping tuple.

    The signature deliberately drops the absolute row index, so the two
    phases are placed in DIFFERENT rows here: K and V never share a row in a
    real run, and letting them alias would under-count row openings.
    """
    if mapping is None:
        return None
    ch, pch, rank, bg, ba, off = mapping
    return (ch * GS_CH + pch * GS_PCH + rank * GS_RANK + bg * GS_BG +
            ba * GS_BA + row * GS_ROW + off)


def _generate(meta, out, generator):
    args = [sys.executable, str(generator),
            "--dhead", str(meta["dhead"]), "--nhead", str(meta["num_ops_per_hbm"]),
            "--seqlen", str(meta["run_length"]), "--dbyte", str(meta["dbyte"]),
            "--output", str(out)]
    key = _absolute(meta["key_addr"], 0)
    value = _absolute(meta["value_addr"], 4096)
    if meta["trace_revision"] == "chunkstripe1":
        args += ["--head-hbm-stripe"]
    if key is not None:
        args += ["--key-addr", hex(key)]
    if value is not None:
        args += ["--value-addr", hex(value)]
    if meta["channel_count"] != 16:
        args += ["--channels", str(meta["channel_count"])]
    if meta["channel_base"] is not None and meta["channel_count"] != 16:
        args += ["--pool-base", str(meta["channel_base"])]
    if meta["shared_kv"]:
        args += ["--shared-kv"]
    if meta["shared_queries"] != 1:
        args += ["--shared-queries", str(meta["shared_queries"])]
    if meta["mq_command"]:
        args += ["--mq"]
    if meta["phase"] != "full":
        args += ["--phase", meta["phase"]]
    done = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if done.returncode:
        raise RuntimeError(done.stderr.decode()[-300:])


def _measure(path):
    """Ground truth for Layer 2, read straight off the generated trace."""
    sequences = collections.defaultdict(list)
    groups = 0
    previous_barrier = False
    with path.open() as handle:
        for line in handle:
            op, text = line.split()
            address = int(text, 16)
            if op == "PIM_BARRIER":
                if not previous_barrier:
                    groups += 1
                previous_barrier = True
                continue
            previous_barrier = False
            if op != "PIM_MAC_AB":
                continue
            key = ((address // GS_CH) % 16, (address // GS_PCH) % 2)
            row = (address % GS_BA) // GS_ROW
            if not sequences[key] or sequences[key][-1] != row:
                sequences[key].append(row)
    openings = max((len(v) for v in sequences.values()), default=0)
    return groups, openings


def check_layer2(data, sample_size, seed, generator):
    by_regime = collections.defaultdict(list)
    for _, meta in data:
        by_regime[regime_key(meta)].append(meta)
    rng = random.Random(seed)
    report = {}
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "layer2.trace"
        for key, metas in by_regime.items():
            picked = rng.sample(metas, min(sample_size, len(metas)))
            exact_bar = exact_row = 0
            for meta in picked:
                try:
                    _generate(meta, out, generator)
                except (RuntimeError, OSError) as error:
                    failures.append((key, "generate", str(error)[:200]))
                    continue
                groups, openings = _measure(out)
                predicted = trace_structure(**{k: meta[k] for k in STRUCT_FIELDS})
                if predicted["barrier_groups"] == groups:
                    exact_bar += 1
                else:
                    failures.append((key, "barrier_groups", predicted["barrier_groups"],
                                     groups, meta["run_length"]))
                if predicted["row_openings"] == openings:
                    exact_row += 1
                else:
                    failures.append((key, "row_openings", predicted["row_openings"],
                                     openings, meta["run_length"]))
            report[key] = {"checked": len(picked), "exact_barrier_groups": exact_bar,
                           "exact_row_openings": exact_row}
    return report, failures


# --------------------------------------------------------------------------
# Layer 3
# --------------------------------------------------------------------------
def _config_id(meta):
    """One 'configuration': everything the FEATURES see except run length.

    Deliberately excludes ``channel_base``, which never reaches the model
    (see ``_geometry``), and includes the K/V byte offsets, which do -- they
    drive ``row_openings``.  The previous version had this exactly backwards,
    so its "held-out configuration" protocol held out a distinction the model
    cannot see while leaving one it can on both sides.
    """
    def offset(address):
        return address[-1] if isinstance(address, (list, tuple)) else address
    return (meta["trace_revision"], meta["mq_command"], meta["power_constraint"],
            meta["num_ops_per_hbm"], meta["channel_count"], meta["shared_queries"],
            meta["nccdab_override"], meta["phase"], meta["dbyte"], meta["dhead"],
            offset(meta["key_addr"]), offset(meta["value_addr"]))


def _input_id(meta):
    """The model's ACTUAL input: its feature vector.

    Splitting on anything coarser leaks.  ``ceil`` steps collapse hundreds of
    run lengths onto one feature vector -- 11,716 legacy cache rows carry only
    49 distinct model inputs -- so a run_length split puts byte-identical
    samples on both sides and reports training error as generalisation.
    """
    return tuple(features_from_meta(meta))


def _dedupe(data):
    """One sample per distinct model input, target = mean cycles there.

    Without this a regime's "n" counts cache rows, not information: the fit
    and every error statistic would be dominated by whichever input happens
    to have been simulated most often.
    """
    grouped = collections.OrderedDict()
    for result, meta in data:
        grouped.setdefault(_input_id(meta), []).append((result, meta))
    reduced = []
    for rows in grouped.values():
        cycles = sum(r[0] for r, _ in rows) / len(rows)
        result, meta = rows[0]
        reduced.append(([int(round(cycles))] + list(result[1:]), meta))
    return reduced


def _split(data, key_of, fraction, seed):
    keys = sorted({key_of(meta) for _, meta in data}, key=repr)
    rng = random.Random(seed)
    rng.shuffle(keys)
    held = set(keys[:max(1, int(len(keys) * fraction))])
    return {index: key_of(meta) in held for index, (_, meta) in enumerate(data)}


def _repeat(data, key_of, fraction, seed, repeats):
    """Cross-validated error of the FITTING PROCEDURE, over several splits.

    A single split of ~50 distinct inputs has a large seed effect: reporting
    one number invites reporting the lucky one.
    """
    import numpy as np
    per_regime = collections.defaultdict(list)
    for offset in range(repeats):
        split = _split(data, key_of, fraction, seed + offset)
        try:
            models = fit_timing(data, validation=split)
        except ValueError:
            continue
        for key, regime in models["regimes"].items():
            metrics = regime.get("validation")
            if metrics:
                per_regime[key].append(metrics)
    summary = {}
    for key, runs in per_regime.items():
        def stat(field):
            values = [r[field] for r in runs if field in r]
            return {"mean": float(np.mean(values)), "std": float(np.std(values))} if values else None
        summary[key] = {"splits": len(runs), "n_validation": runs[0]["n"],
                        "mape": stat("mape"), "p95": stat("p95"),
                        "mae_cycles": stat("mae_cycles"),
                        "aggregate_ratio": stat("aggregate_ratio")}
    return summary


def fit_and_validate(data, fraction, seed, repeats=8):
    """Cross-validate the procedure, then ship a model fitted on everything.

    The shipped coefficients see every sample, so they have no held-out set of
    their own -- the honest claim is about the PROCEDURE, measured by repeated
    leak-free splits.  The previous version quoted one protocol's error next
    to another protocol's coefficients, which described neither.
    """
    reduced = _dedupe(data)
    protocols = {
        "held_out_inputs": _repeat(reduced, _input_id, fraction, seed, repeats),
        "held_out_configs": _repeat(reduced, _config_id, fraction, seed + 1000, repeats),
    }
    shipped = fit_timing(reduced, validation={})
    for key, regime in shipped["regimes"].items():
        regime["cross_validated"] = {name: protocols[name].get(key)
                                     for name in protocols}
        regime["fitted_on_all_inputs"] = True
    return shipped, protocols


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "ramulator2/signature_cache.jsonl")
    parser.add_argument("--extra-cache", type=Path, action="append", default=[],
                        help="additional completed Ramulator signature caches")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/analytic_a1_0902/timing_models.json")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--layer2-samples", type=int, default=40,
                        help="traces regenerated per regime for the Layer-2 check")
    parser.add_argument("--skip-layer2", action="store_true",
                        help="skip trace regeneration (Layer 2 then stays unverified)")
    parser.add_argument("--generator", type=Path,
                        default=ROOT / "ramulator2/trace_gen/gen_trace_attacc_bank.py")
    args = parser.parse_args()

    data = load([args.cache] + list(args.extra_cache))
    ok, mismatches = check_layer1(data)

    layer2, layer2_failures = ({}, [])
    if not args.skip_layer2:
        layer2, layer2_failures = check_layer2(ok, args.layer2_samples,
                                               args.seed, args.generator)

    models, per_protocol = fit_and_validate(ok, args.holdout_fraction, args.seed)
    models["calibration"] = {
        "caches": [str(c) for c in [args.cache] + list(args.extra_cache)],
        "cache_entries": len(data),
        "layer1": {"checked": len(data), "exact": len(ok), "mismatches": len(mismatches)},
        "layer2": layer2,
        "layer3_protocols": per_protocol,
        "holdout_fraction": args.holdout_fraction,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")

    # Human-readable summary -----------------------------------------------
    print("cache entries         : {}".format(len(data)))
    print("Layer 1 command counts: {}/{} exact".format(len(ok), len(data)))
    for key, row in sorted(layer2.items()):
        print("Layer 2 {:22s}: barriers {}/{} exact, row openings {}/{} exact".format(
            key, row["exact_barrier_groups"], row["checked"],
            row["exact_row_openings"], row["checked"]))
    print("Layer 3 (CROSS-VALIDATED on distinct model inputs, mean +/- std):")
    for key, regime in sorted(models["regimes"].items()):
        if regime.get("insufficient_data"):
            print("  {:22s} INSUFFICIENT DATA (n={})".format(key, regime["n"]))
            continue
        print("  {:22s} distinct model inputs={:4d}  effective parameters={}"
              .format(key, regime["n_train"], regime["effective_parameters"]))
        for name, metrics in sorted((regime.get("cross_validated") or {}).items()):
            if not metrics:
                print("      {:18s} n/a".format(name))
                continue
            print("      {:18s} n_val={:3d} over {} splits: MAPE={:.2f}%+/-{:.2f}"
                  "  p95={:.2f}%  MAE={:.0f} cyc  aggregate={:.4f}".format(
                      name, metrics["n_validation"], metrics["splits"],
                      100 * metrics["mape"]["mean"], 100 * metrics["mape"]["std"],
                      100 * metrics["p95"]["mean"], metrics["mae_cycles"]["mean"],
                      metrics["aggregate_ratio"]["mean"]))
        print("      coefficients " + ", ".join(
            "{}={:.3f}".format(n, c)
            for n, c in zip(FEATURE_NAMES, regime["coefficients"]) if c > 0))
        domain = regime.get("domain", {})
        print("      run_length domain {} over {} sampled points, largest "
              "interior gap {}".format(domain.get("run_length"),
                                       domain.get("run_length_sampled_points"),
                                       domain.get("run_length_largest_interior_gap")))
    print("model written to {}".format(args.output))

    if mismatches:
        print("first Layer-1 mismatch: " + json.dumps(mismatches[0], default=str),
              file=sys.stderr)
        raise SystemExit(1)
    if layer2_failures:
        print("Layer-2 failures: " + json.dumps(layer2_failures[:5], default=str),
              file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
