#!/usr/bin/env python3
"""Transfer test: score a calibrated timing model on a cache it never saw.

This is the strongest available check on the analytic PIM model, and the one
a same-cache held-out split cannot give: the model is fitted on one model's
runs (LLAMA-7B) and scored against Ramulator ground truth produced by a
DIFFERENT transformer (e.g. LLAMA3-8B, whose GQA grouping changes the query
batch and the per-HBM head count).  No refit happens here.

Errors are reported per layer and per prediction path, because they fail for
different reasons and a single averaged number hides all of it:

  * Layer 1  command counts -- exact or broken; there is no tolerance.
  * Layer 3  cycles         -- split into runs INSIDE the calibrated feature
                               envelope and runs outside it (extrapolation),
                               and separately for any regime the model has no
                               parameters for at all.

With ``--refit-output`` the script then refits INCLUDING the new cache and
writes an extended model, so the transfer number above stays an honest
before-the-fact measurement rather than a post-hoc fit.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/analytic_a1_0902"))
from src.analytic_pim import (command_counts, estimate,  # noqa: E402
                              features_from_meta, regime_key, _outside_domain)
import calibrate  # noqa: E402


def _percentiles(values):
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    return {"n": n,
            "mape": 100 * sum(values) / n,
            "p50": 100 * values[n // 2],
            "p95": 100 * values[min(n - 1, int(0.95 * (n - 1)))],
            "max": 100 * values[-1]}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-file", type=Path, required=True,
                        help="calibrated timing model to score (NOT refitted here)")
    parser.add_argument("--cache", type=Path, action="append", required=True,
                        help="unseen Ramulator signature cache(s) to score against")
    parser.add_argument("--label", default="transfer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refit-output", type=Path,
                        help="also write a model refitted on old + new caches")
    parser.add_argument("--baseline-cache", type=Path,
                        default=ROOT / "ramulator2/signature_cache.jsonl",
                        help="the cache the model was originally fitted on")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    models = json.loads(args.model_file.read_text())
    if models.get("version") != 2:
        raise SystemExit("{}: not a version-2 model".format(args.model_file))

    unseen = calibrate.load(args.cache)
    baseline_keys = set()
    if args.baseline_cache.exists():
        for signature, _, _ in calibrate.records(args.baseline_cache):
            baseline_keys.add(json.dumps(signature, separators=(",", ":")))
    if not unseen:
        raise SystemExit("no usable entries in {}".format(args.cache))

    # Layer 1 ---------------------------------------------------------------
    ok, mismatches = calibrate.check_layer1(unseen)

    # Layer 3 ---------------------------------------------------------------
    buckets = collections.defaultdict(list)
    per_regime = collections.defaultdict(list)
    diagnostics = {}
    for result, meta in ok:
        key = regime_key(meta)
        regime = models["regimes"].get(key)
        predicted = estimate(**meta, timing_models=models, diagnostics=diagnostics)[0]
        error = abs(predicted - result[0]) / max(1, result[0])
        per_regime[key].append(error)
        if regime is None or regime.get("insufficient_data"):
            buckets["uncalibrated_regime"].append(error)
            continue
        outside = _outside_domain(regime.get("domain", {}),
                                  features_from_meta(meta), meta["run_length"])
        buckets["extrapolated" if outside else "inside_domain"].append(error)

    report = {
        "label": args.label,
        "model_file": str(args.model_file),
        "unseen_caches": [str(c) for c in args.cache],
        "entries": len(unseen),
        "entries_also_in_baseline": sum(
            1 for signature, _, _ in
            (r for cache in args.cache if cache.exists()
             for r in calibrate.records(cache))
            if json.dumps(signature, separators=(",", ":")) in baseline_keys),
        "layer1_command_counts": {"checked": len(unseen), "exact": len(ok),
                                  "mismatches": len(mismatches)},
        "layer3_by_prediction_path": {k: _percentiles(v) for k, v in buckets.items()},
        "layer3_by_regime": {k: _percentiles(v) for k, v in per_regime.items()},
        "estimate_diagnostics": diagnostics,
    }

    if args.refit_output:
        combined = calibrate.load([args.baseline_cache] + list(args.cache))
        combined_ok, combined_mismatch = calibrate.check_layer1(combined)
        refitted, protocols = calibrate.fit_and_validate(
            combined_ok, args.holdout_fraction, args.seed)
        refitted["calibration"] = {
            "caches": [str(args.baseline_cache)] + [str(c) for c in args.cache],
            "cache_entries": len(combined),
            "layer1": {"checked": len(combined), "exact": len(combined_ok),
                       "mismatches": len(combined_mismatch)},
            "layer2": "not re-checked here; run calibrate.py for the trace check",
            "layer3_protocols": protocols,
            "holdout_fraction": args.holdout_fraction, "seed": args.seed,
        }
        args.refit_output.parent.mkdir(parents=True, exist_ok=True)
        args.refit_output.write_text(json.dumps(refitted, indent=2, sort_keys=True) + "\n")
        report["refit_output"] = str(args.refit_output)
        report["refit_validation"] = {
            key: regime.get("validation_protocols")
            for key, regime in refitted["regimes"].items()}

    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
