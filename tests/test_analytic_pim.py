"""Layer-by-layer checks for the analytic PIM pricing path.

Each layer is asserted against ITS OWN ground truth, and the cycle layer is
asserted on a HELD-OUT split only.  An earlier version asserted a
training-set MAPE, which a lookup table passes by construction; the version
after that split on ``run_length``, which leaks because ceil() steps collapse
many lengths onto one feature vector.  Both could pass while the model
generalised badly, so the split here is on the model's actual input.
"""
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/analytic_a1_0902"))
from src.analytic_pim import (UncalibratedRegime, command_counts,  # noqa: E402
                              estimate, fit_timing, regime_key, trace_structure)
import calibrate  # noqa: E402  (the calibration script is the reference impl)

CACHE = ROOT / "ramulator2" / "signature_cache.jsonl"
MODEL = ROOT / "experiments/analytic_a1_0902/timing_models.json"


def _data():
    return calibrate.load([CACHE])


class Layer1CommandCounts(unittest.TestCase):
    def test_every_cached_command_count_is_exact(self):
        """Layer 1 is a transcription, not a fit: it is exact or it is wrong."""
        ok, mismatches = calibrate.check_layer1(_data())
        self.assertEqual(mismatches, [], "command counts drifted from the generator")
        self.assertGreater(len(ok), 0)


class Layer2TraceStructure(unittest.TestCase):
    """Barrier groups and row openings must match a REGENERATED trace.

    This is what keeps the cycle model physical: it is checked against the
    command stream itself, never against a cycle count.
    """

    def test_structure_matches_generated_traces(self):
        data = _data()
        generator = ROOT / "ramulator2/trace_gen/gen_trace_attacc_bank.py"
        if not generator.exists():          # pragma: no cover - checkout without ramulator
            self.skipTest("trace generator not present")
        report, failures = calibrate.check_layer2(data, sample_size=8, seed=7,
                                                  generator=generator)
        self.assertEqual(failures, [], "closed-form trace structure diverged")
        self.assertTrue(report, "no regime was sampled")
        for key, row in report.items():
            self.assertEqual(row["exact_barrier_groups"], row["checked"], key)
            self.assertEqual(row["exact_row_openings"], row["checked"], key)

    def test_row_openings_track_the_row_boundary(self):
        """A run that straddles a DRAM row must cost one more ACT than one
        that does not -- the effect the v1 model averaged away."""
        common = dict(run_length=65, num_ops_per_hbm=1, dbyte=2, dhead=128,
                      channel_count=15, channel_base=None, phase="full",
                      trace_revision="legacy")
        inside = trace_structure(key_addr=(0, 0, 0, 0, 2, 0),
                                 value_addr=(0, 0, 0, 0, 2, 0), **common)
        straddling = trace_structure(key_addr=(0, 0, 0, 0, 2, 768),
                                     value_addr=(0, 0, 0, 0, 2, 768), **common)
        self.assertEqual(inside["row_openings"], 2)
        self.assertEqual(straddling["row_openings"], 4)


class Layer3Cycles(unittest.TestCase):
    def test_shipped_model_reports_cross_validated_error(self):
        """The model file must carry out-of-sample metrics, not training ones."""
        if not MODEL.exists():              # pragma: no cover
            self.skipTest("timing model not calibrated yet")
        models = json.loads(MODEL.read_text())
        self.assertEqual(models.get("version"), 2)
        for key, regime in models["regimes"].items():
            if regime.get("insufficient_data"):
                continue
            cross = regime.get("cross_validated") or {}
            self.assertTrue(cross, "{}: no cross-validated metrics".format(key))
            for name, metrics in cross.items():
                self.assertIsNotNone(metrics, "{}/{} missing".format(key, name))
                self.assertLess(metrics["mape"]["mean"], 0.15,
                                "{}/{} cross-validated MAPE regressed".format(key, name))
                self.assertGreaterEqual(metrics["splits"], 2,
                                        "a single split invites reporting the lucky one")

    def test_split_is_on_the_models_actual_input(self):
        """A run_length split leaks: ceil() steps collapse many lengths onto
        one feature vector, so both sides of the split hold the same sample."""
        data = _data()
        ok, _ = calibrate.check_layer1(data)
        lengths = {meta["run_length"] for _, meta in ok}
        inputs = {calibrate._input_id(meta) for _, meta in ok}
        self.assertLess(len(inputs), len(lengths),
                        "if these were equal the leak would not exist and this "
                        "test would be pointless")
        # channel_base must not reach the features, or a config split leaks too.
        base = dict(run_length=1024, num_ops_per_hbm=1, dbyte=2, dhead=128,
                    channel_count=1, shared_queries=1, mq_command=False,
                    phase="full", trace_revision="chunkstripe1",
                    key_addr=(0, 0, 0, 0, 0, 0), value_addr=(0, 0, 0, 0, 0, 0),
                    power_constraint=True, nccdab_override=None)
        vectors = {calibrate._input_id(dict(base, channel_base=b))
                   for b in (None, 0, 8, 15)}
        self.assertEqual(len(vectors), 1, "channel_base reached the features")
        # ... and the config split must not key on that inert field either.
        config_keys = {calibrate._config_id(dict(base, channel_base=b))
                       for b in (None, 0, 8, 15)}
        self.assertEqual(len(config_keys), 1,
                         "_config_id still splits on a field the model cannot see")

    def test_held_out_inputs_generalise(self):
        """Refit here, so the assertion cannot be satisfied by a stale file."""
        data = _data()
        ok, _ = calibrate.check_layer1(data)
        reduced = calibrate._dedupe(ok)
        split = calibrate._split(reduced, calibrate._input_id, 0.2, 11)
        models = fit_timing(reduced, validation=split)
        for key, regime in models["regimes"].items():
            if regime.get("insufficient_data"):
                continue
            validation = regime.get("validation")
            self.assertIsNotNone(validation, key)
            self.assertLess(validation["mape"], 0.20,
                            "{} held-out MAPE {:.3f}".format(key, validation["mape"]))

    def test_refresh_stretch_would_be_unidentifiable(self):
        """Guard the reason the refresh term was removed: the prediction is
        invariant under coefficients -> stretch * coefficients, so a fitted
        stretch is a gauge, not physics."""
        import numpy as np
        from src.analytic_pim import _nnls
        rng = np.random.default_rng(0)
        matrix = np.abs(rng.normal(size=(40, 4))) * 100
        truth = matrix @ np.array([1.0, 2.0, 3.0, 4.0])
        predictions = []
        for stretch in (1.0, 0.9, 0.75):
            coefficients = _nnls(matrix, truth * stretch,
                                 1.0 / np.maximum(1.0, truth * stretch))
            predictions.append((matrix @ coefficients) / stretch)
        for other in predictions[1:]:
            self.assertLess(float(np.abs(other - predictions[0]).max()), 1e-6)

    def test_fit_requires_an_explicit_split(self):
        with self.assertRaises(ValueError):
            fit_timing([], validation=None)

    def test_uncalibrated_regime_is_loud(self):
        """An unseen regime must never be priced silently."""
        arguments = dict(pim_type="BA", run_length=1024, num_ops_per_hbm=1,
                         dbyte=2, power_constraint=True, dhead=128,
                         channel_count=1, shared_queries=1, channel_base=0,
                         mq_command=False, phase="full",
                         trace_revision="chunkstripe1")
        empty = {"version": 2, "regimes": {}}
        with self.assertRaises(UncalibratedRegime):
            estimate(timing_models=empty, **arguments)
        diagnostics = {}
        estimate(timing_models=empty, diagnostics=diagnostics, **arguments)
        self.assertEqual(diagnostics["uncalibrated"], 1)
        self.assertIn("chunkstripe1|replicate", diagnostics["uncalibrated_regimes"])

    def test_regime_key_is_physical_not_identity(self):
        """Channel, head count and batch size must NOT create new regimes:
        that is how the v1 model fragmented 444 samples over 92 buckets."""
        base = {"trace_revision": "chunkstripe1", "mq_command": False}
        self.assertEqual(regime_key(dict(base, channel_base=0, num_ops_per_hbm=1)),
                         regime_key(dict(base, channel_base=13, num_ops_per_hbm=7)))
        self.assertNotEqual(regime_key(base),
                            regime_key(dict(base, mq_command=True)))


class ExtrapolationIsReported(unittest.TestCase):
    def test_out_of_domain_run_is_flagged(self):
        if not MODEL.exists():              # pragma: no cover
            self.skipTest("timing model not calibrated yet")
        models = json.loads(MODEL.read_text())
        regime = models["regimes"].get("chunkstripe1|replicate")
        if not regime or regime.get("insufficient_data"):
            self.skipTest("regime not calibrated")
        far = int(regime["domain"]["run_length"][1]) * 50
        diagnostics = {}
        estimate(pim_type="BA", run_length=far, num_ops_per_hbm=1, dbyte=2,
                 power_constraint=True, dhead=128, channel_count=1,
                 shared_queries=1, channel_base=0, mq_command=False,
                 phase="full", trace_revision="chunkstripe1",
                 timing_models=models, diagnostics=diagnostics)
        self.assertEqual(diagnostics.get("extrapolated"), 1)
        self.assertIn("run_length", diagnostics["extrapolated_features"])


if __name__ == "__main__":
    unittest.main()
