#!/usr/bin/env python3
"""Versioned point-estimate policy over the unchanged command model.

V1's training-maximum multiplier remains an error diagnostic. It is neither
a probability guarantee nor a default expected-latency estimate. This policy
revision follows an observed regression failure and is not a blind comparison
on the legacy evaluation set. It changes no features or fitted coefficients.
"""
import hashlib
import json
import math
from pathlib import Path
import selector_cost_model as base

SCHEMA = 'command-aware-point-selector-v2'


class PointEstimateSelector(base.CommandAwareSelector):
    def __init__(self, calibration, h, group_size=1, dhead=128, dbyte=2, *, policy=None):
        super().__init__(calibration, h, group_size, dhead, dbyte)
        if policy is None:
            policy = Path(__file__).resolve().parents[2] / 'archive/selector-model-v2/policy.json'
        if isinstance(policy, dict):
            self.policy = dict(policy)
            self.policy_sha256 = hashlib.sha256((json.dumps(self.policy, indent=2, ensure_ascii=False)+'\n').encode()).hexdigest()
        else:
            self.policy_sha256 = base.sha(policy)
            self.policy = json.loads(Path(policy).read_text())
        assert self.policy['schema'] == SCHEMA
        assert self.policy['status'] == 'frozen'
        assert self.policy['decision_rule'] == 'point-estimate-argmin'
        assert self.policy['calibration_sha256'] == self.calibration_sha256
        assert self.policy['wrapper_sha256'] == base.sha(__file__)
        assert self.policy['base_prediction_sha256'] == base.sha(base.__file__)

    def predict_views(self, *args, **kwargs):
        result = super().predict_views(*args, **kwargs)
        diagnostic_bound = result['selection_scan_us']
        result['schema'] = SCHEMA
        result['selection_scan_us'] = result['scan_us']
        result['components']['scan_us'] = result['scan_us']
        result['service_us'] = math.fsum(result['components'].values())
        result['choice'] = 'PIM' if result['service_us'] < result['gpu_us'] else 'GPU'
        result['policy_sha256'] = self.policy_sha256
        result['guard_source'] = 'Training maximum multiplier is diagnostic only; not used for device selection.'
        result['diagnostics'] = dict(training_guard_scan_us=diagnostic_bound,
                                     training_guard_multiplier=self.calibration['train_guard_multiplier'])
        result['development_scope'] = self.policy['development_disclosure']
        self.last_prediction = result
        return result
