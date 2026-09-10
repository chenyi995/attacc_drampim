#!/usr/bin/env python3
"""Logical-head GPU execution with unchanged native HBM traffic accounting.

This revision removes the previous wrapper's GQA L2 sharing discount.  Query
heads remain independent native MATMUL operations, including repeated K/V HBM
traffic.  Geometry, PIM command grouping, and fitted coefficients are inherited
unchanged from the frozen repair.  Importing this module runs no experiment and
does not write a policy; orchestration must explicitly call write_policy().
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import repair_0908_model as prior
import repair_0908_traces as traces

ROOT = prior.ROOT
STAGE = prior.STAGE
native = prior.native
geometry = prior.geometry
capacity_spec = traces.capacity_spec
schedule_views = traces.schedule_views
TraceBank = traces.TraceBank
CALIBRATION_PATH = prior.CALIBRATION_PATH
PARENT_POLICY_PATH = prior.POLICY_PATH
POLICY_PATH = ROOT / 'archive/hbm-2k-batch/policy.json'
REVISION = 'hbm-2k-batch-native-hbm-logical-head-v1'
GPU_TRAFFIC_POLICY = 'native-per-query-head-HBM-no-GQA-L2-discount'
sha = prior.sha
encoded = prior.encoded


def policy_manifest(additional_sources=()):
    """Describe a new identity without changing or refitting the parent policy."""
    parent = json.loads(PARENT_POLICY_PATH.read_text())
    assert parent['schema'] == prior.REVISION
    assert parent['status'] == 'frozen-repair-policy'
    assert parent['calibration_sha256'] == sha(CALIBRATION_PATH)
    for source in parent['sources']:
        assert sha(source['path']) == source['sha256'], source['path']
    paths = [Path(source['path']) for source in parent['sources']]
    paths += [Path(__file__), Path(prior.__file__), Path(traces.__file__),
              PARENT_POLICY_PATH, CALIBRATION_PATH,
              Path(prior.command_model.__file__),
              ROOT / 'docs/analysis/selector_point_model.py',
              STAGE / 'fugue/official-gqa-geometry.json', *map(Path, additional_sources)]
    paths = sorted({p.resolve() for p in paths}, key=str)
    return dict(schema=REVISION, status='frozen-hbm-policy',
        parent_policy_path=str(PARENT_POLICY_PATH), parent_policy_sha256=sha(PARENT_POLICY_PATH),
        coefficient_parent_policy_path=parent['parent_policy_path'],
        coefficient_parent_policy_sha256=parent['parent_policy_sha256'],
        calibration_path=str(CALIBRATION_PATH), calibration_sha256=sha(CALIBRATION_PATH),
        coefficient_policy='Reuse unchanged frozen calibration coefficients; no refit.',
        decision_rule='point-estimate-argmin',
        geometry_repairs=deepcopy(parent['geometry_repairs']),
        capacity_by_dhead=deepcopy(parent['capacity_by_dhead']),
        gpu_traffic_policy=GPU_TRAFFIC_POLICY,
        gpu_gqa=dict(parallelism='Native MATMUL m=q and numOp=batch times local query heads.',
            sharing='No wrapper sharing discount: native off-HBM, L2, L1 and register operands are returned unchanged.',
            KV_location='Repeated K/V operands are read from GPU-local HBM; this is not additional GPU-PIM link traffic.',
            bandwidth='Unchanged native HBM bandwidth and native compute/memory utilization formulas.',
            limitation='Analytical native operator model; no claim of measured GPU kernel or physical cache behavior.'),
        preserved_scope=['Corrected MT head geometry and all projection/KV capacity identities.',
                         'Same dimension-dependent resident cap, QK/PV tails, state assignments and predictor schedule.',
                         'Runner retains the separately implemented F2 readback + max(attention, export) timeline.'],
        limitations=deepcopy(parent['limitations']) + [
            'Per-query-head repeated HBM traffic is the chosen abstraction; no GPU GQA cache/broadcast benefit is modeled.'],
        sources=[dict(path=str(path), sha256=sha(path)) for path in paths])


def write_policy(path=POLICY_PATH, additional_sources=()):
    """Explicit freeze operation; never overwrite a different existing policy."""
    path = Path(path)
    content = encoded(policy_manifest(additional_sources))
    if path.exists():
        assert path.read_bytes() == content, 'Existing HBM policy is immutable; use a new revision.'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return path


class LogicalHeadGPU(native.xPU):
    def _get_traffic(self, layer):
        traffic = super()._get_traffic(layer)
        group = getattr(layer, 'kv_group_size', 1)
        if group > 1:
            assert layer.type == native.LayerType.MATMUL and layer.numOp % group == 0
            off, l2, l1, _ = traffic
            layer.kv_traffic_audit = dict(group_size=group,
                logical_head_operations=layer.numOp, unique_KV_heads=layer.numOp//group,
                unique_operand_bytes=layer.n*layer.k*layer.dbyte*(layer.numOp//group),
                configured_L2_bytes=self.l2_cache_size, shared_off_HBM=False,
                original_off_HBM_operands=list(off), charged_off_HBM_operands=list(off),
                charged_L2_operands=list(l2), charged_L1_operands=list(l1),
                traffic_policy=GPU_TRAFFIC_POLICY,
                scope='Native per-query-head GPU-local HBM traffic; L2 capacity is diagnostic only and never discounts traffic.')
        return traffic


class Model(prior.Model):
    def __init__(self, name, tp, bank, prepare_profiles=False):
        super().__init__(name, tp, bank, prepare_profiles=False)
        self.gpu = LogicalHeadGPU(native.DeviceType.GPU, self.gconf, native.SCALING_FACTOR)
        self.gpu_traffic_policy = GPU_TRAFFIC_POLICY
        if prepare_profiles:
            self.bank.prepare([(1024, self.h, self.mq_capacity), (4096, self.h, self.mq_capacity)],
                              dhead=self.trace_dhead, dbyte=self.dbyte)
            self.selector = PointSelector(CALIBRATION_PATH, self.h, self.gqa,
                                          self.trace_dhead, self.dbyte)


class PointSelector(prior.PointSelector):
    def __init__(self, calibration, h, group_size=1, dhead=128, dbyte=2, *, policy=None):
        # The parent constructor verifies the unchanged coefficients and their
        # original policy lineage.  Prediction arithmetic remains its method.
        super().__init__(calibration, h, group_size, dhead, dbyte, policy=PARENT_POLICY_PATH)
        coefficient_parent = self.parent_policy_sha256
        if policy is None:
            policy = POLICY_PATH if POLICY_PATH.exists() else policy_manifest()
        if isinstance(policy, (str, Path)):
            self.policy_path = str(Path(policy).resolve())
            self.policy_sha256 = sha(policy)
            self.policy = json.loads(Path(policy).read_text())
        else:
            self.policy_path = None
            self.policy = deepcopy(policy)
            self.policy_sha256 = hashlib.sha256(encoded(policy)).hexdigest()
        assert self.policy['schema'] == REVISION and self.policy['status'] == 'frozen-hbm-policy'
        assert self.policy['decision_rule'] == 'point-estimate-argmin'
        assert self.policy['gpu_traffic_policy'] == GPU_TRAFFIC_POLICY
        assert self.policy['parent_policy_sha256'] == sha(PARENT_POLICY_PATH)
        assert self.policy['coefficient_parent_policy_sha256'] == coefficient_parent
        assert self.policy['calibration_sha256'] == self.calibration_sha256
        for source in self.policy['sources']:
            assert sha(source['path']) == source['sha256'], ('HBM policy source changed', source['path'])
        assert self.policy['capacity_by_dhead'][str(dhead)] == capacity_spec(dhead, dbyte)
        self.parent_policy_sha256 = self.policy['parent_policy_sha256']
        self.coefficient_parent_policy_sha256 = coefficient_parent

    def predict_views(self, *args, **kwargs):
        result = super().predict_views(*args, **kwargs)
        result.update(schema=REVISION, gpu_traffic_policy=GPU_TRAFFIC_POLICY,
            coefficient_parent_policy_sha256=self.coefficient_parent_policy_sha256,
            development_scope='New policy identity removes the GPU GQA L2 discount; inherited coefficients, PIM schedule and service arithmetic are unchanged.')
        self.last_prediction = result
        return result
