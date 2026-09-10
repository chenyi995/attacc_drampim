#!/usr/bin/env python3
"""Explicit repaired model abstraction over immutable AttAcc source operators.

No workload execution, calibration, policy write, or simulator is run at import.
GPU GQA treats logical query heads as the original model's batch dimension.
Only off-HBM KV traffic is shared when the unique operand fits configured L2;
all per-query computation, L2/L1 traffic, and energy remain charged.
"""
from copy import deepcopy
import ast
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / 'archive/coverage-expansion/attacc-fugue'
if 'fugue.kvchime_model' not in sys.modules:
    sys.path.insert(0, str(STAGE))
from fugue import kvchime_model as native
from repair_0908_traces import capacity_spec, schedule_views, materialize_tile
import repair_0908_traces
import selector_cost_model as command_model
from selector_point_model import PointEstimateSelector

REVISION = 'repair-0908-consistent-model-v1'
POLICY_PATH = ROOT / 'archive/repair-0908/correction-policy.json'
CALIBRATION_PATH = ROOT / 'archive/selector-model/calibration.json'
PARENT_POLICY_PATH = ROOT / 'archive/selector-model-v2/policy.json'
CORRECTED_MT = ('MT-76B', 'MT-146B')
PRIMARY_MT_SOURCE = ROOT / 'docs/session/local/attention-shape-residency/megatron-source.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)+'\n').encode()


@lru_cache(maxsize=1)
def remote_stacks_per_gpu():
    tree = ast.parse(Path(native.__file__).read_text())
    values = [ast.literal_eval(k.value) for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == 'make_pim_config' for k in node.keywords if k.arg == 'num_hbm']
    assert len(set(values)) == 1 and values[0] > 0
    return values[0]


def geometry(name):
    cfg = deepcopy(native.geometry(name))
    if name in CORRECTED_MT:
        assert cfg['hdim'] % cfg['dhead'] == 0
        cfg['num_heads'] = cfg['hdim'] // cfg['dhead']
        cfg['num_kv_heads'] = cfg['num_heads']
        primary = json.loads(PRIMARY_MT_SOURCE.read_text())
        assert sha(primary['path']) == primary['sha256']
        matches = [row for row in primary['rows'] if row['hidden'] == cfg['hdim']
                   and row['layers'] == cfg['ndec']]
        assert len(matches) == 1 and matches[0]['heads'] == cfg['num_heads']
    assert cfg['hdim'] == cfg['num_heads']*cfg['dhead'], (name, 'hidden/head geometry mismatch')
    assert cfg['num_heads'] == cfg['num_kv_heads']*cfg['gqa_size']
    return cfg


def policy_manifest(additional_sources=()):
    primary = json.loads(PRIMARY_MT_SOURCE.read_text())
    paths = [Path(__file__).resolve(), Path(repair_0908_traces.__file__).resolve(),
             Path(native.__file__).resolve(), STAGE/'src/config.py', STAGE/'src/model.py',
             STAGE/'src/devices.py', CALIBRATION_PATH, PARENT_POLICY_PATH,
             repair_0908_traces.BUFFER_SOURCE, PRIMARY_MT_SOURCE, Path(primary['path']),
             *map(Path, additional_sources)]
    fixes = {}
    for name in CORRECTED_MT:
        old, new = native.geometry(name), geometry(name)
        fixes[name] = dict(hidden=old['hdim'], dhead=old['dhead'],
                          old_heads=old['num_heads'], heads=new['num_heads'],
                          derivation='Source hidden divided by explicit source dhead; matched against primary Table 1 hidden, layers and heads.')
    dimensions = sorted({m['dhead'] for m in [geometry(name) for name in CORRECTED_MT]} |
                        set(json.loads(CALIBRATION_PATH.read_text())['device']['supported_dhead']))
    return dict(schema=REVISION, status='frozen-repair-policy',
        parent_policy_path=str(PARENT_POLICY_PATH), parent_policy_sha256=sha(PARENT_POLICY_PATH),
        calibration_path=str(CALIBRATION_PATH), calibration_sha256=sha(CALIBRATION_PATH),
        coefficient_policy='Reuse unchanged parent coefficients, with the byte-bounded execution schedule; no refit.',
        decision_rule='point-estimate-argmin', geometry_repairs=fixes,
        capacity_by_dhead={str(d): capacity_spec(d) for d in dimensions},
        gpu_gqa=dict(parallelism='Original AttAcc utilization applied to batch times logical query heads.',
                     sharing='KV operand off-HBM traffic divided by GQA group size only if its unique whole-operand footprint fits configured L2.',
                     retained_costs='Per-query FLOPs, scores, probabilities, outputs, L2/L1/register traffic and energy.',
                     limitation='Explicit ideal L2 cache/broadcast residency abstraction; no new kernel or cache-contention benchmark.'),
        limitations=['QK buffer bound does not validate PV/state buffering or pooled double-buffer operation.',
                     'Native command-address defects remain inherited and separately qualified.',
                     'No numerical attention, RoPE arithmetic, online waits, or general DMA contention is added.'],
        sources=[dict(path=str(p.resolve()), sha256=sha(p)) for p in paths])


def write_policy(path=POLICY_PATH, additional_sources=()):
    """Explicit orchestration operation; refuse to overwrite a different freeze."""
    path = Path(path)
    content = encoded(policy_manifest(additional_sources))
    if path.exists():
        assert path.read_bytes() == content, 'Existing correction policy is immutable; use a new revision.'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return path


class LogicalHeadGPU(native.xPU):
    def _get_traffic(self, layer):
        off, l2, l1, reg = super()._get_traffic(layer)
        group = getattr(layer, 'kv_group_size', 1)
        if group == 1:
            return off, l2, l1, reg
        assert layer.type == native.LayerType.MATMUL and layer.numOp % group == 0
        unique_kv_bytes = layer.n*layer.k*layer.dbyte*(layer.numOp//group)
        shared = unique_kv_bytes <= self.l2_cache_size
        original = list(off)
        off = list(off)
        if shared:
            assert off[1] % group == 0
            off[1] //= group
        layer.kv_traffic_audit = dict(group_size=group, logical_head_operations=layer.numOp,
            unique_KV_heads=layer.numOp//group, unique_operand_bytes=unique_kv_bytes,
            configured_L2_bytes=self.l2_cache_size, shared_off_HBM=shared,
            original_off_HBM_operands=original, charged_off_HBM_operands=off,
            charged_L2_operands=list(l2), charged_L1_operands=list(l1),
            scope='Read-only K or V operand shared through ideal L2/cache broadcast; all query-private traffic remains.')
        return off, l2, l1, reg


class Model(native.Model):
    def __init__(self, name, tp, bank, prepare_profiles=False):
        self.name, self.tp, self.bank = name, tp, bank
        self.m = geometry(name)
        self.layers = self.m['ndec']
        assert self.m['num_heads'] % tp == 0 and self.m['num_kv_heads'] % tp == 0
        self.heads = self.m['num_heads']//tp
        self.kvheads = self.m['num_kv_heads']//tp
        self.gqa = self.m['gqa_size']
        self.trace_dhead = self.logical_dhead = self.m['dhead']
        self.dbyte = native.Layer('sum', 'dtype', native.LayerType.FC, True,
                                  native.DataType.W16A16, 1, 1, 1, 1).dbyte
        self.geometry_consistent = True
        self.geometry_scope = 'Repaired head geometry; GPU/PIM share one attention dimension.'
        self.remote_stacks_per_gpu = remote_stacks_per_gpu()
        self.h = math.ceil(self.kvheads/self.remote_stacks_per_gpu)
        self.qbytes = self.heads*self.logical_dhead*self.dbyte
        self.kvbytes = 2*self.kvheads*self.logical_dhead*self.dbyte
        self.mq_capacity = capacity_spec(self.trace_dhead, self.dbyte)['capacity']
        self.gconf = native.make_xpu_config(native.GPUType.A100a, num_gpu=tp)['GPU']
        self.gpu = LogicalHeadGPU(native.DeviceType.GPU, self.gconf, native.SCALING_FACTOR)
        self.pim = native.PIM(native.make_pim_config(native.PIMType.BA, native.InterfaceType.NVLINK3,
                            num_attacc=tp, num_hbm=self.remote_stacks_per_gpu, power_constraint=True), native.SCALING_FACTOR, None)
        self.selector = None
        self.ops, self.shared_cache, self.gpu_traffic_audit = {}, {}, {}
        self.geometry_audit = self.check_geometry()
        if prepare_profiles:
            self.bank.prepare([(1024, self.h, self.mq_capacity), (4096, self.h, self.mq_capacity)],
                              dhead=self.trace_dhead, dbyte=self.dbyte)
            self.selector = PointSelector(CALIBRATION_PATH, self.h, self.gqa, self.trace_dhead, self.dbyte)

    def check_geometry(self):
        assert self.heads*self.logical_dhead == self.m['hdim']//self.tp
        assert self.heads == self.kvheads*self.gqa
        assert self.qbytes == self.m['hdim']//self.tp*self.dbyte
        assert self.kvbytes == 2*self.m['hdim']//self.tp*self.dbyte//self.gqa
        tr = native.Transformer(self.m, tensor_parallel=self.tp)
        tr.build(1, 1, 2, False)
        qkv = next(l for l in tr.sum_decoder if l.name == 'qkv')
        projection = next(l for l in tr.sum_decoder if l.name == 'proj')
        qkv_width = (self.heads+2*self.kvheads)*self.logical_dhead
        assert projection.k == self.heads*self.logical_dhead
        if self.gqa == 1:
            assert qkv.n == qkv_width
        return dict(hidden=self.m['hdim'], heads=self.m['num_heads'], kv_heads=self.m['num_kv_heads'],
                    dhead=self.logical_dhead, tensor_parallel=self.tp, local_query_heads=self.heads,
                    local_kv_heads=self.kvheads, QKV_output_width=qkv_width,
                    PV_output_width=self.heads*self.logical_dhead, projection_input_width=projection.k,
                    Q_bytes_per_token=self.qbytes, KV_bytes_per_token=self.kvbytes,
                    MQ_capacity=self.mq_capacity, passed=True)

    def op(self, stage, name, kind, m, n, k, num=1, weight=False, device='GPU', kv_group=1):
        if name == 'qkv':
            assert n == (self.heads+2*self.kvheads)*self.logical_dhead and k == self.m['hdim']
        elif name == 'proj':
            assert k == self.heads*self.logical_dhead
        elif name in ('score', 'context') and kind == native.LayerType.MATMUL:
            assert num % self.heads == 0 and device == 'GPU'
            assert (k if name == 'score' else n) == self.trace_dhead
            assert kv_group == self.gqa
        key = (stage, name, kind, m, n, k, num, weight, device, kv_group)
        if key not in self.ops:
            layer = native.Layer(stage, name, kind, weight, native.DataType.W16A16, m, n, k, num)
            layer.kv_group_size = kv_group
            target = self.gpu if device == 'GPU' else self.pim
            t, e = target.get_time_and_energy(layer)
            self.ops[key] = t*1e6, sum(e)*1e-6
            if kv_group > 1:
                self.gpu_traffic_audit[str(key)] = dict(shape=dict(m=m, n=n, k=k, numOp=num),
                    flops=layer.get_flops(), **layer.kv_traffic_audit)
        return self.ops[key]

    def gpu_attention(self, q, n, b):
        count = self.heads*b
        values = [self.op('sum', 'score', native.LayerType.MATMUL, q, n, self.logical_dhead,
                          count, kv_group=self.gqa),
                  self.op('sum', 'softmax', native.LayerType.SOFTMAX, q, n, 1, count),
                  self.op('sum', 'context', native.LayerType.MATMUL, q, self.logical_dhead, n,
                          count, kv_group=self.gqa)]
        return tuple(sum(row[i] for row in values) for i in [0, 1])

    def dense_scan(self, q, n, b, mq=False):
        h = math.ceil(self.kvheads*b/self.remote_stacks_per_gpu)
        total = q*self.gqa
        full, tail = divmod(total, self.mq_capacity)
        sizes = ([self.mq_capacity]*full + ([tail] if tail else [])) if mq else [1]*total
        assert sum(sizes) == total and all(0 < size <= self.mq_capacity for size in sizes)
        self.last_dense_schedule = dict(expanded_queries=total, resident_groups=sizes,
            QK_groups=sizes, PV_groups=sizes, heads_per_HBM=h,
            query_capacity=capacity_spec(self.trace_dhead, self.dbyte))
        self.bank.prepare([(n, h, r) for r in set(sizes)], dhead=self.trace_dhead, dbyte=self.dbyte)
        return self.combine([self.bank.get(n, h, r, dhead=self.trace_dhead, dbyte=self.dbyte) for r in sizes])

    def shared_scan(self, objects, views, mq):
        plans = schedule_views(objects, views, self.gqa, self.trace_dhead, self.dbyte, mq)
        self.last_shared_schedule = deepcopy(plans)
        rows = []
        for plan in plans:
            tile = materialize_tile(plan, views, self.gqa, self.mq_capacity)
            key = hashlib.sha256(command_model.canonical([REVISION, self.trace_dhead, self.dbyte,
                                 self.h, self.mq_capacity, objects, tile, plan['mq']])).hexdigest()[:20]
            if key not in self.bank.shared_cache:
                self.bank.shared_cache[key] = self.bank.shared('repair-shared-'+key, objects, tile,
                    self.h, plan['mq'], dhead=self.trace_dhead, dbyte=self.dbyte)
            rows.extend([self.bank.shared_cache[key]]*plan['repeats'])
        return rows[0] if len(rows) == 1 else self.combine(rows)


class PointSelector(PointEstimateSelector):
    def __init__(self, calibration, h, group_size=1, dhead=128, dbyte=2, *, policy=None):
        super().__init__(calibration, h, group_size, dhead, dbyte, policy=PARENT_POLICY_PATH)
        self.parent_policy_sha256 = self.policy_sha256
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
        assert self.policy['schema'] == REVISION and self.policy['status'] == 'frozen-repair-policy'
        assert self.policy['parent_policy_sha256'] == self.parent_policy_sha256
        assert self.policy['calibration_sha256'] == self.calibration_sha256
        for source in self.policy['sources']:
            assert sha(source['path']) == source['sha256'], ('Repair policy source changed', source['path'])
        self.capacity = capacity_spec(dhead, dbyte)['capacity']

    def predict_views(self, objects, views, gpu_us, softmax_us, qin_us, write_us,
                      *, output_us=0., proven_write_overlap_us=0.):
        costs = [gpu_us, softmax_us, qin_us, write_us, output_us, proven_write_overlap_us]
        assert all(math.isfinite(x) and x >= 0 for x in costs)
        c = self.calibration
        plans = schedule_views(objects, views, self.group_size, self.dhead, self.dbyte, True)
        details = []
        for tile in plans:
            counted = command_model.tile_features(tile, self.h, self.dhead, self.dbyte, c['device'])
            cycles = max(1., math.fsum(counted['features'][name]*c['coefficients_cycles'][name]
                                     for name in command_model.FEATURES))
            details.append(dict(schedule=tile, **counted, predicted_cycles=cycles,
                                scan_us=cycles*c['device']['tCK_ns']/1000))
        scan = math.fsum(t['scan_us']*t['schedule']['repeats'] for t in details)
        parts = dict(scan_us=scan, softmax_us=softmax_us, query_and_descriptor_input_us=qin_us,
                     output_us=output_us, exposed_write_us=max(0., write_us-proven_write_overlap_us))
        service = math.fsum(parts.values())
        result = dict(schema=REVISION, scan_us=scan, selection_scan_us=scan,
            service_us=service, raw_service_us=service, gpu_us=gpu_us,
            choice='PIM' if service < gpu_us else 'GPU', components=parts, tiles=details,
            calibration_sha256=self.calibration_sha256, policy_sha256=self.policy_sha256,
            parent_policy_sha256=self.parent_policy_sha256, capacity=capacity_spec(self.dhead, self.dbyte),
            guard_source='Parent training multiplier remains diagnostic; point estimate selects the device.',
            diagnostics=dict(training_guard_scan_us=scan*c['train_guard_multiplier'],
                             training_guard_multiplier=c['train_guard_multiplier']),
            write_overlap_scope='Only supplied proven static write overlap; no candidate timing label.',
            exclusions='No numerical Q/K/V, RoPE, PV buffer validation or DMA contention is introduced.',
            development_scope='Authorized repair; unchanged coefficients with corrected grouping and GPU cost abstraction.')
        self.last_prediction = result
        return result
