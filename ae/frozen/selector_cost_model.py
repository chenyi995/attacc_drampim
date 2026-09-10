#!/usr/bin/env python3
"""Command-aware, shape-only prefill service estimator.

No simulator invocation, trace generation, evaluation label, or evaluation
profile is reachable from prediction. Device constants/coefficients must come
from an independently frozen calibration artifact. The command counting rules
follow the selected AttAcc single-wave generator and current view/tile wrapper.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'command-aware-view-selector-v1'
FEATURES = ('launch', 'mac_interval_work', 'query_writes', 'score_moves',
            'probability_moves', 'context_moves', 'softmax_commands',
            'barrier_rounds', 'row_visits', 'frontend_requests')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ceildiv(value, unit):
    return (value + unit - 1) // unit


def schedule_views(objects, views, group_size=1, capacity=8):
    """Match Model.shared_scan, including its per-reader outer tiling.

    A descriptor with >capacity expanded query heads triggers independent
    reader tiles for *every* view, just as in the executed wrapper. A stream
    can contain several object groups, whose maximum fanout sets nCCDAB.
    """
    assert capacity > 0 and group_size > 0 and objects and views
    ids = [obj['id'] for obj in objects]
    assert len(set(ids)) == len(ids)
    counts = [int(view.get('query_count', 1)) * group_size for view in views]
    assert min(counts) > 0
    all_mq = sum(counts) != 1
    if max(counts) <= capacity:
        raw_tiles = [(list(zip(range(len(views)), views, counts)), 1)]
    else:
        raw_tiles = []
        for index, (view, count) in enumerate(zip(views, counts)):
            full, tail = divmod(count, capacity)
            if full:
                raw_tiles.append(([(index, view, capacity)], full))
            if tail:
                raw_tiles.append(([(index, view, tail)], 1))
    tiles = []
    for members, repeats in raw_tiles:
        uses = {id: 0 for id in ids}
        for index, view, count in members:
            for ref in view['references']:
                assert ref['object'] in uses
                uses[ref['object']] += count
        groups = []
        if all_mq:
            for obj in objects:
                count = uses[obj['id']]
                groups.extend(dict(object=obj['id'], tokens=int(obj['tokens']),
                                   private=bool(obj.get('private', False)), resident_queries=min(capacity, count-start))
                              for start in range(0, count, capacity))
        else:
            for index, view, count in members:
                for _ in range(count):
                    for ref in view['references']:
                        obj = next(obj for obj in objects if obj['id'] == ref['object'])
                        groups.append(dict(object=obj['id'], tokens=int(obj['tokens']),
                                           private=bool(obj.get('private', False)), resident_queries=1))
        assert groups and min(group['tokens'] for group in groups) > 0
        tiles.append(dict(members=[dict(view_index=i, query_count=c) for i, v, c in members], repeats=repeats,
                          query_count=sum(c for i, v, c in members), groups=groups,
                          mq=all_mq, max_resident_queries=max(g['resident_queries'] for g in groups)))
    return tiles


def tile_features(tile, h, dhead, dbyte, device):
    """Analytical native counts. No O(tokens) work and no command trace reads.

    Counts apply to one native head wave. Address-row visits are a structural
    feature, not a claim that all such visits produce an ACT in the simulator.
    Channels execute the same head shape; frontend_requests captures the
    additional channel and all-channel barrier traffic.
    """
    g = device['geometry']
    assert 1 <= h <= g['channels'] and dbyte > 0
    assert dhead > 0 and g['column_bytes'] % dbyte == 0
    token_lanes = g['token_lanes']
    vector_chunks = ceildiv(dhead, g['banks_per_group'] * (g['column_bytes'] // dbyte))
    row_columns = g['row_bytes'] // g['column_bytes']
    # Native QK flushes after token_lanes query-output batches; MVGB uses a
    # column's scalar lanes. These happen to agree at FP16, but are not conflated.
    query_flush = token_lanes
    probability_lanes = g['column_bytes'] // dbyte
    interval = device['interval_cycles'][str(tile['max_resident_queries'])]
    f = dict.fromkeys(FEATURES, 0.)
    f['launch'] = 1.
    commands = Counter()
    barriers = 3  # Explicit QK-complete / softmax-complete / PV-complete rounds.
    for group in tile['groups']:
        n, r = group['tokens'], group['resident_queries']
        token_chunks = ceildiv(n, token_lanes)
        macs_one_phase = token_chunks * vector_chunks
        wr = g['banks_per_group'] * vector_chunks * r
        sb_score = g['bank_groups'] * g['ranks'] * ceildiv(token_chunks, query_flush) * r
        gb = g['ranks'] * g['bank_groups'] * ceildiv(n, token_lanes * probability_lanes) * r
        sb_context = g['banks_per_group'] * g['ranks'] * vector_chunks * r
        object_barriers = 2 + ceildiv(token_chunks, query_flush) + vector_chunks
        f['mac_interval_work'] += 2 * macs_one_phase * interval
        f['query_writes'] += wr
        f['score_moves'] += sb_score
        f['probability_moves'] += gb
        f['context_moves'] += sb_context
        f['row_visits'] += 2 * ceildiv(macs_one_phase, row_columns)
        barriers += object_barriers
        commands.update(PIM_WR_GB=wr*h, PIM_MV_SB=(sb_score+sb_context)*h,
                        PIM_MV_GB=gb*h, PIM_MAC_AB=2*macs_one_phase*h)
    f['softmax_commands'] = tile['query_count']
    f['barrier_rounds'] = barriers
    commands['PIM_SFM'] = tile['query_count'] * h
    commands['PIM_BARRIER'] = barriers * g['channels']
    f['frontend_requests'] = sum(commands.values())
    assert all(math.isfinite(v) and v >= 0 for v in f.values())
    return dict(features=f, command_counts=dict(commands), heads_per_HBM=h,
                dhead=dhead, dbyte=dbyte, interval_cycles=interval)


class CommandAwareSelector:
    def __init__(self, calibration, h, group_size=1, dhead=128, dbyte=2):
        if isinstance(calibration, (str, Path)):
            self.calibration_path = str(Path(calibration).resolve())
            self.calibration_sha256 = sha(calibration)
            calibration = json.loads(Path(calibration).read_text())
        else:
            self.calibration_path = None
            self.calibration_sha256 = hashlib.sha256(canonical(calibration)).hexdigest()
        assert calibration['schema'] == SCHEMA and calibration['status'] == 'frozen-trained'
        assert calibration['features'] == list(FEATURES)
        assert set(calibration['coefficients_cycles']) == set(FEATURES)
        assert all(math.isfinite(v) and v >= 0 for v in calibration['coefficients_cycles'].values())
        assert calibration['prediction_script_sha256'] == sha(__file__), 'Prediction implementation changed after coefficient freeze.'
        self.calibration = calibration
        self.h = int(h); self.group_size = int(group_size); self.dhead = int(dhead); self.dbyte = int(dbyte)
        assert self.dhead in calibration['device']['supported_dhead']
        assert self.dbyte in calibration['device']['supported_dbyte']
        self.last_prediction = None

    def predict_views(self, objects, views, gpu_us, softmax_us, qin_us, write_us,
                      *, output_us=0., proven_write_overlap_us=0.):
        """qin_us may already include output for the legacy choose_views API.

        proven_write_overlap_us must be a static dependency lower bound, never
        a measured/simulated first-private timestamp from this candidate. The
        default charges all writes. Confidence guard is frozen from train only.
        """
        costs = dict(gpu_us=gpu_us, softmax_us=softmax_us, query_input_us=qin_us,
                     output_us=output_us, write_us=write_us, proven_write_overlap_us=proven_write_overlap_us)
        assert all(math.isfinite(value) and value >= 0 for value in costs.values())
        c = self.calibration; device = c['device']
        tiles = schedule_views(objects, views, self.group_size, device['capacity'])
        details = []
        for tile in tiles:
            counted = tile_features(tile, self.h, self.dhead, self.dbyte, device)
            cycles = max(1., math.fsum(counted['features'][name] * c['coefficients_cycles'][name] for name in FEATURES))
            details.append(dict(schedule=tile, **counted, predicted_cycles=cycles,
                                scan_us=cycles * device['tCK_ns'] / 1000))
        scan = math.fsum(row['scan_us'] * row['schedule']['repeats'] for row in details)
        selection_scan = scan * c['train_guard_multiplier']
        exposed = max(0., write_us - proven_write_overlap_us)
        components = dict(scan_us=selection_scan, softmax_us=softmax_us,
                          query_and_descriptor_input_us=qin_us, output_us=output_us, exposed_write_us=exposed)
        service = math.fsum(components.values())
        result = dict(schema=SCHEMA, scan_us=scan, selection_scan_us=selection_scan,
                      service_us=service, raw_service_us=scan+softmax_us+qin_us+output_us+exposed,
                      choice='PIM' if service < gpu_us else 'GPU', gpu_us=gpu_us,
                      components=components, tiles=details,
                      calibration_sha256=self.calibration_sha256,
                      guard_source='Maximum underprediction multiplier on calibration training traces only.',
                      write_overlap_scope='Conservative full write by default; explicit static dependency lower bound only.',
                      exclusions='No RoPE arithmetic, numerical Q/K/V, online waiting or general DMA contention is introduced by this estimator.')
        self.last_prediction = result
        return result

    def choose_views(self, objects, views, gpu_us, softmax_us, qin_us, write_us):
        row = self.predict_views(objects, views, gpu_us, softmax_us, qin_us, write_us)
        return row['choice'], row['service_us']
