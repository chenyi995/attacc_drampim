#!/usr/bin/env python3
"""Byte-bounded MQ command adaptation; the frozen native generator is unchanged.

This module does not run anything at import time. TraceBank runs Ramulator only
when its caller explicitly requests a profile that is absent from the exact
command cache. The capacity is the QK compute-side query budget, not a claim of
validated PV buffering or pooling of the two native double-buffer halves.
"""
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / 'archive/coverage-expansion/attacc-fugue'
if 'fugue.kvchime_traces' not in sys.modules:
    sys.path.insert(0, str(STAGE))
from fugue import kvchime_traces as frozen_traces
from fugue.attention import CAP as FROZEN_CAP
from coverage_trace_cache import ExactTraceBank
import selector_cost_model as command_model

REVISION = 'repair-0908-byte-bounded-mq-v1'
BUFFER_SOURCE = ROOT / 'archive/attacc-fugue/Fugue-paper/Fugue-asplos-experiment-1-prefill-attention-crossover/README.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def capacity_spec(dhead, dbyte=2):
    """Use the declared bank query budget and native padded column width."""
    g = frozen_traces.LAYOUT_GEOMETRY
    buffer_bytes = int(re.search(r'MQ \u6bcf bank (\d+) B buffer', BUFFER_SOURCE.read_text())[1])
    assert dhead > 0 and dbyte > 0 and g['column_bytes'] % dbyte == 0
    bank_lanes = g['column_bytes'] // dbyte
    columns = math.ceil(dhead / (g['banks_per_group'] * bank_lanes))
    query_bytes = columns * g['column_bytes']
    limit = min(FROZEN_CAP, buffer_bytes // query_bytes)
    if limit < 1:
        raise ValueError('One padded query exceeds the declared compute-side bank buffer.')
    return dict(dhead=dhead, dbyte=dbyte, buffer_bytes=buffer_bytes,
                bank_query_columns=columns, padded_query_bytes=query_bytes,
                unpadded_query_bytes=dhead*dbyte/g['banks_per_group'], capacity=limit,
                interval_model_limit=FROZEN_CAP,
                source_path=str(BUFFER_SOURCE), source_sha256=sha(BUFFER_SOURCE),
                scope='QK query compute-side budget; PV/state buffering and double-buffer pooling are not validated.')


def schedule_views(objects, views, group_size=1, dhead=128, dbyte=2, mq=True):
    """One schedule supplies both predictor features and executed query tiles."""
    cap = capacity_spec(dhead, dbyte)['capacity']
    plans = command_model.schedule_views(objects, views, group_size, cap)
    cursors = [0] * len(views)
    for plan in plans:
        for member in plan['members']:
            index = member['view_index']
            member['query_start'] = cursors[index]
            member['repeat_query_stride'] = member['query_count']
            cursors[index] += member['query_count'] * plan['repeats']
    assert cursors == [int(v.get('query_count', 1))*group_size for v in views]
    if not mq:
        by_id = {o['id']: o for o in objects}
        for plan in plans:
            plan['mq'] = False
            plan['groups'] = []
            for member in plan['members']:
                view = views[member['view_index']]
                for _ in range(member['query_count']):
                    for ref in view['references']:
                        obj = by_id[ref['object']]
                        plan['groups'].append(dict(object=obj['id'], tokens=int(obj['tokens']),
                            private=bool(obj.get('private', False)), resident_queries=1))
            plan['max_resident_queries'] = 1
    return plans


def query_assignments(plan, repeat_index=0, group_size=1):
    """Map tile-local states to original query/head identities, without timing."""
    assert 0 <= repeat_index < plan['repeats'] and group_size > 0
    result = []
    for member in plan['members']:
        start = member['query_start'] + repeat_index*member['repeat_query_stride']
        for expanded in range(start, start+member['query_count']):
            result.append(dict(tile_query=len(result), view_index=member['view_index'],
                               expanded_query=expanded, query_index=expanded//group_size,
                               query_head_in_KV_group=expanded % group_size))
    return result


def materialize_tile(plan, views, group_size, capacity):
    """Preserve the frozen rectangular-tiling position convention explicitly."""
    outer_tiled = max(int(v.get('query_count', 1))*group_size for v in views) > capacity
    result = []
    for member in plan['members']:
        source = views[member['view_index']]
        tile = {key: deepcopy(value) for key, value in source.items() if key != 'query_positions'}
        tile['query_count'] = member['query_count']
        tile['source_query_state'] = dict(view_index=member['view_index'],
            expanded_query_start=member['query_start'], repeat_count=plan['repeats'],
            repeat_query_stride=member['repeat_query_stride'], group_size=group_size,
            scope='State correspondence for repeated timing tiles, not simulated numerical operands.')
        if outer_tiled:
            tile['query_positions'] = []
            tile['position_scope'] = 'Rectangular timing tile; source logical query positions remain in the workload.'
        else:
            tile['query_positions'] = [p for p in source.get('query_positions', []) for _ in range(group_size)]
        result.append(tile)
    return result


def build_shared_commands(base_loader, objects, views, h, mq, dhead=128, dbyte=2):
    """Compose existing native object traces; no simulator or filesystem writes."""
    g = frozen_traces.LAYOUT_GEOMETRY
    assert 0 < h <= g['channels']
    spec = capacity_spec(dhead, dbyte)
    cap = spec['capacity']
    allocation = frozen_traces.object_layout(objects, dhead, dbyte)
    refs = {o['id']: [] for o in objects}
    query_ranges, nqueries = [], 0
    for view in views:
        qs = list(range(nqueries, nqueries + view.get('query_count', 1)))
        query_ranges.append(qs)
        nqueries += len(qs)
        positions = []
        for ref in view['references']:
            refs[ref['object']].extend(qs)
            positions.extend(ref['valid_logical_positions'])
        assert sorted(positions) == list(range(view['logical_length']))
    phases, sfm = {}, None
    for obj in objects:
        source = base_loader(obj['tokens'], h, dhead, dbyte)
        markers = [i for i, line in enumerate(source) if line.startswith('PIM_SFM')]
        assert len(markers) == h and markers == list(range(markers[0], markers[-1]+1))
        phases[obj['id']] = {'QK': source[:markers[0]], 'PV': source[markers[-1]+1:]}
        sfm = source[markers[0]:markers[-1]+1]
    lines, residents, blocks = [], [], []
    barriers = [f'PIM_BARRIER 0x{ch*g["channel_bytes"]:08x}' for ch in range(g['channels'])]

    def emit(oid, queries, phase):
        r, start = len(queries), len(lines)
        assert 1 <= r <= cap and r*spec['padded_query_bytes'] <= spec['buffer_bytes']
        offset = allocation[oid]['K_base']
        for line in phases[oid][phase]:
            op, address = line.split()
            address = int(address, 16) + (0 if op == 'PIM_BARRIER' else offset)
            command = f'{op} 0x{address:08x}'
            if op == 'PIM_MAC_AB':
                lines.append(command)
                residents.append(r)
            elif op == 'PIM_BARRIER':
                lines.append(command)
            else:
                lines.extend([command]*r)
        blocks.append(dict(phase=phase, object=oid, queries=queries, resident_queries=r,
                           command_start=start, command_end=len(lines), row_base_bytes=offset))

    for phase in ['QK', 'PV']:
        if mq:
            for obj in objects:
                qs = refs[obj['id']]
                for start in range(0, len(qs), cap):
                    emit(obj['id'], qs[start:start+cap], phase)
        else:
            for qs, view in zip(query_ranges, views):
                for query in qs:
                    for ref in view['references']:
                        emit(ref['object'], [query], phase)
        lines.extend(barriers)
        if phase == 'QK':
            for _ in range(nqueries):
                lines.extend(sfm)
            lines.extend(barriers)
    metadata = dict(objects=objects, views=views, query_ranges=query_ranges, blocks=blocks,
        physical_layout=allocation, dhead=dhead, dbyte=dbyte,
        layout_revision=frozen_traces.LAYOUT_REVISION, execution_revision=REVISION,
        query_capacity=spec, repair_source_sha256=sha(__file__),
        semantics='All QK before query-global softmax, then all PV; private positions replace old positions.',
        timing='Frozen native object traces and maximum resident interval per mixed stream; byte-bounded query groups.',
        arithmetic='Command timing and logical descriptors; no numerical Q/K/V or PV buffer-capacity validation.')
    return lines, residents, metadata


class TraceBank(ExactTraceBank):
    """Reuse the unchanged physical exact cache, with repaired logical schedules."""
    def prepare(self, keys, dhead=128, dbyte=2):
        keys = list(keys)
        cap = capacity_spec(dhead, dbyte)['capacity']
        assert all(1 <= r <= cap for n, h, r in keys), 'Query residents exceed the padded bank budget.'
        return super().prepare(keys, dhead, dbyte)

    def shared(self, name, objects, views, h, mq, dhead=128, dbyte=2):
        lines, residents, metadata = build_shared_commands(self.base, objects, views, h, mq, dhead, dbyte)
        row = self.run(name, lines, residents, metadata)
        row['logical_tokens'] = sum(v['logical_length']*v.get('query_count', 1) for v in views)
        row['object_tokens'] = sum(o['tokens'] for o in objects)
        row['scan_kind'] = 'shared_MQ' if mq else 'shared_single_query'
        row['query_capacity'] = metadata['query_capacity']
        row['execution_revision'] = REVISION
        return row
