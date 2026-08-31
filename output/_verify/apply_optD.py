#!/usr/bin/env python3
"""Incrementalise the decode loop.

Physics: decode step n+1 differs from step n by exactly ONE resident KV row --
the token just generated.  The loop nevertheless rebuilt, for every
(output token, layer, request), the whole history: an O(L) list concatenation
and an O(L) physical-read expansion whose prefix never changes.  It also
recomputed the prefill position-delta map, whose arguments do not depend on the
output row at all.

All three are replaced by state carried across steps.  Nothing about the values
changes: the list is built from the same elements in the same order, the
physical-read expansion is a per-element order-preserving map, and the delta map
is a pure function of (request, bindings[request][layer]).
"""
p = "src/workload_runner.py"
s = open(p).read()

# ---- D1/D2: carry decode state across output rows --------------------------
old_init = '''    previous_output: Dict[str, Dict[int, List[KVLocation]]] = {
        request.request_id: {layer: [] for layer in range(system.model.ndec)}
        for request in requests
    }
'''
new_init = '''    previous_output: Dict[str, Dict[int, List[KVLocation]]] = {
        request.request_id: {layer: [] for layer in range(system.model.ndec)}
        for request in requests
    }
    # Decode appends exactly one resident KV row per output token, so the
    # per-step rebuild of the whole history was recomputing an unchanged
    # prefix.  These carry that state instead.  ``_physical_reads`` is a
    # per-element order-preserving expansion and ``_prefill_location_deltas``
    # is a pure function of (request, its layer bindings), so the values are
    # the same ones the rebuild produced.  Every consumer of these structures
    # reads them (checked: no call site mutates the list, the mask set, or the
    # delta map), which is what makes sharing them across steps safe.
    _old_state: Dict[str, Dict[int, List[KVLocation]]] = {
        request.request_id: {} for request in requests
    }
    _reads_state: Dict[str, Dict[int, Tuple[List[KVLocation], set]]] = {
        request.request_id: {} for request in requests
    }
    _delta_state: Dict[Tuple[str, int], Dict[Tuple[int, int], int]] = {}
    _shadow_gate = getattr(tlb, "shadow_reads", True)

    def _decode_state(request_id: str, layer: int):
        """(resident locations, (physical reads, masked keys)) for this layer."""
        per_layer = _old_state[request_id]
        if layer not in per_layer:
            locations = [location for _, _, _, location
                         in bindings[request_id][layer]]
            locations.extend(previous_output[request_id][layer])
            per_layer[layer] = locations
            _reads_state[request_id][layer] = _pool_reads(tlb, locations)
        return per_layer[layer], _reads_state[request_id][layer]

    def _decode_state_append(request_id: str, layer: int,
                             location: KVLocation) -> None:
        """Mirror one generated KV row into the carried state."""
        per_layer = _old_state[request_id]
        if layer not in per_layer:
            return                      # not materialised yet; lazy init folds it in
        per_layer[layer].append(location)
        reads, masked = _reads_state[request_id][layer]
        reads.append(location)
        if _shadow_gate and location.shadow is not None:
            reads.append(location.shadow)
            masked.add(_address_key(location.shadow))

    def _decode_deltas(request, layer: int):
        """Prefill position deltas: independent of the output row."""
        key = (request.request_id, layer)
        cached = _delta_state.get(key)
        if cached is None:
            cached = _prefill_location_deltas(
                request, bindings[request.request_id][layer])
            _delta_state[key] = cached
        return cached
'''
assert s.count(old_init) == 1, "D1 init anchor"
s = s.replace(old_init, new_init)

old_build = '''            old_by_request = {
                request.request_id: [location for _, _, _, location
                                     in bindings[request.request_id][layer_index]] +
                previous_output[request.request_id][layer_index]
                for request in active
            }
            # Physical master/diff streams per request (masked shadow rows
            # included) -- see ``_physical_reads``.
            reads_by_request = {
                request.request_id: _pool_reads(tlb, old_by_request[request.request_id])
                for request in active
            }
'''
new_build = '''            # Carried across output rows -- see _decode_state.  The rebuild
            # this replaces was O(L) per (token, layer, request).
            old_by_request = {}
            reads_by_request = {}
            for request in active:
                _rid = request.request_id
                old_by_request[_rid], reads_by_request[_rid] = _decode_state(
                    _rid, layer_index)
'''
assert s.count(old_build) == 1, "D1 build anchor"
s = s.replace(old_build, new_build)

old_delta = '''                    location_deltas=_prefill_location_deltas(
                        request, bindings[request.request_id][layer_index]),
'''
new_delta = '''                    location_deltas=_decode_deltas(request, layer_index),
'''
assert s.count(old_delta) == 1, "D2 anchor"
s = s.replace(old_delta, new_delta)

old_append = '''                    previous_output[request_id][layer_index].append(location)
'''
new_append = '''                    previous_output[request_id][layer_index].append(location)
                    _decode_state_append(request_id, layer_index, location)
'''
assert s.count(old_append) == 1, "D1 append anchor"
s = s.replace(old_append, new_append)

# ---- D3: same throwaway-set bug as the one already fixed at the private read
old_common = '''                common = [location for location in reads_by_request[group[0].request_id][0]
                          if _address_key(location) in (common_keys or set())]
'''
new_common = '''                _common_addresses = common_keys or frozenset()
                common = [location for location in reads_by_request[group[0].request_id][0]
                          if (location.key_address, location.value_address)
                          in _common_addresses]
'''
assert s.count(old_common) == 1, "D3 anchor"
s = s.replace(old_common, new_common)

open(p, "w").write(s)
print("applied D1 (incremental decode state) + D2 (delta memo) + D3 (set hoist)")
