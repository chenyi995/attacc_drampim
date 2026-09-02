"""DAG-free input enumeration for the A1 private-KV baseline.

This module deliberately derives PIM work from the A1 algorithmic contract,
not from ``SplitEvent`` objects, TLB entries, or an existing DAG.  It is the
front end for a later macro scheduler: identical signatures are aggregated
before the calibrated analytic PIM model is consulted.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .ramulator_wrapper import MQ_DEFAULT_GEMV_BUFFER_BYTES, mq_query_capacity


_CHANNELS = 16
_CHUNK_ROWS = 256
_HBM_CHANNEL_BYTES = 1 << 30
_KV_GAP_BYTES = 1 << 23


@dataclass(frozen=True)
class A1PimInput:
    """One analytically-derived PIM invocation, before timing evaluation.

    ``queries``/``kv_heads`` and ``energy_replicas`` are carried because the
    DAG's ENERGY account needs them and they cannot be recovered from the
    Ramulator signature: the ALU term is a function of the logical GEMM shape
    (m*n*k*numOp), and the decode placement scan is charged once per HBM that
    holds a copy of the KV.  ``Ramulator.PRICING_FIELDS`` drops both, so a
    signature-log comparison can check the work but not the energy.
    """
    phase: str
    run_length: int
    num_ops_per_hbm: int
    dbyte: int
    dhead: int
    channel_count: int
    shared_queries: int
    channel_base: int
    key_addr: int
    value_addr: int
    queries: int = 1              # op.m -- resident query rows in this sweep
    kv_heads: int = 1             # op.numOp -- KV heads the op stands for
    energy_replicas: int = 1      # extra whole-run multiplier (num_hbm_used)
    shared_kv: bool = False       # mirrors op.pim_shared_kv, not q > 1

    def signature(self, *, pim_type: str = "BA", power_constraint: bool = True,
                  num_hbm: int = 5) -> Tuple:
        """Primitive analytic-PIM input, in Ramulator signature field order."""
        return (pim_type, self.run_length, self.num_ops_per_hbm, self.dbyte,
                power_constraint, self.dhead, num_hbm, self.channel_count,
                self.shared_kv, self.shared_queries, self.channel_base,
                False, None, self.key_addr, self.value_addr, self.phase,
                "chunkstripe1")


def _ceil(x: int, y: int) -> int:
    return -(-int(x) // int(y))


def _slice_channel_rows(context_rows: int, heads_per_hbm: int) -> Tuple[Tuple[int, int], ...]:
    """Return ``(channel, folded_rows)`` for private 16-channel decode.

    A1's private layout uses the original AttAcc ``slice`` rule: each KV head
    occupies a channel slice and 256-row chunks rotate within that slice.

    The rotation is closed-form rather than a loop over chunks: within a head's
    stripe, ``chunk % stripe`` visits each offset ``chunks // stripe`` times,
    plus one more for the first ``chunks % stripe`` offsets.  That makes the
    cost O(16) instead of O(heads x chunks) -- the loop version re-derived the
    same answer for every one of millions of decode steps.
    """
    return _slice_by_chunks(_ceil(context_rows, _CHUNK_ROWS), heads_per_hbm)


def _slice_by_chunks(chunks: int, heads_per_hbm: int) -> Tuple[Tuple[int, int], ...]:
    heads = max(1, int(heads_per_hbm))
    stripe = max(1, _CHANNELS // heads)
    whole, remainder = divmod(int(chunks), stripe)
    loads = [0] * _CHANNELS
    for head in range(heads):
        base = (head * stripe) % _CHANNELS
        for offset in range(stripe):
            loads[(base + offset) % _CHANNELS] += whole + (1 if offset < remainder else 0)
    return tuple((channel, load * _CHUNK_ROWS)
                 for channel, load in enumerate(loads) if load)


def enumerate_a1_pim_inputs(workload, model, *, dbyte: int,
                            num_hbm: int = 5,
                            gemv_buffer_bytes: int = MQ_DEFAULT_GEMV_BUFFER_BYTES
                            ) -> Dict[A1PimInput, int]:
    """Enumerate and aggregate A1 PIM inputs without creating a DAG.

    The model must expose ``ndec``, ``dhead``, ``num_heads``, ``tp`` and
    ``gqa_size`` (the normal :class:`Transformer` interface).  The result's
    multiplicity is the number of algorithmic sweeps/runs, so it is usually
    tiny even for workloads with millions of would-be DAG nodes.
    """
    q_heads = max(1, int(model.num_heads) // max(1, int(model.tp)))
    gqa = max(1, int(getattr(model, "gqa_size", 1) or 1))
    kv_heads = max(1, q_heads // gqa)
    heads_per_hbm = _ceil(kv_heads, max(1, int(num_hbm)))
    # HBM stacks that actually hold a copy of this request's KV.  Only the
    # energy account uses it: the busiest channel sets the time, but every
    # stack pays.
    hbm_used = max(1, min(int(num_hbm), _ceil(kv_heads, heads_per_hbm)))
    cap_rows = max(1, mq_query_capacity(gemv_buffer_bytes) // gqa)
    result: Counter[A1PimInput] = Counter()
    decode: Counter = Counter()
    slice_cache: Dict[int, Tuple[Tuple[int, int], ...]] = {}
    # Rank once, O(R log R), instead of an O(R^2) scan inside the request loop.
    ordinals = {request.request_id: index for index, request in
                enumerate(sorted(workload.requests, key=lambda r: r.request_id))}

    for request in workload.requests:
        prefill_rows = int(request.total_length)
        resident_rows = prefill_rows + int(request.history_len)
        full, tail = divmod(prefill_rows, cap_rows)
        # A private extent is affine.  Absolute bases distinguish channel
        # pools for auditability but are not obtained from a TLB/DAG.
        request_ordinal = ordinals[request.request_id]
        for layer in range(int(model.ndec)):
            base = (layer * max(1, len(workload.requests)) + request_ordinal) * (2 * _KV_GAP_BYTES)
            # ``shared_kv`` is True for EVERY prefill sweep, mirroring
            # workload_runner's ``op.pim_shared_kv = True``.  Inferring it from
            # ``shared_queries > 1`` is wrong for the one-row tail sweep.
            if full:
                result[A1PimInput("full", resident_rows, heads_per_hbm, dbyte,
                                  model.dhead, _CHANNELS, cap_rows * gqa, 0,
                                  base, base + _KV_GAP_BYTES,
                                  queries=cap_rows, kv_heads=kv_heads,
                                  energy_replicas=1, shared_kv=True)] += full
            if tail:
                result[A1PimInput("full", resident_rows, heads_per_hbm, dbyte,
                                  model.dhead, _CHANNELS, tail * gqa, 0,
                                  base, base + _KV_GAP_BYTES,
                                  queries=tail, kv_heads=kv_heads,
                                  energy_replicas=1, shared_kv=True)] += 1

        # Decode attends over prefill/history plus the already-generated rows.
        # Each active private-layout channel is one independent head-folded
        # run.  No field of that run depends on the layer or on the request, so
        # the per-layer and per-output-token loops collapse into a walk over
        # 256-row chunk bands: within a band the channel loads are constant, so
        # one band contributes ``rows_in_band * ndec`` to each of its channels.
        # The loop version built one frozen dataclass per PHYSICAL RUN, which
        # cost 107 s on wl_N64 and 352 s on wl_N64/GPT-175B.
        if request.lout > 0:
            first = resident_rows
            last = resident_rows + int(request.lout) - 1
            for band in range(_ceil(first, _CHUNK_ROWS), _ceil(last, _CHUNK_ROWS) + 1):
                low = max(first, (band - 1) * _CHUNK_ROWS + 1)
                high = min(last, band * _CHUNK_ROWS)
                rows_in_band = high - low + 1
                if rows_in_band <= 0:
                    continue
                pairs = slice_cache.get(band)
                if pairs is None:
                    pairs = slice_cache[band] = _slice_by_chunks(band, heads_per_hbm)
                weight = rows_in_band * int(model.ndec)
                for pair in pairs:
                    decode[pair] += weight

    for (channel, folded_rows), count in decode.items():
        key = channel * _HBM_CHANNEL_BYTES
        # numOp = 1: heads are folded into the row count.  The energy, however,
        # is every HBM's copy of the KV, so the DAG multiplies the priced run
        # by num_hbm_used (workload_runner _append_placement_pim_scan).
        result[A1PimInput("full", folded_rows, 1, dbyte, model.dhead, 1, gqa,
                          channel, key, key + _KV_GAP_BYTES,
                          queries=1, kv_heads=1, energy_replicas=hbm_used,
                          shared_kv=gqa > 1)] += count
    return dict(result)


# Bytes moved per PIM command, and the DRAM fan-out of an all-bank MAC.
# Both are read off ``Ramulator.run``'s ``postprocess``, not re-derived.
_BYTES_PER_COMMAND = 32
_MAC_BANK_FANOUT = {"BA": 2 * 2 * 4 * 4, "BG": 2 * 2 * 4}


def pim_energy_nj(item: A1PimInput, counts: Sequence[int], *, num_hbm: int,
                  num_attacc: int, energy_table: Mapping,
                  pim_type: str = "BA") -> float:
    """PIM energy for one priced invocation, in nJ.

    This reimplements ``Ramulator.run``'s ``postprocess`` followed by
    ``devices.PIM.get_time_and_energy``.  It is a transcription on purpose:
    the point of the comparison is that the two engines charge the SAME
    energy for the same work, so the enumerator must not invent its own
    accounting.  Note what it implies -- energy is a function of the command
    counts and the logical GEMM shape only.  ``cycle`` never enters it, and
    ``sfm`` is not used at all.  Energy agreement therefore tests Layer 1,
    never the timing model.
    """
    mac, _sfm, mvgb, mvsb, wrgb = counts
    moves = wrgb + mvsb + mvgb
    si_io = wrgb * _BYTES_PER_COMMAND
    bus_io = moves * _BYTES_PER_COMMAND
    mem_acc = mac * _BYTES_PER_COMMAND * _MAC_BANK_FANOUT.get(pim_type, 2)
    traffic = [value * num_hbm for value in (si_io, bus_io, bus_io, bus_io, mem_acc)]
    io_table = energy_table["io"]
    io_energy = sum(traffic[i] * io_table[i] for i in range(len(io_table)))
    dram_energy = traffic[-1] * energy_table["mem"] + io_energy
    # get_flops()/2 * alu for a MATMUL, i.e. m*n*k*numOp * alu.
    cal_energy = (item.queries * item.run_length * item.dhead * item.kv_heads
                  * energy_table["alu"])
    picojoules = (dram_energy + cal_energy) * num_attacc
    return picojoules / 1000.0 * max(1, int(item.energy_replicas))


def input_summary(inputs: Mapping[A1PimInput, int]) -> Dict[str, int]:
    """Small audit summary suitable for a DAG-free report."""
    return {
        "unique_signatures": len(inputs),
        "physical_runs": sum(inputs.values()),
        "prefill_sweeps": sum(count for item, count in inputs.items()
                              if item.channel_count == _CHANNELS),
        "decode_channel_runs": sum(count for item, count in inputs.items()
                                   if item.channel_count == 1),
    }
