"""Execution adapters for JSON workloads.

The legacy ``System.simulate`` API models one rectangular request.  This
module retains that API for the no-reuse baseline and adds the trace-informed
GPU/PIM prefill split needed for position-independent KV reuse.
"""

from __future__ import annotations

import math
from array import array
from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .model import Layer
from .ramulator_wrapper import (MQ_DEFAULT_GEMV_BUFFER_BYTES,
                                MQ_DEFAULT_PE_FREQ_GHZ, mq_query_capacity)
from .type import DeviceType, LayerType
from .workload import (ReusePlan, Workload, WorkloadValidationError,
                       validate_reuse_plan)


@dataclass(frozen=True)
class SplitEvent:
    event_id: str
    transformer_layer: int
    tier: int
    request_id: str
    name: str
    device: str
    rows: int
    time_s: float
    energy_nj: float
    link_bytes: int = 0
    depends_on: Tuple[str, ...] = ()
    query_positions: Tuple[int, ...] = ()
    dram_addresses: Tuple[int, ...] = ()
    # Empty for the original per-request events.  A non-empty list identifies
    # a single physical operation that serves several independent agents.
    batch_members: Tuple[str, ...] = ()
    # Physically read K/V rows that carry a softmax mask (shadowed master
    # rows of a CacheBlend/EPIC correction).  They cost DRAM bandwidth and
    # MAC cycles but contribute no score.  Always <= rows.
    masked_rows: int = 0
    start_s: float = 0.0
    end_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.event_id,
            "transformer_layer": self.transformer_layer,
            "tier": self.tier,
            "request": self.request_id,
            "name": self.name,
            "device": self.device,
            "rows": self.rows,
            "time_s": self.time_s,
            "energy_nj": self.energy_nj,
            "link_bytes": self.link_bytes,
            "depends_on": list(self.depends_on),
            "query_positions": list(self.query_positions),
            "dram_addresses": ["0x{:x}".format(address)
                               for address in self.dram_addresses],
            "batch_members": list(self.batch_members),
            "masked_rows": self.masked_rows,
            "start_s": self.start_s,
            "end_s": self.end_s,
        }


def _tier_shapes(workload: Workload,
                 batch_size: Optional[int] = None) -> Iterable[Tuple[int, Tuple[Any, ...], int, int]]:
    """Padded legacy batches per dependency tier.

    ``batch_size`` None (default) keeps the original behaviour: one batch per
    tier.  Otherwise a tier is served as consecutive batches of at most
    ``batch_size`` requests (in workload order), each padded to its own
    maxima, so the legacy model uses the same batch bound as the DAG path.
    """
    for tier, requests in workload.tiers.items():
        chunk = len(requests) if not batch_size else batch_size
        for start in range(0, len(requests), chunk):
            group = tuple(requests[start:start + chunk])
            yield (tier, group, max(request.total_length for request in group),
                   max(request.lout for request in group))


def run_no_reuse_workload(system, workload: Workload, *, pipe: bool,
                          parallel_ff: bool, power_constraint: bool,
                          batch_size: Optional[int] = None) -> List[Any]:
    """Run every tier through the unchanged legacy simulator.

    A tier is the batch unit.  Requests with distinct shapes are padded to the
    tier maxima, exactly as a rectangular serving batch would be.  Therefore a
    one-tier uniform workload calls ``System.simulate`` with exactly the same
    arguments as the old CLI command.
    """
    perfs: List[Any] = []
    for _, requests, lin, lout in _tier_shapes(workload, batch_size):
        system.simulate(len(requests), lin, lout, perfs=perfs, pipe=pipe,
                        parallel_ff=parallel_ff,
                        power_constraint=power_constraint)
    return perfs


def run_no_reuse_report(system, workload: Workload, *, pipe: bool,
                        parallel_ff: bool,
                        power_constraint: bool,
                        batch_size: Optional[int] = None) -> Tuple[List[Any], Dict[str, Any]]:
    """Run the legacy baseline and expose a DAG-comparable wall-clock summary.

    ``System.simulate`` returns the original CSV-oriented record: summarization
    time for the whole tier followed by generation time *per decode step*.
    Reconstructing the latter over ``lout - 1`` gives the legacy simulator's
    full tier duration.  Dependency tiers are submitted serially by the
    workload adapter, so their durations sum to the baseline makespan.

    This intentionally does not invent CacheBlend-only fields such as physical
    TLB addresses or GPU--PIM link bytes.  The legacy energy record is decode
    energy per generated step, so it is reported under that precise name.
    """
    records: List[Any] = []
    tier_reports: List[Dict[str, Any]] = []
    makespan_s = 0.0
    decode_energy_nj = 0.0
    prefill_energy_nj = 0.0

    # The order is defined by System.simulate's legacy dictionaries in
    # src/system.py.  Keep the indices explicit here so the report remains
    # stable even though callers still receive the untouched CSV record.
    summarization_all_index = 0
    generation_all_index = 7

    for tier, requests, lin, lout in _tier_shapes(workload, batch_size):
        tier_records: List[Any] = []
        system.simulate(len(requests), lin, lout, perfs=tier_records,
                        pipe=pipe, parallel_ff=parallel_ff,
                        power_constraint=power_constraint)
        if len(tier_records) != 1:
            raise RuntimeError("legacy simulation must emit one record per tier")
        record = tier_records[0]
        records.append(record)
        _, _, perf_ms, energy_nj_per_step = record
        if lout < 2:
            raise WorkloadValidationError(
                "no-reuse end-to-end reporting requires lout >= 2")

        decode_steps = lout - 1
        prefill_s = perf_ms[summarization_all_index] / 1000.0
        decode_per_token_s = perf_ms[generation_all_index] / 1000.0
        decode_s = decode_per_token_s * decode_steps
        tier_duration_s = prefill_s + decode_s
        tier_decode_energy_nj = energy_nj_per_step[0] * decode_steps
        tier_prefill_energy_nj = getattr(
            system, "last_simulation_summary", {}).get("prefill_energy_nj", 0.0)
        makespan_s += tier_duration_s
        decode_energy_nj += tier_decode_energy_nj
        prefill_energy_nj += tier_prefill_energy_nj
        tier_reports.append({
            "tier": tier,
            "batch_size": len(requests),
            "members": [request.request_id for request in requests],
            "lin": lin,
            "lout": lout,
            "decode_steps": decode_steps,
            "prefill_s": prefill_s,
            "decode_per_token_s": decode_per_token_s,
            "decode_s": decode_s,
            "duration_s": tier_duration_s,
            "prefill_energy_nj": tier_prefill_energy_nj,
            "decode_energy_nj": tier_decode_energy_nj,
            "energy_nj": tier_prefill_energy_nj + tier_decode_energy_nj,
        })

    return records, {
        "policy": "no-reuse",
        "scheduling": ("legacy rectangular batch per dependency tier; tiers serial"
                       if not batch_size else
                       "legacy rectangular batches of <= {} requests per tier; "
                       "batches and tiers serial".format(batch_size)),
        "batch_size": batch_size,
        "tiers": tier_reports,
        "makespan_s": makespan_s,
        "prefill_energy_nj": prefill_energy_nj,
        "decode_energy_nj": decode_energy_nj,
        "energy_nj": prefill_energy_nj + decode_energy_nj,
        "link_bytes": None,
        "pim_time_s_unoverlapped": None,
        "gpu_time_s_unoverlapped": None,
    }


def run_cacheblend_analytic_report(system, workload: Workload, plan: ReusePlan,
                                   *, pipe: bool, parallel_ff: bool,
                                   power_constraint: bool,
                                   batch_size: Optional[int] = None) -> Dict[str, Any]:
    """Estimate CacheBlend or EPIC with the same legacy analytic model as no-reuse.

    Only prefill work changes.  CacheBlend charges a reused row for every full
    recompute layer and for the configured fraction of partial layers; EPIC
    charges only its recomputed prefix rows (in every layer).  Decode retains
    the full KV context and therefore retains the baseline decode cost: the
    PIM scans the same rows, and the reuse-only die work (run descriptors,
    master-row mask between QK^T and softmax, LSE merge of the per-run
    partial softmax) is below 0.2% of a decode step in the physical model, so
    it is deliberately not modeled here.  This is a same-abstraction
    comparison, not a replacement for the physical TLB/Ramulator reference
    model.
    """
    if plan.config.policy not in ("cacheblend", "epic"):
        raise WorkloadValidationError("analytic latency model needs cacheblend or epic")
    _validate_layer_config(plan, system.model.ndec)
    _, baseline = run_no_reuse_report(
        system, workload, pipe=pipe, parallel_ff=parallel_ff,
        power_constraint=power_constraint, batch_size=batch_size)
    reusable_by_request: Dict[str, int] = {}
    epic_prefix_by_request: Dict[str, int] = {}
    for decision in plan.reusable:
        reusable_by_request[decision.request_id] = (
            reusable_by_request.get(decision.request_id, 0) + decision.length)
        epic_prefix_by_request[decision.request_id] = (
            epic_prefix_by_request.get(decision.request_id, 0) +
            len(decision.epic_prefix_rows))
    config = plan.config
    if config.policy == "cacheblend":
        recompute_fraction = (
            len(config.cacheblend_full_recompute_layers) +
            config.cacheblend_recompute_ratio * len(config.cacheblend_partial_recompute_layers)
        ) / system.model.ndec
    else:
        recompute_fraction = None  # EPIC: prefix rows are recomputed in full

    tiers = []
    makespan_s = 0.0
    prefill_energy_nj = 0.0
    decode_energy_nj = 0.0
    for tier in baseline["tiers"]:
        members = tier["members"]
        baseline_work = tier["batch_size"] * tier["lin"]
        reused_work = sum(reusable_by_request.get(member, 0) for member in members)
        fresh_work = sum(
            next(request for request in workload.requests
                 if request.request_id == member).total_length -
            reusable_by_request.get(member, 0)
            for member in members)
        # The baseline batch is padded to ``lin`` rows per member; reuse
        # removes only the reused rows' skipped work, never the padding, so
        # subtract the saving from the padded batch work rather than
        # rebuilding it from the unpadded token counts.
        if config.policy == "cacheblend":
            saved_work = reused_work * (1.0 - recompute_fraction)
        else:
            saved_work = reused_work - sum(epic_prefix_by_request.get(member, 0)
                                           for member in members)
        effective_work = baseline_work - saved_work
        prefill_scale = effective_work / baseline_work
        analytic = dict(tier)
        analytic.update({
            "baseline_prefill_s": tier["prefill_s"],
            "baseline_prefill_energy_nj": tier["prefill_energy_nj"],
            "fresh_prefill_tokens": fresh_work,
            "reused_prefill_tokens": reused_work,
            "padding_prefill_tokens": baseline_work - fresh_work - reused_work,
            "effective_prefill_tokens": effective_work,
            "prefill_scale": prefill_scale,
            "prefill_s": tier["prefill_s"] * prefill_scale,
            "prefill_energy_nj": tier["prefill_energy_nj"] * prefill_scale,
        })
        analytic["duration_s"] = analytic["prefill_s"] + analytic["decode_s"]
        analytic["energy_nj"] = (analytic["prefill_energy_nj"] +
                                  analytic["decode_energy_nj"])
        tiers.append(analytic)
        makespan_s += analytic["duration_s"]
        prefill_energy_nj += analytic["prefill_energy_nj"]
        decode_energy_nj += analytic["decode_energy_nj"]
    return {
        "policy": "{}-analytic".format(config.policy),
        "latency_model": "legacy-analytic",
        "scheduling": baseline["scheduling"],
        "batch_size": batch_size,
        "recompute_fraction_per_reused_row": recompute_fraction,
        "tiers": tiers,
        "makespan_s": makespan_s,
        "prefill_energy_nj": prefill_energy_nj,
        "decode_energy_nj": decode_energy_nj,
        "energy_nj": prefill_energy_nj + decode_energy_nj,
        "link_bytes": None,
        "physical_model_note": "Use --cacheblend-latency-model physical for TLB/Ramulator-DAG validation.",
    }


def _event(events: List[SplitEvent], transformer_layer: int, request_id: str,
           name: str, device: str, rows: int, time_s: float,
           energy: Iterable[float], link_bytes: int = 0) -> None:
    events.append(SplitEvent("legacy-{}".format(len(events)), transformer_layer,
                             0, request_id, name, device, rows, time_s,
                             sum(energy), link_bytes))


def _link_layer(template: Layer, name: str, byte_count: int) -> Layer:
    """Represent an exact byte-count transfer with the legacy X2G model."""
    layer = deepcopy(template)
    layer.name = name
    layer.type = LayerType.X2G
    layer.m = 1
    layer.n = byte_count // layer.dbyte
    layer.k = 1
    layer.numOp = 1
    return layer


def _validate_layer_config(plan: ReusePlan, ndec: int) -> None:
    if plan.config.policy != "cacheblend":
        return
    configured = (set(plan.config.cacheblend_full_recompute_layers) |
                  set(plan.config.cacheblend_partial_recompute_layers))
    expected = set(range(ndec))
    if configured != expected:
        missing, extra = sorted(expected - configured), sorted(configured - expected)
        raise WorkloadValidationError(
            "CacheBlend layers must partition all model layers 0..{}; missing={}, extra={}".format(
                ndec - 1, missing, extra))


def validate_split_events(events: Iterable[SplitEvent]) -> None:
    """Validate split-prefill ordering and communication shapes.

    Within each request/layer the execution order is GPU weight work, Q link,
    PIM attention, context link, then new-KV link.  GPU events may overlap
    later PIM work at runtime, but a PIM score cannot begin before its Q link,
    and the GPU continuation cannot consume PIM context before the return link.
    """
    groups: Dict[Tuple[str, int], List[SplitEvent]] = {}
    for event in events:
        if event.rows <= 0 or event.time_s < 0 or event.energy_nj < 0:
            raise WorkloadValidationError("split event has an invalid shape or cost")
        if event.device == "LINK" and event.link_bytes <= 0:
            raise WorkloadValidationError("link event must describe positive traffic")
        if event.device != "LINK" and event.link_bytes:
            raise WorkloadValidationError("only link events may carry link bytes")
        groups.setdefault((event.request_id, event.transformer_layer), []).append(event)
    for key, group in groups.items():
        names = [event.name for event in group]
        if "q_gpu_to_pim" not in names:
            # Full CacheBlend layers have a GPU score and a K/V placement link.
            if "score" not in names or "kv_gpu_to_pim" not in names:
                raise WorkloadValidationError(
                    "full-prefill event set is incomplete for {} layer {}".format(*key))
            continue
        q = names.index("q_gpu_to_pim")
        score = names.index("score")
        context = names.index("ctx_pim_to_gpu")
        kv = names.index("kv_gpu_to_pim")
        if not q < score < context < kv:
            raise WorkloadValidationError(
                "split-prefill dependency order is invalid for {} layer {}".format(*key))


def _run_legacy_reuse_prefill(system, workload: Workload, plan: ReusePlan) -> Dict[str, Any]:
    """Build and time the split-prefill events used by CacheBlend or EPIC.

    QKV/projection/FFN stay on the GPU.  Queries, newly produced K/V and the
    PIM context result cross the GPU-PIM link.  Score timing is queried from
    AttAcc's PIM device, so its DRAM command timing still comes from Ramulator.
    Old K/V is resident in PIM and is never transferred back to the GPU.
    """
    if system.hetero_name != DeviceType.PIM:
        raise WorkloadValidationError("reuse prefill requires --system dgx-attacc")
    validate_reuse_plan(workload, plan, system.model.ndec)
    _validate_layer_config(plan, system.model.ndec)

    # Build templates for one Transformer layer.  We override the row count
    # for each event below; this does not affect the legacy no-reuse path.
    system.model.build(1, 1, 2, True)
    templates = system.model.sum_decoder
    by_name = {layer.name: layer for layer in templates}
    qkv_template = by_name["qkv"]
    score_template = by_name["score"]
    softmax_template = by_name["softmax"]
    context_template = by_name["context"]
    x2g_template = by_name["comm_x2g"]
    events: List[SplitEvent] = []
    dbyte = qkv_template.dbyte
    hidden = system.model.hdim
    tp = system.model.tp
    local_hidden = hidden // tp
    attention_ops = max(1, system.model.num_heads // tp)

    reused_by_request: Dict[str, List[Any]] = {}
    for decision in plan.reusable:
        reused_by_request.setdefault(decision.request_id, []).append(decision)

    for _, requests, _, _ in _tier_shapes(workload):
        # Cross-tier requests are serialized by this loop; requests inside the
        # same tier are independent and are recorded as separate batch members.
        for request in requests:
            decisions = reused_by_request.get(request.request_id, [])
            reused_rows = sum(decision.length for decision in decisions)
            fresh_rows = request.total_length - reused_rows
            for layer_index in range(system.model.ndec):
                if plan.config.policy == "cacheblend":
                    full = layer_index in plan.config.cacheblend_full_recompute_layers
                    selected = plan.cacheblend_partial_rows.get(layer_index, {}).get(
                        request.request_id, {})
                    corrected_rows = sum(len(rows) for rows in selected.values())
                else:
                    full = False
                    corrected_rows = sum(len(decision.epic_prefix_rows)
                                         for decision in decisions)
                rows = request.total_length if full else fresh_rows + corrected_rows
                if rows <= 0:
                    continue

                # Weight-bearing operators and normalization/activation execute
                # on GPU for only the rows whose hidden states are recomputed.
                for template in templates:
                    if template.name in ("score", "softmax", "context", "comm_x2g"):
                        continue
                    layer = deepcopy(template)
                    layer.m = rows
                    time_s, energy = system.devices["GPU"].get_time_and_energy(layer)
                    _event(events, layer_index, request.request_id, layer.name,
                           "GPU", rows, time_s, energy)

                if full:
                    # Full CacheBlend layers follow original prefill: attention
                    # is computed at GPU, then all new K/V is placed in PIM.
                    for template in (score_template, softmax_template, context_template):
                        layer = deepcopy(template)
                        layer.m = request.total_length
                        layer.n = request.total_length
                        time_s, energy = system.devices["GPU"].get_time_and_energy(layer)
                        _event(events, layer_index, request.request_id, layer.name,
                               "GPU", request.total_length, time_s, energy)
                    kv_bytes = request.total_length * 2 * local_hidden * dbyte
                    layer = _link_layer(x2g_template, "kv_gpu_to_pim", kv_bytes)
                    time_s, energy = system.devices["GPU"].get_time_and_energy(layer)
                    _event(events, layer_index, request.request_id, layer.name,
                           "LINK", request.total_length, time_s, energy, kv_bytes)
                    continue

                # PIM scans old KV plus this layer's fresh/covered rows.  Q is
                # sent first; score/context cannot begin before that dependency.
                q_bytes = rows * local_hidden * dbyte
                layer = _link_layer(x2g_template, "q_gpu_to_pim", q_bytes)
                time_s, energy = system.devices["GPU"].get_time_and_energy(layer)
                _event(events, layer_index, request.request_id, layer.name,
                       "LINK", rows, time_s, energy, q_bytes)
                for template in (score_template, softmax_template, context_template):
                    layer = deepcopy(template)
                    layer.m = rows
                    layer.n = request.total_length
                    layer.k = system.model.dhead
                    # m carries the independently processed query rows; numOp
                    # remains the number of local attention heads.  Multiplying
                    # both would double-count rows in Ramulator's head mapping.
                    layer.numOp = attention_ops
                    time_s, energy = system.devices["Acc"].get_time_and_energy(layer)
                    _event(events, layer_index, request.request_id, layer.name,
                           "PIM", rows, time_s, energy)
                ctx_bytes = rows * local_hidden * dbyte
                layer = _link_layer(x2g_template, "ctx_pim_to_gpu", ctx_bytes)
                time_s, energy = system.devices["GPU"].get_time_and_energy(layer)
                _event(events, layer_index, request.request_id, layer.name,
                       "LINK", rows, time_s, energy, ctx_bytes)
                kv_bytes = rows * 2 * local_hidden * dbyte
                layer = _link_layer(x2g_template, "kv_gpu_to_pim", kv_bytes)
                time_s, energy = system.devices["GPU"].get_time_and_energy(layer)
                _event(events, layer_index, request.request_id, layer.name,
                       "LINK", rows, time_s, energy, kv_bytes)

    # Events are intentionally not summed blindly: GPU K/V writes may overlap
    # with following GPU work, while Q and context are true dependencies.  The
    # event stream keeps the information required for the scheduler to apply
    # AttAcc's pipeline overlap convention.
    validate_split_events(events)
    report = {
        "policy": plan.config.policy,
        "events": [event.to_dict() for event in events],
        "link_bytes": sum(event.link_bytes for event in events),
        "gpu_time_s_unoverlapped": sum(event.time_s for event in events
                                        if event.device == "GPU"),
        "pim_time_s_unoverlapped": sum(event.time_s for event in events
                                        if event.device == "PIM"),
    }
    return report


# CacheBlend uses the same physical bit order as Ramulator's HBM3-PIM mapper:
# CH(4), pCH(1), rank(1), bank-group(2), bank(2), row(14), column(5), byte(5).
# The allocator deliberately exposes those fields instead of keeping an opaque
# byte offset, so a report can be replayed directly as a PIM trace.
_HBM_TX_BYTES = 32
_HBM_CHANNEL_BYTES = 1 << 30
_ORIGINAL_KV_GAP_BYTES = 1 << 23
_ORIGINAL_HEAD_PARTITION_BYTES = 1 << 13

# CacheBlend needs distinct physical destinations for immutable master KV and
# per-request correction (diff) KV.  We preserve AttAcc's channel-first head
# placement inside each pool: a PIM trace maps heads over every channel in its
# pool before reusing a channel at the next 8-KiB partition.  The split is
# explicit here so it can be replaced by a trace-derived topology.
_KV_CHANNELS = {
    "master": tuple(range(0, 8)),
    "diff": tuple(range(8, 16)),
}
_ROTATE_MODES = ("gpu", "die", "bank")
_DIE_ROTATE_CYCLE_S = 1e-9
# After prefill, master and diff already sit as sequential streams in their
# pools, so a scan needs one descriptor per contiguous physical run (base,
# length, pool, mask bit-vector) rather than a lookup per logical row.  The
# mask itself is applied on the die between QK^T and softmax at no extra
# modeled cost; the cross-run softmax merge is the DIE LSE-merge event.
# TODO(manual-audit 2026-08-23, Chenyi): the 5-ns descriptor charge is an
# unsourced modeling constant (introduced upstream in 47ae0c3 without
# derivation or measurement), and it double-charges vs the paper's
# attach-time metadata load (Fugue sec 5.1: the driver loads positions into
# the die once; scans then consult resident metadata for free).  Direction:
# conservative, <1% of a decode step; sweep splitting repeats it per pass.
# Options in docs/README_manual_audit_findings.md -- keep+annotate /
# attach-load event model / calibrate from the RTL decoder.
_TLB_DESCRIPTOR_S = 5e-9
_TLB_DESCRIPTOR_ENERGY = 0.1


def _apply_pim_batch(op, batch_command: str, pe_freq_ghz: float) -> None:
    """Stamp the sweep's batch-command scheme onto a PIM scan op.

    ``replicate`` keeps the legacy one-MAC-per-(column, query) trace.  ``mq``
    is the MQ-MAC command of PLAN_mq_command.md: one MAC_AB per column serves
    every resident Q, and the Ramulator wrapper carries the n-fold PE time
    in the command interval (the DRAM cadence itself is never stretched;
    compute power is accounted separately, see mq_pe_power_w).
    """
    op.pim_batch_command = batch_command
    op.pim_pe_freq_ghz = pe_freq_ghz


def _tlb_plan_cost(runs) -> Tuple[float, Tuple[float, ...]]:
    count = max(1, len(runs))
    return count * _TLB_DESCRIPTOR_S, (_TLB_DESCRIPTOR_ENERGY * count,)


@dataclass(frozen=True)
class KVBlock:
    """One CacheBlend KV allocation in an original-AttAcc-style channel tile.

    The base is the first head's address.  Ramulator's original AttAcc trace
    generator places successive heads in successive channels of this block's
    channel set; only after that set is exhausted does it advance one 8-KiB
    head partition.  V starts at the fixed original AttAcc offset of 8 MiB.
    ``rows`` gives the source-token order inside that allocation, so a TLB
    entry can name both a reusable block and the exact vector within it.
    """
    block_id: str
    layer: int
    owner: str
    fingerprint: str
    kind: str
    rows: Tuple[int, ...]
    key_base: int
    value_base: int
    vector_stride: int
    channel_base: int
    channel_count: int
    channel_tile: int
    partition_offset: int
    # Index of the pool channel actually holding this block's first head.  A
    # pool is ``channel_count`` channels of 1 GiB each; once the first
    # channel's address range is full, allocation continues in the next
    # channel of the same pool (heads then stripe with wrap-around inside the
    # pool, see the trace generator's ``--pool-base``).
    channel_offset: int = 0

    def token_offset(self, owner_row: int) -> int:
        try:
            return self.rows.index(owner_row)
        except ValueError as exc:
            raise WorkloadValidationError(
                "TLB row {} is not reserved in {}".format(owner_row, self.block_id)) from exc

    def to_dict(self) -> Dict[str, Any]:
        count = len(self.rows)
        return {
            "id": self.block_id, "layer": self.layer, "owner": self.owner,
            "fingerprint": self.fingerprint, "kind": self.kind,
            "rows": list(self.rows), "key_base": "0x{:x}".format(self.key_base),
            "value_base": "0x{:x}".format(self.value_base),
            "vector_stride": self.vector_stride, "vector_count": count,
            "key_bytes": count * self.vector_stride,
            "value_bytes": count * self.vector_stride,
            "channel_base": self.channel_base,
            "channel_count": self.channel_count,
            "channel_offset": self.channel_offset,
            "channel_tile": self.channel_tile,
            "partition_offset": self.partition_offset,
        }


@dataclass(frozen=True)
class KVLocation:
    layer: int
    owner: str
    fingerprint: str
    owner_row: int
    kind: str                 # master, diff, or live
    key_address: int
    value_address: int
    bytes_per_vector: int
    block_id: str
    token_offset: int
    channel_base: int
    channel_count: int
    # A ``diff`` overlay names the master row it shadows.  The master stream
    # is still read sequentially through that row; the row is masked out of
    # the score/softmax instead of being skipped by the DRAM access pattern.
    shadow: Optional["KVLocation"] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "layer": self.layer, "owner": self.owner,
            "fingerprint": self.fingerprint, "owner_row": self.owner_row,
            "kind": self.kind, "key_address": "0x{:x}".format(self.key_address),
            "value_address": "0x{:x}".format(self.value_address),
            "bytes_per_vector": self.bytes_per_vector,
            "block_id": self.block_id, "token_offset": self.token_offset,
            "channel_base": self.channel_base,
            "channel_count": self.channel_count,
        }
        if self.shadow is not None:
            result["shadow_master"] = self.shadow.to_dict()
        return result


def _address_key(location: KVLocation) -> Tuple[int, int]:
    return (location.key_address, location.value_address)


def _physical_reads(locations: Sequence[KVLocation]) -> Tuple[List[KVLocation], set]:
    """Expand consumer-visible K/V rows into the rows the PIM actually reads.

    Master and diff live in disjoint channel pools and each pool is streamed
    sequentially.  A corrected row therefore costs two reads: its diff row
    (scored) and the shadowed master row (read in the master stream but
    masked out of the score).  Returns the physical read list and the set of
    masked ``(key, value)`` addresses.
    """
    reads: List[KVLocation] = []
    masked = set()
    for location in locations:
        reads.append(location)
        if location.shadow is not None:
            reads.append(location.shadow)
            masked.add(_address_key(location.shadow))
    return reads, masked


def _masked_rows_per_run(runs: Sequence[Tuple[int, int, int, int, int]],
                         masked: set, bytes_per_vector: int) -> Tuple[int, ...]:
    """Count masked rows inside each coalesced physical run."""
    if not masked:
        return tuple(0 for _ in runs)
    stride = ((bytes_per_vector + _HBM_TX_BYTES - 1) // _HBM_TX_BYTES) * _HBM_TX_BYTES
    # A run is one contiguous K extent (V is the same offsets in the same
    # block), so counting masked K addresses inside [base, base + n*stride)
    # is exact and stays O(log n) per run for long contexts.
    masked_keys = sorted(key for key, _ in masked)
    counts = []
    for key_base, _, count, _, _ in runs:
        low = bisect_left(masked_keys, key_base)
        high = bisect_left(masked_keys, key_base + count * stride)
        counts.append(high - low)
    return tuple(counts)


class CacheBlendTLB:
    """Logical-position to concrete HBM-K/V mapping used by CacheBlend.

    Master rows hold first/full recomputations and immutable reused K/V.  A
    corrected partial-layer row receives a ``diff`` row and shadows the master
    entry in the consumer's TLB; it never overwrites a producer's cache line.
    Master and diff use disjoint physical channel sets, and each pool is
    streamed sequentially at attention time: the shadowed master row is still
    read in the master stream (its ``KVLocation.shadow``) and masked out of
    the score, while the diff pool supplies the corrected row.  Skipping the
    shadowed row instead would break the master stream into one cold-start
    PIM run per correction.
    """

    def __init__(self, bytes_per_vector: int):
        self.bytes_per_vector = bytes_per_vector
        self._locations: Dict[Tuple[int, str, str, int, str], KVLocation] = {}
        self._reserved_rows: Dict[Tuple[int, str, str, str], set] = {}
        self._blocks: Dict[Tuple[int, str, str, str], KVBlock] = {}
        self.entries: List[Dict[str, Any]] = []

    def reserve(self, layer: int, owner: str, fingerprint: str, owner_row: int,
                kind: str) -> None:
        if self._blocks:
            raise WorkloadValidationError("TLB reservations must finish before allocation")
        self._reserved_rows.setdefault((layer, owner, fingerprint, kind), set()).add(owner_row)

    def finalize(self) -> None:
        """Materialize blocks in disjoint original-AttAcc-style channels.

        A channel is a 1-GiB region in the active HBM3-PIM mapper.  Within a
        channel, K starts at an 8-KiB-aligned partition; V is exactly 8 MiB
        above it.  Once the 8-MiB K window is full, allocation continues in
        the next 16-MiB channel tile.  This preserves the original K/V gap
        while preventing K/V or master/diff overlap.
        """
        if self._blocks:
            return
        stride = ((self.bytes_per_vector + _HBM_TX_BYTES - 1) //
                  _HBM_TX_BYTES) * _HBM_TX_BYTES
        # CacheBlend retains AttAcc's original head-to-channel striping.
        # Every master extent spans channels 0--7 and every diff extent spans
        # channels 8--15.  The cursor is therefore shared by all eight
        # channels in a pool: head h is placed on base+(h % 8), and only after
        # those channels are consumed does the trace advance a partition.
        channel_state = {
            kind: {"tile": 0, "cursor": 0, "offset": 0, "channels": channels}
            for kind, channels in _KV_CHANNELS.items()
        }
        for index, key in enumerate(sorted(self._reserved_rows)):
            layer, owner, fingerprint, kind = key
            if kind not in _KV_CHANNELS:
                raise WorkloadValidationError("unknown CacheBlend KV kind '{}'".format(kind))
            rows = tuple(sorted(self._reserved_rows[key]))
            # A pool is one continuous AttAcc-striped K/V stream.  Individual
            # CacheBlend ownership blocks do not reserve a fresh 8-KiB head
            # partition: doing so creates artificial holes and turns one
            # physical scan into many cold-start Ramulator runs.
            span = len(rows) * stride
            if span > _ORIGINAL_KV_GAP_BYTES:
                raise WorkloadValidationError(
                    "CacheBlend KV block exceeds the 8-MiB original-AttAcc K partition")
            pool = channel_state[kind]
            channels = pool["channels"]
            if pool["cursor"] + span > _ORIGINAL_KV_GAP_BYTES:
                pool["tile"] += 1
                pool["cursor"] = 0
            tiles_per_channel = _HBM_CHANNEL_BYTES // (2 * _ORIGINAL_KV_GAP_BYTES)
            if pool["tile"] >= tiles_per_channel:
                # This channel's 1-GiB range is full: continue in the next
                # channel of the same pool.  Master and diff never mix.
                pool["offset"] += 1
                pool["tile"] = 0
                pool["cursor"] = 0
                if pool["offset"] >= len(channels):
                    raise WorkloadValidationError(
                        "CacheBlend KV allocation exceeds channel-pool {} capacity".format(kind))
            channel_base = channels[0]
            channel = channels[pool["offset"]]
            key_base = (channel * _HBM_CHANNEL_BYTES +
                        pool["tile"] * (2 * _ORIGINAL_KV_GAP_BYTES) +
                        pool["cursor"])
            value_base = key_base + _ORIGINAL_KV_GAP_BYTES
            if value_base + span > (channel + 1) * _HBM_CHANNEL_BYTES:
                raise WorkloadValidationError(
                    "CacheBlend KV allocation exceeds channel-pool {} capacity".format(kind))
            partition_offset = pool["cursor"]
            channel_tile = pool["tile"]
            pool["cursor"] += span
            self._blocks[key] = KVBlock(
                "kvb-{:06d}".format(index), layer, owner, fingerprint, kind,
                rows, key_base, value_base, stride, channel_base, len(channels), channel_tile,
                partition_offset, pool["offset"])

    def locate(self, layer: int, owner: str, fingerprint: str, owner_row: int,
               kind: str) -> KVLocation:
        key = (layer, owner, fingerprint, owner_row, kind)
        location = self._locations.get(key)
        if location is not None:
            return location
        block_key = (layer, owner, fingerprint, kind)
        block = self._blocks.get(block_key)
        if block is None:
            raise WorkloadValidationError(
                "TLB location requested before its KV block was reserved: {}".format(block_key))
        token_offset = block.token_offset(owner_row)
        location = KVLocation(
            layer, owner, fingerprint, owner_row, kind,
            block.key_base + token_offset * block.vector_stride,
            block.value_base + token_offset * block.vector_stride,
            self.bytes_per_vector, block.block_id, token_offset,
            block.channel_base, block.channel_count)
        self._locations[key] = location
        return location

    def scan_runs(self, locations: Sequence[KVLocation]) -> Tuple[Tuple[int, int, int, int, int], ...]:
        """Coalesce only physically adjacent K/V vectors for Ramulator.

        Every input location is covered exactly once.  A non-adjacent reused
        segment deliberately remains a separate PIM run; treating it as an
        extension of the first block was the old, inaccurate behavior.
        """
        runs: List[Tuple[int, int, int, int, int]] = []
        # Attention reduction is order-independent.  Present each pool to
        # Ramulator in physical-address order so adjacent ownership blocks
        # become one continuous AttAcc command stream.
        ordered = sorted(locations, key=lambda location:
                         (location.channel_base, location.channel_count,
                          location.key_address, location.value_address))
        for location in ordered:
            stride = ((location.bytes_per_vector + _HBM_TX_BYTES - 1) //
                      _HBM_TX_BYTES) * _HBM_TX_BYTES
            if (runs and location.channel_base == runs[-1][3]
                    and location.channel_count == runs[-1][4]
                    and location.key_address == runs[-1][0] + runs[-1][2] * stride
                    and location.value_address == runs[-1][1] + runs[-1][2] * stride):
                key_base, value_base, count, channel_base, channel_count = runs[-1]
                runs[-1] = (key_base, value_base, count + 1,
                             channel_base, channel_count)
            else:
                runs.append((location.key_address, location.value_address, 1,
                             location.channel_base, location.channel_count))
        return tuple(runs)

    def bind(self, request_id: str, layer: int, position: int, location: KVLocation,
             position_delta: int, reused: bool) -> None:
        self.entries.append({
            "request": request_id, "layer": layer, "position": position,
            "position_delta": position_delta, "reused": reused,
            "location": location.to_dict(),
        })

    def report(self) -> Dict[str, Any]:
        return {"mapping": "Ramulator HBM3-PIM physical byte address",
                "layout": "master channels 0-7, diff channels 8-15; "
                          "8-KiB partitions with V at K + 8 MiB",
                "scan": "each pool is streamed sequentially; a diff entry's "
                        "shadow_master row is read in the master stream and "
                        "masked out of the score (event masked_rows)",
                "head_mapping": "head h uses channel_base + ((channel_offset + h) % "
                                "channel_count); after channel_count heads, advance by "
                                "one 8-KiB partition (implemented by the Ramulator trace "
                                "generator, --pool-base)",
                "channel_capacity_bytes": _HBM_CHANNEL_BYTES,
                "transaction_bytes": _HBM_TX_BYTES,
                "channel_sets": {kind: list(channels)
                                 for kind, channels in _KV_CHANNELS.items()},
                "blocks": [block.to_dict() for _, block in sorted(self._blocks.items())],
                "entries": self.entries}


class NoReuseKVLayout:
    """Private, affine KV placement for the physical no-reuse baseline.

    This is intentionally not a CacheBlend TLB.  A request/layer receives one
    contiguous K/V extent spanning all 16 regular HBM channels, so a logical
    position is resolved by ``base + position * stride`` without a metadata
    lookup.  The class merely supplies the address objects required by the
    shared Ramulator/DAG interface.
    """

    def __init__(self, bytes_per_vector: int):
        self.bytes_per_vector = bytes_per_vector
        self._reserved: Dict[Tuple[int, str, str], set] = {}
        self._blocks: Dict[Tuple[int, str, str], KVBlock] = {}
        self._locations: Dict[Tuple[int, str, str, int], KVLocation] = {}
        self.entries: List[Dict[str, Any]] = []

    def reserve(self, layer: int, owner: str, fingerprint: str, owner_row: int,
                kind: str = "private") -> None:
        self._reserved.setdefault((layer, owner, fingerprint), set()).add(owner_row)

    def finalize(self) -> None:
        if self._blocks:
            return
        stride = ((self.bytes_per_vector + _HBM_TX_BYTES - 1) //
                  _HBM_TX_BYTES) * _HBM_TX_BYTES
        tile, cursor = 0, 0
        for index, key in enumerate(sorted(self._reserved)):
            layer, owner, fingerprint = key
            rows = tuple(sorted(self._reserved[key]))
            span = ((len(rows) * stride + _ORIGINAL_HEAD_PARTITION_BYTES - 1) //
                    _ORIGINAL_HEAD_PARTITION_BYTES) * _ORIGINAL_HEAD_PARTITION_BYTES
            if span > _ORIGINAL_KV_GAP_BYTES:
                raise WorkloadValidationError("no-reuse KV extent exceeds 8-MiB K partition")
            if cursor + span > _ORIGINAL_KV_GAP_BYTES:
                tile += 1
                cursor = 0
            key_base = tile * (2 * _ORIGINAL_KV_GAP_BYTES) + cursor
            self._blocks[key] = KVBlock(
                "nrb-{:06d}".format(index), layer, owner, fingerprint, "private",
                rows, key_base, key_base + _ORIGINAL_KV_GAP_BYTES, stride,
                0, 16, tile, cursor)
            cursor += span

    def locate(self, layer: int, owner: str, fingerprint: str, owner_row: int,
               kind: str = "private") -> KVLocation:
        key = (layer, owner, fingerprint)
        block = self._blocks.get(key)
        if block is None:
            raise WorkloadValidationError("no-reuse location was not reserved: {}".format(key))
        cache_key = (layer, owner, fingerprint, owner_row)
        if cache_key not in self._locations:
            offset = block.token_offset(owner_row)
            self._locations[cache_key] = KVLocation(
                layer, owner, fingerprint, owner_row, "private",
                block.key_base + offset * block.vector_stride,
                block.value_base + offset * block.vector_stride,
                self.bytes_per_vector, block.block_id, offset, 0, 16)
        return self._locations[cache_key]

    def bind(self, request_id: str, layer: int, position: int, location: KVLocation,
             position_delta: int, reused: bool) -> None:
        self.entries.append({"request": request_id, "layer": layer,
                             "position": position, "position_delta": position_delta,
                             "reused": False, "location": location.to_dict()})

    def scan_runs(self, locations: Sequence[KVLocation]) -> Tuple[Tuple[int, int, int, int, int], ...]:
        if not locations:
            return ()
        first = locations[0]
        stride = ((first.bytes_per_vector + _HBM_TX_BYTES - 1) //
                  _HBM_TX_BYTES) * _HBM_TX_BYTES
        if any(location.block_id != first.block_id or
               location.key_address != first.key_address + index * stride or
               location.value_address != first.value_address + index * stride
               for index, location in enumerate(locations)):
            # Decode may include a separately allocated generated-output tail.
            return CacheBlendTLB.scan_runs(self, locations)
        return ((first.key_address, first.value_address, len(locations), 0, 16),)

    def report(self) -> Dict[str, Any]:
        return {"mapping": "affine private no-reuse HBM3-PIM physical byte address",
                "layout": "one private contiguous request/layer extent; no reuse TLB",
                "channel_sets": {"private": list(range(16))},
                "blocks": [block.to_dict() for _, block in sorted(self._blocks.items())],
                "entries": self.entries}


def _cacheblend_event(events: List[SplitEvent], *, layer: int, tier: int,
                      request: str, name: str, device: str, rows: int,
                      time_s: float, energy: Iterable[float],
                      deps: Sequence[str] = (), link_bytes: int = 0,
                      positions: Sequence[int] = (),
                      addresses: Sequence[int] = (),
                      batch_members: Sequence[str] = (),
                      masked_rows: int = 0) -> str:
    event_id = "cb-{}".format(len(events))
    # A prefill TLB/link event names every visible K/V address; over a long
    # context that is O(L) per fresh token.  Keep those lists as packed
    # 64-bit arrays rather than tuples of Python ints to bound host memory.
    # Device energy tables are in pJ (the legacy record divides by 1000 to
    # report nJ, see System.simulate); keep the DAG report in the same nJ.
    events.append(SplitEvent(event_id, layer, tier, request, name, device, rows,
                             time_s, sum(energy) / 1000.0, link_bytes, tuple(deps),
                             tuple(positions), array("Q", addresses),
                             tuple(batch_members), masked_rows))
    return event_id


def _append_physical_pim_scan(system, events: List[SplitEvent], *, op: Layer,
                              layer: int, tier: int, request: str, name: str,
                              rows: int, deps: Sequence[str], positions: Sequence[int],
                              runs: Sequence[Tuple[int, int, int, int, int]],
                              batch_members: Sequence[str] = (),
                              masked: Sequence[int] = ()) -> Tuple[str, ...]:
    """Schedule each HBM channel pool independently and return all scan events.

    ``rows`` of a scan event is the number of K/V rows physically streamed
    by that run; ``masked`` gives, per run, how many of those rows are
    shadowed master rows that are read but excluded from the score.
    """
    accelerator = system.devices["Acc"]
    masked = tuple(masked) if masked else tuple(0 for _ in runs)
    if len(masked) != len(runs):
        raise WorkloadValidationError("masked-row count must be given per physical run")
    if hasattr(accelerator, "get_time_and_energy_runs"):
        measured = accelerator.get_time_and_energy_runs(op)
    else:
        # Preserve the aggregate mock-device API used by lightweight tests.
        measured = [accelerator.get_time_and_energy(op)]
        runs = (runs[0],) if runs else ()
        masked = masked[:len(runs)]
    if len(measured) != len(runs):
        raise WorkloadValidationError("Ramulator physical-run result count mismatch")
    scan_events = []
    for run, (time_s, energy), masked_rows in zip(runs, measured, masked):
        key_addr, value_addr, run_rows, channel, channel_count = run
        if masked_rows > run_rows:
            raise WorkloadValidationError("masked rows exceed the rows read by a PIM run")
        scan_events.append(_cacheblend_event(
            events, layer=layer, tier=tier, request=request, name=name,
            device="PIM:pool{}-{}".format(channel, channel + channel_count - 1),
            rows=run_rows,
            time_s=time_s, energy=energy, deps=tuple(deps),
            positions=tuple(positions), addresses=(key_addr, value_addr),
            batch_members=tuple(batch_members), masked_rows=masked_rows))
    return tuple(scan_events)


def _append_channel_kv_stores(system, events: List[SplitEvent], *, layer: int,
                              tier: int, request: str, name: str,
                              locations: Sequence[KVLocation], dbyte: int,
                              deps: Sequence[str], positions: Sequence[int]) -> Tuple[str, ...]:
    """Write K/V in parallel across concrete TLB channels.

    The configured PIM bandwidth is the 16-channel aggregate.  A CacheBlend
    master/diff pool has eight channels, so each channel receives one
    sixteenth of that aggregate and the DAG completion is the slowest active
    channel rather than a fictitious one-device write.
    """
    by_channel: Dict[Tuple[int, int], List[KVLocation]] = {}
    for location in locations:
        by_channel.setdefault((location.channel_base, location.channel_count), []).append(location)
    result = []
    for (channel, channel_count), channel_locations in sorted(by_channel.items()):
        byte_count = sum(2 * location.bytes_per_vector for location in channel_locations)
        bandwidth = (system.devices["Acc"].peak_memory_bandwidth *
                     channel_count / 16)
        result.append(_cacheblend_event(
            events, layer=layer, tier=tier, request=request, name=name,
            device="PIM:pool{}-{}".format(channel, channel + channel_count - 1),
            rows=len(channel_locations),
            time_s=byte_count / bandwidth,
            energy=(byte_count * system.devices["Acc"].energy_table["mem"],),
            deps=tuple(deps), positions=tuple(positions),
            addresses=tuple(address for location in channel_locations
                            for address in (location.key_address, location.value_address))))
    return tuple(result)


def _schedule_cacheblend(events: Sequence[SplitEvent], *, pipe: bool) -> List[SplitEvent]:
    """Schedule CacheBlend with AttAcc's optional pipeline convention.

    ``pipe=False`` is an intentionally conservative dependency-ordered
    execution: no GPU/PIM/link overlap is credited.  ``pipe=True`` gives each
    trace unit an independent busy timeline, matching AttAcc's ``--pipeopt``
    convention for overlap between computation and communication.
    """
    finish: Dict[str, float] = {}
    availability: Dict[str, float] = {}
    scheduled: List[SplitEvent] = []
    for event in events:
        if any(dep not in finish for dep in event.depends_on):
            raise WorkloadValidationError("CacheBlend event depends on a future event")
        # Links, DIE/TLB and banks are each ordered resources; GPU and PIM
        # work can overlap a link exactly as in the CacheBlend trace.  Without
        # --pipeopt, all operations share one serial timeline.
        resource = event.device if pipe else "SERIAL"
        start = max([availability.get(resource, 0.0)] +
                    [finish[dep] for dep in event.depends_on])
        end = start + event.time_s
        availability[resource] = end
        finish[event.event_id] = end
        scheduled.append(replace(event, start_s=start, end_s=end))
    return scheduled


def _schedule_cacheblend_incremental(events: Sequence[SplitEvent], *, pipe: bool,
                                     start_index: int,
                                     finish: Dict[str, float],
                                     availability: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Extend an append-only CacheBlend schedule without replaying its prefix.

    Events are never inserted before an existing event.  Therefore the finish
    times and device/link availability of a scheduled prefix are immutable;
    extending that prefix gives exactly the same Q-link completions as a full
    later call to :func:`_schedule_cacheblend`.
    """
    finish = dict(finish)
    availability = dict(availability)
    for event in events[start_index:]:
        if any(dep not in finish for dep in event.depends_on):
            raise WorkloadValidationError("CacheBlend event depends on a future event")
        resource = event.device if pipe else "SERIAL"
        start = max([availability.get(resource, 0.0)] +
                    [finish[dep] for dep in event.depends_on])
        availability[resource] = start + event.time_s
        finish[event.event_id] = start + event.time_s
    return finish, availability


def _annotate_batch_q_arrivals(scheduled: Sequence[SplitEvent],
                               batches: List[Dict[str, Any]]) -> None:
    """Record and check ready-queue PIM admission from actual Q arrivals.

    A decode Q cannot arrive before the GPU QKV that generated it and its
    individual GPU-to-PIM link transfer.  The link is a single ordered AttAcc
    resource in ``_schedule_cacheblend``.  A shared-KV PIM batch is therefore
    allowed to start only after the final member Q link has completed.  This is
    deliberately stronger than grouping requests by source order alone: the
    report exposes the measured admission timestamp so it can be audited.
    """
    for batch in batches:
        members = set(batch["members"])
        layer = batch["transformer_layer"]
        q_events = [event for event in scheduled
                    if event.name in ("decode_q_gpu_to_pim",
                                      "decode_gpu_rotate_q_extra_to_pim") and
                    event.transformer_layer == layer and
                    event.request_id in members]
        q_events = [event for event in q_events
                    if all(position == batch["query_positions"][event.request_id]
                           for position in event.query_positions)]
        raw_q = [event for event in q_events if event.name == "decode_q_gpu_to_pim"]
        if len(raw_q) != batch["size"]:
            raise WorkloadValidationError(
                "CacheBlend batch '{}' has incomplete Q arrivals".format(batch["id"]))
        arrival = max(event.end_s for event in q_events)
        batch["q_arrival_s"] = arrival
        batch["q_arrival_event_ids"] = [event.event_id for event in q_events]
        arrival_by_member: Dict[str, float] = {}
        for event in q_events:
            arrival_by_member[event.request_id] = max(
                arrival_by_member.get(event.request_id, 0.0), event.end_s)
        # The admitted batch may be served by several consecutive PIM sweeps
        # (the GEMV-buffer capacity splits it).  Each sweep needs only its own
        # members' Q arrivals; audit every sweep against exactly that set.
        sweep_events = [event for event in scheduled
                        if event.request_id == batch["id"] and
                        event.name == "decode_batch_tlb_lookup_and_bank_plan"]
        if sweep_events:
            sweeps = []
            for event in sweep_events:
                sweep_members = list(event.batch_members) or batch["members"]
                sweep_arrival = max(arrival_by_member[member]
                                    for member in sweep_members)
                if event.start_s + 1e-18 < sweep_arrival:
                    raise WorkloadValidationError(
                        "CacheBlend shared-KV sweep begins before its final Q arrival")
                sweeps.append({"members": sweep_members,
                               "admission_s": sweep_arrival,
                               "start_s": event.start_s})
            batch["sweeps"] = sweeps
            batch["attention_admission_s"] = max(item["admission_s"]
                                                 for item in sweeps)
            batch["attention_start_s"] = min(item["start_s"] for item in sweeps)
            batch["admission"] = "global-q-ready-queue"
        else:
            batch["attention_admission_s"] = None
            batch["attention_start_s"] = None
            batch["admission"] = "global-q-ready-queue-no-shared-kv"


def validate_cacheblend_attacc_overlap_contract(scheduled: Sequence[SplitEvent],
                                                *, pipe: bool) -> Dict[str, Any]:
    """Audit the event-level overlap rules inherited from AttAcc.

    Original AttAcc keeps one outstanding ``comm_x2g`` write timeline
    (``wrt_io_busy``); a later transfer starts at the later of its producer and
    that timeline.  With pipeline disabled it executes the decoder in one
    serial timeline.  CacheBlend extends this exact contract to explicitly
    named GPU/PIM/DIE/TLB resources, which the original rectangular model did
    not expose.  This checker replays those rules from the emitted DAG rather
    than trusting the scheduling function that produced it.
    """
    finish: Dict[str, float] = {}
    available: Dict[str, float] = {}
    tolerance = 1e-18
    for event in scheduled:
        resource = event.device if pipe else "SERIAL"
        expected = max([available.get(resource, 0.0)] +
                       [finish[dependency] for dependency in event.depends_on])
        if abs(event.start_s - expected) > tolerance:
            raise WorkloadValidationError(
                "CacheBlend overlap diverges from AttAcc {} timeline at {}: {} != {}".format(
                    resource, event.event_id, event.start_s, expected))
        if abs(event.end_s - (event.start_s + event.time_s)) > tolerance:
            raise WorkloadValidationError(
                "CacheBlend event duration is inconsistent with the AttAcc timeline")
        available[resource] = event.end_s
        finish[event.event_id] = event.end_s
    report = {
        "passed": True,
        "pipe": pipe,
        "contract": ("original AttAcc serial decoder" if not pipe else
                     "AttAcc comm_x2g busy timeline plus explicit CacheBlend resources"),
        "events_checked": len(scheduled),
    }
    return report


def validate_cacheblend_events(events: Sequence[SplitEvent], workload: Workload,
                               *, local_hidden: int, dbyte: int, dhead: int,
                               heads: int) -> None:
    """Reject malformed CacheBlend event graphs before time/energy scheduling.

    This is deliberately separate from ``validate_reuse_plan``: the latter
    validates policy rows, whereas this validator checks the materialized
    hardware DAG (links, DIE/TLB/PIM ordering and decode-KV materialization).
    """
    by_id = {event.event_id: event for event in events}
    if len(by_id) != len(events):
        raise WorkloadValidationError("CacheBlend event ids must be unique")
    order = {event.event_id: index for index, event in enumerate(events)}
    for event in events:
        if event.rows <= 0 or event.time_s < 0 or event.energy_nj < 0:
            raise WorkloadValidationError("CacheBlend event has an invalid shape or cost")
        if event.masked_rows < 0 or event.masked_rows > event.rows:
            raise WorkloadValidationError("CacheBlend event masks more rows than it reads")
        if event.device == "LINK":
            if event.link_bytes <= 0:
                raise WorkloadValidationError("CacheBlend link event has no traffic")
        elif event.link_bytes:
            raise WorkloadValidationError("only CacheBlend link events may carry traffic")
        for dependency in event.depends_on:
            parent = by_id.get(dependency)
            if parent is None or order[dependency] >= order[event.event_id]:
                raise WorkloadValidationError("CacheBlend event dependency is not topological")
            if parent.tier > event.tier:
                raise WorkloadValidationError("CacheBlend event depends on a later tier")

    def named(event_id: str) -> str:
        return by_id[event_id].name

    q_bytes_per_row = local_hidden * dbyte
    kv_bytes_per_row = 2 * q_bytes_per_row
    tuple_bytes = heads * (dhead + 2) * dbyte
    for event in events:
        if event.name in ("q_gpu_to_pim", "ctx_pim_to_gpu",
                          "decode_q_gpu_to_pim", "decode_ctx_pim_to_gpu"):
            if event.link_bytes != event.rows * q_bytes_per_row:
                raise WorkloadValidationError("CacheBlend Q/context link byte count is invalid")
        elif event.name in ("kv_gpu_to_pim", "decode_kv_gpu_to_pim"):
            if event.link_bytes != event.rows * kv_bytes_per_row:
                raise WorkloadValidationError("CacheBlend KV link byte count is invalid")
        elif event.name in ("gpu_partial_lse_to_pim", "decode_gpu_partial_lse_to_pim"):
            if event.rows != 1 or event.link_bytes != tuple_bytes:
                raise WorkloadValidationError("CacheBlend local LSE tuple shape is invalid")
        if "pim_kv_scan" in event.name:
            if not event.dram_addresses or len(event.dram_addresses) % 2:
                raise WorkloadValidationError("CacheBlend PIM scan lacks K/V addresses")
            if not event.depends_on or not any(
                    ("tlb_lookup_and_bank_plan" in named(dependency) or
                     named(dependency) in ("contiguous_address_plan",
                                           "decode_contiguous_address_plan"))
                    for dependency in event.depends_on):
                raise WorkloadValidationError("PIM scan must depend on an address plan")
        if "die_lse_merge" in event.name:
            dependency_names = {named(dependency) for dependency in event.depends_on}
            if not any("pim_kv_scan" in name for name in dependency_names) or not any(
                    "partial_lse_to_pim" in name for name in dependency_names):
                raise WorkloadValidationError("CacheBlend DIE merge lacks a local contribution")

    groups: Dict[Tuple[str, int], List[SplitEvent]] = {}
    for event in events:
        groups.setdefault((event.request_id, event.transformer_layer), []).append(event)
    for (request_id, layer), group in groups.items():
        names = {event.name for event in group}
        for context_name, tuple_name, merge_name in (
                ("ctx_pim_to_gpu", "gpu_partial_lse_to_pim", "die_lse_merge"),
                ("decode_ctx_pim_to_gpu", "decode_gpu_partial_lse_to_pim",
                 "decode_die_lse_merge")):
            contexts = [event for event in group if event.name == context_name]
            if not contexts:
                continue
            context = contexts[-1]
            relevant = [event for event in group
                        if set(event.query_positions).intersection(context.query_positions)]
            merge_events = {event.event_id for event in relevant
                            if event.name == merge_name}
            scan_events = {event.event_id for event in relevant
                           if "pim_kv_scan" in event.name}
            # A DIE merge already depends on its GPU tuple.  If no old KV was
            # scanned, context waits for the local GPU tuple.  Physical
            # no-reuse instead has one full contiguous PIM scan and therefore
            # waits directly for that scan.
            required = (merge_events or
                        {event.event_id for event in relevant
                         if event.name == tuple_name} or scan_events)
            if not required.issubset(set(context.depends_on)):
                raise WorkloadValidationError(
                    "CacheBlend context return starts before all local results: {}".format(
                        sorted(required - set(context.depends_on))))
        for kv_name, qkv_name, store_names in (
                ("kv_gpu_to_pim", "qkv",
                 ("dram_store_master", "dram_store_diff_and_live")),
                ("decode_kv_gpu_to_pim", "decode_qkv",
                 ("decode_dram_store_master",))):
            for kv_link in (event for event in group if event.name == kv_name):
                qkvs = [event.event_id for event in group
                        if event.name == qkv_name and
                        set(event.query_positions).intersection(kv_link.query_positions)]
                if not set(qkvs).intersection(kv_link.depends_on):
                    raise WorkloadValidationError(
                        "CacheBlend KV transfer is not ordered after QKV")
                stores = [event for event in group if event.name in store_names and
                          set(event.query_positions).intersection(kv_link.query_positions)]
                if not any(kv_link.event_id in store.depends_on for store in stores):
                    raise WorkloadValidationError(
                        "CacheBlend KV store is not ordered after its transfer")

    expected_decode_stores = {
        request.request_id: request.lout * len(range(max(event.transformer_layer
                                                         for event in events) + 1))
        for request in workload.requests
    }
    actual_decode_stores: Dict[str, int] = {}
    for event in events:
        if event.name == "decode_dram_store_master":
            actual_decode_stores[event.request_id] = actual_decode_stores.get(event.request_id, 0) + 1
    for request in workload.requests:
        if actual_decode_stores.get(request.request_id, 0) != expected_decode_stores[request.request_id]:
            raise WorkloadValidationError(
                "CacheBlend request '{}' did not materialize every output KV layer".format(
                    request.request_id))


def _gpu_layer_event(system, events, template, *, layer, tier, request, name,
                     rows, deps=(), positions=()):
    op = deepcopy(template)
    op.m = rows
    time_s, energy = system.devices["GPU"].get_time_and_energy(op)
    return _cacheblend_event(events, layer=layer, tier=tier, request=request,
                             name=name, device="GPU", rows=rows, time_s=time_s,
                             energy=energy, deps=deps, positions=positions)


def _post_attention_gpu(system, events, templates, *, layer, tier, request,
                        rows, dependency, positions):
    last = dependency
    for template in templates:
        if template.name in ("qkv", "score", "softmax", "context", "comm_x2g"):
            continue
        last = _gpu_layer_event(system, events, template, layer=layer, tier=tier,
                                request=request, name="gpu_" + template.name,
                                rows=rows, deps=(last,), positions=positions)
    return last


def _prefill_location_deltas(request, bindings):
    """Map a consumer-visible resident K/V vector to its RoPE position delta."""
    result = {}
    position = 0
    for segment in request.segments:
        for _ in range(segment.length):
            location = bindings[position][3]
            result[_address_key(location)] = segment.position_delta
            if location.shadow is not None:
                # The shadowed master row is streamed with the same segment
                # position shift as the diff row that replaces it.
                result[_address_key(location.shadow)] = segment.position_delta
            position += 1
    return result


def _append_q_rotate_distribution(system, events, x2g, *, layer, tier, request,
                                  q_dependency, q_bytes, locations,
                                  location_deltas, rotate_mode, name_prefix,
                                  positions):
    """Make the Q variants needed by the resident KV blocks visible in the DAG.

    KV itself never crosses the link: ``locations`` already names PIM-resident
    master/diff vectors.  GPU rotation sends one additional Q shard for every
    extra distinct delta; die rotation receives one raw Q and serializes
    shifted variants master-first at one cycle each; bank rotation is the
    requested zero-overhead local-rotate assumption.
    """
    if rotate_mode not in _ROTATE_MODES:
        raise WorkloadValidationError("unknown CacheBlend rotate mode '{}'".format(rotate_mode))
    targets = {}
    for location in locations:
        key = (location.key_address, location.value_address)
        delta = location_deltas.get(key, 0)  # generated decode KV has delta 0
        targets.setdefault(delta, set()).add(location.kind)
    if not targets:
        return q_dependency
    ordered = tuple(sorted(targets, key=lambda delta:
                           (0 if "master" in targets[delta] else 1, delta)))
    shifted = tuple(delta for delta in ordered if delta != 0)
    if not shifted:
        return q_dependency
    if rotate_mode == "gpu":
        # The regular Q link carries one variant.  Only the additional distinct
        # variants add traffic; no dense RoPE compute charge is applied.
        extra_variants = len(ordered) - 1
        if extra_variants == 0:
            return q_dependency
        extra_bytes = extra_variants * q_bytes
        transfer = _link_layer(x2g, name_prefix + "gpu_rotate_q_extra_to_pim",
                               extra_bytes)
        time_s, energy = system.devices["GPU"].get_time_and_energy(transfer)
        return _cacheblend_event(
            events, layer=layer, tier=tier, request=request,
            name=name_prefix + "gpu_rotate_q_extra_to_pim", device="LINK",
            rows=extra_variants, time_s=time_s, energy=energy,
            deps=(q_dependency,), link_bytes=extra_bytes, positions=positions)
    if rotate_mode == "bank":
        return _cacheblend_event(
            events, layer=layer, tier=tier, request=request,
            name=name_prefix + "bank_rotate_q_local", device="PIM", rows=len(shifted),
            time_s=0.0, energy=(), deps=(q_dependency,), positions=positions)

    # One die rotate unit: master targets issue before diff targets.  The
    # chain makes the later diff Q depend on every earlier shifted variant.
    last = q_dependency
    for delta in shifted:
        kind = "master" if "master" in targets[delta] else "diff"
        last = _cacheblend_event(
            events, layer=layer, tier=tier, request=request,
            name=name_prefix + "die_rotate_q_" + kind, device="DIE", rows=1,
            time_s=_DIE_ROTATE_CYCLE_S, energy=(), deps=(last,),
            positions=positions)
    return last


def _policy_corrected_rows(plan: ReusePlan, layer: int, request) -> set:
    """Return consumer positions whose reusable master KV is overlaid by diff.

    CacheBlend chooses correction rows independently per layer.  EPIC uses
    its deterministic leading correction prefix on each shifted segment.
    Both policies then share the same address-resolved master/diff DAG.
    """
    corrected = set()
    if plan.config.policy == "cacheblend":
        by_segment = plan.cacheblend_partial_rows.get(layer, {}).get(
            request.request_id, {})
        for index, rows in by_segment.items():
            offset = sum(segment.length for segment in request.segments[:index])
            corrected.update(offset + row for row in rows)
    elif plan.config.policy == "epic":
        decisions = {decision.segment_index: decision for decision in plan.reusable
                     if decision.request_id == request.request_id}
        for index, decision in decisions.items():
            offset = sum(segment.length for segment in request.segments[:index])
            corrected.update(offset + row for row in decision.epic_prefix_rows)
    return corrected


def _cacheblend_tlb_rows(workload: Workload, plan: ReusePlan, layer: int,
                         request, tlb: CacheBlendTLB, force_fresh: bool = False):
    decisions = {d.segment_index: d for d in plan.reusable
                 if d.request_id == request.request_id}
    corrected = _policy_corrected_rows(plan, layer, request)
    bindings = []
    position = 0
    for index, segment in enumerate(request.segments):
        decision = decisions.get(index)
        for row in range(segment.length):
            reused = decision is not None and not force_fresh
            if reused:
                kind = "diff" if position in corrected else "master"
                if kind == "diff":
                    # A correction is consumer-private: two workers may
                    # correct the same source row differently.
                    owner, owner_row = request.request_id, position
                else:
                    owner, owner_row = decision.owner_request_id, row
            else:
                # A producer's row is addressed by its offset inside the
                # fingerprinted segment, exactly as a later consumer resolves
                # it.  Keying it by absolute request position would place
                # producer writes and consumer reads at different rows of the
                # same master block.
                owner, owner_row = request.request_id, row
                # ``diff`` is an overlay, not a synonym for newly generated
                # KV.  Trace rows such as B-pos7/8 are new live KV and remain
                # in the master cache; only a corrected reused row shadows a
                # master row through the diff cache.
                kind = "master"
            location = tlb.locate(layer, owner, segment.fingerprint, owner_row, kind)
            if kind == "diff":
                # The master pool is streamed sequentially through the
                # shadowed row; the TLB supplies a mask rather than a hole.
                shadow = tlb.locate(layer, decision.owner_request_id,
                                    segment.fingerprint, row, "master")
                location = replace(location, shadow=shadow)
            tlb.bind(request.request_id, layer, position, location,
                     segment.position_delta, reused)
            bindings.append((position, reused, position in corrected, location))
            position += 1
    return bindings


def _reserve_cacheblend_tlb_rows(workload: Workload, plan: ReusePlan, layer: int,
                                 request, tlb: CacheBlendTLB,
                                 force_fresh: bool = False) -> None:
    """Reserve exactly the entries that ``_cacheblend_tlb_rows`` will bind."""
    decisions = {d.segment_index: d for d in plan.reusable
                 if d.request_id == request.request_id}
    corrected = _policy_corrected_rows(plan, layer, request)
    position = 0
    for index, segment in enumerate(request.segments):
        decision = decisions.get(index)
        for row in range(segment.length):
            reused = decision is not None and not force_fresh
            if reused:
                kind = "diff" if position in corrected else "master"
                if kind == "diff":
                    owner, owner_row = request.request_id, position
                    # The shadowed master row is still streamed (masked).
                    tlb.reserve(layer, decision.owner_request_id,
                                segment.fingerprint, row, "master")
                else:
                    owner, owner_row = decision.owner_request_id, row
            else:
                owner, owner_row = request.request_id, row
                # Keep reservations exactly consistent with the binding rule:
                # live/new rows are master, corrected reused rows are diff.
                kind = "master"
            tlb.reserve(layer, owner, segment.fingerprint, owner_row, kind)
            position += 1


def _prepare_cacheblend_tlb(workload: Workload, plan: ReusePlan, ndec: int,
                            tlb: CacheBlendTLB,
                            output_fingerprints: Mapping[str, str],
                            *, contiguous_no_reuse: bool = False) -> None:
    """Reserve all prefill and decode blocks before assigning physical bytes."""
    for layer in range(ndec):
        force_fresh = (plan.config.policy == "cacheblend" and
                       layer in plan.config.cacheblend_full_recompute_layers)
        for request in workload.requests:
            if contiguous_no_reuse:
                fingerprint = "{}::no-reuse-input".format(request.request_id)
                for position in range(request.total_length):
                    tlb.reserve(layer, request.request_id, fingerprint, position,
                                "private")
            else:
                _reserve_cacheblend_tlb_rows(workload, plan, layer, request, tlb,
                                             force_fresh=force_fresh)
            output_fingerprint = output_fingerprints.get(
                request.request_id, "{}::output".format(request.request_id))
            for output_row in range(request.lout):
                tlb.reserve(layer, request.request_id, output_fingerprint,
                            output_row, "master")
    tlb.finalize()


def _contiguous_no_reuse_tlb_rows(request, layer: int, tlb: CacheBlendTLB):
    """Bind one request's input to one contiguous physical master extent."""
    fingerprint = "{}::no-reuse-input".format(request.request_id)
    bindings = []
    for position in range(request.total_length):
        location = tlb.locate(layer, request.request_id, fingerprint, position,
                              "private")
        tlb.bind(request.request_id, layer, position, location, 0, False)
        bindings.append((position, False, False, location))
    return bindings


def _parent_output_fingerprints(workload: Workload) -> Dict[str, str]:
    """Return the one cache fingerprint under which each parent's output lives."""
    fingerprints: Dict[str, str] = {}
    for request in workload.requests:
        if request.parent_id is None:
            continue
        for segment in request.segments:
            if segment.role != "parent_out":
                continue
            previous = fingerprints.setdefault(request.parent_id, segment.fingerprint)
            if previous != segment.fingerprint:
                raise WorkloadValidationError(
                    "children of parent '{}' disagree on parent_out fingerprint".format(
                        request.parent_id))
    return fingerprints


def _append_cacheblend_decode(system, events: List[SplitEvent], tlb: CacheBlendTLB,
                              templates: Mapping[str, Layer], post: Sequence[Layer],
                              *, request, tier: int,
                              prefill_bindings: Mapping[int, Sequence[Tuple[int, bool, bool, KVLocation]]],
                              output_fingerprint: str, initial_deps: Sequence[str],
                              rotate_mode: str,
                              contiguous_no_reuse: bool = False) -> Tuple[str, ...]:
    """Materialize an agent's generated tokens and their per-layer KV cache.

    The parent output is no longer a logical-only reuse decision.  Every decode
    token writes K/V to the same ``(parent, fingerprint, output row, layer)``
    location subsequently resolved by a child's ``parent_out`` TLB entry.
    """
    qkv, score, softmax, context, x2g = (templates[name] for name in
                                         ("qkv", "score", "softmax", "context",
                                          "comm_x2g"))
    dbyte = qkv.dbyte
    local_hidden = system.model.hdim // system.model.tp
    heads = max(1, system.model.num_heads // system.model.tp)
    previous_output: Dict[int, List[KVLocation]] = {
        layer: [] for layer in range(system.model.ndec)}
    prefill_deltas = {
        layer: _prefill_location_deltas(request, prefill_bindings[layer])
        for layer in range(system.model.ndec)
    }
    token_deps: Tuple[str, ...] = tuple(initial_deps)
    for output_row in range(request.lout):
        layer_deps = token_deps
        for layer_index in range(system.model.ndec):
            output_location = tlb.locate(layer_index, request.request_id,
                                          output_fingerprint, output_row, "master")
            # Positions continue directly after the request prefill context.
            tlb.bind(request.request_id, layer_index,
                     request.total_length + output_row, output_location, 0, False)
            q = _gpu_layer_event(system, events, qkv, layer=layer_index, tier=tier,
                                 request=request.request_id, name="decode_qkv", rows=1,
                                 deps=layer_deps,
                                 positions=(request.total_length + output_row,))
            q_bytes = local_hidden * dbyte
            q_transfer = _link_layer(x2g, "decode_q_gpu_to_pim", q_bytes)
            time_s, energy = system.devices["GPU"].get_time_and_energy(q_transfer)
            q_link = _cacheblend_event(
                events, layer=layer_index, tier=tier, request=request.request_id,
                name="decode_q_gpu_to_pim", device="LINK", rows=1, time_s=time_s,
                energy=energy, deps=(q,), link_bytes=q_bytes,
                positions=(request.total_length + output_row,))
            # K/V is a QKV output, not an attention output.  Send it after
            # the critical Q transfer and allow it to overlap the local/PIM
            # attention path below.  The PIM write remains ordered after the
            # scan in event construction, but only depends on this transfer.
            kv_bytes = 2 * local_hidden * dbyte
            kv_transfer = _link_layer(x2g, "decode_kv_gpu_to_pim", kv_bytes)
            time_s, energy = system.devices["GPU"].get_time_and_energy(kv_transfer)
            kv_link = _cacheblend_event(
                events, layer=layer_index, tier=tier, request=request.request_id,
                name="decode_kv_gpu_to_pim", device="LINK", rows=1, time_s=time_s,
                energy=energy, deps=(q,), link_bytes=kv_bytes,
                positions=(request.total_length + output_row,),
                addresses=(output_location.key_address, output_location.value_address))

            local_last = q
            for template, name in ((score, "decode_gpu_local_score"),
                                   (softmax, "decode_gpu_local_softmax"),
                                   (context, "decode_gpu_local_context")):
                op = deepcopy(template)
                op.m, op.n, op.numOp = 1, 1, heads
                time_s, energy = system.devices["GPU"].get_time_and_energy(op)
                local_last = _cacheblend_event(
                    events, layer=layer_index, tier=tier, request=request.request_id,
                    name=name, device="GPU", rows=1, time_s=time_s, energy=energy,
                    deps=(local_last,), positions=(request.total_length + output_row,))
            tuple_bytes = heads * (system.model.dhead + 2) * dbyte
            tuple_transfer = _link_layer(x2g, "decode_gpu_partial_lse_to_pim",
                                          tuple_bytes)
            time_s, energy = system.devices["GPU"].get_time_and_energy(tuple_transfer)
            tuple_link = _cacheblend_event(
                events, layer=layer_index, tier=tier, request=request.request_id,
                name="decode_gpu_partial_lse_to_pim", device="LINK", rows=1,
                time_s=time_s, energy=energy, deps=(local_last,), link_bytes=tuple_bytes,
                positions=(request.total_length + output_row,))

            old = [location for _, _, _, location in prefill_bindings[layer_index]]
            old.extend(previous_output[layer_index])
            # ``old`` is the consumer-visible KV (one entry per position);
            # ``reads`` is what the master/diff pools physically stream, with
            # shadowed master rows masked rather than skipped.
            reads, masked_keys = _physical_reads(old)
            context_ready = local_last
            if old:
                rotate_ready = _append_q_rotate_distribution(
                    system, events, x2g, layer=layer_index, tier=tier,
                    request=request.request_id, q_dependency=q_link, q_bytes=q_bytes,
                    locations=reads, location_deltas=prefill_deltas[layer_index],
                    rotate_mode=rotate_mode, name_prefix="decode_",
                    positions=(request.total_length + output_row,))
                die_q = _cacheblend_event(
                    events, layer=layer_index, tier=tier, request=request.request_id,
                    name="decode_die_query_position_transform", device="DIE", rows=1,
                    time_s=q_bytes / system.devices["Acc"].softmax_peak_bandwidth,
                    energy=(q_bytes * system.devices["Acc"].energy_table["sram"],),
                    deps=(rotate_ready,), positions=(request.total_length + output_row,))
                op = deepcopy(score)
                op.m, op.n, op.k, op.numOp = 1, len(reads), system.model.dhead, heads
                op.pim_kv_runs = tlb.scan_runs(reads)
                plan_time_s, plan_energy = _tlb_plan_cost(op.pim_kv_runs)
                address_plan = _cacheblend_event(
                    events, layer=layer_index, tier=tier, request=request.request_id,
                    name=("decode_contiguous_address_plan" if contiguous_no_reuse
                          else "decode_tlb_lookup_and_bank_plan"),
                    device=("ADDR" if contiguous_no_reuse else "TLB"), rows=len(old),
                    time_s=(0.0 if contiguous_no_reuse else plan_time_s),
                    energy=(() if contiguous_no_reuse else plan_energy),
                    deps=(die_q,), positions=(request.total_length + output_row,),
                    addresses=[address for location in reads
                               for address in (location.key_address, location.value_address)])
                scan = _append_physical_pim_scan(
                    system, events, op=op, layer=layer_index, tier=tier,
                    request=request.request_id,
                    name="decode_pim_kv_scan_score_softmax_pv", rows=len(reads),
                    deps=(address_plan,), positions=(request.total_length + output_row,),
                    runs=op.pim_kv_runs,
                    masked=_masked_rows_per_run(op.pim_kv_runs, masked_keys,
                                                tlb.bytes_per_vector))
                # Every physical run yields one local softmax tuple; the DIE
                # merges those with the GPU tuple.
                merge_width = len(scan) + 1
                die_merge = _cacheblend_event(
                    events, layer=layer_index, tier=tier, request=request.request_id,
                    name="decode_die_lse_merge", device="DIE", rows=1,
                    time_s=(merge_width * tuple_bytes /
                            system.devices["Acc"].softmax_peak_bandwidth),
                    energy=(merge_width * tuple_bytes *
                            system.devices["Acc"].energy_table["sram"],),
                    deps=tuple(scan) + (tuple_link,),
                    positions=(request.total_length + output_row,))
                ctx_transfer = _link_layer(x2g, "decode_ctx_pim_to_gpu", q_bytes)
                time_s, energy = system.devices["GPU"].get_time_and_energy(ctx_transfer)
                context_ready = _cacheblend_event(
                    events, layer=layer_index, tier=tier, request=request.request_id,
                    name="decode_ctx_pim_to_gpu", device="LINK", rows=1,
                    time_s=time_s, energy=energy, deps=(die_merge,), link_bytes=q_bytes,
                    positions=(request.total_length + output_row,))

            post_last = _post_attention_gpu(
                system, events, post, layer=layer_index, tier=tier,
                request=request.request_id, rows=1, dependency=context_ready,
                positions=(request.total_length + output_row,))
            store = _cacheblend_event(
                events, layer=layer_index, tier=tier, request=request.request_id,
                name="decode_dram_store_master", device="PIM", rows=1,
                time_s=kv_bytes / system.devices["Acc"].peak_memory_bandwidth,
                energy=(kv_bytes * system.devices["Acc"].energy_table["mem"],),
                deps=(kv_link,), positions=(request.total_length + output_row,),
                addresses=(output_location.key_address, output_location.value_address))
            previous_output[layer_index].append(output_location)
            layer_deps = (post_last, store)
        token_deps = layer_deps
    return token_deps


def _append_cacheblend_decode_batched(
        system, events: List[SplitEvent], tlb: CacheBlendTLB,
        templates: Mapping[str, Layer], post: Sequence[Layer], *, tier: int,
        inputs: Sequence[Tuple[Any, Mapping[int, Sequence[Tuple[int, bool, bool, KVLocation]]],
                               str, Sequence[str]]], batch_size: int,
        batch_records: List[Dict[str, Any]], rotate_mode: str,
        pipe: bool, contiguous_no_reuse: bool = False,
        pim_batch_command: str = "replicate",
        pim_pe_freq_ghz: float = MQ_DEFAULT_PE_FREQ_GHZ,
        gemv_buffer_bytes: int = MQ_DEFAULT_GEMV_BUFFER_BYTES) -> Dict[str, Tuple[str, ...]]:
    """Decode a tier, admitting PIM-attention batches by real Q arrival.

    Q does not exist until QKV has run, so QKV is first batched by its GPU
    input-ready order.  All resulting GPU->PIM Q/KV transfers are then
    scheduled on AttAcc's actual link resource.  Only after that do we form
    attention/FFN batches: repeatedly take the earliest ``batch_size`` Q-link
    completions from the global ready queue.  This deliberately separates the
    upstream GPU batch from the downstream PIM-ready batch.
    """
    if batch_size < 2:
        raise WorkloadValidationError("batched CacheBlend decode requires batch_size >= 2")
    qkv, score, softmax, context, x2g = (templates[name] for name in
                                         ("qkv", "score", "softmax", "context",
                                          "comm_x2g"))
    dbyte = qkv.dbyte
    local_hidden = system.model.hdim // system.model.tp
    heads = max(1, system.model.num_heads // system.model.tp)
    q_bytes = local_hidden * dbyte
    kv_bytes = 2 * q_bytes
    tuple_bytes = heads * (system.model.dhead + 2) * dbyte
    requests = [item[0] for item in inputs]
    bindings = {item[0].request_id: item[1] for item in inputs}
    fingerprints = {item[0].request_id: item[2] for item in inputs}
    token_deps = {item[0].request_id: tuple(item[3]) for item in inputs}
    previous_output: Dict[str, Dict[int, List[KVLocation]]] = {
        request.request_id: {layer: [] for layer in range(system.model.ndec)}
        for request in requests
    }
    # Cached prefix state for the real, append-only AttAcc event schedule.
    # It avoids replaying a long decode trace solely to find newly emitted Q
    # completion times.
    provisional_finish: Dict[str, float] = {}
    provisional_availability: Dict[str, float] = {}
    provisional_index = 0

    def batch_request_label(output_row: int, layer: int, ordinal: int) -> str:
        return "batch:t{}:o{}:l{}:g{}".format(tier, output_row, layer, ordinal)

    for output_row in range(max((request.lout for request in requests), default=0)):
        layer_deps = {request.request_id: token_deps[request.request_id]
                      for request in requests if output_row < request.lout}
        for layer_index in range(system.model.ndec):
            active = [request for request in requests if output_row < request.lout]
            next_layer_deps: Dict[str, Tuple[str, ...]] = {}
            # Stage A: QKV is a GPU operation, so it can only be batched from
            # the preceding hidden-state readiness.  It emits every Q/KV link
            # before any PIM attention is admitted.
            qkv_aliases: Dict[str, str] = {}
            q_links: Dict[str, str] = {}
            kv_links: Dict[str, str] = {}
            output_locations: Dict[str, KVLocation] = {}
            qkv_batch_members: Dict[str, Tuple[str, ...]] = {}
            for group_index in range(0, len(active), batch_size):
                group = active[group_index:group_index + batch_size]
                members = tuple(request.request_id for request in group)
                gpu_label = "gpu-" + batch_request_label(
                    output_row, layer_index, group_index // batch_size)
                positions = tuple(request.total_length + output_row for request in group)
                op = deepcopy(qkv)
                op.m = len(group)
                time_s, energy = system.devices["GPU"].get_time_and_energy(op)
                batch_qkv = _cacheblend_event(
                    events, layer=layer_index, tier=tier, request=gpu_label,
                    name="decode_batch_qkv", device="GPU", rows=len(group),
                    time_s=time_s, energy=energy,
                    deps=tuple(dep for request in group
                               for dep in layer_deps[request.request_id]),
                    positions=positions, batch_members=members)
                for request in group:
                    request_id = request.request_id
                    position = request.total_length + output_row
                    qkv_batch_members[request_id] = members
                    qkv_aliases[request_id] = _cacheblend_event(
                        events, layer=layer_index, tier=tier, request=request_id,
                        name="decode_qkv", device="GPU", rows=1, time_s=0.0,
                        energy=(), deps=(batch_qkv,), positions=(position,))
                    location = tlb.locate(layer_index, request_id,
                                          fingerprints[request_id], output_row,
                                          "master")
                    tlb.bind(request_id, layer_index, position, location, 0, False)
                    output_locations[request_id] = location
                    transfer = _link_layer(x2g, "decode_q_gpu_to_pim", q_bytes)
                    time_s, energy = system.devices["GPU"].get_time_and_energy(transfer)
                    q_links[request_id] = _cacheblend_event(
                        events, layer=layer_index, tier=tier, request=request_id,
                        name="decode_q_gpu_to_pim", device="LINK", rows=1,
                        time_s=time_s, energy=energy, deps=(qkv_aliases[request_id],),
                        link_bytes=q_bytes, positions=(position,))
                    transfer = _link_layer(x2g, "decode_kv_gpu_to_pim", kv_bytes)
                    time_s, energy = system.devices["GPU"].get_time_and_energy(transfer)
                    kv_links[request_id] = _cacheblend_event(
                        events, layer=layer_index, tier=tier, request=request_id,
                        name="decode_kv_gpu_to_pim", device="LINK", rows=1,
                        time_s=time_s, energy=energy, deps=(qkv_aliases[request_id],),
                        link_bytes=kv_bytes, positions=(position,),
                        addresses=(location.key_address, location.value_address))

            old_by_request = {
                request.request_id: [location for _, _, _, location
                                     in bindings[request.request_id][layer_index]] +
                previous_output[request.request_id][layer_index]
                for request in active
            }
            # Physical master/diff streams per request (masked shadow rows
            # included) -- see ``_physical_reads``.
            reads_by_request = {
                request.request_id: _physical_reads(old_by_request[request.request_id])
                for request in active
            }
            # GPU rotation creates additional Q variants that have their own
            # GPU->PIM transfers.  Emit them before admission, so the ready
            # timestamp denotes the final required external Q arrival.
            rotate_ready = {
                request.request_id: _append_q_rotate_distribution(
                    system, events, x2g, layer=layer_index, tier=tier,
                    request=request.request_id, q_dependency=q_links[request.request_id],
                    q_bytes=q_bytes, locations=reads_by_request[request.request_id][0],
                    location_deltas=_prefill_location_deltas(
                        request, bindings[request.request_id][layer_index]),
                    rotate_mode=rotate_mode, name_prefix="decode_",
                    positions=(request.total_length + output_row,))
                for request in active if old_by_request[request.request_id]
            }
            # This provisional schedule is intentionally taken after every
            # external Q link (including GPU-rotated variants) has been
            # emitted.  Later attention events cannot move a completed link
            # on its ordered resource.
            provisional_finish, provisional_availability = _schedule_cacheblend_incremental(
                events, pipe=pipe, start_index=provisional_index,
                finish=provisional_finish, availability=provisional_availability)
            provisional_index = len(events)
            input_rank = {request.request_id: index for index, request in enumerate(active)}
            ready = sorted(active, key=lambda request:
                           (provisional_finish[
                                rotate_ready[request.request_id]
                                if rotate_mode == "gpu" and request.request_id in rotate_ready
                                else q_links[request.request_id]],
                            input_rank[request.request_id]))

            # Stage B: attention, context and FFN use the global Q-ready
            # order.  A later request can therefore join an earlier PIM batch
            # when its actual link completes first.
            for group_index in range(0, len(ready), batch_size):
                group = ready[group_index:group_index + batch_size]
                members = tuple(request.request_id for request in group)
                label = batch_request_label(output_row, layer_index,
                                            group_index // batch_size)
                positions = tuple(request.total_length + output_row for request in group)
                batch_records.append({
                    "id": label, "tier": tier, "output_row": output_row,
                    "transformer_layer": layer_index, "members": list(members),
                    "query_positions": {request.request_id:
                                        request.total_length + output_row
                                        for request in group},
                    "size": len(group),
                    "admission": "global-q-ready-queue",
                    "qkv_batch_members": {request.request_id:
                                          list(qkv_batch_members[request.request_id])
                                          for request in group},
                })

                tuple_links: Dict[str, str] = {}
                local_last = None
                for template, name in ((score, "decode_batch_gpu_local_score"),
                                       (softmax, "decode_batch_gpu_local_softmax"),
                                       (context, "decode_batch_gpu_local_context")):
                    op = deepcopy(template)
                    op.m, op.n, op.numOp = len(group), 1, heads
                    time_s, energy = system.devices["GPU"].get_time_and_energy(op)
                    local_last = _cacheblend_event(
                        events, layer=layer_index, tier=tier, request=label,
                        name=name, device="GPU", rows=len(group), time_s=time_s,
                        energy=energy,
                        deps=(tuple(qkv_aliases[request.request_id] for request in group)
                              if local_last is None else (local_last,)), positions=positions,
                        batch_members=members)
                for request in group:
                    request_id = request.request_id
                    position = request.total_length + output_row
                    transfer = _link_layer(x2g, "decode_gpu_partial_lse_to_pim", tuple_bytes)
                    time_s, energy = system.devices["GPU"].get_time_and_energy(transfer)
                    tuple_links[request_id] = _cacheblend_event(
                        events, layer=layer_index, tier=tier, request=request_id,
                        name="decode_gpu_partial_lse_to_pim", device="LINK", rows=1,
                        time_s=time_s, energy=energy, deps=(local_last,),
                        link_bytes=tuple_bytes, positions=(position,))

                # A shared master stream is common to the group when every
                # member reads the same physical master rows; each member's
                # own correction mask stays query-private, exactly like Q.
                common_keys = None
                for request in group:
                    keys = {_address_key(location)
                            for location in reads_by_request[request.request_id][0]
                            if location.kind == "master"}
                    common_keys = keys if common_keys is None else common_keys & keys
                common = [location for location in reads_by_request[group[0].request_id][0]
                          if _address_key(location) in (common_keys or set())]
                # The mask is query-private; the shared event reports the
                # rows masked for at least one member of the batch.
                common_masked = set().union(*(reads_by_request[request.request_id][1]
                                              for request in group))
                scan_deps: Dict[str, List[str]] = {request.request_id: [] for request in group}
                if common:
                    common_addresses = [address for location in common
                                        for address in (location.key_address, location.value_address)]
                    # A sweep can hold at most the queries whose slices fit
                    # the per-bank GEMV buffer; a larger admitted batch is
                    # served by consecutive sweeps over the same rows
                    # (Fugue: "beyond which the sweep splits").
                    sweep_cap = mq_query_capacity(gemv_buffer_bytes)
                    for sweep_start in range(0, len(group), sweep_cap):
                        sweep = group[sweep_start:sweep_start + sweep_cap]
                        sweep_members = tuple(request.request_id for request in sweep)
                        sweep_positions = tuple(request.total_length + output_row
                                                for request in sweep)
                        die_qs = []
                        for request in sweep:
                            request_id = request.request_id
                            position = request.total_length + output_row
                            die_qs.append(_cacheblend_event(
                                events, layer=layer_index, tier=tier, request=request_id,
                                name="decode_die_query_position_transform", device="DIE", rows=1,
                                time_s=q_bytes / system.devices["Acc"].softmax_peak_bandwidth,
                                energy=(q_bytes * system.devices["Acc"].energy_table["sram"],),
                                deps=(rotate_ready.get(request_id, q_links[request_id]),), positions=(position,)))
                        op = deepcopy(score)
                        op.m, op.n, op.k, op.numOp = len(sweep), len(common), system.model.dhead, heads
                        op.pim_kv_runs = tlb.scan_runs(common)
                        plan_time_s, plan_energy = _tlb_plan_cost(op.pim_kv_runs)
                        tlb_event = _cacheblend_event(
                            events, layer=layer_index, tier=tier, request=label,
                            name="decode_batch_tlb_lookup_and_bank_plan", device="TLB",
                            rows=len(common), time_s=plan_time_s,
                            energy=plan_energy, deps=tuple(die_qs),
                            positions=sweep_positions, addresses=common_addresses,
                            batch_members=sweep_members)
                        op.pim_shared_kv = True
                        op.pim_shared_queries = len(sweep)
                        _apply_pim_batch(op, pim_batch_command, pim_pe_freq_ghz)
                        shared_scan = _append_physical_pim_scan(
                            system, events, op=op, layer=layer_index, tier=tier,
                            request=label,
                            name="decode_batch_pim_kv_scan_score_softmax_pv", rows=len(common),
                            deps=(tlb_event,), positions=sweep_positions, runs=op.pim_kv_runs,
                            batch_members=sweep_members,
                            masked=_masked_rows_per_run(op.pim_kv_runs, common_masked,
                                                        tlb.bytes_per_vector))
                        for request in sweep:
                            scan_deps[request.request_id].extend(shared_scan)

                for request in group:
                    request_id = request.request_id
                    position = request.total_length + output_row
                    private = [location for location in reads_by_request[request_id][0]
                               if _address_key(location) not in (common_keys or set())]
                    private_masked = reads_by_request[request_id][1]
                    if private:
                        die_q = _cacheblend_event(
                            events, layer=layer_index, tier=tier, request=request_id,
                            name="decode_die_query_position_transform", device="DIE", rows=1,
                            time_s=q_bytes / system.devices["Acc"].softmax_peak_bandwidth,
                            energy=(q_bytes * system.devices["Acc"].energy_table["sram"],),
                            deps=(rotate_ready.get(request_id, q_links[request_id]),), positions=(position,))
                        addresses = [address for location in private
                                     for address in (location.key_address, location.value_address)]
                        op = deepcopy(score)
                        op.m, op.n, op.k, op.numOp = 1, len(private), system.model.dhead, heads
                        op.pim_kv_runs = tlb.scan_runs(private)
                        plan_time_s, plan_energy = _tlb_plan_cost(op.pim_kv_runs)
                        address_plan = _cacheblend_event(
                            events, layer=layer_index, tier=tier, request=request_id,
                            name=("decode_contiguous_address_plan" if contiguous_no_reuse
                                  else "decode_tlb_lookup_and_bank_plan"),
                            device=("ADDR" if contiguous_no_reuse else "TLB"), rows=len(private),
                            time_s=(0.0 if contiguous_no_reuse else plan_time_s),
                            energy=(() if contiguous_no_reuse else plan_energy),
                            deps=(die_q,), positions=(position,), addresses=addresses)
                        scan_deps[request_id].extend(_append_physical_pim_scan(
                            system, events, op=op, layer=layer_index, tier=tier,
                            request=request_id, name="decode_pim_kv_scan_score_softmax_pv",
                            rows=len(private), deps=(address_plan,), positions=(position,),
                            runs=op.pim_kv_runs,
                            masked=_masked_rows_per_run(op.pim_kv_runs, private_masked,
                                                        tlb.bytes_per_vector)))

                context_links: Dict[str, str] = {}
                for request in group:
                    request_id = request.request_id
                    position = request.total_length + output_row
                    contribution = scan_deps[request_id]
                    if contribution:
                        # One local softmax tuple per physical run plus the
                        # GPU tuple.
                        merge_width = len(contribution) + 1
                        merge = _cacheblend_event(
                            events, layer=layer_index, tier=tier, request=request_id,
                            name="decode_die_lse_merge", device="DIE", rows=1,
                            time_s=(merge_width * tuple_bytes /
                                    system.devices["Acc"].softmax_peak_bandwidth),
                            energy=(merge_width * tuple_bytes *
                                    system.devices["Acc"].energy_table["sram"],),
                            deps=tuple(contribution + [tuple_links[request_id]]),
                            positions=(position,))
                        context_deps = (merge,)
                    else:
                        context_deps = (tuple_links[request_id],)
                    transfer = _link_layer(x2g, "decode_ctx_pim_to_gpu", q_bytes)
                    time_s, energy = system.devices["GPU"].get_time_and_energy(transfer)
                    context_links[request_id] = _cacheblend_event(
                        events, layer=layer_index, tier=tier, request=request_id,
                        name="decode_ctx_pim_to_gpu", device="LINK", rows=1,
                        time_s=time_s, energy=energy, deps=context_deps,
                        link_bytes=q_bytes, positions=(position,))

                post_last = None
                for template in post:
                    if template.name in ("qkv", "score", "softmax", "context", "comm_x2g"):
                        continue
                    op = deepcopy(template)
                    op.m = len(group)
                    time_s, energy = system.devices["GPU"].get_time_and_energy(op)
                    post_last = _cacheblend_event(
                        events, layer=layer_index, tier=tier, request=label,
                        name="decode_batch_gpu_" + template.name, device="GPU", rows=len(group),
                        time_s=time_s, energy=energy,
                        deps=tuple(context_links.values()) if post_last is None else (post_last,),
                        positions=positions, batch_members=members)
                if post_last is None:
                    raise WorkloadValidationError("CacheBlend post-attention GPU sequence is empty")
                for request in group:
                    request_id = request.request_id
                    position = request.total_length + output_row
                    location = output_locations[request_id]
                    store = _cacheblend_event(
                        events, layer=layer_index, tier=tier, request=request_id,
                        name="decode_dram_store_master", device="PIM", rows=1,
                        time_s=kv_bytes / system.devices["Acc"].peak_memory_bandwidth,
                        energy=(kv_bytes * system.devices["Acc"].energy_table["mem"],),
                        deps=(kv_links[request_id],), positions=(position,),
                        addresses=(location.key_address, location.value_address))
                    previous_output[request_id][layer_index].append(location)
                    next_layer_deps[request_id] = (post_last, store)
            layer_deps = next_layer_deps
        token_deps.update(layer_deps)
    return {request.request_id: token_deps[request.request_id] for request in requests}


def _append_physical_no_reuse_prefill_layer(
        system, events: List[SplitEvent], tlb: CacheBlendTLB,
        templates: Mapping[str, Layer], post: Sequence[Layer], *, layer: int,
        tier: int, request, bindings: Sequence[Tuple[int, bool, bool, KVLocation]],
        initial_deps: Sequence[str]) -> Tuple[str, str]:
    """Emit one continuous PIM attention scan for a no-reuse prefill layer.

    This deliberately uses the CacheBlend physical DAG and scheduler, but a
    single contiguous TLB extent rather than a set of reused/diff extents.
    The ``m=n=L`` score shape matches the existing full-prefill abstraction;
    it must be kept identical when comparing this baseline with CacheBlend.
    """
    qkv, score, x2g = (templates[name] for name in ("qkv", "score", "comm_x2g"))
    rows = request.total_length
    dbyte = qkv.dbyte
    local_hidden = system.model.hdim // system.model.tp
    heads = max(1, system.model.num_heads // system.model.tp)
    positions = range(rows)
    q = _gpu_layer_event(system, events, qkv, layer=layer, tier=tier,
                         request=request.request_id, name="qkv", rows=rows,
                         deps=initial_deps, positions=positions)
    q_bytes = rows * local_hidden * dbyte
    q_transfer = _link_layer(x2g, "q_gpu_to_pim", q_bytes)
    time_s, energy = system.devices["GPU"].get_time_and_energy(q_transfer)
    q_link = _cacheblend_event(
        events, layer=layer, tier=tier, request=request.request_id,
        name="q_gpu_to_pim", device="LINK", rows=rows, time_s=time_s,
        energy=energy, deps=(q,), link_bytes=q_bytes, positions=positions)
    kv_bytes = 2 * q_bytes
    kv_transfer = _link_layer(x2g, "kv_gpu_to_pim", kv_bytes)
    time_s, energy = system.devices["GPU"].get_time_and_energy(kv_transfer)
    addresses = [address for _, _, _, location in bindings
                 for address in (location.key_address, location.value_address)]
    kv_link = _cacheblend_event(
        events, layer=layer, tier=tier, request=request.request_id,
        name="kv_gpu_to_pim", device="LINK", rows=rows, time_s=time_s,
        energy=energy, deps=(q,), link_bytes=kv_bytes, positions=positions,
        addresses=addresses)
    address_plan = _cacheblend_event(
        events, layer=layer, tier=tier, request=request.request_id,
        name="contiguous_address_plan", device="ADDR", rows=rows,
        time_s=0.0, energy=(), deps=(q_link,),
        positions=positions, addresses=addresses)
    op = deepcopy(score)
    op.m, op.n, op.k, op.numOp = rows, rows, system.model.dhead, heads
    locations = [location for _, _, _, location in bindings]
    op.pim_kv_runs = tlb.scan_runs(locations)
    op.pim_shared_kv = True
    op.pim_shared_queries = rows
    time_s, energy = system.devices["Acc"].get_time_and_energy(op)
    scan = _cacheblend_event(
        events, layer=layer, tier=tier, request=request.request_id,
        name="pim_kv_scan_score_softmax_pv", device="PIM", rows=rows,
        time_s=time_s, energy=energy, deps=(address_plan,), positions=positions,
        addresses=addresses)
    ctx_transfer = _link_layer(x2g, "ctx_pim_to_gpu", q_bytes)
    time_s, energy = system.devices["GPU"].get_time_and_energy(ctx_transfer)
    ctx_link = _cacheblend_event(
        events, layer=layer, tier=tier, request=request.request_id,
        name="ctx_pim_to_gpu", device="LINK", rows=rows, time_s=time_s,
        energy=energy, deps=(scan,), link_bytes=q_bytes, positions=positions)
    post_last = _post_attention_gpu(
        system, events, post, layer=layer, tier=tier, request=request.request_id,
        rows=rows, dependency=ctx_link, positions=positions)
    store = _cacheblend_event(
        events, layer=layer, tier=tier, request=request.request_id,
        name="dram_store_master", device="PIM", rows=rows,
        time_s=kv_bytes / system.devices["Acc"].peak_memory_bandwidth,
        energy=(kv_bytes * system.devices["Acc"].energy_table["mem"],),
        deps=(kv_link,), positions=positions, addresses=addresses)
    return post_last, store


def summarize_cacheblend_schedule(scheduled: Sequence[SplitEvent],
                                  workload: Workload) -> Dict[str, Any]:
    """Compact per-request / per-tier completion times of a scheduled DAG.

    ``prefill_end_s`` is the last non-decode event of the request,
    ``first_token_s`` the completion of its first generated token (last event
    at query position ``total_length``), and ``end_s`` its final event.  Batch
    events are attributed to every member.
    """
    per_request: Dict[str, Dict[str, float]] = {
        request.request_id: {"tier": request.tier, "prefill_end_s": 0.0,
                             "first_token_s": 0.0, "end_s": 0.0}
        for request in workload.requests}
    first_position = {request.request_id: request.total_length
                      for request in workload.requests}
    for event in scheduled:
        members = event.batch_members or (event.request_id,)
        for index, member in enumerate(members):
            record = per_request.get(member)
            if record is None:
                continue
            record["end_s"] = max(record["end_s"], event.end_s)
            if event.name.startswith("decode_"):
                # Batch events list one query position per member, in member
                # order; a private event lists only its own positions.
                if event.batch_members and len(event.query_positions) == len(members):
                    positions = (event.query_positions[index],)
                else:
                    positions = event.query_positions
                if first_position[member] in positions:
                    record["first_token_s"] = max(record["first_token_s"], event.end_s)
            else:
                record["prefill_end_s"] = max(record["prefill_end_s"], event.end_s)
    tiers: Dict[str, Dict[str, float]] = {}
    for request in workload.requests:
        record = per_request[request.request_id]
        tier = tiers.setdefault(str(request.tier), {"start_s": None, "end_s": 0.0,
                                                    "requests": 0})
        tier["end_s"] = max(tier["end_s"], record["end_s"])
        tier["requests"] += 1
    for event in scheduled:
        tier = tiers.get(str(event.tier))
        if tier is not None:
            tier["start_s"] = (event.start_s if tier["start_s"] is None
                               else min(tier["start_s"], event.start_s))
    return {"requests": per_request, "tiers": tiers}


def _run_cacheblend_prefill(system, workload: Workload, plan: ReusePlan,
                            *, pipe: bool, batch_size: int = 1,
                            physical_no_reuse: bool = False,
                            rotate_mode: str = "gpu",
                            include_events: bool = True,
                            pim_batch_command: str = "replicate",
                            pim_pe_freq_ghz: float = MQ_DEFAULT_PE_FREQ_GHZ,
                            gemv_buffer_bytes: int = MQ_DEFAULT_GEMV_BUFFER_BYTES) -> Dict[str, Any]:
    if system.hetero_name != DeviceType.PIM:
        raise WorkloadValidationError("reuse prefill requires --system dgx-attacc")
    validate_reuse_plan(workload, plan, system.model.ndec)
    _validate_layer_config(plan, system.model.ndec)
    system.model.build(1, 1, 2, True)
    templates = {layer.name: layer for layer in system.model.sum_decoder}
    qkv, score, softmax, context = (templates[name] for name in
                                    ("qkv", "score", "softmax", "context"))
    x2g = templates["comm_x2g"]
    post = list(system.model.sum_decoder)
    dbyte, local_hidden = qkv.dbyte, system.model.hdim // system.model.tp
    heads = max(1, system.model.num_heads // system.model.tp)
    # Ramulator's original AttAcc generator receives the address of one head
    # vector and distributes heads across channels / 8-KiB partitions itself.
    # Using the concatenated local-hidden vector here would consume one K/V
    # address interval per *all-head* token and incorrectly overflow the
    # fixed 8-MiB K-to-V window for long contexts.
    tlb = (NoReuseKVLayout(system.model.dhead * dbyte)
           if physical_no_reuse else CacheBlendTLB(system.model.dhead * dbyte))
    events: List[SplitEvent] = []
    previous_tier_done: Tuple[str, ...] = ()
    parent_output_fingerprints = _parent_output_fingerprints(workload)
    _prepare_cacheblend_tlb(workload, plan, system.model.ndec, tlb,
                            parent_output_fingerprints,
                            contiguous_no_reuse=physical_no_reuse)
    if batch_size < 1:
        raise WorkloadValidationError("--cacheblend-batch-size must be at least 1")
    if rotate_mode not in _ROTATE_MODES:
        raise WorkloadValidationError("--cacheblend-rotate-mode must be one of {}".format(
            ", ".join(_ROTATE_MODES)))
    if pim_batch_command not in ("replicate", "mq"):
        raise WorkloadValidationError(
            "--pim-batch-command must be 'replicate' or 'mq'")
    if pim_pe_freq_ghz <= 0:
        raise WorkloadValidationError("--pe-freq-ghz must be positive")
    if gemv_buffer_bytes < 64:
        raise WorkloadValidationError(
            "--gemv-buffer-bytes must hold at least one 64-B query slice")
    batch_records: List[Dict[str, Any]] = []

    for tier, requests, _, _ in _tier_shapes(workload):
        tier_done: List[str] = []
        decode_inputs = []
        for request in requests:
            request_ready: Tuple[str, ...] = previous_tier_done
            # A layer's K/V writeback is not an input to the next layer's GPU
            # QKV.  Keep it pending so it can overlap that compute exactly as
            # in the CacheBlend trace, then join it before decode first reads
            # the completed prefill cache.
            prefill_store_events: List[str] = []
            prefill_bindings: Dict[int, Sequence[Tuple[int, bool, bool, KVLocation]]] = {}
            for layer_index in range(system.model.ndec):
                full = (plan.config.policy == "cacheblend" and
                        layer_index in plan.config.cacheblend_full_recompute_layers)
                bindings = (_contiguous_no_reuse_tlb_rows(request, layer_index, tlb)
                            if physical_no_reuse else
                            _cacheblend_tlb_rows(workload, plan, layer_index,
                                                 request, tlb, force_fresh=full))
                prefill_bindings[layer_index] = bindings
                if physical_no_reuse:
                    request_ready = _append_physical_no_reuse_prefill_layer(
                        system, events, tlb, templates, post, layer=layer_index,
                        tier=tier, request=request, bindings=bindings,
                        initial_deps=request_ready)
                    continue
                reusable = [item for item in bindings if item[1]]
                # A partial layer without an old cache is simply an ordinary
                # GPU prefill; do not fabricate PIM traffic for it.
                if full or not reusable:
                    q = _gpu_layer_event(system, events, qkv, layer=layer_index,
                                         tier=tier, request=request.request_id,
                                         name="qkv", rows=request.total_length,
                                         deps=request_ready,
                                         positions=range(request.total_length))
                    # QKV produces K/V as well as Q.  Put new KV on the link
                    # immediately, in parallel with GPU attention; its PIM
                    # write is emitted later so it cannot delay the critical
                    # attention scan on our single-PIM-resource model.
                    kv_bytes = request.total_length * 2 * local_hidden * dbyte
                    transfer = _link_layer(x2g, "kv_gpu_to_pim", kv_bytes)
                    time_s, energy = system.devices["GPU"].get_time_and_energy(transfer)
                    kv_link = _cacheblend_event(
                        events, layer=layer_index, tier=tier,
                        request=request.request_id, name="kv_gpu_to_pim",
                        device="LINK", rows=request.total_length, time_s=time_s,
                        energy=energy, deps=(q,), link_bytes=kv_bytes,
                        positions=range(request.total_length),
                        addresses=[address for _, _, _, loc in bindings
                                   for address in (loc.key_address, loc.value_address)])
                    attn_last = q
                    for template, name in ((score, "gpu_score"),
                                           (softmax, "gpu_softmax"),
                                           (context, "gpu_context")):
                        op = deepcopy(template)
                        op.m = request.total_length
                        op.n = request.total_length
                        time_s, energy = system.devices["GPU"].get_time_and_energy(op)
                        attn_last = _cacheblend_event(
                            events, layer=layer_index, tier=tier,
                            request=request.request_id, name=name, device="GPU",
                            rows=request.total_length, time_s=time_s, energy=energy,
                            deps=(attn_last,), positions=range(request.total_length))
                    post_last = _post_attention_gpu(
                        system, events, post, layer=layer_index, tier=tier,
                        request=request.request_id, rows=request.total_length,
                        dependency=attn_last, positions=range(request.total_length))
                    store = _append_channel_kv_stores(
                        system, events, layer=layer_index, tier=tier,
                        request=request.request_id, name="dram_store_master",
                        locations=[loc for _, _, _, loc in bindings], dbyte=dbyte,
                        deps=(kv_link,), positions=range(request.total_length))
                    prefill_store_events.extend(store)
                    request_ready = (post_last,)
                    continue

                compute_positions = [position for position, reused, corrected, _ in bindings
                                     if not reused or corrected]
                q = _gpu_layer_event(system, events, qkv, layer=layer_index,
                                     tier=tier, request=request.request_id,
                                     name="qkv", rows=len(compute_positions),
                                     deps=request_ready, positions=compute_positions)
                q_bytes = len(compute_positions) * local_hidden * dbyte
                q_transfer = _link_layer(x2g, "q_gpu_to_pim", q_bytes)
                time_s, energy = system.devices["GPU"].get_time_and_energy(q_transfer)
                q_link = _cacheblend_event(events, layer=layer_index, tier=tier,
                                            request=request.request_id,
                                            name="q_gpu_to_pim", device="LINK",
                                            rows=len(compute_positions), time_s=time_s,
                                            energy=energy, deps=(q,), link_bytes=q_bytes,
                                            positions=compute_positions)
                # Q gets the link first because it is on the PIM-attention
                # critical path.  K/V is already available at QKV completion,
                # so its transfer may proceed while GPU/PIM attention runs.
                kv_bytes = len(compute_positions) * 2 * local_hidden * dbyte
                writes = [loc for pos, reused, corrected, loc in bindings
                          if pos in compute_positions]
                kv_transfer = _link_layer(x2g, "kv_gpu_to_pim", kv_bytes)
                time_s, energy = system.devices["GPU"].get_time_and_energy(kv_transfer)
                kv_link = _cacheblend_event(
                    events, layer=layer_index, tier=tier, request=request.request_id,
                    name="kv_gpu_to_pim", device="LINK", rows=len(compute_positions),
                    time_s=time_s, energy=energy, deps=(q,), link_bytes=kv_bytes,
                    positions=compute_positions, addresses=[address for loc in writes
                                                            for address in (loc.key_address,
                                                                            loc.value_address)])
                # The GPU attends the fresh rows to each other as one
                # rectangular block, exactly like an ordinary prefill over
                # ``len(compute_positions)`` tokens (m = n = fresh rows, the
                # same convention as the no-reuse path).  Old cache rows are
                # deliberately excluded from this GPU op: they are scanned by
                # the PIM and merged at the DIE.  A per-token m = 1 loop would
                # re-stream the whole fresh K/V from GPU memory for every
                # query and overstate this stage by more than an order of
                # magnitude for a mostly-fresh request.
                fresh_rows = len(compute_positions)
                local_last = q
                for template, name in ((score, "gpu_local_score"),
                                       (softmax, "gpu_local_softmax"),
                                       (context, "gpu_local_context")):
                    op = deepcopy(template)
                    op.m, op.n, op.numOp = fresh_rows, fresh_rows, heads
                    time_s, energy = system.devices["GPU"].get_time_and_energy(op)
                    local_last = _cacheblend_event(
                        events, layer=layer_index, tier=tier,
                        request=request.request_id, name=name, device="GPU",
                        rows=fresh_rows, time_s=time_s, energy=energy,
                        deps=(local_last,), positions=compute_positions)
                # Each fresh query still hands its own local softmax tuple to
                # the DIE, which merges it with that query's PIM partials.
                local_tuple_events: List[str] = []
                tuple_bytes = heads * (system.model.dhead + 2) * dbyte
                tuple_transfer = _link_layer(x2g, "gpu_partial_lse_to_pim", tuple_bytes)
                tuple_time_s, tuple_energy = system.devices["GPU"].get_time_and_energy(
                    tuple_transfer)
                for position in compute_positions:
                    local_tuple_events.append(_cacheblend_event(
                        events, layer=layer_index, tier=tier,
                        request=request.request_id, name="gpu_partial_lse_to_pim",
                        device="LINK", rows=1, time_s=tuple_time_s, energy=tuple_energy,
                        deps=(local_last,), link_bytes=tuple_bytes,
                        positions=(position,)))

                die_last = q_link
                pim_results: List[str] = []
                location_deltas = _prefill_location_deltas(request, bindings)
                # Group queries with an identical visible old-KV physical run.
                # This preserves causality (the key includes every visible
                # location) while issuing one multi-Q Ramulator scan instead
                # of one process invocation per fresh token.
                # A prefill scan streams the reused *master* rows visible to
                # the query.  A corrected row's diff is being computed on the
                # GPU in this very layer, so only its shadowed master row is
                # streamed here, and it is masked out of the score.
                old_groups: Dict[Tuple[Tuple[int, int], ...], List[Tuple[int, int, str, List[KVLocation]]]] = {}
                masked_prefill_keys = {_address_key(loc.shadow) for _, reused, corrected, loc
                                       in bindings if reused and corrected and loc.shadow is not None}
                for ordinal, (position, tuple_event) in enumerate(zip(compute_positions,
                                                                       local_tuple_events)):
                    old = [loc.shadow if corrected else loc
                           for pos, reused, corrected, loc in bindings
                           if reused and pos <= position and
                           (not corrected or loc.shadow is not None)]
                    key = tuple(_address_key(loc) for loc in old)
                    old_groups.setdefault(key, []).append((ordinal, position, tuple_event, old))
                prefill_sweep = max(1, min(batch_size,
                                           mq_query_capacity(gemv_buffer_bytes)))
                for compatible_queries in old_groups.values():
                    # Prefill follows the same user-visible admission bound as
                    # decode, further capped by the per-bank GEMV buffer's
                    # resident-Q capacity (the sweep splits beyond it).
                    for first in range(0, len(compatible_queries), prefill_sweep):
                        grouped = compatible_queries[first:first + prefill_sweep]
                        old = grouped[0][3]
                        if not old:
                            continue
                        die_qs = []
                        for _, position, _, _ in grouped:
                            rotate_ready = _append_q_rotate_distribution(
                                system, events, x2g, layer=layer_index, tier=tier,
                                request=request.request_id, q_dependency=q_link,
                                q_bytes=local_hidden * dbyte, locations=old,
                                location_deltas=location_deltas, rotate_mode=rotate_mode,
                                name_prefix="", positions=(position,))
                            die_q_time = (local_hidden * dbyte) / system.devices["Acc"].softmax_peak_bandwidth
                            die_qs.append(_cacheblend_event(
                                events, layer=layer_index, tier=tier, request=request.request_id,
                                name="die_query_position_transform", device="DIE", rows=1,
                                time_s=die_q_time,
                                energy=(local_hidden * dbyte * system.devices["Acc"].energy_table["sram"],),
                                deps=tuple(dict.fromkeys((die_last, rotate_ready))),
                                positions=(position,)))
                        op = deepcopy(score)
                        op.m, op.n, op.k, op.numOp = len(grouped), len(old), system.model.dhead, heads
                        # Each contiguous physical TLB run is supplied to
                        # Ramulator.  Do not collapse a multi-block scan to its
                        # first K/V base address.
                        op.pim_kv_runs = tlb.scan_runs(old)
                        plan_time_s, plan_energy = _tlb_plan_cost(op.pim_kv_runs)
                        tlb_event = _cacheblend_event(
                            events, layer=layer_index, tier=tier, request=request.request_id,
                            name="tlb_lookup_and_bank_plan", device="TLB", rows=len(old),
                            time_s=plan_time_s, energy=plan_energy,
                            deps=tuple(die_qs), positions=tuple(item[1] for item in grouped),
                            addresses=[address for loc in old
                                       for address in (loc.key_address, loc.value_address)])
                        op.pim_shared_kv = True
                        op.pim_shared_queries = len(grouped)
                        _apply_pim_batch(op, pim_batch_command, pim_pe_freq_ghz)
                        scan = _append_physical_pim_scan(
                            system, events, op=op, layer=layer_index, tier=tier,
                            request=request.request_id, name="pim_kv_scan_score_softmax_pv",
                            rows=len(old), deps=(tlb_event,),
                            positions=tuple(item[1] for item in grouped),
                            runs=op.pim_kv_runs,
                            masked=_masked_rows_per_run(op.pim_kv_runs, masked_prefill_keys,
                                                        tlb.bytes_per_vector))
                        for ordinal, position, tuple_event, _ in grouped:
                            # One local softmax tuple per physical run plus
                            # the query's GPU tuple.
                            merge_width = len(scan) + 1
                            die_merge = _cacheblend_event(
                                events, layer=layer_index, tier=tier, request=request.request_id,
                                name="die_lse_merge", device="DIE", rows=1,
                                time_s=(merge_width * heads * (system.model.dhead + 2) * dbyte /
                                        system.devices["Acc"].softmax_peak_bandwidth),
                                energy=(merge_width * heads * (system.model.dhead + 2) * dbyte *
                                        system.devices["Acc"].energy_table["sram"],),
                                deps=tuple(scan) + (tuple_event,), positions=(position,))
                            die_last = die_merge
                            pim_results.append(die_merge)
                ctx_bytes = len(compute_positions) * local_hidden * dbyte
                ctx_transfer = _link_layer(x2g, "ctx_pim_to_gpu", ctx_bytes)
                time_s, energy = system.devices["GPU"].get_time_and_energy(ctx_transfer)
                # Rows without an old-cache contribution still require their
                # GPU local tuple before the aggregate context is usable.
                context_dep = tuple(dict.fromkeys(pim_results + local_tuple_events))
                ctx_link = _cacheblend_event(
                    events, layer=layer_index, tier=tier, request=request.request_id,
                    name="ctx_pim_to_gpu", device="LINK", rows=len(compute_positions),
                    time_s=time_s, energy=energy, deps=context_dep, link_bytes=ctx_bytes,
                    positions=compute_positions)
                post_last = _post_attention_gpu(
                    system, events, post, layer=layer_index, tier=tier,
                    request=request.request_id, rows=len(compute_positions),
                    dependency=ctx_link, positions=compute_positions)
                store = _append_channel_kv_stores(
                    system, events, layer=layer_index, tier=tier,
                    request=request.request_id, name="dram_store_diff_and_live",
                    locations=writes, dbyte=dbyte, deps=(kv_link,),
                    positions=compute_positions)
                prefill_store_events.extend(store)
                request_ready = (post_last,)
            # Decode consumes the prefill KV of every layer, so this is the
            # first required join point for asynchronous prefill writeback.
            request_ready = tuple(dict.fromkeys(request_ready +
                                                 tuple(prefill_store_events)))
            output_fingerprint = parent_output_fingerprints.get(
                request.request_id, "{}::output".format(request.request_id))
            if batch_size == 1:
                request_ready = _append_cacheblend_decode(
                    system, events, tlb, templates, post, request=request, tier=tier,
                    prefill_bindings=prefill_bindings,
                    output_fingerprint=output_fingerprint, initial_deps=request_ready,
                    rotate_mode=rotate_mode,
                    contiguous_no_reuse=physical_no_reuse)
                tier_done.extend(request_ready)
            else:
                decode_inputs.append((request, prefill_bindings, output_fingerprint, request_ready))
        if batch_size > 1 and decode_inputs:
            tier_done.extend(dep for deps in _append_cacheblend_decode_batched(
                system, events, tlb, templates, post, tier=tier, inputs=decode_inputs,
                batch_size=batch_size, batch_records=batch_records,
                rotate_mode=rotate_mode, pipe=pipe,
                contiguous_no_reuse=physical_no_reuse,
                pim_batch_command=pim_batch_command,
                pim_pe_freq_ghz=pim_pe_freq_ghz,
                gemv_buffer_bytes=gemv_buffer_bytes).values() for dep in deps)
        previous_tier_done = tuple(tier_done)

    validate_cacheblend_events(events, workload, local_hidden=local_hidden,
                               dbyte=dbyte, dhead=system.model.dhead,
                               heads=heads)
    scheduled = _schedule_cacheblend(events, pipe=pipe)
    _annotate_batch_q_arrivals(scheduled, batch_records)
    overlap_validation = validate_cacheblend_attacc_overlap_contract(scheduled, pipe=pipe)
    tlb_report = tlb.report()
    if not include_events:
        # The per-position TLB entry list is as large as the event list.
        tlb_report = dict(tlb_report, entries_omitted=len(tlb_report["entries"]))
        del tlb_report["entries"]
    report = {
        "policy": "no-reuse-physical" if physical_no_reuse else plan.config.policy,
        "latency_model": "physical-dag-ramulator" if physical_no_reuse else None,
        "cacheblend_batch_size": batch_size,
        "cacheblend_rotate_mode": rotate_mode,
        "pim_batch_command": pim_batch_command,
        "pim_pe_freq_ghz": pim_pe_freq_ghz,
        "gemv_buffer_bytes": gemv_buffer_bytes,
        "pim_sweep_query_capacity": mq_query_capacity(gemv_buffer_bytes),
        "batches": batch_records,
        "overlap_validation": overlap_validation,
        "events": ([event.to_dict() for event in scheduled] if include_events
                   else None),
        "event_count": len(scheduled),
        "summary": summarize_cacheblend_schedule(scheduled, workload),
        "tlb": tlb_report,
        "link_bytes": sum(event.link_bytes for event in scheduled),
        "makespan_s": max((event.end_s for event in scheduled), default=0.0),
        "gpu_time_s_unoverlapped": sum(event.time_s for event in scheduled
                                        if event.device == "GPU"),
        "pim_time_s_unoverlapped": sum(event.time_s for event in scheduled
                                        if event.device == "PIM"),
        # Address-resolved scans run on per-pool resources (PIM:poolA-B).
        "pim_pool_time_s_unoverlapped": sum(event.time_s for event in scheduled
                                             if event.device.startswith("PIM:")),
        "die_time_s_unoverlapped": sum(event.time_s for event in scheduled
                                        if event.device == "DIE"),
        "energy_nj": sum(event.energy_nj for event in scheduled),
        "energy_unit": "nJ",
    }
    ramulator = getattr(system.devices.get("Acc"), "ramulator", None)
    if ramulator is not None and hasattr(ramulator, "cache_report"):
        report["ramulator_signature_cache"] = ramulator.cache_report()
    return report


def run_reuse_prefill(system, workload: Workload, plan: ReusePlan,
                      *, pipe: bool = False, cacheblend_batch_size: int = 1,
                      cacheblend_rotate_mode: str = "gpu",
                      include_events: bool = True,
                      pim_batch_command: str = "replicate",
                      pim_pe_freq_ghz: float = MQ_DEFAULT_PE_FREQ_GHZ,
                      gemv_buffer_bytes: int = MQ_DEFAULT_GEMV_BUFFER_BYTES) -> Dict[str, Any]:
    """Dispatch address-resolved CacheBlend and EPIC to the shared DAG.

    CacheBlend samples correction rows per layer; EPIC overlays the fixed
    leading prefix of each shifted segment.  Both use one GPU/PIM/DIE/TLB/link
    event path and the same physical master/diff KV layout.
    """
    if plan.config.policy in ("cacheblend", "epic", "no-reuse"):
        return _run_cacheblend_prefill(system, workload, plan, pipe=pipe,
                                       batch_size=cacheblend_batch_size,
                                       physical_no_reuse=(plan.config.policy == "no-reuse"),
                                       rotate_mode=cacheblend_rotate_mode,
                                       include_events=include_events,
                                       pim_batch_command=pim_batch_command,
                                       pim_pe_freq_ghz=pim_pe_freq_ghz,
                                       gemv_buffer_bytes=gemv_buffer_bytes)
    return _run_legacy_reuse_prefill(system, workload, plan)
