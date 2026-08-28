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
from .cpp_eventcore import new_core as _new_event_core
from .workload import (CACHEBLEND_FAMILY, EPIC_FAMILY,
                       ReusePlan, Workload, WorkloadValidationError,
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
    if any(request.history_len for request in workload.requests):
        raise WorkloadValidationError(
            "history_len is not modeled by the unchanged legacy latency model; "
            "use the physical DAG path (--no-reuse-latency-model physical / "
            "--cacheblend-latency-model physical) or an --ablation config")
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
    if config.policy in CACHEBLEND_FAMILY:
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
        if config.policy in CACHEBLEND_FAMILY:
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
    if _EC is not None:
        _EC.add(device, time_s, ())


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
                if plan.config.policy in CACHEBLEND_FAMILY:
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
                    if _WARM_LEDGER is not None:
                        _WARM_LEDGER.append(("agg", deepcopy(layer),
                                             (len(events) - 1,)))
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
# Pool widths (ruling 2026-08-25, audit issue 3a): diff rows are few (a
# handful of corrected rows per agent), so the diff pool gets ONE channel
# and the master stream keeps the other fifteen -- the old 8/8 split halved
# the dominant master bandwidth for almost no diff traffic.  PROVISIONAL:
# the split may become density-driven (rho_b) later.
_KV_CHANNELS = {
    "master": tuple(range(0, 15)),
    "diff": tuple(range(15, 16)),
}
_ROTATE_MODES = ("gpu", "die", "bank")
_DIE_ROTATE_CYCLE_S = 1e-9
# After prefill, master and diff already sit as sequential streams in their
# pools, so a scan needs one descriptor per contiguous physical run (base,
# length, pool, mask bit-vector) rather than a lookup per logical row.  The
# mask itself is applied on the die between QK^T and softmax at no extra
# modeled cost; the cross-run softmax merge is the DIE LSE-merge event.
_TLB_DESCRIPTOR_S = 5e-9
_TLB_DESCRIPTOR_ENERGY = 0.1


def _apply_pim_batch(op, batch_command: str, pe_freq_ghz: float) -> None:
    """Stamp the sweep's batch-command scheme onto a PIM scan op.

    ``replicate`` keeps the legacy one-MAC-per-(column, query) trace.  ``mq``
    is the MQ-MAC command of PLAN_mq_command.md: one MAC_AB per column serves
    every resident Q, and the Ramulator wrapper carries the n-fold PE time
    plus the power stretch in the command interval.
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


def _pool_reads(tlb, locations: Sequence[KVLocation]) -> Tuple[List[KVLocation], set]:
    """Physical read stream per the layout's mask capability (see
    CacheBlendTLB.shadow_reads)."""
    if getattr(tlb, "shadow_reads", True):
        return _physical_reads(locations)
    return list(locations), set()


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

    ``shadow_reads`` declares the die-side read-mask gate: the master pool
    streams THROUGH a corrected row and masks it out of the score.  This is
    a Fugue hardware feature -- layouts without the gate (naive) must set it
    False, and their scans then SKIP the corrected master row: the enclosing
    run splits at the gap and the corrected row is read from its own page
    elsewhere (act a run, act one row, act the next run -- ruling chenyi9
    2026-08-26).

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

    shadow_reads = True

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
        # Every master extent spans channels 0--14 and every diff extent
        # spans channel 15.  The cursor is shared by all channels in a
        # pool: head h is placed on base+(h % pool width), and only after
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
                "layout": "master channels 0-14, diff channel 15; "
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


# Naive/software stores are PAGE-granular (ruling chenyi9 2026-08-26): large
# segments are chunked into 256-token pages BEFORE placement, exactly like a
# vLLM-style paged KV store, so no monolithic extent ever monopolizes one
# channel -- the row conflicts must come from the rotation itself.
_NAIVE_PAGE_ROWS = 256

# Serving batch width for concurrent decodes (ruling chenyi9 2026-08-26):
# the GPU weight pass always batches across concurrently decoding requests
# (standard serving behavior, every rung); capped at the MQ residency max
# n_cap = 8 so the GPU wave and the A5/A6 PIM wave describe the same
# concurrency.
_DECODE_SERVE_WAVE = 8

# W2 build-once machinery (ruling chenyi9 2026-08-28): during a warm build
# the accelerator entries return zero placeholders; every placeholder-priced
# event is recorded here as (kind, deepcopied op, event indexes) and
# repriced from the warm cache before finalization -- the second full
# construction of the old warm flow is gone.  _WARM_BYPASS carries the
# ORIGINAL accelerator entries so the A6 estimator prices its probe for
# real (a placeholder would skew the side decision).
_WARM_LEDGER: Optional[List[Tuple[str, Any, Tuple[int, ...]]]] = None
_WARM_BYPASS: Optional[Tuple[Any, Any]] = None
# W3: with --workload-report-events none the per-event K/V address arrays
# are pure run-time dead weight (they only ever reach the report when
# events are included); construction skips storing them.  The validator's
# scan-address audit is gated accordingly.
_RETAIN_EVENT_ADDRESSES = True
# Native event-schedule core (experimental C++ branch, chenyi9 2026-08-28):
# one instance per report run, mirroring every event as (device, duration,
# deps) and owning the schedule state machine.  None = pure-Python paths
# (KVPIM_CPPCORE=0 or library missing); all Python logic is preserved and
# the overlap-contract validator replays the native schedule every run.
_EC = None


def _gqa_group(system) -> int:
    """Q heads per shared KV head (1 = MHA; ruling chenyi9 2026-08-27)."""
    return max(1, int(getattr(system.model, "gqa_size", 1) or 1))


def _gqa_kv_heads_local(system, q_heads_local: int) -> int:
    """Local resident KV heads for a local Q-head count."""
    return max(1, int(q_heads_local) // _gqa_group(system))


def _hbm_seq_split(system) -> int:
    """REVERTED to constant 1 (chenyi9 2026-08-27 night): sequence split
    presupposed token striping; AttAcc's head-per-channel is restored.
    The original computation is kept below for the record but the
    function short-circuits."""
    return 1
    accelerator = system.devices.get("Acc") if isinstance(
        getattr(system, "devices", None), dict) else None
    if accelerator is None:
        return 1
    num_hbm = max(1, int(getattr(accelerator, "num_hbm", 1) or 1))
    q_local = max(1, int(getattr(system.model, "num_heads", 1)) //
                  max(1, int(getattr(system.model, "tp", 1))))
    kv_local = _gqa_kv_heads_local(system, q_local)
    return max(1, num_hbm // kv_local) if kv_local < num_hbm else 1


class NaiveKVLayout(CacheBlendTLB):
    """Scattered software-order PAGED KV placement -- the A3 rung.

    No die-side mask gate (``shadow_reads = False``): a corrected row's
    master copy is skipped, splitting the master run, and the corrected row
    is a separate single-row activation at its own append position.

    No master/diff channel split and no PIM-aware remap: every reservation
    (master, diff and live alike, INCLUDING large history/live extents) is
    first split into 256-token pages, and the pages are appended round-robin
    over the 16 channels in reservation (= software append) order.  A scan
    of one logical context therefore touches many single-channel pools; the
    pages that share a channel sit at non-adjacent addresses (other
    requests' pages appended between them), so each one is a separate
    physical run with its own row activation -- the row-conflict penalty of
    a PIM-oblivious paged store emerges from the rotation, never from a
    constructed layout.
    """

    shadow_reads = False

    def finalize(self) -> None:
        if self._blocks:
            return
        stride = ((self.bytes_per_vector + _HBM_TX_BYTES - 1) //
                  _HBM_TX_BYTES) * _HBM_TX_BYTES
        channel_state = {channel: {"tile": 0, "cursor": 0}
                         for channel in range(16)}
        tiles_per_channel = _HBM_CHANNEL_BYTES // (2 * _ORIGINAL_KV_GAP_BYTES)
        self._pages: Dict[Tuple, Dict[int, KVBlock]] = {}
        rotation = 0
        # Python dicts preserve insertion order, so iterating the reservation
        # dict walks pages in software append order.
        for index, key in enumerate(self._reserved_rows):
            layer, owner, fingerprint, kind = key
            rows = tuple(sorted(self._reserved_rows[key]))
            page_map: Dict[int, KVBlock] = {}
            for page_index, start in enumerate(
                    range(0, len(rows), _NAIVE_PAGE_ROWS)):
                page_rows = rows[start:start + _NAIVE_PAGE_ROWS]
                span = len(page_rows) * stride
                channel = rotation % 16
                rotation += 1
                state = channel_state[channel]
                if state["cursor"] + span > _ORIGINAL_KV_GAP_BYTES:
                    state["tile"] += 1
                    state["cursor"] = 0
                if state["tile"] >= tiles_per_channel:
                    raise WorkloadValidationError(
                        "naive KV allocation exceeds channel {} capacity".format(
                            channel))
                key_base = (channel * _HBM_CHANNEL_BYTES +
                            state["tile"] * (2 * _ORIGINAL_KV_GAP_BYTES) +
                            state["cursor"])
                block = KVBlock(
                    "nvp-{:06d}-{:03d}".format(index, page_index),
                    layer, owner, fingerprint, kind, page_rows,
                    key_base, key_base + _ORIGINAL_KV_GAP_BYTES, stride,
                    channel, 1, state["tile"], state["cursor"], 0)
                state["cursor"] += span
                self._blocks[(layer, owner, fingerprint, kind,
                              page_index)] = block
                for row in page_rows:
                    page_map[row] = block
            self._pages[key] = page_map

    def locate(self, layer: int, owner: str, fingerprint: str, owner_row: int,
               kind: str) -> KVLocation:
        cache_key = (layer, owner, fingerprint, owner_row, kind)
        location = self._locations.get(cache_key)
        if location is not None:
            return location
        page_map = getattr(self, "_pages", {}).get(
            (layer, owner, fingerprint, kind))
        if page_map is None or owner_row not in page_map:
            raise WorkloadValidationError(
                "naive KV page requested before it was reserved: {}".format(
                    (layer, owner, fingerprint, kind, owner_row)))
        block = page_map[owner_row]
        token_offset = block.token_offset(owner_row)
        location = KVLocation(
            layer, owner, fingerprint, owner_row, kind,
            block.key_base + token_offset * block.vector_stride,
            block.value_base + token_offset * block.vector_stride,
            self.bytes_per_vector, block.block_id, token_offset,
            block.channel_base, block.channel_count)
        self._locations[cache_key] = location
        return location

    def report(self) -> Dict[str, Any]:
        return {"mapping": "Ramulator HBM3-PIM physical byte address",
                "layout": "naive scattered PAGED (A3): every reservation "
                          "split into 256-token pages, pages appended "
                          "round-robin over 16 single-channel pools in "
                          "software order; no master/diff split",
                "page_rows": _NAIVE_PAGE_ROWS,
                "channel_capacity_bytes": _HBM_CHANNEL_BYTES,
                "transaction_bytes": _HBM_TX_BYTES,
                "channel_sets": {"naive": list(range(16))},
                "page_count": len(self._blocks),
                "entries": self.entries}


class NaiveMaskKVLayout(NaiveKVLayout):
    """A3a (ruling chenyi9 2026-08-26): the SAME paged software rotation as
    A3, but the consumer CAN mask -- a corrected row's stale master copy is
    streamed with the run and masked out of the score (GPU-side maskable),
    so the run does not split; the corrected row still reads from its own
    page.  Differs from A3 only in ``shadow_reads``."""

    shadow_reads = True

    def report(self) -> Dict[str, Any]:
        report = super().report()
        report["layout"] = ("naive scattered PAGED with read-mask (A3a): "
                           "same page rotation as A3, stale rows streamed "
                           "and masked instead of splitting the run")
        return report


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
                             tuple(positions),
                             array("Q", addresses if _RETAIN_EVENT_ADDRESSES
                                   else ()),
                             tuple(batch_members), masked_rows))
    if _EC is not None:
        _EC.add(device, time_s, deps)
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
    first_event_index = len(events)
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
    if _WARM_LEDGER is not None and hasattr(accelerator, "get_time_and_energy_runs"):
        _WARM_LEDGER.append(("runs", deepcopy(op),
                             tuple(range(first_event_index,
                                         first_event_index + len(scan_events)))))
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
        # head->HBM remap (2026-08-27): a pool event carries ONE head's bytes
        # on ONE HBM's channels; the concurrent copies on the other stacks
        # are the head-parallel dimension, so the wall time uses the
        # PER-HBM bandwidth share, not the aggregate.
        # Final placement ruling (chenyi9 2026-08-27 night): a pool holds
        # ONE head's chunks on ONE HBM's channels (softmax accumulates
        # inside that HBM's die; heads parallelize across HBMs outside the
        # event).  The store therefore gets the PER-HBM bandwidth share.
        accelerator = system.devices["Acc"]
        bandwidth = (accelerator.peak_memory_bandwidth /
                     max(1, getattr(accelerator, "num_hbm", 1)) *
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
    if _EC is not None and _EC.size() == len(events):
        # Native fast path: recompute the whole schedule from the mirrored
        # graph (a reprice may have updated durations), then attach times.
        _EC.reset()
        _EC.advance()
        start_arr, end_arr = _EC.bulk_times(len(events))
        return [replace(event, start_s=start_arr[index], end_s=end_arr[index])
                for index, event in enumerate(events)]
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
    # Index once instead of rescanning every event per batch: the old
    # listcomps were O(batches x events) -- an accidental quadratic that
    # dominated construction at suite scale (py-spy evidence, chenyi9
    # 2026-08-27).  Buckets keep (scan_index, event) so the per-batch lists
    # reproduce the original scheduled-order output exactly.
    q_by_layer_request: Dict[Tuple[int, str], List[Tuple[int, Any]]] = {}
    sweeps_by_batch_id: Dict[str, List[Any]] = {}
    for scan_index, event in enumerate(scheduled):
        if event.name in ("decode_q_gpu_to_pim",
                          "decode_gpu_rotate_q_extra_to_pim"):
            q_by_layer_request.setdefault(
                (event.transformer_layer, event.request_id), []).append(
                    (scan_index, event))
        elif event.name == "decode_batch_tlb_lookup_and_bank_plan":
            sweeps_by_batch_id.setdefault(event.request_id, []).append(event)
    for batch in batches:
        members = set(batch["members"])
        layer = batch["transformer_layer"]
        indexed_q = sorted(
            (pair for member in members
             for pair in q_by_layer_request.get((layer, member), ())),
            key=lambda pair: pair[0])
        q_events = [event for _, event in indexed_q
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
        sweep_events = sweeps_by_batch_id.get(batch["id"], [])
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
        elif event.name in ("kv_gpu_to_pim", "decode_kv_gpu_to_pim",
                            "kv_pim_to_gpu"):
            if event.link_bytes != event.rows * kv_bytes_per_row:
                raise WorkloadValidationError("CacheBlend KV link byte count is invalid")
        elif event.name in ("gpu_partial_lse_to_pim", "decode_gpu_partial_lse_to_pim"):
            if event.rows != 1 or event.link_bytes != tuple_bytes:
                raise WorkloadValidationError("CacheBlend local LSE tuple shape is invalid")
        if "pim_kv_scan" in event.name:
            if _RETAIN_EVENT_ADDRESSES and (
                    not event.dram_addresses or len(event.dram_addresses) % 2):
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
        if event.name == "die_score_assembly":
            # Bank-whole prefill: the DIE assembles one score vector per query
            # from the pool scans alone (causal drop, no GPU tuple).
            dependency_names = {named(dependency) for dependency in event.depends_on}
            if not any("pim_kv_scan" in name for name in dependency_names):
                raise WorkloadValidationError(
                    "bank-whole DIE score assembly lacks a PIM scan contribution")

    groups: Dict[Tuple[str, int], List[SplitEvent]] = {}
    for event in events:
        groups.setdefault((event.request_id, event.transformer_layer), []).append(event)
    for (request_id, layer), group in groups.items():
        names = {event.name for event in group}
        for context_name, tuple_name, merge_names in (
                ("ctx_pim_to_gpu", "gpu_partial_lse_to_pim",
                 ("die_lse_merge", "die_score_assembly")),
                ("decode_ctx_pim_to_gpu", "decode_gpu_partial_lse_to_pim",
                 ("decode_die_lse_merge",))):
            contexts = [event for event in group if event.name == context_name]
            if not contexts:
                continue
            context = contexts[-1]
            relevant = [event for event in group
                        if set(event.query_positions).intersection(context.query_positions)]
            merge_events = {event.event_id for event in relevant
                            if event.name in merge_names}
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
    if plan.config.policy in CACHEBLEND_FAMILY:
        by_segment = plan.cacheblend_partial_rows.get(layer, {}).get(
            request.request_id, {})
        for index, rows in by_segment.items():
            offset = sum(segment.length for segment in request.segments[:index])
            corrected.update(offset + row for row in rows)
    elif plan.config.policy in EPIC_FAMILY:
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
    bindings.extend(_history_tlb_rows(request, layer, tlb, "master"))
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
        force_fresh = (plan.config.policy in CACHEBLEND_FAMILY and
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
            # The agent's own earlier-turn KV: one private resident extent
            # per request, in the master pool (private under the affine
            # no-reuse layout).  It is never written during the run.
            history_fingerprint = _history_fingerprint(request.request_id)
            for row in range(request.history_len):
                tlb.reserve(layer, request.request_id, history_fingerprint,
                            row, "private" if contiguous_no_reuse else "master")
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
    bindings.extend(_history_tlb_rows(request, layer, tlb, "private"))
    return bindings


def _history_fingerprint(request_id: str) -> str:
    return "{}::history".format(request_id)


def _history_tlb_rows(request, layer: int, tlb, kind: str):
    """Bind an agent's resident earlier-turn KV as one private extent.

    History rows are the agent's own KV from earlier turns: already resident
    in PIM memory, attended by every query, never recomputed and never
    corrected.  They are appended after the segment bindings (so positional
    indexing of ``bindings[0:total_length]`` stays valid) at query positions
    ``-H..-1``, which precede every prefill position: causal filters
    (``pos <= position`` grouping, the bank-whole comparator) always keep
    them visible.  Delta is 0 -- the KV was computed in place by this agent,
    so no Q rotation variant is required.
    """
    bindings = []
    fingerprint = _history_fingerprint(request.request_id)
    for row in range(request.history_len):
        location = tlb.locate(layer, request.request_id, fingerprint, row, kind)
        tlb.bind(request.request_id, layer, row - request.history_len,
                 location, 0, True)
        bindings.append((row - request.history_len, True, False, location))
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
            reads, masked_keys = _pool_reads(tlb, old)
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
                op.m, op.n, op.k, op.numOp = (1, len(reads), system.model.dhead,
                                              _gqa_kv_heads_local(system, heads))
                if _gqa_group(system) > 1:
                    # GQA: the group's Q heads are resident queries against
                    # the ONE shared KV head.
                    op.pim_shared_kv = True
                    op.pim_shared_queries = _gqa_group(system)
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
                request.request_id: _pool_reads(tlb, old_by_request[request.request_id])
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
            if _EC is not None:
                # Native core: incremental advance without the per-round
                # dict copies of the Python fallback (those were an
                # accidental quadratic at suite scale).
                _EC.advance()
            else:
                provisional_finish, provisional_availability = _schedule_cacheblend_incremental(
                    events, pipe=pipe, start_index=provisional_index,
                    finish=provisional_finish, availability=provisional_availability)
            provisional_index = len(events)
            input_rank = {request.request_id: index for index, request in enumerate(active)}
            # WARM-BUILD INVARIANT (review note 2026-08-28): under a W2
            # warm build this sort runs on placeholder-priced provisional
            # finishes.  It stays identical to the cold order only because
            # every sort-key event is a positive-duration LINK event whose
            # same-resource finish order is monotone in append order --
            # placeholder-independent.  Do not key this sort on Acc-priced
            # (placeholder-able) or zero-duration events.
            def _q_key(request):
                key_id = (rotate_ready[request.request_id]
                          if rotate_mode == "gpu" and request.request_id in rotate_ready
                          else q_links[request.request_id])
                finish_s = (_EC.end(key_id) if _EC is not None
                            else provisional_finish[key_id])
                return (finish_s, input_rank[request.request_id])
            ready = sorted(active, key=_q_key)

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
                    sweep_cap = max(1, mq_query_capacity(gemv_buffer_bytes) //
                                    _gqa_group(system))
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
                        op.m, op.n, op.k, op.numOp = (len(sweep), len(common),
                                                      system.model.dhead,
                                                      _gqa_kv_heads_local(system, heads))
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
                        op.pim_shared_queries = len(sweep) * _gqa_group(system)
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
                        op.m, op.n, op.k, op.numOp = (1, len(private),
                                                      system.model.dhead,
                                                      _gqa_kv_heads_local(system, heads))
                        if _gqa_group(system) > 1:
                            op.pim_shared_kv = True
                            op.pim_shared_queries = _gqa_group(system)
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
    # The agent's resident earlier-turn KV (bindings flagged reused) extends
    # the scan but is never computed, transferred, or stored in this run.
    fresh = [item for item in bindings if not item[1]]
    history = [location for _, reused, _, location in bindings if reused]
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
    addresses = [address for _, _, _, location in fresh
                 for address in (location.key_address, location.value_address)]
    kv_link = _cacheblend_event(
        events, layer=layer, tier=tier, request=request.request_id,
        name="kv_gpu_to_pim", device="LINK", rows=rows, time_s=time_s,
        energy=energy, deps=(q,), link_bytes=kv_bytes, positions=positions,
        addresses=addresses)
    # The scan walks the resident history extent before this turn's rows;
    # every query sees all history keys (they precede position 0).
    locations = history + [location for _, _, _, location in fresh]
    scan_addresses = [address for location in locations
                      for address in (location.key_address, location.value_address)]
    address_plan = _cacheblend_event(
        events, layer=layer, tier=tier, request=request.request_id,
        name="contiguous_address_plan", device="ADDR", rows=rows + len(history),
        time_s=0.0, energy=(), deps=(q_link,),
        positions=positions, addresses=scan_addresses)
    # GEMV residency cap (bug fix, chenyi9 2026-08-28): the old code put
    # ALL rows x gqa query slices into ONE op (thousands of "resident"
    # queries -- physically impossible against the 512-B buffer, and it
    # exploded the replicate trace to tens of millions of commands).  The
    # scan is served by ceil(rows / cap) consecutive full-range sweeps of
    # at most cap resident rows; sweeps share one shape, so ONE priced op
    # (plus one partial) covers them and the event carries the summed
    # time/energy.  The warm ledger records the sweep decomposition.
    gqa_group = _gqa_group(system)
    cap_rows = max(1, mq_query_capacity(MQ_DEFAULT_GEMV_BUFFER_BYTES) // gqa_group)
    full_sweeps, partial_rows = divmod(rows, cap_rows)

    def _sweep_op(sweep_rows: int):
        op = deepcopy(score)
        op.m, op.n, op.k, op.numOp = (sweep_rows, rows + len(history),
                                      system.model.dhead,
                                      _gqa_kv_heads_local(system, heads))
        op.pim_kv_runs = tlb.scan_runs(locations)
        op.pim_shared_kv = True
        op.pim_shared_queries = sweep_rows * gqa_group
        return op

    total_time = 0.0
    total_energy = 0.0
    ledger_parts = []
    op_full = None
    if full_sweeps:
        op_full = _sweep_op(cap_rows)
        time_s, energy = system.devices["Acc"].get_time_and_energy(op_full)
        total_time += full_sweeps * time_s
        total_energy += full_sweeps * sum(energy)
        ledger_parts.append((op_full, full_sweeps))
    if partial_rows:
        op_partial = _sweep_op(partial_rows)
        time_s, energy = system.devices["Acc"].get_time_and_energy(op_partial)
        total_time += time_s
        total_energy += sum(energy)
        ledger_parts.append((op_partial, 1))
    scan = _cacheblend_event(
        events, layer=layer, tier=tier, request=request.request_id,
        name="pim_kv_scan_score_softmax_pv", device="PIM", rows=rows + len(history),
        time_s=total_time, energy=(total_energy,), deps=(address_plan,),
        positions=positions, addresses=scan_addresses)
    if _WARM_LEDGER is not None:
        _WARM_LEDGER.append(("agg_cb_sweeps",
                             tuple((deepcopy(op), count)
                                   for op, count in ledger_parts),
                             (len(events) - 1,)))
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


def _energy_breakdown(scheduled: Sequence[SplitEvent]) -> Dict[str, Any]:
    """Per-part energy of a scheduled DAG (nJ), two granularities.

    ``by_class`` folds the device string into GPU / LINK / PIM (plain and
    per-pool) / DIE / TLB; ``by_event`` keys by event name so e.g. the PIM
    scan, the KV stores and each link direction can be read separately
    (chenyi9 order 2026-08-26: the ladder CSV needs every part's energy).
    """
    by_class: Dict[str, float] = {}
    by_event: Dict[str, float] = {}
    for event in scheduled:
        device = event.device
        device_class = ("PIM" if device == "PIM" or device.startswith("PIM:")
                        else device)
        by_class[device_class] = by_class.get(device_class, 0.0) + event.energy_nj
        by_event[event.name] = by_event.get(event.name, 0.0) + event.energy_nj
    return {"by_class": {key: by_class[key] for key in sorted(by_class)},
            "by_event": {key: by_event[key] for key in sorted(by_event)}}


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


def _software_reuse_rows(plan: ReusePlan, layer: int, request):
    """Classify one request's rows for a GPU-resident-KV layer (A2 rung).

    Returns ``(compute_positions, reused_row_count)``: positions the GPU must
    (re)compute -- fresh rows plus policy-corrected rows -- and the count of
    rows served straight from the GPU-resident KV cache.  Mirrors the
    ``(position, reused, corrected)`` classification of
    ``_cacheblend_tlb_rows`` without materializing TLB locations.
    """
    decisions = {d.segment_index for d in plan.reusable
                 if d.request_id == request.request_id}
    corrected = _policy_corrected_rows(plan, layer, request)
    compute: List[int] = []
    reused_rows = 0
    position = 0
    for index, segment in enumerate(request.segments):
        reused_segment = index in decisions
        for _ in range(segment.length):
            if reused_segment and position not in corrected:
                reused_rows += 1
            else:
                compute.append(position)
            position += 1
    return compute, reused_rows


def _prefill_side_summary(workload: Workload,
                          prefill_attn_rows: Mapping[str, Mapping[str, int]]
                          ) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Uniform-denominator prefill-attention accounting for the ladder CSV.

    Returns (total attention rows per side, per-request class over EVERY
    workload request): "pim" / "gpu" / "mixed" (layers on both sides) /
    "none" (a fully reused, zero-correction request runs no prefill
    attention at all).
    """
    sides: Dict[str, str] = {}
    totals = {"pim": 0, "gpu": 0}
    for request in workload.requests:
        rows = prefill_attn_rows.get(request.request_id, {"pim": 0, "gpu": 0})
        totals["pim"] += rows["pim"]
        totals["gpu"] += rows["gpu"]
        sides[request.request_id] = ("mixed" if rows["pim"] and rows["gpu"]
                                     else "pim" if rows["pim"]
                                     else "gpu" if rows["gpu"] else "none")
    return totals, sides


def _run_gpu_software_only(system, workload: Workload, plan: ReusePlan,
                           *, pipe: bool, include_events: bool = True) -> Dict[str, Any]:
    """A2 rung of the ladder on the event DAG: software reuse, GPU compute,
    KV cache in REMOTE DUMB STORAGE.

    Ruling (chenyi9 2026-08-26): the link-bytes metric counts the GPU <->
    remote-storage interconnect (NVLink/PCIe; the remote side may be
    PIM-HBM or plain DRAM), and in A2 the KV cache lives ENTIRELY in that
    remote storage with no PIM compute.  Consequences per layer/step:

    * prefill: fresh/corrected rows are computed on the GPU and their K/V is
      written out over the link (``kv_gpu_to_remote``); reused rows and the
      agent's resident history K/V stream BACK over the link
      (``kv_remote_to_gpu``) before the full-context GPU attention;
    * decode: every generated token streams the whole per-layer context K/V
      back over the link (one aggregated ``kv_remote_to_gpu`` event per step,
      bytes x ndec layers) before the GPU generation block, then writes the
      new row's K/V out.  The link, not the GPU, is expected to dominate --
      that is the point of this rung.

    The remote DRAM's own access time is folded into the link event (the
    interconnect is far slower than the remote stack's internal bandwidth).
    Requests of one tier contend on the single GPU resource; tiers keep the
    engine's tier-barrier convention.  NOTE two deliberate differences from
    the analytic A2 (which is still the GPU-local-KV model of the matrix):
    KV residency (remote here, GPU-local there) and decode batching
    (per-request batch 1 here, padded tier batch there).
    """
    # Keep the W3 retention gate coherent on this path too (review
    # hardening 2026-08-28): without this a prior events-none cacheblend
    # run would leave the module global stale for a same-process A2 run.
    global _RETAIN_EVENT_ADDRESSES, _EC
    _RETAIN_EVENT_ADDRESSES = bool(include_events)
    if _EC is not None:
        _EC.close()
    _EC = _new_event_core(pipe)
    validate_reuse_plan(workload, plan, system.model.ndec)
    _validate_layer_config(plan, system.model.ndec)
    system.model.build(1, 1, 2, True)
    templates = {layer.name: layer for layer in system.model.sum_decoder}
    qkv, score, softmax, context = (templates[name] for name in
                                    ("qkv", "score", "softmax", "context"))
    post = list(system.model.sum_decoder)
    x2g = templates["comm_x2g"]
    heads = max(1, system.model.num_heads // system.model.tp)
    ndec = system.model.ndec
    dbyte = qkv.dbyte
    local_hidden = system.model.hdim // system.model.tp
    kv_row_bytes = (2 * max(1, local_hidden // _gqa_group(system)) *
                    dbyte)  # one K row + one V row (KV heads only)

    def link_event(name, byte_count, *, layer, tier, request, rows, deps):
        op = _link_layer(x2g, name, byte_count)
        time_s, energy = system.devices["GPU"].get_time_and_energy(op)
        return _cacheblend_event(events, layer=layer, tier=tier,
                                 request=request, name=name, device="LINK",
                                 rows=rows, time_s=time_s, energy=energy,
                                 deps=deps, link_bytes=byte_count)

    events: List[SplitEvent] = []
    prefill_attn_rows: Dict[str, Dict[str, int]] = {}
    previous_tier_done: Tuple[str, ...] = ()
    for tier, requests, _, _ in _tier_shapes(workload):
        tier_done: List[str] = []
        decode_ready: Dict[str, Tuple[str, ...]] = {}
        decode_totals: Dict[str, int] = {}
        for request in requests:
            request_ready: Tuple[str, ...] = previous_tier_done
            total_rows = request.total_length + request.history_len
            prefill_store_events: List[str] = []
            for layer_index in range(ndec):
                full = (plan.config.policy in CACHEBLEND_FAMILY and
                        layer_index in plan.config.cacheblend_full_recompute_layers)
                if full:
                    compute, reused_rows = list(range(request.total_length)), 0
                else:
                    compute, reused_rows = _software_reuse_rows(
                        plan, layer_index, request)
                if compute:
                    prefill_attn_rows.setdefault(
                        request.request_id,
                        {"pim": 0, "gpu": 0})["gpu"] += len(compute)
                if not compute:
                    # Every row of this layer is reused with no correction
                    # (e.g. the same instance re-run): nothing to recompute
                    # or store; decode streams the remote rows later anyway.
                    continue
                q = _gpu_layer_event(
                    system, events, qkv, layer=layer_index, tier=tier,
                    request=request.request_id, name="qkv", rows=len(compute),
                    deps=request_ready, positions=compute)
                # Reused rows and the agent's resident history K/V live in the
                # remote store: stream them back before the GPU can attend.
                remote_resident = reused_rows + request.history_len
                attn_deps = (q,)
                if remote_resident:
                    readback = link_event(
                        "kv_remote_to_gpu", remote_resident * kv_row_bytes,
                        layer=layer_index, tier=tier,
                        request=request.request_id, rows=remote_resident,
                        deps=request_ready)
                    attn_deps = (q, readback)
                gpu_last = None
                for template, name, wide in ((score, "gpu_prefill_score", "n"),
                                             (softmax, "gpu_prefill_softmax", "n"),
                                             (context, "gpu_prefill_context", "k")):
                    op = deepcopy(template)
                    op.m, op.numOp = len(compute), heads
                    setattr(op, wide, total_rows)
                    time_s, energy = system.devices["GPU"].get_time_and_energy(op)
                    gpu_last = _cacheblend_event(
                        events, layer=layer_index, tier=tier,
                        request=request.request_id, name=name, device="GPU",
                        rows=len(compute), time_s=time_s, energy=energy,
                        deps=(attn_deps if gpu_last is None else (gpu_last,)),
                        positions=compute)
                # Fresh/corrected K/V leaves for the remote store as soon as
                # QKV produced it; the write overlaps the attention block and
                # is joined before decode first reads the completed cache.
                prefill_store_events.append(link_event(
                    "kv_gpu_to_remote", len(compute) * kv_row_bytes,
                    layer=layer_index, tier=tier, request=request.request_id,
                    rows=len(compute), deps=(q,)))
                request_ready = (_post_attention_gpu(
                    system, events, post, layer=layer_index, tier=tier,
                    request=request.request_id, rows=len(compute),
                    dependency=gpu_last, positions=compute),)
            request_ready = tuple(dict.fromkeys(request_ready +
                                                 tuple(prefill_store_events)))
            decode_ready[request.request_id] = request_ready
            decode_totals[request.request_id] = total_rows
        # Decode, BATCHED across the tier's concurrent requests (ruling
        # chenyi9 2026-08-26: the weight pass is always GPU work and always
        # batches as far as possible -- one weight pass serves the whole
        # wave, while each query still streams its OWN context K/V back
        # over the link and pays its own attention).  Wave width follows
        # the serving batch (8).  Per wave and step: per-request read link
        # -> one batched GPU generation block -> per-request write-back.
        last = dict(decode_ready)
        last_write: Dict[str, str] = {}
        max_lout = max((request.lout for request in requests), default=0)
        for step in range(max_lout):
            active = [request for request in requests if step < request.lout]
            for start in range(0, len(active), _DECODE_SERVE_WAVE):
                group = active[start:start + _DECODE_SERVE_WAVE]
                members = tuple(request.request_id for request in group)
                positions = tuple(decode_totals[request.request_id] + step
                                  for request in group)
                reads = []
                for request in group:
                    context_rows = decode_totals[request.request_id] + step + 1
                    reads.append(link_event(
                        "kv_remote_to_gpu", context_rows * kv_row_bytes * ndec,
                        layer=ndec - 1, tier=tier, request=request.request_id,
                        rows=context_rows, deps=last[request.request_id]))
                step_time = 0.0
                step_energy = 0.0
                for template in system.model.sum_decoder:
                    if template.name == "comm_x2g":
                        continue  # the remote link is modeled by the explicit
                                  # kv_remote_to_gpu / kv_gpu_to_remote events
                    if template.name in ("score", "softmax", "context"):
                        # Attention is per query (own context width).
                        for request in group:
                            context_rows = (decode_totals[request.request_id] +
                                            step + 1)
                            op = deepcopy(template)
                            if template.name == "context":
                                op.m, op.k, op.numOp = 1, context_rows, heads
                            else:
                                op.m, op.n, op.numOp = 1, context_rows, heads
                            time_s, energy = system.devices["GPU"].get_time_and_energy(op)
                            step_time += time_s
                            step_energy += sum(energy)
                    else:
                        # Weight work: ONE pass serves the whole wave.
                        op = deepcopy(template)
                        op.m = len(group)
                        time_s, energy = system.devices["GPU"].get_time_and_energy(op)
                        step_time += time_s
                        step_energy += sum(energy)
                compute_event = _cacheblend_event(
                    events, layer=ndec - 1, tier=tier,
                    request="gpu-decode-t{}-s{}-w{}".format(tier, step,
                                                            start // _DECODE_SERVE_WAVE),
                    name="decode_gpu_step_batch", device="GPU",
                    rows=len(group), time_s=step_time * ndec,
                    energy=(step_energy * ndec,),
                    deps=tuple(dict.fromkeys(tuple(reads))),
                    positions=positions, batch_members=members)
                for request in group:
                    last[request.request_id] = (compute_event,)
                    last_write[request.request_id] = link_event(
                        "kv_gpu_to_remote", kv_row_bytes * ndec,
                        layer=ndec - 1, tier=tier,
                        request=request.request_id, rows=1,
                        deps=(compute_event,))
        for request in requests:
            tier_done.extend(dict.fromkeys(
                last[request.request_id] +
                ((last_write[request.request_id],)
                 if request.request_id in last_write else ())))
        previous_tier_done = tuple(dict.fromkeys(tier_done))

    # validate_split_events checks the split-era GPU/PIM naming contract and
    # does not apply to a GPU-only stream; keep only its shape sanity part.
    for event in events:
        if event.rows <= 0 or event.time_s < 0 or event.energy_nj < 0:
            raise WorkloadValidationError(
                "GPU-only event has an invalid shape or cost")
    scheduled = _schedule_cacheblend(events, pipe=pipe)
    attn_rows_total, attn_sides = _prefill_side_summary(workload,
                                                        prefill_attn_rows)
    return {
        "policy": plan.config.policy,
        "latency_model": "physical-dag-gpu-remote-kv",
        "decode_attn": "gpu",
        "kv_mapping": "none",
        "prefill_attention_rows": attn_rows_total,
        "prefill_attention_sides": attn_sides,
        "kv_residency": "remote-dumb-storage (ruling 2026-08-26: link bytes "
                        "= GPU <-> remote storage over NVLink/PCIe)",
        "pim_prefill_mode": "gpu",
        "history_rows": sum(request.history_len for request in workload.requests),
        "events": ([event.to_dict() for event in scheduled] if include_events
                   else None),
        "event_count": len(scheduled),
        "summary": summarize_cacheblend_schedule(scheduled, workload),
        "link_bytes": sum(event.link_bytes for event in scheduled),
        "makespan_s": max((event.end_s for event in scheduled), default=0.0),
        "gpu_time_s_unoverlapped": sum(event.time_s for event in scheduled
                                        if event.device == "GPU"),
        "pim_pool_time_s_unoverlapped": 0.0,
        "die_time_s_unoverlapped": 0.0,
        "energy_nj": sum(event.energy_nj for event in scheduled),
        "energy_breakdown_nj": _energy_breakdown(scheduled),
        "energy_unit": "nJ",
    }


def _run_cacheblend_prefill(system, workload: Workload, plan: ReusePlan,
                            *, pipe: bool, batch_size: int = 1,
                            physical_no_reuse: bool = False,
                            kv_mapping: str = "master-diff",
                            rotate_mode: str = "gpu",
                            include_events: bool = True,
                            pim_prefill_mode: str = "dynamic",
                            pim_batch_command: str = "replicate",
                            pim_pe_freq_ghz: float = MQ_DEFAULT_PE_FREQ_GHZ,
                            gemv_buffer_bytes: int = MQ_DEFAULT_GEMV_BUFFER_BYTES,
                            _build_sink: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # W3 (2026-08-28): event address arrays only ever reach the report when
    # events are included; otherwise skip storing them (and the validator's
    # scan-address audit follows the same gate).
    global _RETAIN_EVENT_ADDRESSES, _EC
    _RETAIN_EVENT_ADDRESSES = bool(include_events)
    if _EC is not None:
        _EC.close()
    _EC = _new_event_core(pipe)
    if system.hetero_name != DeviceType.PIM:
        raise WorkloadValidationError("reuse prefill requires --system dgx-attacc")
    validate_reuse_plan(workload, plan, system.model.ndec)
    _validate_layer_config(plan, system.model.ndec)
    system.model.build(1, 1, 2, True)
    templates = {layer.name: layer for layer in system.model.sum_decoder}
    qkv, score, softmax, context = (templates[name] for name in
                                    ("qkv", "score", "softmax", "context"))
    # Per-request side chosen by the dynamic placement rule (stable across
    # layers of one request; reported for auditability).
    dynamic_prefill_sides: Dict[str, str] = {}
    # Where prefill attention ACTUALLY ran, in attention rows summed over
    # layers (ruling chenyi9 2026-08-26): the dynamic decision record alone
    # under-counts -- an all-fresh request takes the ordinary-GPU branch and
    # a fully-reused zero-correction request runs no prefill attention at
    # all, yet both must appear in the ladder statistic with a uniform
    # denominator (every request of the workload).
    prefill_attn_rows: Dict[str, Dict[str, int]] = {}
    x2g = templates["comm_x2g"]
    post = list(system.model.sum_decoder)
    dbyte, local_hidden = qkv.dbyte, system.model.hdim // system.model.tp
    heads = max(1, system.model.num_heads // system.model.tp)
    # Ramulator's original AttAcc generator receives the address of one head
    # vector and distributes heads across channels / 8-KiB partitions itself.
    # Using the concatenated local-hidden vector here would consume one K/V
    # address interval per *all-head* token and incorrectly overflow the
    # fixed 8-MiB K-to-V window for long contexts.
    if kv_mapping not in ("master-diff", "naive", "naive-mask", "private"):
        raise WorkloadValidationError(
            "physical decode-on-PIM needs --kv-mapping master-diff, naive, "
            "naive-mask or private, got '{}'".format(kv_mapping))
    if kv_mapping == "private" and not physical_no_reuse:
        raise WorkloadValidationError(
            "--kv-mapping private is the no-reuse layout; use --reuse no-reuse")
    if physical_no_reuse:
        tlb = NoReuseKVLayout(system.model.dhead * dbyte)
    elif kv_mapping == "naive":
        tlb = NaiveKVLayout(system.model.dhead * dbyte)
    elif kv_mapping == "naive-mask":
        tlb = NaiveMaskKVLayout(system.model.dhead * dbyte)
    else:
        tlb = CacheBlendTLB(system.model.dhead * dbyte)
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
    if pim_prefill_mode not in ("gpu", "pim", "dynamic"):
        raise WorkloadValidationError(
            "--pim-prefill-mode must be 'gpu', 'pim' or 'dynamic'")
    if pim_pe_freq_ghz <= 0:
        raise WorkloadValidationError("--pe-freq-ghz must be positive")
    if gemv_buffer_bytes < 64:
        raise WorkloadValidationError(
            "--gemv-buffer-bytes must hold at least one 64-B query slice")
    batch_records: List[Dict[str, Any]] = []
    # (1) D_i bitmap loads (master-write filter): EPIC's correction set is
    # layer-invariant (one load per agent); CacheBlend samples per layer.
    epic_bitmap_loaded: set = set()
    di_bitmap_bytes_total = 0

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
                full = (plan.config.policy in CACHEBLEND_FAMILY and
                        layer_index in plan.config.cacheblend_full_recompute_layers)
                bindings = (_contiguous_no_reuse_tlb_rows(request, layer_index, tlb)
                            if physical_no_reuse else
                            _cacheblend_tlb_rows(workload, plan, layer_index,
                                                 request, tlb, force_fresh=full))
                prefill_bindings[layer_index] = bindings
                side_rows = prefill_attn_rows.setdefault(
                    request.request_id, {"pim": 0, "gpu": 0})
                if physical_no_reuse:
                    side_rows["gpu"] += request.total_length
                    request_ready = _append_physical_no_reuse_prefill_layer(
                        system, events, tlb, templates, post, layer=layer_index,
                        tier=tier, request=request, bindings=bindings,
                        initial_deps=request_ready)
                    continue
                reusable = [item for item in bindings if item[1]]
                # (1) D_i bitmap master-write filter (2026-08-21): before this
                # layer's masked scans, the driver loads a per-agent bitmap of
                # the overridden positions into the DIE.  Master-side score
                # writes at D_i are dropped against it, so diff/master arrival
                # order is immaterial -- the same information the mask gate
                # consults, given a second (score-side) consumer.
                bitmap_corrected = any(item[2] for item in bindings if item[1])
                if (not physical_no_reuse and not full and reusable and
                        bitmap_corrected and
                        (plan.config.policy not in ("epic", "recompute") or
                         request.request_id not in epic_bitmap_loaded)):
                    if plan.config.policy in ("epic", "recompute"):
                        epic_bitmap_loaded.add(request.request_id)
                    bitmap_bytes = (request.total_length + 7) // 8
                    di_bitmap_bytes_total += bitmap_bytes
                    transfer = _link_layer(x2g, "di_bitmap_gpu_to_die", bitmap_bytes)
                    time_s, energy = system.devices["GPU"].get_time_and_energy(transfer)
                    bitmap_link = _cacheblend_event(
                        events, layer=layer_index, tier=tier,
                        request=request.request_id, name="di_bitmap_gpu_to_die",
                        device="LINK", rows=1, time_s=time_s, energy=energy,
                        deps=request_ready, link_bytes=bitmap_bytes)
                    bitmap_load = _cacheblend_event(
                        events, layer=layer_index, tier=tier,
                        request=request.request_id, name="die_load_di_bitmap",
                        device="DIE", rows=1,
                        # Broadcast, not split: every stack's die stores the
                        # SAME bitmap for its own head, so the wall time uses
                        # one die's bandwidth share and the energy counts one
                        # copy per stack (head->HBM remap, 2026-08-27).
                        time_s=(bitmap_bytes /
                                (system.devices["Acc"].softmax_peak_bandwidth /
                                 max(1, getattr(system.devices["Acc"],
                                                "num_hbm", 1)))),
                        energy=(bitmap_bytes *
                                max(1, getattr(system.devices["Acc"],
                                               "num_hbm", 1)) *
                                system.devices["Acc"].energy_table["sram"],),
                        deps=(bitmap_link,))
                    request_ready = tuple(dict.fromkeys(
                        request_ready + (bitmap_load,)))
                # A layer with nothing resident to scan is simply an ordinary
                # GPU prefill; do not fabricate PIM traffic for it.  A full
                # recompute layer rebuilds every shared row, but the agent's
                # own history KV stays valid in every layer, so with history
                # the layer takes the split path below: the GPU attends the
                # rebuilt rows to each other while the PIM scans the resident
                # history extent.  (``full`` forces every segment row fresh,
                # so at history 0 this condition is identical to the previous
                # ``full or not reusable``.)
                if not reusable:
                    side_rows["gpu"] += request.total_length
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
                if not compute_positions:
                    # Fully reused layer with zero corrections (e.g. the same
                    # instance re-run under a second model): no fresh rows to
                    # compute, transfer or store -- the resident rows serve
                    # decode directly.
                    continue
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
                # PAPER TIE (Question 1): the physical event DAG carries
                # the same placement menu as the analytic A ladder so one
                # orchestration can be measured on either path and the
                # paper's placement numbers never depend on which simulator
                # abstraction produced them.
                # Prefill-placement menu of the 2026-08-24 ladder: "gpu"
                # (software prefill: resident rows come back over the link,
                # the GPU runs one full-context block -- the A1-A4 rung),
                # "pim" (bank-whole: everything lands and scans in the banks
                # -- the A5 rung), "dynamic" (Fugue/A6: per-request the two
                # sides are priced with the same models the branches use and
                # the cheaper one is committed, ties to the PIM).  The former
                # GPU/PIM "split" hybrid is abolished.
                location_deltas = _prefill_location_deltas(request, bindings)
                if getattr(tlb, "shadow_reads", True):
                    masked_prefill_keys = {_address_key(loc.shadow)
                                           for _, reused_flag, corrected, loc in bindings
                                           if reused_flag and corrected and
                                           loc.shadow is not None}
                    old_reads = [loc.shadow if corrected else loc
                                 for _, reused_flag, corrected, loc in bindings
                                 if reused_flag and
                                 (not corrected or loc.shadow is not None)]
                else:
                    # No mask gate (naive): read the corrected row from its
                    # own page and SKIP the master copy -- the master run
                    # splits at the gap (act a run, act one row, act on).
                    masked_prefill_keys = set()
                    old_reads = [loc for _, reused_flag, _, loc in bindings
                                 if reused_flag]
                scan_locations = old_reads + list(writes)
                readback_rows = [loc for _, reused_flag, corrected, loc in bindings
                                 if reused_flag and not corrected]
                if pim_prefill_mode == "dynamic":
                    prefill_side = dynamic_prefill_sides.get(request.request_id)
                    if prefill_side is None:
                        # xPU path: resident-row readback + full-context GPU
                        # attention block.
                        t_xpu = 0.0
                        op = _link_layer(x2g, "kv_pim_to_gpu",
                                         len(readback_rows) * 2 * local_hidden * dbyte)
                        t_xpu += system.devices["GPU"].get_time_and_energy(op)[0]
                        full_rows = len(readback_rows) + len(compute_positions)
                        for template, wide in ((score, "n"), (softmax, "n"),
                                               (context, "k")):
                            op = deepcopy(template)
                            op.m, op.numOp = len(compute_positions), heads
                            setattr(op, wide, full_rows)
                            t_xpu += system.devices["GPU"].get_time_and_energy(op)[0]
                        # bank path: the same sweep-set the "pim" branch
                        # prices (Q/ctx links + TLB plan + shared scans).
                        cap = max(1, min(batch_size,
                                         mq_query_capacity(gemv_buffer_bytes) //
                                         _gqa_group(system)))
                        sweeps = max(1, math.ceil(len(compute_positions) / cap))
                        est = deepcopy(score)
                        est.m, est.n, est.k, est.numOp = (
                            min(cap, len(compute_positions)),
                            len(scan_locations), system.model.dhead,
                            _gqa_kv_heads_local(system, heads))
                        est.pim_kv_runs = tlb.scan_runs(scan_locations)
                        est.pim_shared_kv = est.m * _gqa_group(system) > 1
                        est.pim_shared_queries = est.m * _gqa_group(system)
                        _apply_pim_batch(est, pim_batch_command, pim_pe_freq_ghz)
                        accelerator = system.devices["Acc"]
                        # Under a warm build the patched entries return zero
                        # placeholders; the side DECISION must be real, so
                        # the probe prices through the original entry (an
                        # inline cold simulation that also warms the cache).
                        if _WARM_BYPASS is not None:
                            measured = _WARM_BYPASS[0](est)
                        elif hasattr(accelerator, "get_time_and_energy_runs"):
                            measured = accelerator.get_time_and_energy_runs(est)
                        else:
                            # Aggregate mock-device API of lightweight tests.
                            measured = [accelerator.get_time_and_energy(est)]
                        t_bank = sum(item[0] for item in measured) * sweeps
                        t_bank += _tlb_plan_cost(est.pim_kv_runs)[0] * sweeps
                        for name in ("q_gpu_to_pim", "ctx_pim_to_gpu"):
                            op = _link_layer(x2g, name,
                                             len(compute_positions) * local_hidden * dbyte)
                            t_bank += system.devices["GPU"].get_time_and_energy(op)[0]
                        prefill_side = "pim" if t_bank <= t_xpu else "gpu"
                        dynamic_prefill_sides[request.request_id] = prefill_side
                else:
                    prefill_side = pim_prefill_mode
                side_rows["pim" if prefill_side == "pim" else "gpu"] += (
                    len(compute_positions))
                if prefill_side == "pim":
                    # (2) Bank-whole prefill with causal drop (2026-08-21):
                    # the batch's own K/V lands in the stack first (Fugue
                    # sec 4.5.2 landing order), every query of the sweep scans
                    # the full landed range -- reused masters (shadowed rows
                    # read-masked) plus all fresh/corrected rows -- and the
                    # DIE drops non-causal positions (key position > query
                    # position) while assembling each query's score vector:
                    # one comparator, no GPU triangle, no LSE tuple, a single
                    # sequential softmax per query.  The dropped upper
                    # triangle is scanned and therefore costed.
                    store = _append_channel_kv_stores(
                        system, events, layer=layer_index, tier=tier,
                        request=request.request_id, name="dram_store_diff_and_live",
                        locations=writes, dbyte=dbyte, deps=(kv_link,),
                        positions=compute_positions)
                    prefill_store_events.extend(store)
                    scan_addresses = [address for loc in scan_locations
                                      for address in (loc.key_address, loc.value_address)]
                    pim_results = []
                    sweep_cap = max(1, min(batch_size,
                                           mq_query_capacity(gemv_buffer_bytes) //
                                           _gqa_group(system)))
                    for first in range(0, len(compute_positions), sweep_cap):
                        grouped_positions = compute_positions[first:first + sweep_cap]
                        die_qs = []
                        for position in grouped_positions:
                            rotate_ready = _append_q_rotate_distribution(
                                system, events, x2g, layer=layer_index, tier=tier,
                                request=request.request_id, q_dependency=q_link,
                                q_bytes=local_hidden * dbyte, locations=scan_locations,
                                location_deltas=location_deltas,
                                rotate_mode=rotate_mode, name_prefix="",
                                positions=(position,))
                            die_qs.append(_cacheblend_event(
                                events, layer=layer_index, tier=tier,
                                request=request.request_id,
                                name="die_query_position_transform", device="DIE",
                                rows=1,
                                time_s=((local_hidden * dbyte) /
                                        system.devices["Acc"].softmax_peak_bandwidth),
                                energy=(local_hidden * dbyte *
                                        system.devices["Acc"].energy_table["sram"],),
                                deps=tuple(dict.fromkeys((rotate_ready,) + tuple(store))),
                                positions=(position,)))
                        op = deepcopy(score)
                        op.m, op.n, op.k, op.numOp = (len(grouped_positions),
                                                      len(scan_locations),
                                                      system.model.dhead,
                                                      _gqa_kv_heads_local(system, heads))
                        op.pim_kv_runs = tlb.scan_runs(scan_locations)
                        plan_time_s, plan_energy = _tlb_plan_cost(op.pim_kv_runs)
                        tlb_event = _cacheblend_event(
                            events, layer=layer_index, tier=tier,
                            request=request.request_id,
                            name="tlb_lookup_and_bank_plan", device="TLB",
                            rows=len(scan_locations), time_s=plan_time_s,
                            energy=plan_energy, deps=tuple(die_qs),
                            positions=tuple(grouped_positions),
                            addresses=scan_addresses)
                        op.pim_shared_kv = (len(grouped_positions) *
                                            _gqa_group(system)) > 1
                        op.pim_shared_queries = (len(grouped_positions) *
                                                 _gqa_group(system))
                        _apply_pim_batch(op, pim_batch_command, pim_pe_freq_ghz)
                        scan = _append_physical_pim_scan(
                            system, events, op=op, layer=layer_index, tier=tier,
                            request=request.request_id,
                            name="pim_kv_scan_score_softmax_pv",
                            rows=len(scan_locations), deps=(tlb_event,),
                            positions=tuple(grouped_positions), runs=op.pim_kv_runs,
                            masked=_masked_rows_per_run(op.pim_kv_runs,
                                                        masked_prefill_keys,
                                                        tlb.bytes_per_vector))
                        assembly_bytes = heads * (system.model.dhead + 2) * dbyte
                        for position in grouped_positions:
                            pim_results.append(_cacheblend_event(
                                events, layer=layer_index, tier=tier,
                                request=request.request_id,
                                name="die_score_assembly", device="DIE", rows=1,
                                time_s=(len(scan) * assembly_bytes /
                                        system.devices["Acc"].softmax_peak_bandwidth),
                                energy=(len(scan) * assembly_bytes *
                                        system.devices["Acc"].energy_table["sram"],),
                                deps=tuple(scan), positions=(position,)))
                    ctx_bytes = len(compute_positions) * local_hidden * dbyte
                    ctx_transfer = _link_layer(x2g, "ctx_pim_to_gpu", ctx_bytes)
                    time_s, energy = system.devices["GPU"].get_time_and_energy(ctx_transfer)
                    ctx_link = _cacheblend_event(
                        events, layer=layer_index, tier=tier,
                        request=request.request_id, name="ctx_pim_to_gpu",
                        device="LINK", rows=len(compute_positions), time_s=time_s,
                        energy=energy, deps=tuple(dict.fromkeys(pim_results)),
                        link_bytes=ctx_bytes, positions=compute_positions)
                    post_last = _post_attention_gpu(
                        system, events, post, layer=layer_index, tier=tier,
                        request=request.request_id, rows=len(compute_positions),
                        dependency=ctx_link, positions=compute_positions)
                    request_ready = (post_last,)
                    continue
                # prefill_side == "gpu": software reuse prefill (the
                # ladder's A1-A4 placement).  The resident reused/history
                # rows come back over the link, the GPU runs one ordinary
                # full-context attention block (fresh queries against
                # readback + fresh rows), and the fresh K/V still lands in
                # the banks because decode attention stays on the PIM.
                # Corrected rows are recomputed on the GPU in this very
                # layer, so their stale masters are not read back.
                gpu_last = q
                if readback_rows:
                    # DRAM-side read of the resident rows feeding the link
                    # (ruling chenyi9 2026-08-26): a scattered layout pays
                    # its activations here too -- per-channel pool events,
                    # so naive fragmentation surfaces in the readback.
                    dram_reads = _append_channel_kv_stores(
                        system, events, layer=layer_index, tier=tier,
                        request=request.request_id, name="dram_read_resident",
                        locations=readback_rows, dbyte=dbyte,
                        deps=request_ready, positions=compute_positions)
                    readback_bytes = len(readback_rows) * 2 * local_hidden * dbyte
                    op = _link_layer(x2g, "kv_pim_to_gpu", readback_bytes)
                    time_s, energy = system.devices["GPU"].get_time_and_energy(op)
                    gpu_last = _cacheblend_event(
                        events, layer=layer_index, tier=tier,
                        request=request.request_id, name="kv_pim_to_gpu",
                        device="LINK", rows=len(readback_rows), time_s=time_s,
                        energy=energy, deps=(q,) + tuple(dram_reads), link_bytes=readback_bytes,
                        positions=compute_positions,
                        addresses=[address for loc in readback_rows
                                   for address in (loc.key_address,
                                                   loc.value_address)])
                full_rows = len(readback_rows) + len(compute_positions)
                for template, name in ((score, "gpu_prefill_score"),
                                       (softmax, "gpu_prefill_softmax"),
                                       (context, "gpu_prefill_context")):
                    op = deepcopy(template)
                    op.m, op.numOp = len(compute_positions), heads
                    if name == "gpu_prefill_context":
                        op.k = full_rows
                    else:
                        op.n = full_rows
                    time_s, energy = system.devices["GPU"].get_time_and_energy(op)
                    gpu_last = _cacheblend_event(
                        events, layer=layer_index, tier=tier,
                        request=request.request_id, name=name, device="GPU",
                        rows=len(compute_positions), time_s=time_s,
                        energy=energy, deps=(gpu_last,),
                        positions=compute_positions)
                post_last = _post_attention_gpu(
                    system, events, post, layer=layer_index, tier=tier,
                    request=request.request_id, rows=len(compute_positions),
                    dependency=gpu_last, positions=compute_positions)
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

    ctx = {
        "events": events, "batch_records": batch_records, "tlb": tlb,
        "dynamic_prefill_sides": dynamic_prefill_sides,
        "prefill_attn_rows": prefill_attn_rows,
        "di_bitmap_bytes_total": di_bitmap_bytes_total,
        "local_hidden": local_hidden, "dbyte": dbyte, "heads": heads,
        "physical_no_reuse": physical_no_reuse, "batch_size": batch_size,
        "rotate_mode": rotate_mode, "pim_batch_command": pim_batch_command,
        "pim_prefill_mode": pim_prefill_mode, "kv_mapping": kv_mapping,
        "pim_pe_freq_ghz": pim_pe_freq_ghz,
        "gemv_buffer_bytes": gemv_buffer_bytes,
    }
    if _build_sink is not None:
        # W1/W2 warm build: hand the constructed graph to the warm driver;
        # validation/scheduling/annotation/report run exactly once, after
        # repricing (the old warm flow burned all of them on placeholders
        # and threw the result away).
        _build_sink.update(ctx)
        return {}
    return _finalize_cacheblend_report(system, workload, plan, ctx,
                                       include_events=include_events, pipe=pipe)


def _finalize_cacheblend_report(system, workload: Workload, plan: ReusePlan,
                                ctx: Dict[str, Any], *, include_events: bool,
                                pipe: bool) -> Dict[str, Any]:
    """Validation, scheduling, annotation and report assembly.

    Split out of _run_cacheblend_prefill (2026-08-28) so the warm path can
    build once, reprice the Acc events from the warm cache, and finalize --
    identical code path to a cold run from this point on.
    """
    events = ctx["events"]
    batch_records = ctx["batch_records"]
    tlb = ctx["tlb"]
    dynamic_prefill_sides = ctx["dynamic_prefill_sides"]
    prefill_attn_rows = ctx["prefill_attn_rows"]
    di_bitmap_bytes_total = ctx["di_bitmap_bytes_total"]
    local_hidden = ctx["local_hidden"]
    dbyte = ctx["dbyte"]
    heads = ctx["heads"]
    physical_no_reuse = ctx["physical_no_reuse"]
    batch_size = ctx["batch_size"]
    rotate_mode = ctx["rotate_mode"]
    pim_batch_command = ctx["pim_batch_command"]
    pim_prefill_mode = ctx["pim_prefill_mode"]
    kv_mapping = ctx["kv_mapping"]
    pim_pe_freq_ghz = ctx["pim_pe_freq_ghz"]
    gemv_buffer_bytes = ctx["gemv_buffer_bytes"]
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
        "pim_prefill_mode": pim_prefill_mode,
        "kv_mapping": "private" if physical_no_reuse else kv_mapping,
        "decode_attn": "pim",
        "pim_prefill_sides": dict(sorted(dynamic_prefill_sides.items())),
        "prefill_attention_rows": _prefill_side_summary(
            workload, prefill_attn_rows)[0],
        "prefill_attention_sides": _prefill_side_summary(
            workload, prefill_attn_rows)[1],
        "di_bitmap_bytes": di_bitmap_bytes_total,
        "di_write_filter": "master-side score writes at D_i dropped against the "
                           "per-agent bitmap; diff/master arrival order immaterial",
        "pim_pe_freq_ghz": pim_pe_freq_ghz,
        "gemv_buffer_bytes": gemv_buffer_bytes,
        "pim_sweep_query_capacity": mq_query_capacity(gemv_buffer_bytes),
        # Resident earlier-turn KV rows summed over all agents (per layer);
        # each agent's extent is scanned by every attention pass but never
        # computed, transferred, or stored in this run.
        "history_rows": sum(request.history_len for request in workload.requests),
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
        "energy_breakdown_nj": _energy_breakdown(scheduled),
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
                      pim_prefill_mode: str = "dynamic",
                      pim_batch_command: str = "replicate",
                      pim_pe_freq_ghz: float = MQ_DEFAULT_PE_FREQ_GHZ,
                      gemv_buffer_bytes: int = MQ_DEFAULT_GEMV_BUFFER_BYTES,
                      decode_attn: str = "pim",
                      kv_mapping: str = "master-diff",
                      warm: bool = True) -> Dict[str, Any]:
    """Dispatch address-resolved CacheBlend and EPIC to the shared DAG.

    CacheBlend samples correction rows per layer; EPIC overlays the fixed
    leading prefix of each shifted segment.  Both use one GPU/PIM/DIE/TLB/link
    event path.  ``kv_mapping``/``decode_attn`` carry the remaining ladder
    axes onto the event DAG (2026-08-26): master-diff = A4-A6 layout, naive =
    A3 scattered layout, private = A1 no-reuse layout, and decode_attn "gpu"
    (+ kv_mapping "none") = the A2 GPU-only rung.
    """
    if decode_attn not in ("pim", "gpu"):
        raise WorkloadValidationError("--decode-attn must be 'pim' or 'gpu'")
    if decode_attn == "gpu":
        if kv_mapping != "none":
            raise WorkloadValidationError(
                "--decode-attn gpu keeps the KV cache in GPU memory; "
                "use --kv-mapping none")
        if plan.config.policy not in ("cacheblend", "epic", "recompute",
                                      "no-reuse", "promptcache", "cachecraft",
                                      "cachetune"):
            raise WorkloadValidationError(
                "unknown reuse policy for the GPU-only event path")
        return _run_gpu_software_only(system, workload, plan, pipe=pipe,
                                      include_events=include_events)
    if plan.config.policy in ("cacheblend", "epic", "recompute", "no-reuse"):
        run_kwargs = dict(pipe=pipe,
                          batch_size=cacheblend_batch_size,
                          physical_no_reuse=(plan.config.policy == "no-reuse"),
                          kv_mapping=kv_mapping,
                          rotate_mode=cacheblend_rotate_mode,
                          pim_prefill_mode=pim_prefill_mode,
                          pim_batch_command=pim_batch_command,
                          pim_pe_freq_ghz=pim_pe_freq_ghz,
                          gemv_buffer_bytes=gemv_buffer_bytes)
        if warm:
            report = _warm_build_price_finalize(
                system, workload, plan, include_events=include_events,
                run_kwargs=run_kwargs)
            if report is not None:
                return report
        return _run_cacheblend_prefill(system, workload, plan,
                                       include_events=include_events,
                                       **run_kwargs)
    return _run_legacy_reuse_prefill(system, workload, plan)


def _warm_build_price_finalize(system, workload: Workload, plan: ReusePlan,
                               *, include_events: bool,
                               run_kwargs: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Cache-first warm flow, build-once edition (W1+W2, chenyi9 2026-08-28).

    The old flow built the FULL graph with zero placeholders, ran every
    post-build phase, threw all of it away, simulated the collected shapes,
    then built everything a second time against the warm cache.  This
    edition builds ONCE: the accelerator entries return placeholders while
    a ledger records (op, event indexes) for every placeholder-priced
    event; the A6 estimator prices its probes through the ORIGINAL entries
    (real inline simulations -- decisions must never see placeholders);
    after the parallel simulation phase the ledger events are repriced
    from the warm cache and the single graph proceeds to the one and only
    validation/schedule/annotation/report pass.  Returns None when
    ineligible; the caller falls back to the plain cold path.
    """
    global _WARM_LEDGER, _WARM_BYPASS
    accelerator = system.devices.get("Acc") if hasattr(system, "devices") else None
    if accelerator is None or not hasattr(accelerator, "get_time_and_energy_runs"):
        return None
    workers = getattr(getattr(accelerator, "ramulator", None), "workers", 1)
    if workers <= 1:
        return None
    import sys as _sys
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    ledger: List[Tuple[str, Any, Tuple[int, ...]]] = []
    sink: Dict[str, Any] = {}
    original_runs = accelerator.get_time_and_energy_runs
    original_agg = accelerator.get_time_and_energy

    def _collect_runs(op):
        runs = getattr(op, "pim_kv_runs", None) or ((0, 0, 1, 0, 16),)
        return [(0.0, (0.0,)) for _ in runs]

    def _collect_agg(op):
        return (0.0, (0.0,))

    accelerator.get_time_and_energy_runs = _collect_runs
    accelerator.get_time_and_energy = _collect_agg
    _WARM_LEDGER = ledger
    _WARM_BYPASS = (original_runs, original_agg)
    started = _time.time()
    try:
        _run_cacheblend_prefill(system, workload, plan,
                                include_events=include_events,
                                _build_sink=sink, **run_kwargs)
    except Exception:
        sink = {}                  # fall back to the plain cold path
    finally:
        accelerator.get_time_and_energy_runs = original_runs
        accelerator.get_time_and_energy = original_agg
        _WARM_LEDGER = None
        _WARM_BYPASS = None
    if not sink:
        return None

    # Exact-repeat dedupe, conservative key = everything that can influence
    # the simulation (unchanged from the old flow).
    unique: Dict[Tuple, Tuple[str, Layer]] = {}
    flat_ops = []
    for kind, op, _ in ledger:
        if kind == "agg_cb_sweeps":
            flat_ops.extend(("agg_cb", sweep_op) for sweep_op, _ in op)
        else:
            flat_ops.append((kind, op))
    for kind, op in flat_ops:
        key = (kind, getattr(op, "name", ""), op.m, op.n, op.k,
               getattr(op, "numOp", 1), getattr(op, "dbyte", 2),
               tuple(getattr(op, "pim_kv_runs", ()) or ()),
               getattr(op, "pim_shared_kv", False),
               getattr(op, "pim_shared_queries", 1),
               getattr(op, "pim_batch_command", None),
               getattr(op, "pim_pe_freq_ghz", None))
        unique.setdefault(key, (kind, op))
    jobs = list(unique.values())
    print("[warm] collected {} Acc ops -> {} unique shapes (single build "
          "{:.0f}s); simulating with up to {} workers".format(
              len(ledger), len(jobs), _time.time() - started, workers),
          file=_sys.stderr, flush=True)
    if jobs:
        fire = {"runs": original_runs, "agg": original_agg,
                "agg_cb": original_agg}
        # Pool width follows --ramulator-workers (96-core budget ruling);
        # the wrapper's per-call inner pool is suppressed while the outer
        # pool runs (outer x inner nesting blew the budget, 2026-08-26).
        ramulator = getattr(accelerator, "ramulator", None)
        saved_inner = getattr(ramulator, "workers", None)
        if ramulator is not None:
            ramulator.workers = 1
        try:
            with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                list(pool.map(lambda item: fire[item[0]](item[1]), jobs))
        finally:
            if ramulator is not None and saved_inner is not None:
                ramulator.workers = saved_inner

    # Reprice the placeholder events from the now-warm cache.  Every call
    # below is a signature-cache hit (the firing phase covered a superset
    # of the ledger's simulation signatures in this same process).
    events = sink["events"]
    repriced = 0
    for kind, op, indexes in ledger:
        if kind == "runs":
            measured = original_runs(op)
            if len(measured) != len(indexes):
                raise WorkloadValidationError(
                    "warm reprice run-count mismatch")
            for index, (time_s, energy) in zip(indexes, measured):
                events[index] = replace(events[index], time_s=time_s,
                                        energy_nj=sum(energy) / 1000.0)
                if _EC is not None:
                    _EC.set_duration(index, time_s)
                repriced += 1
        elif kind == "agg_cb":
            # A _cacheblend_event priced by the aggregate entry (the
            # bank-whole contiguous scan): nJ convention (sum/1000).
            time_s, energy = original_agg(op)
            events[indexes[0]] = replace(events[indexes[0]], time_s=time_s,
                                         energy_nj=sum(energy) / 1000.0)
            if _EC is not None:
                _EC.set_duration(indexes[0], time_s)
            repriced += 1
            continue
        elif kind == "agg_cb_sweeps":
            # Capacity-swept contiguous scan: op is a tuple of
            # (sweep_op, count); total = sum(count * priced sweep).
            total_time = 0.0
            total_energy_pj = 0.0
            for sweep_op, count in op:
                time_s, energy = original_agg(sweep_op)
                total_time += count * time_s
                total_energy_pj += count * sum(energy)
            events[indexes[0]] = replace(events[indexes[0]], time_s=total_time,
                                         energy_nj=total_energy_pj / 1000.0)
            if _EC is not None:
                _EC.set_duration(indexes[0], total_time)
            repriced += 1
            continue
        else:
            time_s, energy = original_agg(op)
            # Mirrors _event exactly (legacy aggregate events store the
            # raw sum; the unit question is tracked separately).
            events[indexes[0]] = replace(events[indexes[0]], time_s=time_s,
                                         energy_nj=sum(energy))
            if _EC is not None:
                _EC.set_duration(indexes[0], time_s)
            repriced += 1
    print("[warm] cache ready after {:.0f}s total; repriced {} events, "
          "finalizing once".format(_time.time() - started, repriced),
          file=_sys.stderr, flush=True)
    return _finalize_cacheblend_report(system, workload, plan, sink,
                                       include_events=include_events,
                                       pipe=run_kwargs["pipe"])
