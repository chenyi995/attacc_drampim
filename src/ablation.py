"""Legacy-model ablation driver for the PIM KV-reuse study.

This module answers one question with one code path: given a workload and a
reuse policy, what do latency, energy, and AttAcc memory occupancy look like
when the *placement* of prefill attention, decode attention, and the KV cache
changes?

It deliberately stays inside the original AttAcc evaluation abstraction:

* every layer is priced by the same ``devices['GPU']`` / ``devices['Acc']``
  models that :meth:`System.simulate` uses,
* decode attention on PIM is measured by the original AttAcc Ramulator trace
  (one trace per contiguous K/V extent, deduplicated by run signature),
* a dependency tier is served as rectangular padded batches, exactly as the
  legacy no-reuse adapter does.

The physical TLB/event-DAG model in :mod:`src.workload_runner` is a different
abstraction and is not used here.

Six named configurations (``--ablation``) span the study:

===  ==============  ============  ============  =============================
key  prefill attn    decode attn   KV mapping    meaning
===  ==============  ============  ============  =============================
A1   gpu             pim           private       AttAcc, no reuse (reference)
A2   gpu             gpu           none          software reuse on the GPU only
A3   gpu             pim           naive         SW reuse + AttAcc, scattered
                                                 layout (no channel split)
A4   gpu             pim           master-diff   SW reuse + AttAcc, channel-
                                                 split master/diff pools
A5   pim             pim           master-diff   A4 + ALL prefill attention on
                                                 the PIM + MQ batching
A6   dynamic         pim           master-diff   Fugue: A5 + dynamic per-class
                                                 GPU/PIM prefill placement
===  ==============  ============  ============  =============================

The ladder of 2026-08-24; the former "split" A6 (GPU/PIM hybrid prefill) is
abolished.  Attention batching follows the rung (A1-A4 replicate, A5/A6
mq).  The reuse policy (``--reuse``, incl. the promptcache/cachecraft/
cachetune enrichment) is independent of the placement configuration; ``A1``
is only meaningful with ``no-reuse`` and ``A3``--``A6`` with a reuse policy.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .model import Layer
from .ramulator_wrapper import (MQ_DEFAULT_GEMV_BUFFER_BYTES,
                                MQ_DEFAULT_PE_FREQ_GHZ, mq_query_capacity)
from .system import apply_attacc_pipeline
from .type import DeviceType, LayerType
from .workload import (CACHEBLEND_FAMILY, EPIC_FAMILY, ReusePlan, Request,
                       Workload, WorkloadValidationError)
from .workload_runner import _tier_shapes

# A channel is one 1-GiB region of the HBM3-PIM address mapper; K starts at an
# 8-KiB-aligned partition and V sits 8 MiB above it.  These are the original
# AttAcc trace-generator constants, repeated here so an ablation run produces
# the same addresses as the physical model's TLB.
_HBM_CHANNEL_BYTES = 1 << 30
_ORIGINAL_KV_GAP_BYTES = 1 << 23
_TOTAL_CHANNELS = 16

PREFILL_ATTN_MODES = ("gpu", "pim", "dynamic")
DECODE_ATTN_MODES = ("gpu", "pim")
KV_MAPPINGS = ("none", "private", "naive", "master-diff")
MASTER_SHADOW_MODES = ("read-mask", "skip")

# The A1-A6 ladder (Chenyi's ruling, 2026-08-24).  PAPER TIE (Question 1):
# the paper asks WHERE prefill attention, decode attention and the KV cache
# should live once requests share KV -- each rung isolates exactly one
# placement decision, so the ladder differences ARE the paper's evidence:
# A2-A1 = software reuse alone, A3-A2 = PIM decode + KV residency,
# A4-A3 = PIM-aware layout, A5-A4 = prefill attention on the PIM (with the
# batching it enables), A6-A5 = the dynamic per-request rule (Fugue).
# The former A6 -- the
# GPU/PIM "split" prefill hybrid (GPU attends the fresh rows, the PIM scans
# the reused KV, LSE merge at the DIE) -- is ABOLISHED as a ladder point:
# the design menu is either all-GPU or all-PIM prefill attention, never
# half and half.  Attention batching (the MQ command) is coupled to
# prefill-on-PIM: A1-A4 run the legacy replicate command, A5/A6 run MQ.
# Every rung assumes multi-round agentic orchestration (per-request
# ``history_len``) -- that is workload configuration, not a switch here.
PRESETS: Dict[str, Dict[str, str]] = {
    "A1": {"prefill_attn": "gpu", "decode_attn": "pim", "kv_mapping": "private",
           "pim_batch_command": "replicate"},
    "A2": {"prefill_attn": "gpu", "decode_attn": "gpu", "kv_mapping": "none",
           "pim_batch_command": "replicate"},
    "A3": {"prefill_attn": "gpu", "decode_attn": "pim", "kv_mapping": "naive",
           "pim_batch_command": "replicate"},
    "A4": {"prefill_attn": "gpu", "decode_attn": "pim", "kv_mapping": "master-diff",
           "pim_batch_command": "replicate"},
    # A5/A6 carry the C3 microarchitecture point they are defined with
    # (ruling 2026-08-25): prefill-on-PIM, attention batching and the
    # bank-PE design point are ONE package.  Values = the derived balance
    # point where the PE, the TSV stream and the in-bank area budget meet:
    # n_cap = 12 resident queries @ nCCDAB floor 6, f* = 12/(6 x 0.769 ns)
    # ~= 2.6 GHz, buffer = 12 x 64 B = 768 B.  PROVISIONAL -- Chenyi will
    # retune these knobs later; an explicit CLI value still overrides.
    "A5": {"prefill_attn": "pim", "decode_attn": "pim", "kv_mapping": "master-diff",
           "pim_batch_command": "mq", "pim_pe_freq_ghz": 2.6,
           "gemv_buffer_bytes": 768},
    "A6": {"prefill_attn": "dynamic", "decode_attn": "pim", "kv_mapping": "master-diff",
           "pim_batch_command": "mq", "pim_pe_freq_ghz": 2.6,
           "gemv_buffer_bytes": 768},
}

PRESET_LABELS = {
    "A1": "AttAcc, no reuse (reference)",
    "A2": "software reuse on the GPU only",
    "A3": "software reuse + AttAcc, scattered layout (no channel split)",
    "A4": "software reuse + AttAcc, channel-split master/diff pools",
    "A5": "A4 + all prefill attention on the PIM + MQ attention batching",
    "A6": "Fugue: A5 + dynamic per-class GPU/PIM prefill placement",
}


@dataclass(frozen=True)
class AblationConfig:
    """Where each stage runs and how the KV cache is laid out in the PIM."""

    preset: Optional[str]
    prefill_attn: str
    decode_attn: str
    kv_mapping: str
    # Queries that share one PIM K/V stream when prefill attention runs on the
    # PIM (the GEMV buffer's query capacity).  1 reproduces AttAcc's native
    # one-query-per-scan decode behaviour applied to prefill.
    pim_prefill_query_batch: int = 4
    # Channels of the 16-channel device given to the master pool; the diff
    # pool receives the rest.  Only used by ``master-diff``.  15/1 by the
    # 2026-08-25 ruling (audit issue 3a): diff rows are few, so the diff
    # pool must not halve the master stream.
    master_pool_channels: int = 15
    # Software prefill (A3/A4) reads reused K/V back from PIM over the link.
    prefill_kv_readback: bool = True
    # What the master pool does with a row that a correction has overwritten.
    # ``read-mask`` streams it with the master run and drops it from the score,
    # keeping the run contiguous; ``skip`` leaves it out of the address stream
    # and therefore breaks the master run at every correction.
    master_shadow: str = "read-mask"
    # How a multi-query PIM sweep issues its MACs (PLAN_mq_command.md).
    # ``replicate``: one MAC_AB per (column, query), the legacy expansion.
    # ``mq``: one MAC_AB per column serves every resident Q; the PE multiplies
    # internally and the command interval carries the n-fold time and the
    # power stretch.
    pim_batch_command: str = "replicate"
    # Bank-PE clock for the MQ command (AttAcc synthesizes 666 MHz).
    pim_pe_freq_ghz: float = MQ_DEFAULT_PE_FREQ_GHZ
    # Per-bank GEMV (input-vector) buffer size; one Q slice is 64 B, so this
    # caps the queries resident in one sweep (the sweep splits beyond it).
    gemv_buffer_bytes: int = MQ_DEFAULT_GEMV_BUFFER_BYTES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "label": PRESET_LABELS.get(self.preset or "", "custom"),
            "prefill_attn": self.prefill_attn,
            "decode_attn": self.decode_attn,
            "kv_mapping": self.kv_mapping,
            "pim_prefill_query_batch": self.pim_prefill_query_batch,
            "master_pool_channels": self.master_pool_channels,
            "prefill_kv_readback": self.prefill_kv_readback,
            "master_shadow": self.master_shadow,
            "pim_batch_command": self.pim_batch_command,
            "pim_pe_freq_ghz": self.pim_pe_freq_ghz,
            "gemv_buffer_bytes": self.gemv_buffer_bytes,
        }


def resolve_config(preset: Optional[str], prefill_attn: Optional[str],
                   decode_attn: Optional[str], kv_mapping: Optional[str],
                   *, policy: str,
                   pim_prefill_query_batch: Optional[int] = None,
                   master_pool_channels: int = 15,
                   prefill_kv_readback: bool = True,
                   master_shadow: str = "read-mask",
                   pim_batch_command: Optional[str] = None,
                   pim_pe_freq_ghz: Optional[float] = None,
                   gemv_buffer_bytes: Optional[int] = None) -> AblationConfig:
    """Expand a preset and let explicit switches override it.

    ``pim_batch_command=None`` follows the ladder: the preset's coupling
    (A1-A4 replicate, A5/A6 mq) or replicate for a preset-less run.  An
    explicit value always wins.
    """
    base = dict(PRESETS[preset]) if preset else {}
    if not base and not (prefill_attn and decode_attn):
        raise WorkloadValidationError(
            "--ablation preset or both --prefill-attn and --decode-attn are required")
    resolved = {
        "prefill_attn": prefill_attn or base.get("prefill_attn", "gpu"),
        "decode_attn": decode_attn or base.get("decode_attn", "pim"),
        "kv_mapping": kv_mapping or base.get("kv_mapping", "private"),
    }
    pim_batch_command = (pim_batch_command or
                         base.get("pim_batch_command", "replicate"))
    # The microarchitecture knobs follow the rung (A5/A6 carry the balance
    # point) unless the CLI sets them explicitly; a preset-less run keeps
    # the stock AttAcc values.  PROVISIONAL values, see PRESETS.
    if pim_pe_freq_ghz is None:
        pim_pe_freq_ghz = base.get("pim_pe_freq_ghz", MQ_DEFAULT_PE_FREQ_GHZ)
    if gemv_buffer_bytes is None:
        gemv_buffer_bytes = base.get("gemv_buffer_bytes",
                                     MQ_DEFAULT_GEMV_BUFFER_BYTES)
    if pim_prefill_query_batch is None:
        # Under the mq command the prefill sweep is bounded by the GEMV
        # buffer's resident-Q capacity, not by a separate knob (audit
        # issue 4); replicate keeps the legacy default of 4.
        pim_prefill_query_batch = (mq_query_capacity(gemv_buffer_bytes)
                                   if pim_batch_command == "mq" else 4)
    if resolved["prefill_attn"] not in PREFILL_ATTN_MODES:
        raise WorkloadValidationError("--prefill-attn must be one of {}".format(
            ", ".join(PREFILL_ATTN_MODES)))
    if resolved["decode_attn"] not in DECODE_ATTN_MODES:
        raise WorkloadValidationError("--decode-attn must be one of {}".format(
            ", ".join(DECODE_ATTN_MODES)))
    if resolved["kv_mapping"] not in KV_MAPPINGS:
        raise WorkloadValidationError("--kv-mapping must be one of {}".format(
            ", ".join(KV_MAPPINGS)))
    if resolved["decode_attn"] == "gpu" and resolved["kv_mapping"] != "none":
        raise WorkloadValidationError(
            "--decode-attn gpu keeps the KV cache in GPU memory; use --kv-mapping none")
    if resolved["decode_attn"] == "pim" and resolved["kv_mapping"] == "none":
        raise WorkloadValidationError(
            "--decode-attn pim needs a PIM KV mapping (private, naive, master-diff)")
    if policy == "no-reuse" and resolved["kv_mapping"] in ("naive", "master-diff"):
        raise WorkloadValidationError(
            "--kv-mapping {} describes reused/recomputed KV; --reuse no-reuse uses "
            "private".format(resolved["kv_mapping"]))
    if policy != "no-reuse" and resolved["kv_mapping"] == "private":
        raise WorkloadValidationError(
            "--kv-mapping private is the no-reuse layout; use naive or master-diff")
    if not 1 <= master_pool_channels < _TOTAL_CHANNELS:
        raise WorkloadValidationError(
            "--kv-pool-split must leave both pools at least one channel")
    if pim_prefill_query_batch < 1:
        raise WorkloadValidationError("--pim-prefill-query-batch must be >= 1")
    if master_shadow not in MASTER_SHADOW_MODES:
        raise WorkloadValidationError("--master-shadow must be one of {}".format(
            ", ".join(MASTER_SHADOW_MODES)))
    if pim_batch_command not in ("replicate", "mq"):
        raise WorkloadValidationError(
            "--pim-batch-command must be 'replicate' or 'mq'")
    if pim_pe_freq_ghz <= 0:
        raise WorkloadValidationError("--pe-freq-ghz must be positive")
    if gemv_buffer_bytes < 64:
        raise WorkloadValidationError(
            "--gemv-buffer-bytes must hold at least one 64-B query slice")
    return AblationConfig(preset=preset, pim_prefill_query_batch=pim_prefill_query_batch,
                          master_pool_channels=master_pool_channels,
                          prefill_kv_readback=prefill_kv_readback,
                          master_shadow=master_shadow,
                          pim_batch_command=pim_batch_command,
                          pim_pe_freq_ghz=pim_pe_freq_ghz,
                          gemv_buffer_bytes=gemv_buffer_bytes, **resolved)


# ---------------------------------------------------------------------------
# Reuse bookkeeping
# ---------------------------------------------------------------------------


def _reused_tokens_by_request(plan: ReusePlan) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for decision in plan.reusable:
        totals[decision.request_id] = totals.get(decision.request_id, 0) + decision.length
    return totals


def _reused_segments_by_request(plan: ReusePlan) -> Dict[str, List[Tuple[int, int, str]]]:
    """``request -> [(segment index, length, fingerprint)]`` for reused segments."""
    result: Dict[str, List[Tuple[int, int, str]]] = {}
    for decision in plan.reusable:
        result.setdefault(decision.request_id, []).append(
            (decision.segment_index, decision.length, decision.fingerprint))
    for rows in result.values():
        rows.sort()
    return result


def _epic_prefix_by_request(plan: ReusePlan) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for decision in plan.reusable:
        totals[decision.request_id] = (totals.get(decision.request_id, 0) +
                                       len(decision.epic_prefix_rows))
    return totals


def _corrected_rows(plan: ReusePlan, layer: int,
                    request_id: str) -> Dict[int, Tuple[int, ...]]:
    """Recomputed rows of one request in one transformer layer, by segment.

    CacheBlend samples them per layer; EPIC fixes the leading rows of every
    shifted segment in every layer.  Returns ``{}`` when the layer reuses the
    old K/V unchanged.
    """
    policy = plan.config.policy
    if policy in CACHEBLEND_FAMILY:
        if layer in plan.config.cacheblend_full_recompute_layers:
            return {}
        return dict(plan.cacheblend_partial_rows.get(layer, {}).get(request_id, {}))
    if policy in EPIC_FAMILY:
        return {decision.segment_index: decision.epic_prefix_rows
                for decision in plan.reusable
                if decision.request_id == request_id and decision.epic_prefix_rows}
    return {}


def _layer_classes(plan: ReusePlan, ndec: int,
                   requests: Sequence[Request]) -> List[Tuple[int, int]]:
    """Group transformer layers that share a decode K/V access pattern.

    Returns ``[(representative layer, number of layers in the class)]``.  A
    CacheBlend full-recompute layer holds no diff rows and therefore scans a
    different physical pattern from a partial layer; EPIC applies the same
    correction in every layer.
    """
    policy = plan.config.policy
    if policy in CACHEBLEND_FAMILY:
        full = tuple(layer for layer in plan.config.cacheblend_full_recompute_layers
                     if layer < ndec)
        partial = tuple(layer for layer in plan.config.cacheblend_partial_recompute_layers
                        if layer < ndec)
        classes = []
        if full:
            classes.append((full[0], len(full)))
        if partial:
            classes.append((partial[0], len(partial)))
        return classes
    return [(0, ndec)]


# ---------------------------------------------------------------------------
# Physical K/V run structure seen by one decode scan
# ---------------------------------------------------------------------------


def _channel_extent(channel: int, order: int, rows: int, stride: int) -> Tuple[int, int]:
    """Base K/V byte addresses of the ``order``-th extent inside a channel."""
    key_base = (channel * _HBM_CHANNEL_BYTES +
                (order * rows * stride) % _ORIGINAL_KV_GAP_BYTES)
    return key_base, key_base + _ORIGINAL_KV_GAP_BYTES


def _naive_run_lengths(request: Request, plan: ReusePlan, layer: int) -> List[int]:
    """Runs of a naive, non-PIM-aware layout.

    The software's chunk/block layout is preserved and every block -- reused,
    recomputed, and private -- is allocated from one pool spanning all 16
    channels.  A scan walks the request's K/V in logical token order, so it
    leaves a contiguous extent at every segment boundary and at every
    recomputed row (whose new K/V lives in a separately allocated block).
    The stale master row underneath a recomputed row is skipped, not read.
    """
    reused = {index: length for index, length, _ in
              _reused_segments_by_request(plan).get(request.request_id, ())}
    corrected = _corrected_rows(plan, layer, request.request_id)
    lengths: List[int] = []
    for index, segment in enumerate(request.segments):
        if index not in reused:
            # A privately computed segment is one freshly allocated block.
            lengths.append(segment.length)
            continue
        rows = sorted(set(corrected.get(index, ())))
        cursor = 0
        for row in rows:
            if row > cursor:
                lengths.append(row - cursor)  # master piece before the patch
            lengths.append(1)                 # the recomputed row, elsewhere
            cursor = row + 1
        if cursor < segment.length:
            lengths.append(segment.length - cursor)
    return [length for length in lengths if length > 0]


def _master_diff_lengths(request: Request, plan: ReusePlan, layer: int,
                         shadow: str = "read-mask") -> Tuple[List[int], int]:
    """Runs of the PIM-aware master/diff layout, one tuple per channel pool.

    Master holds every immutable row -- including the rows shadowed by a
    correction, which are streamed and masked out of the score rather than
    breaking the stream -- and is co-located per segment.  Diff holds the
    recomputed rows as one contiguous extent in the other pool.  The two pools
    use disjoint channels and therefore run concurrently.
    """
    reused = {index: length for index, length, _ in
              _reused_segments_by_request(plan).get(request.request_id, ())}
    corrected = _corrected_rows(plan, layer, request.request_id)
    diff_rows = sum(len(set(rows)) for rows in corrected.values())
    master_lengths: List[int] = []
    private_rows = 0
    for index, segment in enumerate(request.segments):
        if index not in reused:
            private_rows += segment.length
            continue
        if shadow == "read-mask":
            # One shared, PIM-aware extent per reused chunk: the overwritten
            # rows stay in the address stream and are masked out of the score.
            master_lengths.append(segment.length)
            continue
        # ``skip`` leaves every overwritten row out of the address stream, so
        # the chunk's extent breaks at each correction.
        cursor = 0
        for row in sorted(set(corrected.get(index, ()))):
            if row > cursor:
                master_lengths.append(row - cursor)
            cursor = row + 1
        if cursor < segment.length:
            master_lengths.append(segment.length - cursor)
    if private_rows:
        master_lengths.append(private_rows)
    return [length for length in master_lengths if length > 0], diff_rows


def _private_runs(context_rows: int, stride: int
                  ) -> Tuple[Tuple[int, int, int, int, int], ...]:
    key_base, value_base = _channel_extent(0, 0, context_rows, stride)
    return ((key_base, value_base, context_rows, 0, _TOTAL_CHANNELS),)


@dataclass(frozen=True)
class ScanProfile:
    """Physical K/V runs one decode step performs, grouped by channel pool."""

    pools: Tuple[Tuple[Tuple[int, int, int, int, int], ...], ...]
    rows_read: int
    run_count: int
    # True when the scan is one full-length extent over all 16 channels, i.e.
    # exactly the original AttAcc measurement.
    legacy_shape: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"pools": [[{"rows": run[2], "channel_base": run[3],
                            "channels": run[4]} for run in pool]
                          for pool in self.pools],
                "rows_read": self.rows_read, "run_count": self.run_count,
                "legacy_shape": self.legacy_shape}


def _runs_from_lengths(lengths: Sequence[int], stride: int, channel_base: int,
                       channels: int) -> Tuple[Tuple[int, int, int, int, int], ...]:
    runs = []
    for order, length in enumerate(lengths):
        if length <= 0:
            continue
        key_base, value_base = _channel_extent(channel_base, order, length, stride)
        runs.append((key_base, value_base, length, channel_base, channels))
    return tuple(runs)


def _naive_channel_pools(requests: Sequence[Request], plan: ReusePlan,
                         layer: int, input_rows: int, tail_rows: int,
                         stride: int, history_rows: int) -> Tuple[Tuple, int, int]:
    """Naive layout with per-chunk channel tracking (ruling 2026-08-25).

    The naive allocator is SEQUENTIAL: every block (shared chunk, private
    segment, per-request correction/history/generated block) takes the next
    channel in order, wrapping over the 16 channels.  A decode scan touches
    its blocks on whichever channel they landed: blocks on different
    channels stream in parallel, blocks that COLLIDE on one channel
    serialize -- that queueing, not the ACT-scale run boundaries, is the
    real cost of a non-PIM-aware layout (audit issue 3b).  Each channel's
    load becomes one single-channel pool, so ``_decode_block_time``'s
    max-over-pools is exactly the serialized-channel completion time.
    """
    cursor = 0
    channel_of: Dict[Any, int] = {}

    def channel(key):
        nonlocal cursor
        if key not in channel_of:
            channel_of[key] = cursor % _TOTAL_CHANNELS
            cursor += 1
        return channel_of[key]

    ordered = sorted(requests, key=lambda item: (item.tier, item.request_id))
    reused_by_request = {request.request_id: _reused_segments_by_request(plan).get(
        request.request_id, ()) for request in ordered}
    # Allocation pass: walk arrival order once so shared chunks get the
    # owner's channel and every later block advances the cursor.
    for request in ordered:
        reused = {index for index, _, _ in reused_by_request[request.request_id]}
        for index, segment in enumerate(request.segments):
            if index in reused:
                channel(("chunk", segment.fingerprint))
            else:
                channel(("own", request.request_id, index))
        channel(("corr", request.request_id))
        if history_rows:
            channel(("hist", request.request_id))
        channel(("tail", request.request_id))

    # Per-request per-channel loads, then the batch mean per channel.
    loads = [0.0] * _TOTAL_CHANNELS
    for request in ordered:
        reused = {index for index, _, _ in reused_by_request[request.request_id]}
        corrected = _corrected_rows(plan, layer, request.request_id)
        for index, segment in enumerate(request.segments):
            patched = len(set(corrected.get(index, ())))
            if index in reused:
                # Stale rows under a patch are skipped; patched rows are
                # read from this request's correction block.
                loads[channel(("chunk", segment.fingerprint))] += (
                    segment.length - patched)
                if patched:
                    loads[channel(("corr", request.request_id))] += patched
            else:
                loads[channel(("own", request.request_id, index))] += segment.length
        if history_rows:
            loads[channel(("hist", request.request_id))] += history_rows
        if tail_rows:
            loads[channel(("tail", request.request_id))] += tail_rows
    members = max(1, len(ordered))
    loads = [value / members for value in loads]
    # Rescale to the padded per-member scan length the legacy trace carries.
    target = history_rows + input_rows + tail_rows
    total = sum(loads)
    if total <= 0:
        return ((_private_runs(target, stride),), target, 1)
    loads = [value * target / total for value in loads]
    pools = []
    for index, value in enumerate(loads):
        rows = int(round(value))
        if rows <= 0:
            continue
        pools.append(_runs_from_lengths([rows], stride, index, 1))
    rows_total = sum(run[2] for pool in pools for run in pool)
    return (tuple(pools), rows_total, sum(len(pool) for pool in pools))


def _batch_scan_profile(requests: Sequence[Request], plan: ReusePlan, layer: int,
                        input_rows: int, tail_rows: int, stride: int,
                        config: AblationConfig,
                        history_rows: int = 0) -> ScanProfile:
    """Representative scan of one padded batch.

    A legacy trace carries the batch in ``numOp`` (heads x batch), so every
    attention unit in the trace shares one run-length profile.  The batch
    profile is therefore the member-average structure, rescaled to the padded
    input length.  Generated tokens are appended as their own extent, so the
    profile changes only in its tail as decode proceeds and the Ramulator
    signature cache keeps serving the fixed part.  ``history_rows`` is the
    agent's own KV from earlier turns: one resident, contiguous extent that
    the scan walks before this turn's rows (private rows in the master pool
    under ``master-diff``).
    """
    if config.kv_mapping == "private":
        return ScanProfile((_private_runs(history_rows + input_rows + tail_rows,
                                          stride),),
                           history_rows + input_rows + tail_rows, 1,
                           legacy_shape=True)
    if config.kv_mapping == "naive":
        pools, rows, run_count = _naive_channel_pools(
            requests, plan, layer, input_rows, tail_rows, stride, history_rows)
        return ScanProfile(pools, rows, run_count)
    master_channels = config.master_pool_channels
    per_request = [_master_diff_lengths(request, plan, layer, config.master_shadow)
                   for request in requests]
    diff_rows = int(round(sum(rows for _, rows in per_request) /
                          max(len(per_request), 1)))
    # ``skip`` drops every overwritten row from the master stream, so the
    # master pool covers the input minus those rows; ``read-mask`` still reads
    # them and covers the whole input.
    master_rows = (input_rows - diff_rows if config.master_shadow == "skip"
                   else input_rows)
    master_lengths = _average_run_lengths([lengths for lengths, _ in per_request],
                                          max(master_rows, 1))
    if history_rows > 0:
        master_lengths = [history_rows] + list(master_lengths)
    if tail_rows > 0:
        master_lengths = list(master_lengths) + [tail_rows]
    pools = [_runs_from_lengths(master_lengths, stride, 0, master_channels)]
    if diff_rows > 0:
        pools.append(_runs_from_lengths([diff_rows], stride, master_channels,
                                        _TOTAL_CHANNELS - master_channels))
    rows = sum(master_lengths) + max(diff_rows, 0)
    return ScanProfile(tuple(pools), rows, sum(len(pool) for pool in pools))


def _average_run_lengths(per_request: Sequence[Sequence[int]],
                         total_rows: int) -> List[int]:
    """Mean run structure of a batch, rescaled to the padded input length.

    Members of a padded batch have different segment compositions and
    different recompute-row counts, but one legacy trace can carry only one
    run-length profile.  The batch is therefore represented by the mean number
    of runs, with the padded rows distributed in the members' mean proportion.
    """
    populated = [list(runs) for runs in per_request if runs]
    if not populated:
        return [total_rows]
    count = max(1, int(round(sum(len(runs) for runs in populated) / len(populated))))
    weights = [0.0] * count
    for runs in populated:
        total = float(sum(runs)) or 1.0
        scale = count / len(runs)
        for index, length in enumerate(runs):
            weights[min(count - 1, int(index * scale))] += length / total
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return [total_rows]
    lengths = [max(1, int(round(total_rows * value / weight_sum))) for value in weights]
    # Rounding must not change how many rows the scan reads.
    drift = total_rows - sum(lengths)
    index = 0
    while drift and index < len(lengths) * 8:
        slot = index % len(lengths)
        step = 1 if drift > 0 else -1
        if lengths[slot] + step >= 1:
            lengths[slot] += step
            drift -= step
        index += 1
    return lengths


# ---------------------------------------------------------------------------
# Layer pricing
# ---------------------------------------------------------------------------


def _scaled(layer: Layer, factor: float) -> Layer:
    op = copy.deepcopy(layer)
    op.m = max(1, int(round(layer.m * factor)))
    return op


def _link_layer(template: Layer, name: str, rows: int, width: int) -> Layer:
    op = copy.deepcopy(template)
    op.name = name
    op.m, op.n, op.k, op.numOp = max(rows, 0), width, 1, 1
    return op


def _pim_scan(system, template: Layer, *, rows: int, queries: int, heads: int,
              runs: Sequence[Tuple[int, int, int, int, int]],
              query_batch: int, batch_command: str = "replicate",
              pe_freq_ghz: float = MQ_DEFAULT_PE_FREQ_GHZ,
              gemv_buffer_bytes: int = MQ_DEFAULT_GEMV_BUFFER_BYTES
              ) -> Tuple[float, List[float]]:
    """Price one PIM attention scan with the original AttAcc Ramulator path.

    ``runs`` are the contiguous K/V extents of one channel pool; they are
    streamed one after another inside that pool.  ``queries`` queries share
    the stream ``query_batch`` at a time, so the pool is re-streamed
    ``ceil(queries / query_batch)`` times -- ``queries = 1`` is the decode
    case and reproduces AttAcc's original measurement exactly.
    """
    if not runs:
        return 0.0, [0.0] * 6
    op = copy.deepcopy(template)
    op.m, op.n, op.k, op.numOp = 1, sum(run[2] for run in runs), template.k, heads
    op.pim_kv_runs = tuple(runs)
    # The resident-Q count of one sweep is bounded by the per-bank GEMV
    # buffer whichever batch command is used (both schemes keep every Q of
    # the sweep loaded); beyond it the sweep splits into more passes.
    shared_queries = min(query_batch, max(queries, 1),
                         mq_query_capacity(gemv_buffer_bytes))
    # ``--shared-kv`` puts every head group of a trace on one K/V partition.
    # Only a multi-query scan needs it; a decode scan keeps AttAcc's original
    # per-head-group placement.
    op.pim_shared_kv = shared_queries > 1
    op.pim_shared_queries = shared_queries
    op.pim_batch_command = batch_command
    op.pim_pe_freq_ghz = pe_freq_ghz
    measured = system.devices["Acc"].get_time_and_energy_runs(op)
    passes = math.ceil(max(queries, 1) / op.pim_shared_queries)
    time_s = sum(item[0] for item in measured) * passes
    energy = [sum(values) * passes for values in
              zip(*[item[1] for item in measured])]
    return time_s, list(energy)


def _accumulate(target: Dict[str, float], key: str, value: float) -> None:
    target[key] = target.get(key, 0.0) + value


# ---------------------------------------------------------------------------
# Prefill and decode
# ---------------------------------------------------------------------------


def _prefill_batch(system, plan: ReusePlan, config: AblationConfig,
                   requests: Sequence[Request], lin: int, ndec: int,
                   *, scale: float, reused_rows: int, effective_rows: int,
                   history_rows: int = 0
                   ) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    """Time and energy of one padded prefill batch, summed over all layers.

    ``history_rows`` is the per-request KV already resident from the agent's
    own earlier turns (padded to the batch maximum).  Those rows are never
    recomputed; prefill only attends over them, so they widen the attention
    context (the score/softmax/context templates arrive here already widened
    by the caller), extend the PIM scan, and -- under a GPU-prefill placement
    with the KV cache in AttAcc memory -- travel back over the link.
    """
    batch = len(requests)
    heads = max(1, system.model.num_heads // system.model.tp)
    dbyte = system.model.sum_decoder[0].dbyte
    local_hidden = system.model.hdim // system.model.tp
    templates = {layer.name: layer for layer in system.model.sum_decoder}
    x2g = templates.get("comm_x2g")
    breakdown: Dict[str, float] = {}
    energy_breakdown: Dict[str, float] = {}
    # Attention under a PIM/split placement is priced per layer class and is
    # already summed over layers; every other term is one layer x ndec.
    attn_breakdown: Dict[str, float] = {}
    attn_energy_breakdown: Dict[str, float] = {}
    attn_time = 0.0
    attn_energy = 0.0
    padded_rows = batch * lin
    time_s = 0.0
    energy_nj = 0.0
    io_busy = 0.0

    for layer in system.model.sum_decoder:
        attention = layer.name in ("score", "softmax", "context")
        if attention and config.prefill_attn != "gpu":
            continue  # priced below on the configured attention device
        op = _scaled(layer, scale) if layer.type != LayerType.MATMUL else None
        if attention:
            # score/softmax/context carry the token count in m (per head), so
            # the reuse saving scales them exactly like a weight layer.
            op = copy.deepcopy(layer)
            op.m = max(1, int(round(layer.m * scale)))
        elif op is None:
            op = _scaled(layer, scale)
        exec_time, energy = system.devices["GPU"].get_time_and_energy(op)
        if layer.type == LayerType.X2G:
            exec_time += max(io_busy - time_s, 0)
            io_busy = time_s + exec_time
        time_s += exec_time
        energy_nj += sum(energy) / 1000.0
        _accumulate(breakdown, "gpu_" + layer.name, exec_time)
        _accumulate(energy_breakdown, "gpu_" + layer.name, sum(energy) / 1000.0)

    if config.prefill_attn == "gpu":
        readback_rows = reused_rows + len(requests) * history_rows
        if config.prefill_kv_readback and readback_rows and config.decode_attn == "pim":
            # The KV cache lives in AttAcc memory, so software prefill has to
            # pull every reused K/V row -- and the agents' resident history
            # rows -- back over the link before the GPU can attend to them.
            op = _link_layer(x2g, "kv_pim_to_gpu", readback_rows, 2 * local_hidden)
            exec_time, energy = system.devices["GPU"].get_time_and_energy(op)
            time_s += exec_time
            energy_nj += sum(energy) / 1000.0
            _accumulate(breakdown, "link_kv_pim_to_gpu", exec_time)
            _accumulate(energy_breakdown, "link_kv_pim_to_gpu", sum(energy) / 1000.0)
    else:
        # Layers differ in how much reuse they carry, and a CacheBlend
        # full-recompute layer carries none at all: it rebuilds the whole
        # request K/V, so its attention is an ordinary GPU prefill with no PIM
        # scan and no Q/context link -- exactly what the reference trace shows
        # for its L0/L1 ("in GPU with all-new KV, old KV unused").  Everything
        # below is therefore accumulated per layer class and already totalled
        # over layers, unlike the uniform terms above.
        score, softmax = templates["score"], templates["softmax"]
        stride = ((system.model.dhead * dbyte + 31) // 32) * 32
        classes = []  # (layer count, recomputed rows in the batch, carries reuse)
        full_layers = (len(plan.config.cacheblend_full_recompute_layers)
                       if plan.config.policy in CACHEBLEND_FAMILY else 0)
        full_layers = min(full_layers, ndec)
        if full_layers:
            classes.append((full_layers, padded_rows, False))
            reuse_layers = ndec - full_layers
            if reuse_layers:
                # Total recomputed rows are conserved: the uniform ``scale``
                # already spreads the saving over every layer, so the partial
                # layers absorb the whole saving here.
                partial_rows = ((effective_rows * ndec - padded_rows * full_layers) /
                                reuse_layers)
                classes.append((reuse_layers, max(partial_rows, 1.0), True))
        else:
            classes.append((ndec, effective_rows, True))

        for count, rows_batch, carries_reuse in classes:
            queries = max(1, int(round(rows_batch / max(batch, 1))))
            if not carries_reuse:
                # Full recompute rebuilds every shared row, but the agent's
                # own history KV stays valid in every layer: the GPU attends
                # the rebuilt rows to each other while the PIM scans the
                # resident history (none at H=0: an ordinary GPU block).
                scan_rows, gpu_rows = history_rows, queries
                gpu_prefix = "gpu_full_"
            else:
                # "pim" and "dynamic" both consider the whole landed context
                # on the PIM (no GPU triangle -- the former "split" hybrid is
                # abolished); "dynamic" may instead commit this class to the
                # GPU below (the Fugue placement rule).
                scan_rows, gpu_rows = lin + history_rows, 0
                gpu_prefix = "gpu_local_"
            if scan_rows <= 0:
                # Nothing resident to scan: the layer degenerates to an
                # ordinary GPU prefill, with no PIM scan and no link traffic.
                for name in ("score", "softmax", "context"):
                    op = copy.deepcopy(templates[name])
                    op.m = max(1, gpu_rows)
                    if name in ("score", "softmax"):
                        op.n = max(1, gpu_rows)
                    else:
                        op.k = max(1, gpu_rows)
                    exec_time, energy = system.devices["GPU"].get_time_and_energy(op)
                    attn_time += exec_time * count
                    attn_energy += sum(energy) / 1000.0 * count
                    _accumulate(attn_breakdown, gpu_prefix + name, exec_time * count)
                    _accumulate(attn_energy_breakdown, gpu_prefix + name,
                                sum(energy) / 1000.0 * count)
                continue

            if carries_reuse and config.prefill_attn == "dynamic":
                # PAPER TIE (Question 1, Fugue rung): a fixed side is
                # provably wrong in both directions -- a mostly-fresh
                # request wastes the readback either way while a
                # mostly-reused one wastes GPU attention over rows the
                # banks already hold -- so the paper's method decides PER
                # REQUEST CLASS.  Fugue placement rule (paper
                # Eq.(placement)), evaluated with the simulator's own cost
                # model instead of the closed form:
                # price the bank path (q link + PIM scan + softmax + ctx
                # link) and the xPU path (read the resident rows back over
                # the link, then an ordinary GPU attention block over the
                # full context) for this layer class, and commit the cheaper
                # one; ties go to the bank.  Decode always stays on the PIM
                # (decode_attn); a full-recompute class never reaches here
                # (cold rows take the GPU-plus-history arm above).
                # TODO(review): swap the decision input to the paper's
                # closed-form t_bank/t_xPU if runtime-policy fidelity is
                # preferred over the oracle model costs.
                est_scan, _ = _pim_scan(
                    system, score, rows=scan_rows, queries=queries,
                    heads=heads * batch, runs=_private_runs(scan_rows, stride),
                    query_batch=config.pim_prefill_query_batch,
                    batch_command=config.pim_batch_command,
                    pe_freq_ghz=config.pim_pe_freq_ghz,
                    gemv_buffer_bytes=config.gemv_buffer_bytes)
                sfm_est = copy.deepcopy(softmax)
                sfm_est.m, sfm_est.n, sfm_est.numOp = queries, scan_rows, heads * batch
                t_bank = est_scan + system.devices["Acc"].get_time_and_energy(sfm_est)[0]
                for name in ("q_gpu_to_pim", "ctx_pim_to_gpu"):
                    op = _link_layer(x2g, name, int(round(rows_batch)), local_hidden)
                    t_bank += system.devices["GPU"].get_time_and_energy(op)[0]
                readback_rows = reused_rows + len(requests) * history_rows
                xpu_pieces = []
                op = _link_layer(x2g, "kv_pim_to_gpu", readback_rows, 2 * local_hidden)
                exec_time, energy = system.devices["GPU"].get_time_and_energy(op)
                xpu_pieces.append(("link_kv_pim_to_gpu", exec_time, energy))
                # Price the class's GPU share EXACTLY the way the "gpu" arm
                # charges it -- the top-loop scale folding (op.m scaled by
                # the class's recompute fraction, n/k/numOp exactly as the
                # model built them).  Audit issue 1 (2026-08-25): the
                # previous per-request op shapes overpriced the xPU path,
                # so the rule flipped classes to the PIM that A4 beat.
                class_scale = (rows_batch / padded_rows) if padded_rows else 1.0
                for name in ("score", "softmax", "context"):
                    op = copy.deepcopy(templates[name])
                    op.m = max(1, int(round(op.m * class_scale)))
                    exec_time, energy = system.devices["GPU"].get_time_and_energy(op)
                    xpu_pieces.append(("gpu_dynamic_" + name, exec_time, energy))
                t_xpu = sum(item[1] for item in xpu_pieces)
                if t_bank > t_xpu:
                    # The class goes to the GPU: readback, then a serial
                    # local attention block; no PIM scan, no q/ctx link.
                    for name, exec_time, energy in xpu_pieces:
                        attn_time += exec_time * count
                        attn_energy += sum(energy) / 1000.0 * count
                        _accumulate(attn_breakdown, name, exec_time * count)
                        _accumulate(attn_energy_breakdown, name,
                                    sum(energy) / 1000.0 * count)
                    continue
                # Otherwise fall through: the class is priced exactly like
                # "pim" below (the estimates above hit the same caches).

            # The two branches of one attention layer.  ``pim_branch`` cannot
            # start before q~ has crossed the link; ``gpu_branch`` needs no
            # PIM data and starts as soon as QKV is done, so the two run
            # concurrently and the layer costs the slower one -- this is what
            # KVpim-sim's trace shows for B-prefill L2, where the GPU ``fresh
            # score`` events overlap the b0/b1 scans and the DIE merges the two
            # partial (m, l, o) triples afterwards.  Energy is additive either
            # way; only the critical path changes.
            pim_branch = 0.0
            gpu_branch = 0.0

            pim_time, pim_energy = _pim_scan(
                system, score, rows=scan_rows, queries=queries,
                heads=heads * batch, runs=_private_runs(scan_rows, stride),
                query_batch=config.pim_prefill_query_batch,
                batch_command=config.pim_batch_command,
                pe_freq_ghz=config.pim_pe_freq_ghz,
                gemv_buffer_bytes=config.gemv_buffer_bytes)
            pim_branch += pim_time
            attn_energy += sum(pim_energy) / 1000.0 * count
            _accumulate(attn_breakdown, "pim_prefill_score", pim_time * count)
            _accumulate(attn_energy_breakdown, "pim_prefill_score",
                        sum(pim_energy) / 1000.0 * count)

            sfm = copy.deepcopy(softmax)
            sfm.m, sfm.n, sfm.numOp = queries, scan_rows, heads * batch
            exec_time, energy = system.devices["Acc"].get_time_and_energy(sfm)
            pim_branch += exec_time
            attn_energy += sum(energy) / 1000.0 * count
            _accumulate(attn_breakdown, "pim_prefill_softmax", exec_time * count)
            _accumulate(attn_energy_breakdown, "pim_prefill_softmax",
                        sum(energy) / 1000.0 * count)

            if gpu_rows:
                for name in ("score", "softmax", "context"):
                    op = copy.deepcopy(templates[name])
                    op.m = max(1, gpu_rows)
                    if name in ("score", "softmax"):
                        op.n = max(1, gpu_rows)
                    else:
                        op.k = max(1, gpu_rows)
                    exec_time, energy = system.devices["GPU"].get_time_and_energy(op)
                    gpu_branch += exec_time
                    attn_energy += sum(energy) / 1000.0 * count
                    _accumulate(attn_breakdown, gpu_prefix + name, exec_time * count)
                    _accumulate(attn_energy_breakdown, gpu_prefix + name,
                                sum(energy) / 1000.0 * count)

            # Q must reach the PIM before it can scan; the merged context comes
            # back afterwards and is on the critical path of both branches.
            link_time = {}
            for name, width in (("q_gpu_to_pim", local_hidden),
                                ("ctx_pim_to_gpu", local_hidden)):
                op = _link_layer(x2g, name, int(round(rows_batch)), width)
                exec_time, energy = system.devices["GPU"].get_time_and_energy(op)
                link_time[name] = exec_time
                attn_energy += sum(energy) / 1000.0 * count
                _accumulate(attn_breakdown, "link_" + name, exec_time * count)
                _accumulate(attn_energy_breakdown, "link_" + name,
                            sum(energy) / 1000.0 * count)

            pim_branch += link_time["q_gpu_to_pim"]
            # The only class with both branches is full recompute (GPU
            # rebuilds the rows while the PIM scans the resident history);
            # the branches have no data dependency, so the layer costs the
            # slower one.
            serial = pim_branch + gpu_branch
            critical = max(pim_branch, gpu_branch)
            attn_time += (critical + link_time["ctx_pim_to_gpu"]) * count
            if critical != serial:
                # Keep the breakdown summing to attn_time: the components above
                # are the per-device costs, this is what the overlap removes.
                _accumulate(attn_breakdown, "prefill_overlap_saving",
                            (critical - serial) * count)

    totals = {key: value * ndec for key, value in breakdown.items()}
    energy_totals = {key: value * ndec for key, value in energy_breakdown.items()}
    for key, value in attn_breakdown.items():
        _accumulate(totals, key, value)
    for key, value in attn_energy_breakdown.items():
        _accumulate(energy_totals, key, value)
    return (time_s * ndec + attn_time, energy_nj * ndec + attn_energy,
            totals, energy_totals)


def _decode_block_time(system, config: AblationConfig, block: Sequence[Layer],
                       profile: Optional[ScanProfile], *, heads: int, batch: int,
                       pipe: bool, parallel_ff: bool
                       ) -> Tuple[float, float, Dict[str, float]]:
    """Price one decoder block of one decode step on the configured devices."""
    breakdown: Dict[str, float] = {}
    energy_nj = 0.0
    for layer in block:
        if config.decode_attn == "pim" and layer.type in (
                LayerType.MATMUL, LayerType.SOFTMAX, LayerType.X2G):
            if layer.name == "score" and profile is not None and profile.legacy_shape:
                # One full-length extent over all sixteen channels is exactly
                # the original AttAcc measurement; keep its shape-indexed
                # Ramulator cache instead of re-running an identical trace.
                exec_time, energy = system.devices["Acc"].get_time_and_energy(layer)
                energy_nj += sum(energy) / 1000.0
            elif layer.name == "score":
                pool_times, pool_energy = [], 0.0
                for pool in profile.pools:
                    time_s, energy = _pim_scan(
                        system, layer, rows=sum(run[2] for run in pool), queries=1,
                        heads=heads * batch, runs=pool, query_batch=1,
                        batch_command=config.pim_batch_command,
                        pe_freq_ghz=config.pim_pe_freq_ghz,
                        gemv_buffer_bytes=config.gemv_buffer_bytes)
                    pool_times.append(time_s)
                    pool_energy += sum(energy) / 1000.0
                # Disjoint channel pools stream concurrently; extents inside a
                # pool are serial.
                exec_time = max(pool_times) if pool_times else 0.0
                energy_nj += pool_energy
            else:
                exec_time, energy = system.devices["Acc"].get_time_and_energy(layer)
                energy_nj += sum(energy) / 1000.0
        else:
            exec_time, energy = system.devices["GPU"].get_time_and_energy(layer)
            energy_nj += sum(energy) / 1000.0
        layer.exec_time = exec_time
        _accumulate(breakdown, layer.name, exec_time)
    if config.decode_attn == "pim":
        apply_attacc_pipeline(block, system.model.num_heads,
                              system.devices["GPU"].num_xpu, pipe)
        if parallel_ff:
            _ff_parallel(system, block, batch)
    return sum(layer.exec_time for layer in block), energy_nj, breakdown


def _ff_parallel(system, layers: Sequence[Layer], batch: int) -> None:
    """AttAcc's ``--ffopt``: feedforward overlaps the PIM attention."""
    bw_scale = (system.devices["Acc"].peak_memory_bandwidth /
                system.devices["GPU"].peak_memory_bandwidth)
    for layer in layers:
        if "ff" not in layer.name:
            continue
        if layer.bound == "compute":
            attn_flops = (system.devices["GPU"].peak_memory_bandwidth /
                          layer.dbyte * 2 * bw_scale)
            ratio = system.devices["GPU"].peak_flops / (
                system.devices["GPU"].peak_flops + attn_flops)
            layer.exec_time *= ratio
        elif layer.bound == "memory":
            attn_eff_bw = (system.devices["GPU"].peak_memory_bandwidth *
                           bw_scale / batch)
            ratio = system.devices["GPU"].peak_memory_bandwidth / (
                system.devices["GPU"].peak_memory_bandwidth + attn_eff_bw)
            layer.exec_time *= ratio


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------


def _memory_report(system, workload: Workload, plan: ReusePlan,
                   config: AblationConfig,
                   batch_size: Optional[int] = None) -> Dict[str, Any]:
    """AttAcc/GPU KV occupancy of the whole workload under this mapping.

    Reused chunks are stored once and shared by every consumer; recomputed
    rows add one row each.  ``private`` stores a full private copy per
    request, which is what the no-reuse baseline needs.
    """
    ndec = system.model.ndec
    dbyte = 2 if system.model.dtype.name.startswith("W16") else 1
    bytes_per_row = 2 * system.model.hdim * dbyte * ndec
    reused_by_request = _reused_tokens_by_request(plan)
    generated = sum(request.lout for request in workload.requests)
    total_tokens = sum(request.total_length for request in workload.requests)
    # Each agent's own earlier-turn KV is private and already resident; it is
    # stored once per agent whatever the reuse policy.
    history_rows = sum(request.history_len for request in workload.requests)

    # Sharing is a property of the reuse policy, not of the PIM mapping: a
    # pure-GPU CacheBlend/EPIC deployment also stores each reused chunk once,
    # it just stores it in GPU memory.
    if plan.config.policy == "no-reuse":
        stored_rows = total_tokens + generated + history_rows
        shared_rows = 0
        diff_rows = 0
    else:
        # One physical copy per distinct reusable chunk.
        distinct: Dict[str, int] = {}
        for decision in plan.reusable:
            distinct[decision.fingerprint] = decision.length
        shared_rows = sum(distinct.values())
        private_rows = total_tokens - sum(reused_by_request.values())
        # A CacheBlend full-recompute layer rebuilds the whole request K/V, so
        # in those layers nothing is shared and no row is a diff.
        full_layers = (len(plan.config.cacheblend_full_recompute_layers)
                       if plan.config.policy in CACHEBLEND_FAMILY else 0)
        reuse_layers = max(ndec - full_layers, 0)
        diff_rows = 0
        for layer in range(ndec):
            for request in workload.requests:
                diff_rows += sum(len(set(rows)) for rows in
                                 _corrected_rows(plan, layer, request.request_id).values())
        diff_rows = diff_rows / max(ndec, 1)
        # ``private_rows`` (total minus consumer-side reused tokens) already
        # CONTAINS the owner's single copy of every shared chunk -- the
        # owner has no reuse decision.  Adding ``shared_rows`` on top
        # double-counted those owner copies and understated the capacity
        # saving (audit 2026-08-25: multihop showed 4.7% saved instead of
        # the true 20.3%).  ``shared_rows`` stays as a reporting field.
        reuse_layer_rows = private_rows + diff_rows * ndec / max(reuse_layers, 1)
        stored_rows = ((reuse_layers * reuse_layer_rows +
                        full_layers * total_tokens) / max(ndec, 1) + generated +
                       history_rows)
        shared_rows = shared_rows * reuse_layers / max(ndec, 1)

    kv_bytes = stored_rows * bytes_per_row
    baseline_bytes = (total_tokens + generated + history_rows) * bytes_per_row
    report: Dict[str, Any] = {
        "kv_bytes": kv_bytes,
        "kv_gib": kv_bytes / (1 << 30),
        "kv_rows": stored_rows,
        "shared_master_rows": shared_rows,
        "diff_rows_per_layer": diff_rows,
        "full_recompute_layers": (len(plan.config.cacheblend_full_recompute_layers)
                                  if plan.config.policy in CACHEBLEND_FAMILY else 0),
        "generated_rows": generated,
        "history_rows": history_rows,
        "no_reuse_kv_bytes": baseline_bytes,
        "kv_bytes_vs_no_reuse": kv_bytes / baseline_bytes if baseline_bytes else None,
        # Marks reports produced after the owner-copy double-count fix
        # (2026-08-25) so repair_memory_column.py leaves them alone.
        "owner_copy_fix": "native",
        "bytes_per_token_all_layers": bytes_per_row,
        "resident_in": "GPU HBM" if config.decode_attn == "gpu" else "AttAcc HBM",
    }
    if config.decode_attn == "pim":
        capacity = system.devices["Acc"].aggregate_memory_capacity
        report["attacc_capacity_bytes"] = capacity
        report["attacc_capacity_used"] = kv_bytes / capacity if capacity else None
        if config.kv_mapping == "master-diff":
            report["master_pool_channels"] = config.master_pool_channels
            report["diff_pool_channels"] = _TOTAL_CHANNELS - config.master_pool_channels
    weights, _, _ = system.get_required_mem_capacity(1, 1, 2)
    report["gpu_weight_bytes"] = weights
    gpu_capacity = system.devices["GPU"].aggregate_memory_capacity
    report["gpu_capacity_bytes"] = gpu_capacity
    # With decode on the PIM the KV cache is resident in AttAcc HBM and the
    # GPU's HBM holds the weights plus, transiently, the K/V of the prefill
    # layer it is currently computing for the largest padded batch (fresh
    # rows it produced and reused rows it read back over the link).
    largest = max((len(requests) * lin for _, requests, lin, _ in
                   _tier_shapes(workload, batch_size)), default=0)
    hidden = system.model.hdim
    dbyte = system.model.sum_decoder[0].dbyte if system.model.sum_decoder else 2
    temp_kv = largest * 2 * hidden * dbyte  # one layer, all tensor-parallel shards
    report["gpu_temp_kv_bytes_per_layer"] = temp_kv
    resident_kv = kv_bytes if config.decode_attn == "gpu" else temp_kv
    report["gpu_capacity_used"] = (
        (weights + resident_kv) / gpu_capacity if gpu_capacity else None)
    return report


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_ablation_report(system, workload: Workload, plan: ReusePlan,
                        config: AblationConfig, *, pipe: bool, parallel_ff: bool,
                        power_constraint: bool,
                        batch_size: Optional[int] = None) -> Dict[str, Any]:
    """Run one ablation configuration over every dependency tier."""
    if config.decode_attn == "pim" and system.hetero_name != DeviceType.PIM:
        raise WorkloadValidationError("--decode-attn pim requires --system dgx-attacc")
    if config.prefill_attn != "gpu" and system.hetero_name != DeviceType.PIM:
        raise WorkloadValidationError(
            "--prefill-attn {} requires --system dgx-attacc".format(config.prefill_attn))
    ndec = system.model.ndec
    reused_by_request = _reused_tokens_by_request(plan)
    epic_prefix = _epic_prefix_by_request(plan)
    policy = plan.config.policy
    if policy in CACHEBLEND_FAMILY:
        recompute_fraction = (
            len(plan.config.cacheblend_full_recompute_layers) +
            plan.config.cacheblend_recompute_ratio *
            len(plan.config.cacheblend_partial_recompute_layers)) / ndec
    else:
        recompute_fraction = None

    tiers: List[Dict[str, Any]] = []
    makespan_s = 0.0
    prefill_energy_nj = 0.0
    decode_energy_nj = 0.0
    prefill_s_total = 0.0
    decode_s_total = 0.0

    for tier, requests, lin, lout in _tier_shapes(workload, batch_size):
        if lout < 2:
            raise WorkloadValidationError(
                "end-to-end reporting requires lout >= 2")
        batch = len(requests)
        attn_on_hetero = (config.decode_attn == "pim")
        system.model.build(batch, lin, lout, attn_on_hetero)
        # Agentic multi-turn: KV rows the agents already hold from their own
        # earlier turns, padded to the batch maximum like ``lin``.  They are
        # attended, never recomputed, so they widen every attention context
        # (score/softmax read ``n`` more columns, context contracts over ``k``
        # more rows) without touching the projection/FF row counts.
        hist_rows = max((request.history_len for request in requests), default=0)
        if hist_rows:
            for block in [system.model.sum_decoder] + system.model.gen_decoder:
                for layer in block:
                    if layer.name in ("score", "softmax"):
                        layer.n += hist_rows
                    elif layer.name == "context":
                        layer.k += hist_rows
        heads = max(1, system.model.num_heads // system.model.tp)
        dbyte = system.model.sum_decoder[0].dbyte
        stride = ((system.model.dhead * dbyte + 31) // 32) * 32

        padded_rows = batch * lin
        reused_rows = sum(reused_by_request.get(request.request_id, 0)
                          for request in requests)
        if policy in CACHEBLEND_FAMILY:
            saved_rows = reused_rows * (1.0 - recompute_fraction)
        elif policy in EPIC_FAMILY:
            saved_rows = reused_rows - sum(epic_prefix.get(request.request_id, 0)
                                           for request in requests)
        else:
            saved_rows = 0.0
        effective_rows = padded_rows - saved_rows
        scale = effective_rows / padded_rows if padded_rows else 1.0

        prefill_s, prefill_energy, prefill_breakdown, prefill_energy_breakdown = (
            _prefill_batch(system, plan, config, requests, lin, ndec, scale=scale,
                           reused_rows=int(reused_rows),
                           effective_rows=int(round(effective_rows)),
                           history_rows=hist_rows))

        decode_s = 0.0
        decode_energy = 0.0
        decode_breakdown: Dict[str, float] = {}
        scan_profiles: Dict[str, Any] = {}
        classes = _layer_classes(plan, ndec, requests)
        # Legacy-shape decode pricing issues ONE Ramulator run per decode
        # step, serially -- on a long-output workload that is thousands of
        # multi-minute simulations in a row.  The steps are independent, so
        # warm the wrapper's shape cache in PARALLEL first (ruling
        # 2026-08-25: one task may use up to 32 cores); the serial loop
        # below then serves every step from the in-memory cache.  Only the
        # private (legacy-shape) mapping takes this path -- the reuse
        # mappings go through the address-resolved signature cache instead.
        if config.decode_attn == "pim" and config.kv_mapping == "private":
            accelerator = system.devices["Acc"]
            if hasattr(accelerator, "ramulator"):
                warm_layers = [copy.deepcopy(layer)
                               for block in system.model.gen_decoder
                               for layer in block
                               if layer.name == "score" and
                               layer.type == LayerType.MATMUL]
                if len(warm_layers) > 8:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=32) as warm_pool:
                        list(warm_pool.map(accelerator.get_time_and_energy,
                                           warm_layers))
        for step_index, block in enumerate(system.model.gen_decoder):
            context_rows = lin + step_index + 1
            for representative, count in classes:
                profile = None
                if config.decode_attn == "pim":
                    profile = _batch_scan_profile(requests, plan, representative,
                                                  lin, context_rows - lin, stride,
                                                  config, history_rows=hist_rows)
                    if step_index == 0:
                        scan_profiles["layer_{}".format(representative)] = dict(
                            profile.to_dict(), layers=count)
                block_copy = copy.deepcopy(block)
                block_time, block_energy, block_breakdown = _decode_block_time(
                    system, config, block_copy, profile, heads=heads, batch=batch,
                    pipe=pipe, parallel_ff=parallel_ff)
                decode_s += block_time * count
                decode_energy += block_energy * count
                for key, value in block_breakdown.items():
                    _accumulate(decode_breakdown, key, value * count)

        duration_s = prefill_s + decode_s
        makespan_s += duration_s
        prefill_s_total += prefill_s
        decode_s_total += decode_s
        prefill_energy_nj += prefill_energy
        decode_energy_nj += decode_energy
        tiers.append({
            "tier": tier,
            "batch_size": batch,
            "members": [request.request_id for request in requests],
            "lin": lin,
            "lout": lout,
            "decode_steps": lout - 1,
            "padded_prefill_tokens": padded_rows,
            "history_rows_per_request": hist_rows,
            "reused_prefill_tokens": reused_rows,
            "effective_prefill_tokens": effective_rows,
            "prefill_scale": scale,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "decode_per_token_s": decode_s / (lout - 1),
            "duration_s": duration_s,
            "prefill_energy_nj": prefill_energy,
            "decode_energy_nj": decode_energy,
            "energy_nj": prefill_energy + decode_energy,
            "prefill_breakdown_s": prefill_breakdown,
            "prefill_energy_breakdown_nj": prefill_energy_breakdown,
            "decode_breakdown_s": decode_breakdown,
            "decode_scan_profile": scan_profiles,
        })

    report = {
        "policy": policy,
        "latency_model": "legacy-attacc-ablation",
        "gpu_model": getattr(system.devices["GPU"], "gpu_model", "legacy"),
        "ablation": config.to_dict(),
        "scheduling": ("legacy rectangular batches of <= {} requests per tier; "
                       "batches and tiers serial".format(batch_size) if batch_size
                       else "one legacy rectangular batch per dependency tier; tiers serial"),
        "batch_size": batch_size,
        "recompute_fraction_per_reused_row": recompute_fraction,
        "tiers": tiers,
        "makespan_s": makespan_s,
        "prefill_s": prefill_s_total,
        "decode_s": decode_s_total,
        "prefill_energy_nj": prefill_energy_nj,
        "decode_energy_nj": decode_energy_nj,
        "energy_nj": prefill_energy_nj + decode_energy_nj,
        "energy_unit": "nJ",
        "memory": _memory_report(system, workload, plan, config,
                                 batch_size=batch_size),
    }
    ramulator = getattr(system.devices.get("Acc"), "ramulator", None)
    if ramulator is not None and hasattr(ramulator, "cache_report"):
        report["ramulator_signature_cache"] = ramulator.cache_report()
    return report
