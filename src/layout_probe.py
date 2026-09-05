"""Layout probe: dump the real addresses a scan touches, so a human can
hand-check the placement (chenyi9 order 2026-09-04).

Off unless ``KVPIM_LAYOUT_DUMP`` names a file.  It costs nothing when off --
``enabled()`` is a module-level flag read once at import.

What it records, one JSON object per line:

``kind="blocks"``
    The KV store's block table after ``finalize()``: one row per CACHED
    CHUNK -- ``(layer, owner, fingerprint, kind)`` -> its key/value base
    address, its channel set and its token rows.  This is the answer to
    "where did this chunk actually land".

``kind="scan"``
    One PIM scan.  Carries the placement inputs (policy, heads_per_hbm,
    master_channels), the per-channel extents HANDED TO RAMULATOR
    (``extents``: the (key, value, tokens) triples whose row-alignment
    decides the activations), and per channel the measured time and energy.

    The last two fields are the reduction itself, spelled out, because that
    is what separates the rungs: ``scan_time_s`` is the MAX over channels
    (the lanes run concurrently -- fix 75da860) and ``scan_energy_nj`` is
    the SUM over channels times ``num_hbm_used`` (every stack holding KV
    pays its own copy).  ``per_channel`` lists every term of both.

Environment
-----------
``KVPIM_LAYOUT_DUMP``        output path; unset disables the probe
``KVPIM_LAYOUT_DUMP_LAYER``  only this decoder layer (default 0); ``all``
                             keeps every layer and gets very large
``KVPIM_LAYOUT_DUMP_MAX``    stop after this many scan records (default 400)
``KVPIM_LAYOUT_DUMP_TAG``    free-form label written on every record, used
                             to say which rung produced the file
"""

import json
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

# One DRAM row is 1024 B and a token costs 4 B of address space (the MAC_AB
# all-bank broadcast), so a row holds 256 tokens and 32 B is one column.
_ROW_BYTES = 1024
_BYTES_PER_TOKEN = 4
_COL_BYTES = 32
_CHANNEL_BYTES = 1 << 30

_PATH = os.environ.get("KVPIM_LAYOUT_DUMP", "").strip()
_TAG = os.environ.get("KVPIM_LAYOUT_DUMP_TAG", "").strip()
_LAYER_ENV = os.environ.get("KVPIM_LAYOUT_DUMP_LAYER", "0").strip()
_LAYER: Optional[int] = None if _LAYER_ENV == "all" else int(_LAYER_ENV or 0)
_MAX = int(os.environ.get("KVPIM_LAYOUT_DUMP_MAX", "400"))

_lock = threading.Lock()
_scans = 0
_handle = None


def enabled() -> bool:
    return bool(_PATH)


def _write(record: Dict[str, Any]) -> None:
    global _handle
    with _lock:
        if _handle is None:
            _handle = open(_PATH, "a", encoding="utf-8")
        if _TAG:
            record["tag"] = _TAG
        _handle.write(json.dumps(record, sort_keys=True) + "\n")
        _handle.flush()


def decode(address: int) -> Dict[str, int]:
    """Split a byte address into the coordinates a hand-check needs.

    ``channel`` is the 1-GiB region index used by ``_channel_extent_addresses``;
    ``row`` and ``col`` locate the address inside that channel at the 1024-B /
    32-B granularity the trace generator writes.
    """
    channel, offset = divmod(int(address), _CHANNEL_BYTES)
    row, in_row = divmod(offset, _ROW_BYTES)
    return {"channel": channel, "row": row, "col": in_row // _COL_BYTES,
            "byte_in_row": in_row}


def acts_for(tokens: int) -> int:
    """Activations an extent needs: it is row-aligned, so ceil(tokens/256)."""
    span = int(tokens) * _BYTES_PER_TOKEN
    return -(-span // _ROW_BYTES)


def record_blocks(tlb, *, note: str = "") -> None:
    """Dump the KV store's per-chunk block table (once, after finalize).

    Reads ``tlb._blocks`` rather than ``report()``: every layout class fills
    it with ``KVBlock`` objects, but only ``CacheBlendTLB`` and the no-reuse
    layout publish them under ``blocks`` -- ``NaiveKVLayout`` reports
    ``page_count`` instead, so A3/A3b would come back empty.

    TWO ADDRESS SCALES COEXIST and this record shows both, because mixing
    them up is the easiest way to hand-check the wrong thing:

    * the TLB's own space, ``vector_stride`` bytes per token (dhead x dbyte
      rounded up to a 32-B transaction -- 256 B at dhead=128, dbyte=2).  It
      feeds ``_tlb_plan_cost`` and the JSON report.
    * the space Ramulator simulates, 4 B per token.  ``MAC_AB`` is an
      all-bank broadcast, so one address names n_pch x n_rank x n_bg = 16
      partitions and the trace generator steps
      ceil(dhead/n_bank/n_mac) = 2 columns of 32 B per 16 tokens = 4 B each
      (``gen_trace_attacc_bank.py`` score_mac).  ``_channel_extent_addresses``
      SYNTHESIZES the extent addresses at this scale; the TLB's addresses do
      NOT reach Ramulator.

    So ``key_base`` below and ``key_address`` in a scan record are not
    comparable numbers.  Compare CHANNELS and relative ORDER, not addresses.
    """
    if not enabled():
        return
    try:
        blocks = getattr(tlb, "_blocks", None) or {}
        rows = []
        for _key, block in sorted(blocks.items(), key=lambda kv: str(kv[0])):
            if _LAYER is not None and getattr(block, "layer", None) != _LAYER:
                continue
            token_rows = tuple(getattr(block, "rows", ()) or ())
            stride = int(getattr(block, "vector_stride", 0) or 0)
            key_base = int(getattr(block, "key_base", 0) or 0)
            rows.append({
                "id": getattr(block, "block_id", ""),
                "layer": getattr(block, "layer", None),
                "owner": getattr(block, "owner", ""),
                "fingerprint": getattr(block, "fingerprint", ""),
                "kind": getattr(block, "kind", ""),
                "tokens": len(token_rows),
                "row_first": token_rows[0] if token_rows else None,
                "row_last": token_rows[-1] if token_rows else None,
                "key_base": key_base,
                "value_base": int(getattr(block, "value_base", 0) or 0),
                "vector_stride_bytes": stride,
                "channel_base": getattr(block, "channel_base", None),
                "channel_count": getattr(block, "channel_count", None),
                "channel_offset": getattr(block, "channel_offset", None),
                "channel_tile": getattr(block, "channel_tile", None),
                # decoded in the TLB's OWN space (stride bytes per token)
                "tlb_channel": key_base // _CHANNEL_BYTES,
                "tlb_offset_in_channel": key_base % _CHANNEL_BYTES,
                "tlb_bytes": len(token_rows) * stride,
            })
    except Exception as exc:                      # a probe must never break a run
        _write({"kind": "blocks_error", "error": repr(exc)})
        return
    _write({"kind": "blocks", "note": note, "layer_filter": _LAYER,
            "n_blocks_total": len(blocks), "n_blocks_kept": len(rows),
            "scales": {"tlb_bytes_per_token": "block.vector_stride "
                                              "(dhead x dbyte, 32-B aligned)",
                       "ramulator_bytes_per_token": _BYTES_PER_TOKEN,
                       "row_bytes": _ROW_BYTES, "col_bytes": _COL_BYTES,
                       "tokens_per_row": _ROW_BYTES // _BYTES_PER_TOKEN},
            "blocks": rows})


def record_scan(*, layer: int, tier: int, request: str, name: str,
                policy: str, heads_per_hbm: int, master_channels: int,
                kv_heads: int, num_hbm_used: int,
                n_master: int, n_diff: int,
                loads: Sequence[float], active: Sequence[int],
                extent_groups: Sequence[Tuple[int, int, Sequence[Tuple[int, int, int]]]],
                per_channel_rows: Dict[int, int],
                per_channel_masked: Dict[int, int],
                measured: Sequence[Tuple[float, Sequence[float]]]) -> None:
    """Dump one PIM scan: its extents, and both reduction terms per channel."""
    global _scans
    if not enabled():
        return
    if _LAYER is not None and layer != _LAYER:
        return
    with _lock:
        if _scans >= _MAX:
            return
        _scans += 1
    extents = {}
    for channel, _count, placed in extent_groups or ():
        extents[str(channel)] = [
            {"key_address": int(k), "value_address": int(v), "tokens": int(rows),
             "acts": acts_for(rows), "key_decoded": decode(k)}
            for k, v, rows in placed]
    terms: List[Dict[str, Any]] = []
    for channel, (time_s, energy_vec) in zip(active, measured):
        energy_pj = float(sum(energy_vec))
        terms.append({
            "channel": int(channel),
            "rows": int(per_channel_rows.get(channel, 0)),
            "masked_rows": int(per_channel_masked.get(channel, 0)),
            "acts": sum(entry["acts"] for entry in extents.get(str(channel), [])),
            "time_s": float(time_s),
            # the event stores nJ (_cacheblend_event divides the pJ table by
            # 1000) and scales by num_hbm_used; both shown so the sum is
            # reproducible by hand
            "energy_nj_one_stack": energy_pj / 1000.0,
            "energy_nj_charged": energy_pj / 1000.0 * num_hbm_used,
        })
    scan_time = max((entry["time_s"] for entry in terms), default=0.0)
    _write({
        "kind": "scan", "layer": layer, "tier": tier, "request": request,
        "event_name": name, "policy": policy,
        "heads_per_hbm": int(heads_per_hbm),
        "master_channels": int(master_channels),
        "kv_heads": int(kv_heads), "num_hbm_used": int(num_hbm_used),
        "reads_master": int(n_master), "reads_diff": int(n_diff),
        "loads": [float(x) for x in loads],
        "active": [int(c) for c in active],
        "extents": extents,
        "per_channel": terms,
        # the reduction, spelled out
        "scan_time_s": scan_time,
        "scan_time_channel": next((entry["channel"] for entry in terms
                                   if entry["time_s"] == scan_time), None),
        "scan_energy_nj": sum(entry["energy_nj_charged"] for entry in terms),
        "scan_acts": sum(entry["acts"] for entry in terms),
        "reduction": "time = MAX over channels; energy = SUM over channels "
                     "x num_hbm_used",
    })
