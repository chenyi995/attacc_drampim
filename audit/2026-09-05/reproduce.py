"""Independent audit probes. No production code is changed; no speedups are measured.

Run: PYTHONPATH=. python3 audit/2026-09-05/reproduce.py
The device stub records operator shapes and exercises the real DAG/layout code.
"""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json

from src.ablation import PRESETS, resolve_config
from src.config import make_model_config, make_pim_config
from src.model import Transformer
from src.type import DataType, DeviceType, PIMType, InterfaceType
from src.workload import Request, Segment, Workload, build_reuse_plan, load_workload
import src.workload_runner as wr


class Device:
    peak_memory_bandwidth = 10**12
    softmax_peak_bandwidth = 10**12
    energy_table = {"mem": 1, "sram": 1}
    num_hbm = 1

    def __init__(self):
        self.calls = []

    def get_time_and_energy(self, op):
        self.calls.append((op.name, op.m, op.n, op.k, op.numOp))
        return 1e-6, [1, 0, 0, 0, 0, 0]

    def get_time_and_energy_runs(self, op):
        self.calls.append((op.name, op.m, op.n, op.k, op.numOp))
        return [(1e-6, [1, 0, 0, 0, 0, 0]) for _ in op.pim_kv_runs]


def system(gqa=1):
    return SimpleNamespace(
        hetero_name=DeviceType.PIM,
        devices={"GPU": Device(), "Acc": Device()},
        model=Transformer(dict(name="audit", ndec=1, num_heads=8,
                               hdim=1024, dhead=128, ff_scale=4,
                               gqa_size=gqa, dtype=DataType.W16A16), 1))


def run(workload, rung, policy=None, **kw):
    policy = policy or ("no-reuse" if rung == "A1" else "epic")
    plan = build_reuse_plan(workload, policy, epic_prefix_recompute_tokens=8,
        cacheblend_partial_recompute_layers=(0,) if policy == "cachetune" else (),
        cacheblend_recompute_ratio=0.15 if policy == "cachetune" else 0.)
    cfg = resolve_config(rung, None, None, None, policy=policy)
    sys = system()
    report = wr.run_reuse_prefill(
        sys, workload, plan, pipe=True, cacheblend_batch_size=8,
        include_events=True, warm=False, decode_attn=cfg.decode_attn,
        kv_mapping=cfg.kv_mapping, channel_placement=cfg.channel_placement,
        pim_prefill_mode=cfg.prefill_attn, pim_batch_command=cfg.pim_batch_command,
        pim_pe_freq_ghz=cfg.pim_pe_freq_ghz, **kw)
    return report, sys


def main():
    out = {}
    solo = Workload("supervisor", (Request("r", 0, None, 1,
                       (Segment("doc", "doc", 256),), 256),), {})
    for rung in ("A1", "A3b", "A5"):
        report, sys = run(solo, rung)
        out[rung + "_fresh_prefill"] = {
            "reported_attention_rows": report["prefill_attention_rows"],
            "actual_events": dict(Counter((e["device"] + ":" + e["name"])
                for e in report["events"] if not e["name"].startswith("decode_"))),
            "gpu_matmul_shapes": [c for c in sys.devices["GPU"].calls
                                    if c[0] in ("score", "context")],
        }
    # Same TLB and same two master chunks, zero repairs: a pure diff-placement
    # ablation must preserve every master channel. These policies do not.
    tlb = wr.LocalDiffKVLayout(256)
    for i in range(5):
        for row in range(256):
            tlb.reserve(0, "owner", "c" + str(i), row, "master")
    tlb.chunk_order = ["c" + str(i) for i in range(5)]
    tlb.finalize()
    reads = [tlb.locate(0, "owner", "c" + str(i), row, "master")
             for i in (0, 4) for row in range(256)]
    def groups(policy, locations=reads):
        return wr._striped_append_channel_extents(
            locations, policy=policy, heads_per_hbm=4, tlb=tlb)
    out["zero_diff_master_loads"] = {
        p: {c: sum(e[2] for e in es) for c, _, es in groups(p)}
        for p in ("slice-append", "master-diff-local-append")}
    out["a3b_same_chunk_changes_channel"] = {
        "c4_after_c0": groups("slice-append"),
        "c4_alone": groups("slice-append", reads[256:]),
        "c4_persistent_slot": tlb.chunk_slot("c4", 4, "append")}
    large = wr.LocalDiffKVLayout(256)
    for row in range(1024):
        large.reserve(0, "owner", "large", row, "master")
    large.chunk_order = ["large"]
    large.finalize()
    large_reads = [large.locate(0, "owner", "large", i, "master")
                   for i in range(1024)]
    out["1024_token_segment_one_slot"] = wr._striped_append_channel_extents(
        large_reads, policy="master-diff-local-append", heads_per_hbm=4, tlb=large)
    out["future_reader_changes_placement"] = {
        "no_future_reader": wr._chunk_slot_table(["a", "b"], [], 2, "table"),
        "future_reads_both": wr._chunk_slot_table(["a", "b"],
                                                 [frozenset(["a", "b"])], 2, "table")}
    shared = Workload("supervisor", (
        Request("a", 0, None, 1, (Segment("doc", "shared", 256),), 256),
        Request("b", 0, None, 1, (Segment("user", "private", 8),
                Segment("doc", "shared", 256)), 264)), {})
    reversed_shared = Workload(shared.kind, tuple(reversed(shared.requests)), {})
    reverse_report, _ = run(reversed_shared, "A5")
    out["same_tier_read_before_owner_write"] = {
        "consumer_first_scan_end_s": min(e["end_s"] for e in reverse_report["events"]
            if e["request"] == "b" and e["name"] == "pim_kv_scan_score_softmax_pv"),
        "owner_store_start_s": min(e["start_s"] for e in reverse_report["events"]
            if e["request"] == "a" and e["name"] == "dram_store_master")}
    out["rung_dependent_recompute_rows"] = {}
    for rung in ("A3b", "A4c"):
        canonical = PRESETS[rung]["kv_mapping"] != "naive"
        plan = build_reuse_plan(shared, "recompute", epic_prefix_recompute_tokens=8,
                                recompute_canonical=canonical)
        out["rung_dependent_recompute_rows"][rung] = list(plan.reusable[0].epic_prefix_rows)
    out["policy_dispatch"] = {}
    for policy in ("epic", "cachecraft", "cachetune"):
        report, _ = run(shared, "A6", policy=policy)
        out["policy_dispatch"][policy] = {
            "keys": sorted(report),
            "decode_events": sum(e["name"].startswith("decode_")
                                  for e in report.get("events", [])),
            "event_count": len(report.get("events", []))}
    # A real GQA model still constructs a dense-MHA QKV projection.
    model = Transformer(make_model_config("LLAMA3-8B", DataType.W16A16), 1)
    model.build(1, 1, 2, True)
    out["llama3_qkv_width"] = {"actual": model.sum_decoder[0].n,
                               "geometry_expected": 4096 + 2 * 8 * 128}
    out["default_pim_count"] = make_pim_config(PIMType.BA, InterfaceType.NVLINK3,
                                                num_hbm=1)["NUM_ATTACC"]
    if wr._EC is not None:
        wr._EC.close()
        wr._EC = None
    ev = []
    for device in ("PIM:pool0-14", "PIM:pool0-0", "PIM"):
        wr._cacheblend_event(ev, layer=0, tier=0, request="r", name="audit",
                            device=device, rows=1, time_s=1., energy=(), deps=())
    scheduled = wr._schedule_cacheblend(ev, pipe=True)
    out["overlapping_physical_resources"] = [
        {"resource": e.device, "start": e.start_s, "end": e.end_s} for e in scheduled]
    # These are independent paper-contract checks, not speed measurements.
    # False means the recorded implementation violates that specific contract.
    owner_read = out["same_tier_read_before_owner_write"]
    out["paper_contract_checks"] = {
        "A1_fresh_prefill_has_gpu_score": any(
            key.startswith("GPU:") and "score" in key
            for key in out["A1_fresh_prefill"]["actual_events"]),
        "A5_fresh_prefill_has_pim_scan": any(
            key.startswith("PIM") and "scan_score" in key
            for key in out["A5_fresh_prefill"]["actual_events"]),
        "zero_diff_preserves_master_channels":
            out["zero_diff_master_loads"]["slice-append"] ==
            out["zero_diff_master_loads"]["master-diff-local-append"],
        "rungs_preserve_corrected_token_indices":
            out["rung_dependent_recompute_rows"]["A3b"] ==
            out["rung_dependent_recompute_rows"]["A4c"],
        "shared_scan_does_not_finish_before_owner_store_starts":
            owner_read["consumer_first_scan_end_s"] >= owner_read["owner_store_start_s"],
        "all_policy_sweeps_include_decode_and_makespan": all(
            row["decode_events"] > 0 and "makespan_s" in row["keys"]
            for row in out["policy_dispatch"].values()),
        "llama3_qkv_width_matches_gqa_geometry":
            out["llama3_qkv_width"]["actual"] ==
            out["llama3_qkv_width"]["geometry_expected"],
        "overlapping_pim_resources_do_not_all_run_at_once":
            max(e.end_s for e in scheduled) > 1.,
    }
    # Disclose the external location: these inputs are referenced by the paper,
    # but absent from this repository's tracked experiment inputs.
    external = Path("/data2/chenyi9/KV-PIM/attacc_drampim_xinyao")
    paths = sorted((external / "workload/sweep").glob("wl*.json"))
    paths += sorted((external / "experiments/paper_ladder/workloads").glob("*.json"))
    out["external_workload_stats"] = []
    for path in paths:
        try:
            wl = load_workload(path)
            plan = build_reuse_plan(wl, "epic", epic_prefix_recompute_tokens=8)
            row = dict(path=str(path), requests=len(wl.requests),
                tiers=len(wl.tiers), inputs=sum(r.total_length for r in wl.requests),
                outputs=sum(r.lout for r in wl.requests),
                histories=sum(r.history_len for r in wl.requests),
                timestamped=sum("timestamp" in r.raw for r in wl.requests),
                reused=plan.reused_tokens,
                corrections=sum(len(d.epic_prefix_rows) for d in plan.reusable),
                segment_lengths=dict(Counter(s.length for r in wl.requests for s in r.segments)),
                per_tier={str(t): dict(requests=len(rs),
                    corrections=sum(len(d.epic_prefix_rows) for d in plan.reusable
                                    if d.request_id in {r.request_id for r in rs}))
                          for t, rs in wl.tiers.items()})
            # Corpus offsets are directly calculable, independent of planner.
            offsets = {}
            for r in wl.requests:
                pos = 0
                for s in r.segments:
                    if s.role == "doc":
                        offsets.setdefault((r.tier, s.fingerprint), set()).add(pos)
                    pos += s.length
            row["max_distinct_doc_offsets_within_tier"] = max(map(len, offsets.values()), default=0)
            out["external_workload_stats"].append(row)
        except Exception as exc:
            out["external_workload_stats"].append(dict(path=str(path), error=str(exc)))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
