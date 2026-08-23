import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from src.workload import (Request, Segment, Workload, WorkloadValidationError,
                          build_reuse_plan, load_workload, validate_reuse_plan,
                          workload_summary)
from src.workload_runner import (run_cacheblend_analytic_report,
                                 run_no_reuse_report, run_no_reuse_workload)
from src.workload_runner import run_reuse_prefill
from src.ablation import (_batch_scan_profile, _master_diff_lengths,
                         _naive_run_lengths, resolve_config,
                         run_ablation_report)
from src.model import Transformer
from src.system import System, apply_attacc_pipeline
from src.config import make_xpu_config
from src.type import DataType, DeviceType, GPUType


ROOT = Path(__file__).resolve().parents[1]


class WorkloadTests(unittest.TestCase):
    def test_attacc_pipeopt_reference_matches_decoder_events(self):
        """The extracted overlap reference preserves the original event rule."""
        toy = {"name": "toy", "ndec": 1, "num_heads": 4, "hdim": 16,
               "dhead": 4, "ff_scale": 4, "gqa_size": 1,
               "dtype": DataType.W16A16}
        gpu = make_xpu_config(GPUType.A100a, num_gpu=1)["GPU"]
        # The pipeline formula is selected by the PIM execution path.  GPU is
        # sufficient as a deterministic timing backend for this unit test.
        reference_system = System(gpu, toy)
        reference_system.hetero_name = DeviceType.PIM
        reference_system.model.build(1, 6, 2, True)
        reference = copy.deepcopy(reference_system.model.gen_decoder[0])
        for layer in reference:
            device = (reference_system.devices["Acc"]
                      if layer.type.name in ("MATMUL", "SOFTMAX", "X2G")
                      else reference_system.devices["GPU"])
            layer.exec_time, layer.energy = device.get_time_and_energy(layer)
        apply_attacc_pipeline(reference, reference_system.model.num_heads,
                              reference_system.GPU.num_xpu, True)

        prototype = System(gpu, toy)
        prototype.hetero_name = DeviceType.PIM
        prototype.devices["Acc"].pim_type = SimpleNamespace(name="bank")
        prototype.simulate(1, 6, 2, perfs=[], pipe=True,
                           parallel_ff=False, power_constraint=False)
        observed = prototype.model.gen_decoder[0]
        self.assertEqual([layer.name for layer in observed],
                         [layer.name for layer in reference])
        for actual, expected in zip(observed, reference):
            self.assertAlmostEqual(actual.exec_time, expected.exec_time)

    def test_existing_rag_workload_is_valid(self):
        workload = load_workload(ROOT / "workload/workload_2wikimqa_first8.json")
        self.assertEqual(workload.kind, "rag")
        self.assertTrue(all(r.total_length == sum(s.length for s in r.segments)
                            for r in workload.requests))
        self.assertEqual(workload_summary(workload)["tiers"].keys(), {"0"})

    def test_existing_supervisor_workload_is_valid(self):
        workload = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
        self.assertEqual(workload.kind, "supervisor")
        self.assertEqual([r.request_id for r in workload.tiers[1]],
                         ["t1w0", "t1w1", "t1w2", "t1w3"])

    def test_bad_legacy_length_is_rejected(self):
        payload = [{"sample": 0, "seg_lens": [1, 1, 1],
                    "seg_sha": ["s", "d", "q"],
                    "seg_role": ["sys", "doc", "query"], "L": 4,
                    "lout": 1}]
        path = ROOT / "tests/.bad_workload.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(WorkloadValidationError, "sum to"):
            load_workload(path)

    def test_rag_segment_composition_is_checked(self):
        payload = [{"sample": 0, "seg_lens": [1, 1], "seg_sha": ["s", "q"],
                    "seg_role": ["sys", "query"], "L": 2, "lout": 1}]
        path = ROOT / "tests/.bad_workload.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(WorkloadValidationError, "one or more docs"):
            load_workload(path)

    def test_parent_tier_is_checked(self):
        payload = {"agents": [
            {"id": "p", "tier": 0, "parent": None, "lout": 2,
             "segs": [{"role": "sys", "sha": "s", "len": 1}]},
            {"id": "c", "tier": 0, "parent": "p", "lout": 1,
             "segs": [{"role": "parent_out", "sha": "o", "len": 1}]},
        ]}
        path = ROOT / "tests/.bad_workload.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(WorkloadValidationError, "earlier tier"):
            load_workload(path)

    def test_reuse_policy_is_independent_of_workload_kind(self):
        rag = load_workload(ROOT / "workload/workload_2wikimqa_first8.json")
        supervisor = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
        self.assertGreater(build_reuse_plan(rag, "epic").reused_tokens, 0)
        plan = build_reuse_plan(supervisor, "cacheblend", 0.5, 7)
        self.assertGreater(plan.reused_tokens, 0)
        self.assertEqual(plan, build_reuse_plan(supervisor, "cacheblend", 0.5, 7))
        self.assertEqual(build_reuse_plan(rag, "no-reuse").reused_tokens, 0)

    def test_parent_output_owner_is_the_declared_parent(self):
        workload = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
        plan = build_reuse_plan(workload, "epic", epic_prefix_recompute_tokens=2)
        relay_decisions = [decision for decision in plan.reusable
                           if decision.fingerprint == "out-sup"]
        self.assertEqual(len(relay_decisions), 4)
        self.assertTrue(all(decision.owner_request_id == "sup"
                            for decision in relay_decisions))
        self.assertTrue(all(decision.epic_prefix_rows == (0, 1)
                            for decision in relay_decisions))
        stable_sys = [decision for decision in plan.reusable
                      if decision.fingerprint == "sys0"]
        self.assertTrue(all(not decision.epic_prefix_rows
                            for decision in stable_sys))

    def test_no_reuse_preserves_tier_as_batch_unit(self):
        class FakeSystem:
            def __init__(self):
                self.calls = []

            def simulate(self, batch, lin, lout, **kwargs):
                self.calls.append((batch, lin, lout, kwargs))
                kwargs["perfs"].append((batch, lin, lout))

        workload = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
        system = FakeSystem()
        results = run_no_reuse_workload(system, workload, pipe=True,
                                        parallel_ff=True, power_constraint=False)
        self.assertEqual([(batch, lin, lout) for batch, lin, lout, _ in system.calls],
                         [(1, 550, 150), (4, 700, 150)])
        self.assertEqual(results, [(1, 550, 150), (4, 700, 150)])

    def test_no_reuse_report_reconstructs_serial_end_to_end_makespan(self):
        class FakeSystem:
            def simulate(self, batch, lin, lout, **kwargs):
                # Legacy record shape: [tag, config, perf_ms, energy_nj].
                perf_ms = [0.0] * 20
                perf_ms[0] = float(lin)          # whole-tier prefill
                perf_ms[7] = float(batch)        # per-token decode
                kwargs["perfs"].append([[], [], perf_ms, [2.0]])

        workload = Workload("supervisor", (
            Request("root", 0, None, 3,
                    (Segment("sys", "s", 2),), 2),
            Request("child", 1, "root", 2,
                    (Segment("parent_out", "o", 4),), 4),
        ), {})
        _, report = run_no_reuse_report(FakeSystem(), workload, pipe=False,
                                        parallel_ff=False,
                                        power_constraint=False)
        # Tier 0: 2 ms prefill + (3 - 1) * 1 ms decode.
        # Tier 1: 4 ms prefill + (2 - 1) * 1 ms decode.
        self.assertAlmostEqual(report["makespan_s"], .009)
        self.assertEqual([tier["duration_s"] for tier in report["tiers"]],
                         [.004, .005])
        self.assertAlmostEqual(report["decode_energy_nj"], 6.0)

    def test_cacheblend_analytic_scales_only_reused_prefill(self):
        class FakeSystem:
            model = SimpleNamespace(ndec=3)

            def simulate(self, batch, lin, lout, **kwargs):
                perf_ms = [0.0] * 20
                perf_ms[0] = float(lin)
                perf_ms[7] = float(batch)
                kwargs["perfs"].append([[], [], perf_ms, [2.0]])

        workload = Workload("supervisor", (
            Request("root", 0, None, 3,
                    (Segment("sys", "s", 2),), 2),
            Request("child", 1, "root", 2,
                    (Segment("parent_out", "o", 4),), 4),
        ), {})
        plan = build_reuse_plan(workload, "cacheblend", .5, 7, (0,), (1, 2))
        report = run_cacheblend_analytic_report(
            FakeSystem(), workload, plan, pipe=False, parallel_ff=False,
            power_constraint=False)
        child = report["tiers"][1]
        # One reused row costs 1 full + 2 * .5 partial layers = 2/3 work.
        self.assertAlmostEqual(child["prefill_scale"], 2.0 / 3.0)
        self.assertAlmostEqual(child["prefill_s"], .004 * 2.0 / 3.0)
        self.assertAlmostEqual(child["decode_s"], .001)
        self.assertEqual(report["policy"], "cacheblend-analytic")
        # EPIC under the same legacy abstraction: only its recomputed prefix
        # rows are charged.  The child's shifted parent_out segment recomputes
        # one leading row of its four reused rows.
        epic_plan = build_reuse_plan(workload, "epic", epic_prefix_recompute_tokens=1)
        epic = run_cacheblend_analytic_report(
            FakeSystem(), workload, epic_plan, pipe=False, parallel_ff=False,
            power_constraint=False)
        self.assertEqual(epic["policy"], "epic-analytic")
        self.assertAlmostEqual(epic["tiers"][1]["prefill_scale"], 1.0 / 4.0)
        self.assertAlmostEqual(epic["tiers"][1]["decode_s"], .001)
        # A tier batch bound splits a tier into serial padded batches.
        two = Workload("rag", (
            Request("a", 0, None, 2, (Segment("sys", "s", 2),), 2),
            Request("b", 0, None, 2, (Segment("sys", "s", 3),), 3),
            Request("c", 0, None, 2, (Segment("sys", "s", 5),), 5)), {})
        _, batched = run_no_reuse_report(FakeSystem(), two, pipe=False,
                                         parallel_ff=False, power_constraint=False,
                                         batch_size=2)
        self.assertEqual([(t["batch_size"], t["lin"]) for t in batched["tiers"]],
                         [(2, 3), (1, 5)])
        _, whole = run_no_reuse_report(FakeSystem(), two, pipe=False,
                                       parallel_ff=False, power_constraint=False)
        self.assertEqual([(t["batch_size"], t["lin"]) for t in whole["tiers"]],
                         [(3, 5)])

    def test_single_no_reuse_workload_matches_legacy_call_arguments(self):
        payload = [{"sample": 0, "seg_lens": [2, 3, 4],
                    "seg_sha": ["s", "d", "q"],
                    "seg_role": ["sys", "doc", "query"], "L": 9,
                    "lout": 5}]
        path = ROOT / "tests/.single_workload.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(path.unlink)
        workload = load_workload(path)

        class FakeSystem:
            def __init__(self):
                self.calls = []

            def simulate(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                kwargs["perfs"].append("legacy-result")

        system = FakeSystem()
        perfs = run_no_reuse_workload(system, workload, pipe=True,
                                      parallel_ff=False, power_constraint=True)
        self.assertEqual(perfs, ["legacy-result"])
        self.assertEqual(system.calls[0][0], (1, 9, 5))
        self.assertTrue(system.calls[0][1]["pipe"])
        self.assertTrue(system.calls[0][1]["power_constraint"])

    def test_no_reuse_matches_real_attacc_for_a_small_request(self):
        """The JSON entry point must preserve the legacy perf record exactly."""
        toy = {"name": "toy", "ndec": 3, "num_heads": 4, "hdim": 16,
               "dhead": 4, "ff_scale": 4, "gqa_size": 1,
               "dtype": DataType.W16A16}
        gpu = make_xpu_config(GPUType.A100a, num_gpu=1)["GPU"]
        direct_system = System(gpu, toy)
        direct = []
        direct_system.simulate(1, 6, 2, perfs=direct, pipe=True,
                               parallel_ff=False, power_constraint=False)
        workload = Workload("rag", (
            Request("sample-0", 0, None, 2, (
                Segment("sys", "sys", 2), Segment("doc", "doc", 2),
                Segment("query", "query", 2)), 6),), {})
        workload_system = System(gpu, toy)
        via_workload = run_no_reuse_workload(
            workload_system, workload, pipe=True, parallel_ff=False,
            power_constraint=False)
        self.assertEqual(via_workload, direct)

    def test_no_reuse_multi_agent_json_matches_legacy_tier_calls_exactly(self):
        """A supervisor JSON must be only a tiered adapter around legacy AttAcc.

        Legacy AttAcc has no DAG input: its native unit is one padded
        rectangular ``(batch, Lin, Lout)`` call.  This test therefore builds a
        real multi-agent JSON, invokes the old API once for each tier, and
        requires the JSON no-reuse entry point to return the identical records.
        """
        payload = {"meta": {"schema": "v2-dag"}, "agents": [
            {"id": "sup", "tier": 0, "parent": None, "lout": 2,
             "segs": [
                 {"role": "sys", "sha": "sys", "len": 2},
                 {"role": "instr", "sha": "sup-instr", "len": 3},
             ]},
            {"id": "worker-0", "tier": 1, "parent": "sup", "lout": 1,
             "segs": [
                 {"role": "sys", "sha": "sys", "len": 2},
                 {"role": "parent_out", "sha": "out-sup", "len": 2},
                 {"role": "instr", "sha": "w0-instr", "len": 1},
             ]},
            {"id": "worker-1", "tier": 1, "parent": "sup", "lout": 2,
             "segs": [
                 {"role": "sys", "sha": "sys", "len": 2},
                 {"role": "parent_out", "sha": "out-sup", "len": 2},
                 {"role": "instr", "sha": "w1-instr", "len": 2},
             ]},
        ]}
        path = ROOT / "tests/.multi_agent_no_reuse.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(path.unlink)
        workload = load_workload(path)

        toy = {"name": "toy", "ndec": 1, "num_heads": 4, "hdim": 16,
               "dhead": 4, "ff_scale": 4, "gqa_size": 1,
               "dtype": DataType.W16A16}

        def make_system():
            return System(make_xpu_config(GPUType.A100a, num_gpu=1)["GPU"], toy)

        direct = []
        legacy = make_system()
        # Tier 0: one 5-token agent; tier 1: two agents padded to (6, 2).
        legacy.simulate(1, 5, 2, perfs=direct, pipe=True,
                        parallel_ff=False, power_constraint=False)
        legacy.simulate(2, 6, 2, perfs=direct, pipe=True,
                        parallel_ff=False, power_constraint=False)

        via_workload = run_no_reuse_workload(
            make_system(), workload, pipe=True, parallel_ff=False,
            power_constraint=False)
        self.assertEqual(via_workload, direct)

    def test_relay_json_no_reuse_matches_original_attacc_prototype(self):
        """Use the supplied relay JSON, not a synthetic DAG fixture."""
        workload = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
        toy = {"name": "toy", "ndec": 1, "num_heads": 4, "hdim": 16,
               "dhead": 4, "ff_scale": 4, "gqa_size": 1,
               "dtype": DataType.W16A16}

        def make_system():
            return System(make_xpu_config(GPUType.A100a, num_gpu=1)["GPU"], toy)

        direct = []
        legacy = make_system()
        for _, requests in workload.tiers.items():
            legacy.simulate(len(requests), max(request.total_length for request in requests),
                            max(request.lout for request in requests), perfs=direct,
                            pipe=True, parallel_ff=False, power_constraint=False)
        via_json = run_no_reuse_workload(make_system(), workload, pipe=True,
                                         parallel_ff=False, power_constraint=False)
        self.assertEqual(via_json, direct)

    def test_2wikimqa_json_no_reuse_matches_original_attacc_prototype(self):
        """The supplied RAG JSON is likewise only a tiered legacy adapter."""
        workload = load_workload(ROOT / "workload/workload_2wikimqa_first8.json")
        toy = {"name": "toy", "ndec": 1, "num_heads": 4, "hdim": 16,
               "dhead": 4, "ff_scale": 4, "gqa_size": 1,
               "dtype": DataType.W16A16}

        def make_system():
            return System(make_xpu_config(GPUType.A100a, num_gpu=1)["GPU"], toy)

        requests = workload.tiers[0]
        direct = []
        make_system().simulate(len(requests), max(request.total_length for request in requests),
                               max(request.lout for request in requests), perfs=direct,
                               pipe=True, parallel_ff=False, power_constraint=False)
        via_json = run_no_reuse_workload(make_system(), workload, pipe=True,
                                         parallel_ff=False, power_constraint=False)
        self.assertEqual(via_json, direct)

    def _ablation_toy(self):
        """Two-request workload with one shared chunk, on a toy model.

        ``devices['Acc']`` stays the GPU model, so the comparison exercises the
        ablation driver's layer/pipeline arithmetic rather than Ramulator.
        """
        toy = {"name": "toy", "ndec": 4, "num_heads": 4, "hdim": 16,
               "dhead": 4, "ff_scale": 4, "gqa_size": 1,
               "dtype": DataType.W16A16}
        gpu = make_xpu_config(GPUType.A100a, num_gpu=1)["GPU"]
        workload = Workload("rag", (
            Request("r0", 0, None, 3, (
                Segment("sys", "shared-sys", 4), Segment("doc", "own-0", 4),
                Segment("query", "q0", 2)), 10),
            Request("r1", 0, None, 3, (
                Segment("sys", "shared-sys", 4), Segment("doc", "own-1", 4),
                Segment("query", "q1", 2)), 10),), {})
        return toy, gpu, workload

    def test_ablation_a1_reproduces_the_original_attacc_legacy_report(self):
        """A1 must be the untouched AttAcc evaluation, not a re-derivation.

        The legacy CSV's summarization total omits the K/V link layer, which
        the ablation driver counts so that GPU-side and PIM-side prefill
        configurations are accounted the same way.  Everything else -- decode
        time, prefill compute, prefill and decode energy -- must match bit for
        bit.
        """
        toy, gpu, workload = self._ablation_toy()
        plan = build_reuse_plan(workload, "no-reuse", 0.0, 0, (), (), 1)

        legacy_system = System(gpu, toy)
        legacy_system.hetero_name = DeviceType.PIM
        legacy_system.devices["Acc"].pim_type = SimpleNamespace(name="bank")
        _, legacy = run_no_reuse_report(legacy_system, workload, pipe=True,
                                        parallel_ff=True, power_constraint=False)

        system = System(gpu, toy)
        system.hetero_name = DeviceType.PIM
        report = run_ablation_report(
            system, workload, plan,
            resolve_config("A1", None, None, None, policy="no-reuse"),
            pipe=True, parallel_ff=True, power_constraint=False)

        legacy_decode = sum(tier["decode_s"] for tier in legacy["tiers"])
        legacy_prefill = sum(tier["prefill_s"] for tier in legacy["tiers"])
        link_s = sum(tier["prefill_breakdown_s"].get("gpu_comm_x2g", 0.0)
                     for tier in report["tiers"])
        self.assertGreater(link_s, 0.0)
        self.assertAlmostEqual(report["decode_s"] / legacy_decode, 1.0, places=12)
        self.assertAlmostEqual((report["prefill_s"] - link_s) / legacy_prefill,
                               1.0, places=12)
        self.assertAlmostEqual(report["energy_nj"] / legacy["energy_nj"],
                               1.0, places=12)
        self.assertAlmostEqual(
            report["prefill_energy_nj"] / legacy["prefill_energy_nj"], 1.0,
            places=12)

    def test_naive_mapping_fragments_the_scan_that_master_diff_keeps_whole(self):
        """The 3-vs-4 mechanism must be a real address-pattern difference.

        Naive keeps the software's chunk layout in one 16-channel pool, so a
        recomputed row splits the stream and is read from its own block.
        Master/diff streams the immutable rows -- including the ones shadowed
        by a correction -- as whole per-chunk extents in one pool while the
        recomputed rows form a single extent in the other.
        """
        toy, gpu, workload = self._ablation_toy()
        plan = build_reuse_plan(workload, "cacheblend", 0.5, 7, (0,),
                                (1, 2, 3), 1)
        request = workload.requests[1]
        naive_full = _naive_run_lengths(request, plan, 0)      # full-recompute
        naive_partial = _naive_run_lengths(request, plan, 1)   # patched layer
        master, diff_rows = _master_diff_lengths(request, plan, 1)

        self.assertEqual(sum(naive_full), request.total_length)
        self.assertGreater(len(naive_partial), len(naive_full))
        self.assertGreater(diff_rows, 0)
        self.assertEqual(sum(master), request.total_length)
        # Naive reads each logical row once; master/diff also streams the
        # shadowed master row of every correction.
        self.assertEqual(sum(naive_partial), request.total_length)
        self.assertEqual(sum(master) + diff_rows,
                         request.total_length + diff_rows)

        config = resolve_config("A3", None, None, None, policy="cacheblend")
        profile = _batch_scan_profile(workload.requests, plan, 1, 10, 2, 32, config)
        self.assertEqual(len(profile.pools), 1)
        self.assertEqual({run[4] for run in profile.pools[0]}, {16})
        self.assertEqual(sum(run[2] for run in profile.pools[0]), 12)

        config = resolve_config("A4", None, None, None, policy="cacheblend")
        profile = _batch_scan_profile(workload.requests, plan, 1, 10, 2, 32, config)
        self.assertEqual(len(profile.pools), 2)
        self.assertEqual({run[3] for run in profile.pools[0]}, {0})
        self.assertEqual({run[3] for run in profile.pools[1]}, {8})
        self.assertEqual(sum(run[2] for run in profile.pools[0]), 12)

    def test_split_prefill_overlaps_its_gpu_and_pim_branches(self):
        """A6's two branches have no data dependency, so the layer costs max().

        KVpim-sim's trace runs the GPU ``fresh score`` events of B-prefill L2
        concurrently with the ``b0``/``b1`` scans and merges the two partial
        (m, l, o) triples on the DIE afterwards, so charging the branches one
        after the other overstates the split.  A5 has no GPU branch and must be
        bit-identical under both settings.
        """
        toy, gpu, workload = self._ablation_toy()
        plan = build_reuse_plan(workload, "cacheblend", 0.5, 7, (0,),
                                (1, 2, 3), 1)

        def stub_runs(op):
            # Stand in for Ramulator: one measurement per contiguous extent,
            # linear in the rows it streams.  The test is about how the driver
            # combines the branches, not about the memory timing.
            return [(1e-9 * run[2] * op.numOp, [1.0] * 6)
                    for run in op.pim_kv_runs]

        def run(preset, overlap):
            system = System(gpu, toy)
            system.hetero_name = DeviceType.PIM
            system.devices["Acc"].pim_type = SimpleNamespace(name="bank")
            system.devices["Acc"].get_time_and_energy_runs = stub_runs
            return run_ablation_report(
                system, workload, plan,
                resolve_config(preset, None, None, None, policy="cacheblend",
                               split_overlap=overlap),
                pipe=True, parallel_ff=True, power_constraint=False)

        overlap, serial = run("A6", True), run("A6", False)
        self.assertLess(overlap["prefill_s"], serial["prefill_s"])
        # Overlap hides time, it does not remove work.
        self.assertAlmostEqual(overlap["prefill_energy_nj"] /
                               serial["prefill_energy_nj"], 1.0, places=12)
        # The breakdown still sums to the reported prefill time.
        for tier in overlap["tiers"]:
            saving = tier["prefill_breakdown_s"]["split_overlap_saving"]
            self.assertLess(saving, 0.0)

        a5_overlap, a5_serial = run("A5", True), run("A5", False)
        self.assertEqual(a5_overlap["prefill_s"], a5_serial["prefill_s"])
        self.assertNotIn("split_overlap_saving",
                         a5_overlap["tiers"][0]["prefill_breakdown_s"])

    def test_ablation_rejects_incoherent_placement_switches(self):
        for kwargs in (
                {"preset": None, "prefill_attn": "gpu", "decode_attn": "gpu",
                 "kv_mapping": "naive", "policy": "cacheblend"},
                {"preset": None, "prefill_attn": "gpu", "decode_attn": "pim",
                 "kv_mapping": "none", "policy": "cacheblend"},
                {"preset": None, "prefill_attn": "gpu", "decode_attn": "pim",
                 "kv_mapping": "master-diff", "policy": "no-reuse"},
                {"preset": None, "prefill_attn": "gpu", "decode_attn": "pim",
                 "kv_mapping": "private", "policy": "cacheblend"}):
            policy = kwargs.pop("policy")
            with self.assertRaises(WorkloadValidationError):
                resolve_config(kwargs.pop("preset"), kwargs["prefill_attn"],
                               kwargs["decode_attn"], kwargs["kv_mapping"],
                               policy=policy)

    def test_reuse_structure_checker_covers_layers_and_rows(self):
        workload = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
        cacheblend = build_reuse_plan(
            workload, "cacheblend", .25, 7, (0, 1), (2,))
        validate_reuse_plan(workload, cacheblend, model_layers=3)
        with self.assertRaisesRegex(WorkloadValidationError, "cover the model"):
            validate_reuse_plan(workload, cacheblend, model_layers=4)
        epic = build_reuse_plan(workload, "epic", epic_prefix_recompute_tokens=2)
        validate_reuse_plan(workload, epic, model_layers=3)

    def test_split_prefill_emits_ordered_gpu_pim_link_events(self):
        class Device:
            peak_memory_bandwidth = 10**12
            softmax_peak_bandwidth = 10**12
            energy_table = {"mem": 1, "sram": 1}

            def get_time_and_energy(self, layer):
                return .001, [1, 0, 0, 0, 0, 0]

        class System:
            hetero_name = DeviceType.PIM
            devices = {"GPU": Device(), "Acc": Device()}
            model = Transformer({"name": "toy", "ndec": 3, "num_heads": 4,
                                 "hdim": 16, "ff_scale": 4,
                                 "dtype": DataType.W16A16}, tensor_parallel=1)

        workload = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
        plan = build_reuse_plan(workload, "epic", epic_prefix_recompute_tokens=1)
        report = run_reuse_prefill(System(), workload, plan, pipe=True)
        self.assertEqual(report["policy"], "epic")
        self.assertGreater(report["link_bytes"], 0)
        names = [event["name"] for event in report["events"]]
        self.assertIn("q_gpu_to_pim", names)
        self.assertIn("ctx_pim_to_gpu", names)
        self.assertIn("kv_gpu_to_pim", names)
        self.assertIn("dram_store_diff_and_live", names)

    def test_cacheblend_emits_trace_ordered_tlb_and_physical_addresses(self):
        class GPU:
            def get_time_and_energy(self, layer):
                return .001, [1, 0, 0, 0, 0, 0]

        class PIM:
            peak_memory_bandwidth = 10**12
            softmax_peak_bandwidth = 10**12
            energy_table = {"mem": 1, "sram": 1}

            def __init__(self):
                self.kv_runs = []

            def get_time_and_energy(self, layer):
                if getattr(layer, "pim_kv_runs", None) is not None:
                    self.kv_runs.append(layer.pim_kv_runs)
                return .002, [2, 0, 0, 0, 0, 0]

        pim = PIM()

        class System:
            hetero_name = DeviceType.PIM
            devices = {"GPU": GPU(), "Acc": pim}
            model = Transformer({"name": "toy", "ndec": 3, "num_heads": 4,
                                 "hdim": 16, "ff_scale": 4,
                                 "dtype": DataType.W16A16}, tensor_parallel=1)

        workload = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
        plan = build_reuse_plan(workload, "cacheblend", .1, 7, (0, 1), (2,))
        report = run_reuse_prefill(System(), workload, plan, pipe=True)
        self.assertEqual(report["cacheblend_batch_size"], 1)
        self.assertEqual(report["batches"], [])
        self.assertEqual(report["overlap_validation"], {
            "passed": True, "pipe": True,
            "contract": "AttAcc comm_x2g busy timeline plus explicit CacheBlend resources",
            "events_checked": len(report["events"]),
        })
        events = [event for event in report["events"]
                  if event["request"] == "t1w0" and
                  event["transformer_layer"] == 2]
        names = [event["name"] for event in events]
        self.assertLess(names.index("q_gpu_to_pim"),
                        names.index("die_query_position_transform"))
        self.assertLess(names.index("die_query_position_transform"),
                        names.index("tlb_lookup_and_bank_plan"))
        self.assertLess(names.index("tlb_lookup_and_bank_plan"),
                        names.index("pim_kv_scan_score_softmax_pv"))
        self.assertLess(names.index("pim_kv_scan_score_softmax_pv"),
                        names.index("die_lse_merge"))
        self.assertLess(names.index("die_lse_merge"), names.index("ctx_pim_to_gpu"))
        self.assertEqual(report["tlb"]["channel_sets"],
                         {"master": list(range(8)), "diff": list(range(8, 16))})
        blocks = {block["id"]: block for block in report["tlb"]["blocks"]}
        self.assertTrue(blocks)
        for block in blocks.values():
            self.assertIn(block["kind"], ("master", "diff"))
            expected_channels = (range(8) if block["kind"] == "master"
                                 else range(8, 16))
            self.assertIn(block["channel_base"], expected_channels)
            self.assertEqual(block["partition_offset"] % 32, 0)
            self.assertEqual(int(block["value_base"], 0) -
                             int(block["key_base"], 0), 1 << 23)

        for entry in report["tlb"]["entries"]:
            location = entry["location"]
            block = blocks[location["block_id"]]
            offset = location["token_offset"] * block["vector_stride"]
            self.assertEqual(int(location["key_address"], 0),
                             int(block["key_base"], 0) + offset)
            self.assertEqual(int(location["value_address"], 0),
                             int(block["value_base"], 0) + offset)
        # CacheBlend's diff store is an overlay: it can contain only a reused
        # row selected for correction.  Ordinary live/new rows stay in master.
        diff_entries = [entry for entry in report["tlb"]["entries"]
                        if entry["location"]["kind"] == "diff"]
        self.assertTrue(diff_entries)
        self.assertTrue(all(entry["reused"] for entry in diff_entries))
        self.assertTrue(any(not entry["reused"] and
                            entry["location"]["kind"] == "master"
                            for entry in report["tlb"]["entries"]))
        self.assertTrue(any(len(runs) > 1 for runs in pim.kv_runs),
                        "a multi-block CacheBlend scan must retain every TLB extent")
        scan = next(event for event in events
                    if event["name"] == "pim_kv_scan_score_softmax_pv")
        self.assertTrue(scan["dram_addresses"])
        self.assertTrue(scan["dram_addresses"][0].startswith("0x"))
        self.assertGreater(report["makespan_s"], 0)
        parent_store = next(event for event in report["events"]
                            if event["request"] == "sup" and
                            event["transformer_layer"] == 2 and
                            event["name"] == "decode_dram_store_master")
        child_parent_out = next(entry for entry in report["tlb"]["entries"]
                                if entry["request"] == "t1w0" and
                                entry["layer"] == 2 and entry["position"] == 400)
        self.assertEqual(child_parent_out["location"]["fingerprint"], "out-sup")
        self.assertEqual(child_parent_out["location"]["key_address"],
                         parent_store["dram_addresses"][0])
        first_child_qkv = next(event for event in report["events"]
                               if event["request"] == "t1w0" and
                               event["transformer_layer"] == 0 and
                               event["name"] == "qkv")
        final_parent_store = [event for event in report["events"]
                              if event["request"] == "sup" and
                              event["name"] == "decode_dram_store_master"][-1]
        self.assertIn(final_parent_store["id"], first_child_qkv["depends_on"])
        serial = run_reuse_prefill(System(), workload, plan, pipe=False)
        self.assertEqual(serial["overlap_validation"]["contract"],
                         "original AttAcc serial decoder")
        self.assertLessEqual(report["makespan_s"], serial["makespan_s"])
        context = next(event for event in events if event["name"] == "ctx_pim_to_gpu")
        kv_link = next(event for event in events if event["name"] == "kv_gpu_to_pim")
        qkv = next(event for event in events if event["name"] == "qkv")
        post = next(event for event in events if event["name"] == "gpu_ff2")
        self.assertIn(qkv["id"], kv_link["depends_on"])
        self.assertNotIn(context["id"], kv_link["depends_on"])
        self.assertNotIn(post["id"], kv_link["depends_on"])
        self.assertLess(kv_link["end_s"], context["start_s"])

    def test_physical_no_reuse_uses_private_contiguous_16_channel_kv(self):
        class GPU:
            def get_time_and_energy(self, layer):
                return .001, [1, 0, 0, 0, 0, 0]

        class PIM:
            peak_memory_bandwidth = 10**12
            softmax_peak_bandwidth = 10**12
            energy_table = {"mem": 1, "sram": 1}

            def __init__(self):
                self.runs = []

            def get_time_and_energy(self, layer):
                self.runs.append((layer.m, getattr(layer, "pim_kv_runs", None)))
                return .002, [2, 0, 0, 0, 0, 0]

        pim = PIM()

        class System:
            hetero_name = DeviceType.PIM
            devices = {"GPU": GPU(), "Acc": pim}
            model = Transformer({"name": "toy", "ndec": 3, "num_heads": 4,
                                 "hdim": 16, "ff_scale": 4,
                                 "dtype": DataType.W16A16}, tensor_parallel=1)

        workload = Workload("rag", (
            Request("r", 0, None, 1,
                    (Segment("sys", "s", 4),), 4),), {})
        report = run_reuse_prefill(System(), workload,
                                   build_reuse_plan(workload, "no-reuse"),
                                   pipe=True, cacheblend_batch_size=2)
        self.assertEqual(report["policy"], "no-reuse-physical")
        self.assertEqual(report["tlb"]["channel_sets"], {"private": list(range(16))})
        self.assertFalse(any(event["device"] == "TLB" for event in report["events"]))
        prefill_runs = [runs for rows, runs in pim.runs if rows == 4]
        self.assertEqual(len(prefill_runs), 3)
        self.assertTrue(all(len(runs) == 1 and runs[0][4] == 16
                            for runs in prefill_runs))

    def test_cacheblend_same_tier_decode_batches_are_materialized(self):
        class GPU:
            def get_time_and_energy(self, layer):
                return .001, [1, 0, 0, 0, 0, 0]

        class PIM:
            peak_memory_bandwidth = 10**12
            softmax_peak_bandwidth = 10**12
            energy_table = {"mem": 1, "sram": 1}

            def __init__(self):
                self.shared_kv = []

            def get_time_and_energy(self, layer):
                self.shared_kv.append(bool(getattr(layer, "pim_shared_kv", False)))
                return .002, [2, 0, 0, 0, 0, 0]

        pim = PIM()

        class System:
            hetero_name = DeviceType.PIM
            devices = {"GPU": GPU(), "Acc": pim}
            model = Transformer({"name": "toy", "ndec": 3, "num_heads": 4,
                                 "hdim": 16, "ff_scale": 4,
                                 "dtype": DataType.W16A16}, tensor_parallel=1)

        workload = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
        plan = build_reuse_plan(workload, "cacheblend", .1, 7, (0, 1), (2,))
        report = run_reuse_prefill(System(), workload, plan, pipe=True,
                                   cacheblend_batch_size=2)
        self.assertEqual(report["cacheblend_batch_size"], 2)
        self.assertEqual(report["cacheblend_rotate_mode"], "gpu")
        worker_batches = [batch for batch in report["batches"] if batch["tier"] == 1]
        self.assertTrue(worker_batches)
        self.assertTrue(all(batch["size"] == 2 for batch in worker_batches))
        shared_batches = [batch for batch in worker_batches
                          if batch["attention_start_s"] is not None]
        self.assertTrue(shared_batches)
        self.assertTrue(all(batch["admission"] == "global-q-ready-queue"
                            for batch in shared_batches))
        self.assertTrue(all(batch["attention_start_s"] >= batch["q_arrival_s"]
                            for batch in shared_batches))
        measured = [event for event in report["events"]
                    if event["name"] in ("decode_batch_qkv", "decode_batch_gpu_ff1",
                                          "decode_batch_gpu_ff2") and event["tier"] == 1]
        self.assertTrue(measured)
        self.assertTrue(all(event["rows"] == 2 for event in measured))
        self.assertTrue(all(len(event["batch_members"]) == 2 for event in measured))
        # Shared parent-out rows are one batch PIM scan, while agent-private
        # rows remain separate scans and are merged independently at the DIE.
        shared = [event for event in report["events"]
                  if event["name"] == "decode_batch_pim_kv_scan_score_softmax_pv" and
                  event["tier"] == 1]
        self.assertTrue(shared)
        self.assertTrue(all(len(event["batch_members"]) == 2 for event in shared))
        self.assertIn(True, pim.shared_kv)
        gpu_rotate = [event for event in report["events"]
                      if event["name"] == "decode_gpu_rotate_q_extra_to_pim"]
        self.assertTrue(gpu_rotate)
        die_report = run_reuse_prefill(System(), workload, plan, pipe=True,
                                       cacheblend_batch_size=2,
                                       cacheblend_rotate_mode="die")
        bank_report = run_reuse_prefill(System(), workload, plan, pipe=True,
                                        cacheblend_batch_size=2,
                                        cacheblend_rotate_mode="bank")
        self.assertEqual(die_report["cacheblend_rotate_mode"], "die")
        self.assertEqual(bank_report["cacheblend_rotate_mode"], "bank")
        self.assertTrue(any(event["name"].startswith("decode_die_rotate_q_")
                            for event in die_report["events"]))
        self.assertTrue(any(event["name"] == "decode_bank_rotate_q_local"
                            for event in bank_report["events"]))
        self.assertGreater(report["link_bytes"], bank_report["link_bytes"])
        self.assertGreaterEqual(die_report["makespan_s"], bank_report["makespan_s"])

    def test_cacheblend_decode_streams_master_and_diff_pools_sequentially(self):
        """Corrections mask master rows; they must not fragment the master stream.

        Master and diff are disjoint channel pools.  A consumer streams the
        owner's master block sequentially (shadowed rows read but masked) and
        its diff rows sequentially, so the number of physical runs is
        independent of how many rows CacheBlend chose to correct.
        """
        class GPU:
            def get_time_and_energy(self, layer):
                return .001, [1, 0, 0, 0, 0, 0]

        class PIM:
            peak_memory_bandwidth = 10**12
            softmax_peak_bandwidth = 10**12
            energy_table = {"mem": 1, "sram": 1}

            def get_time_and_energy(self, layer):
                return .002, [2, 0, 0, 0, 0, 0]

            def get_time_and_energy_runs(self, layer):
                return [(.002, [2, 0, 0, 0, 0, 0])
                        for _ in getattr(layer, "pim_kv_runs", ((),))]

        class System:
            hetero_name = DeviceType.PIM
            devices = {"GPU": GPU(), "Acc": PIM()}
            model = Transformer({"name": "toy", "ndec": 2, "num_heads": 4,
                                 "hdim": 16, "ff_scale": 4,
                                 "dtype": DataType.W16A16}, tensor_parallel=1)

        def request(request_id, query):
            segments = (Segment("sys", "sys", 2), Segment("doc", "doc", 20),
                        Segment("query", query, 2))
            return Request(request_id, 0, None, 2, segments, 24)

        workload = Workload("rag", (request("r0", "q0"), request("r1", "q1")), {})

        def decode_scans(ratio):
            plan = build_reuse_plan(workload, "cacheblend", ratio, 3, (0,), (1,))
            report = run_reuse_prefill(System(), workload, plan, pipe=True)
            corrected = sum(len(rows) for rows in
                            plan.cacheblend_partial_rows[1].get("r1", {}).values())
            first = min(event["query_positions"][0] for event in report["events"]
                        if event["request"] == "r1" and event["transformer_layer"] == 1
                        and event["name"] == "decode_pim_kv_scan_score_softmax_pv")
            scans = [event for event in report["events"]
                     if event["request"] == "r1" and event["transformer_layer"] == 1
                     and event["name"] == "decode_pim_kv_scan_score_softmax_pv"
                     and event["query_positions"] == [first]]
            return plan, report, corrected, scans

        plan, report, corrected, scans = decode_scans(0.25)
        self.assertGreater(corrected, 0)
        master = [event for event in scans if event["device"] == "PIM:pool0-7"]
        diff = [event for event in scans if event["device"] == "PIM:pool8-15"]
        # The whole 24-row context is streamed from master (shadowed rows
        # included), and only the corrected rows come from the diff pool.
        self.assertEqual(sum(event["rows"] for event in master), 24)
        self.assertEqual(sum(event["masked_rows"] for event in master), corrected)
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff[0]["rows"], corrected)
        self.assertEqual(diff[0]["masked_rows"], 0)
        # Fragmentation-free: exactly as many master runs as with no correction.
        _, _, _, baseline = decode_scans(0.0)
        self.assertEqual(len(master),
                         len([event for event in baseline
                              if event["device"] == "PIM:pool0-7"]))
        # Every event's rows and masks are consistent and the DIE merge is
        # priced per physical run (one local softmax tuple each) plus the GPU
        # tuple, not per K/V row.
        merge = next(event for event in report["events"]
                     if event["request"] == "r1" and event["transformer_layer"] == 1
                     and event["name"] == "decode_die_lse_merge")
        tuple_bytes = 4 * (4 + 2) * 2
        self.assertAlmostEqual(merge["time_s"],
                               (len(scans) + 1) * tuple_bytes / 10**12)
        # A producer's master rows are exactly the rows a consumer resolves,
        # so the shadowed master row of a diff entry is the owner's row.
        entries = report["tlb"]["entries"]
        owner_rows = {(entry["layer"], entry["position"]): entry["location"]
                      for entry in entries if entry["request"] == "r0"
                      and entry["position"] < 24}
        for entry in entries:
            if entry["request"] != "r1" or not entry["reused"]:
                continue
            location = entry["location"]
            owner = owner_rows[(entry["layer"], entry["position"])]
            if location["kind"] == "diff":
                self.assertEqual(location["shadow_master"]["key_address"],
                                 owner["key_address"])
                self.assertEqual(location["channel_base"], 8)
            else:
                self.assertEqual(location["key_address"], owner["key_address"])
        # Prefill scans of a partial layer stream master only (the diff row
        # is being computed on the GPU) and mask the corrected rows.
        prefill = [event for event in report["events"]
                   if event["request"] == "r1" and event["transformer_layer"] == 1
                   and event["name"] == "pim_kv_scan_score_softmax_pv"]
        self.assertTrue(prefill)
        self.assertTrue(all(event["device"] == "PIM:pool0-7" for event in prefill))
        self.assertTrue(any(event["masked_rows"] > 0 for event in prefill))

    def test_cacheblend_pool_spills_into_next_channel_of_the_same_pool(self):
        """A pool is channel_count x 1 GiB; a full first channel spills within the pool."""
        from src.workload_runner import CacheBlendTLB, _HBM_CHANNEL_BYTES
        tlb = CacheBlendTLB(256)
        # 40 blocks of 4 MiB K each: 8 MiB K windows hold two blocks per
        # 16 MiB tile, so 64 tiles = 1 GiB fill channel 0 after 128 blocks;
        # use bigger blocks to keep the test fast: 8 MiB (one block per tile).
        rows_per_block = (1 << 23) // 256
        for index in range(70):
            for row in range(0, rows_per_block, rows_per_block - 1):
                tlb.reserve(0, "o", "fp{}".format(index), row, "master")
        # ``reserve`` keeps only rows; give each block its full span.
        for index in range(70):
            tlb._reserved_rows[(0, "o", "fp{}".format(index), "master")] = set(range(rows_per_block))
        tlb.reserve(0, "c", "d", 0, "diff")
        tlb.finalize()
        blocks = tlb.report()["blocks"]
        master = [b for b in blocks if b["kind"] == "master"]
        self.assertEqual(len(master), 70)
        channels = sorted({int(b["key_base"], 0) // _HBM_CHANNEL_BYTES for b in master})
        self.assertEqual(channels, [0, 1])
        self.assertTrue(all(b["channel_base"] == 0 and b["channel_count"] == 8 for b in master))
        self.assertEqual({b["channel_offset"] for b in master}, {0, 1})
        spilled = [b for b in master if b["channel_offset"] == 1]
        self.assertEqual(len(spilled), 6)
        self.assertEqual(int(spilled[0]["key_base"], 0), 1 * _HBM_CHANNEL_BYTES)
        diff = [b for b in blocks if b["kind"] == "diff"]
        self.assertEqual(int(diff[0]["key_base"], 0) // _HBM_CHANNEL_BYTES, 8)


class MQBatchCommandTests(unittest.TestCase):
    """MQ-MAC batch command (PLAN_mq_command.md): trace shape and timing knobs."""

    def test_interval_is_preset_floor_or_pe_bound(self):
        from src.ramulator_wrapper import (MQ_POWER_BUDGET_W,
                                           mq_interval_cycles, mq_pe_power_w,
                                           mq_query_capacity)
        # 2026-08-23 model revision: the DRAM cadence is never stretched by
        # compute (FIMDRAM precedent).  interval = max(preset floor, PE term).
        self.assertEqual(mq_interval_cycles(1, True, 1.3), 6)
        self.assertEqual(mq_interval_cycles(4, True, 1.3), 6)   # was 7 under
        self.assertEqual(mq_interval_cycles(8, True, 1.3), 9)   # the stretch
        # At AttAcc's synthesized 666 MHz the PE throughput term dominates.
        self.assertEqual(mq_interval_cycles(4, True, 0.666), 8)
        self.assertEqual(mq_interval_cycles(8, True, 0.666), 16)
        # NPC floor is the DRAM datapath's own nCCDAB=4.
        self.assertEqual(mq_interval_cycles(2, False, 1.3), 4)
        # PE power is a separate account: a whole stack running n=32 at the
        # PC floor stays far inside the 116-W IDD7 budget line.
        self.assertLess(mq_pe_power_w(32, 6), MQ_POWER_BUDGET_W)
        # One 64-B query slice per Q in AttAcc's 512-B GEMV buffer.
        self.assertEqual(mq_query_capacity(512), 8)
        self.assertEqual(mq_query_capacity(1024), 16)

    def test_mq_trace_reads_each_column_once(self):
        import subprocess
        import tempfile
        from collections import Counter
        generator = ROOT / "ramulator2" / "trace_gen" / "gen_trace_attacc_bank.py"
        counts = {}
        with tempfile.TemporaryDirectory() as tmp:
            for label, extra in (("replicate", []), ("mq", ["--mq"])):
                trace = Path(tmp) / (label + ".trace")
                subprocess.run(
                    ["python3", str(generator), "--dhead", "128", "--nhead", "1",
                     "--seqlen", "64", "--dbyte", "2", "--output", str(trace),
                     "--shared-kv", "--shared-queries", "4"] + extra,
                    check=True, stdout=subprocess.DEVNULL)
                counts[label] = Counter(line.split()[0]
                                        for line in trace.read_text().splitlines()
                                        if line.strip())
        # MQ issues every MAC_AB once; the query-private movements stay x4.
        self.assertEqual(counts["replicate"]["PIM_MAC_AB"],
                         4 * counts["mq"]["PIM_MAC_AB"])
        for shared in ("PIM_WR_GB", "PIM_MV_SB", "PIM_SFM", "PIM_MV_GB"):
            self.assertEqual(counts["replicate"][shared], counts["mq"][shared])

    def test_phase_slices_partition_the_full_trace(self):
        import subprocess
        import tempfile
        from collections import Counter
        generator = ROOT / "ramulator2" / "trace_gen" / "gen_trace_attacc_bank.py"
        counts = {}
        with tempfile.TemporaryDirectory() as tmp:
            for phase in ("full", "score", "context"):
                trace = Path(tmp) / (phase + ".trace")
                subprocess.run(
                    ["python3", str(generator), "--dhead", "128", "--nhead", "1",
                     "--seqlen", "64", "--dbyte", "2", "--output", str(trace),
                     "--shared-kv", "--shared-queries", "4", "--mq",
                     "--phase", phase],
                    check=True, stdout=subprocess.DEVNULL)
                counts[phase] = Counter(line.split()[0]
                                        for line in trace.read_text().splitlines()
                                        if line.strip())
        merged = counts["score"] + counts["context"]
        self.assertEqual(merged, counts["full"])
        # Q loading and softmax belong to the score phase, P loading to the
        # context phase; the MACs split between the two.
        self.assertNotIn("PIM_MV_GB", counts["score"])
        self.assertNotIn("PIM_WR_GB", counts["context"])
        self.assertNotIn("PIM_SFM", counts["context"])
        self.assertGreater(counts["score"]["PIM_MAC_AB"], 0)
        self.assertGreater(counts["context"]["PIM_MAC_AB"], 0)

    def test_batched_decode_splits_sweeps_at_the_gemv_buffer_capacity(self):
        # Four admitted agents against a 128-B (two-query) GEMV buffer must
        # scan the common master rows in consecutive sweeps of at most two Qs.
        class GPU:
            def get_time_and_energy(self, layer):
                return .001, [1, 0, 0, 0, 0, 0]

        class PIM:
            peak_memory_bandwidth = 10**12
            softmax_peak_bandwidth = 10**12
            energy_table = {"mem": 1, "sram": 1}

            def __init__(self):
                self.sweep_queries = []

            def get_time_and_energy(self, layer):
                if getattr(layer, "pim_shared_kv", False):
                    self.sweep_queries.append(
                        int(getattr(layer, "pim_shared_queries", 1)))
                    self.assertion = getattr(layer, "pim_batch_command", None)
                return .002, [2, 0, 0, 0, 0, 0]

        pim = PIM()

        class System:
            hetero_name = DeviceType.PIM
            devices = {"GPU": GPU(), "Acc": pim}
            model = Transformer({"name": "toy", "ndec": 3, "num_heads": 4,
                                 "hdim": 16, "ff_scale": 4,
                                 "dtype": DataType.W16A16}, tensor_parallel=1)

        workload = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
        plan = build_reuse_plan(workload, "cacheblend", .1, 7, (0, 1), (2,))
        report = run_reuse_prefill(System(), workload, plan, pipe=True,
                                   cacheblend_batch_size=4,
                                   pim_batch_command="mq",
                                   gemv_buffer_bytes=128)
        self.assertEqual(report["pim_batch_command"], "mq")
        self.assertEqual(report["pim_sweep_query_capacity"], 2)
        shared = [event for event in report["events"]
                  if event["name"] == "decode_batch_pim_kv_scan_score_softmax_pv"]
        self.assertTrue(shared)
        self.assertTrue(all(len(event["batch_members"]) <= 2 for event in shared))
        self.assertTrue(pim.sweep_queries)
        self.assertTrue(all(count <= 2 for count in pim.sweep_queries))
        self.assertEqual(pim.assertion, "mq")

    def test_bank_whole_prefill_lands_kv_first_and_loads_di_bitmap(self):
        class GPU:
            def get_time_and_energy(self, layer):
                return .001, [1, 0, 0, 0, 0, 0]

        class PIM:
            peak_memory_bandwidth = 10**12
            softmax_peak_bandwidth = 10**12
            energy_table = {"mem": 1, "sram": 1}

            def get_time_and_energy(self, layer):
                return .002, [2, 0, 0, 0, 0, 0]

        class System:
            hetero_name = DeviceType.PIM
            devices = {"GPU": GPU(), "Acc": PIM()}
            model = Transformer({"name": "toy", "ndec": 3, "num_heads": 4,
                                 "hdim": 16, "ff_scale": 4,
                                 "dtype": DataType.W16A16}, tensor_parallel=1)

        workload = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
        plan = build_reuse_plan(workload, "cacheblend", .1, 7, (0, 1), (2,))
        report = run_reuse_prefill(System(), workload, plan, pipe=True,
                                   cacheblend_batch_size=4,
                                   pim_batch_command="mq",
                                   pim_prefill_mode="bank-whole")
        self.assertEqual(report["pim_prefill_mode"], "bank-whole")
        events = report["events"]
        by_id = {event["id"]: event for event in events}
        prefill = [event for event in events
                   if not event["name"].startswith("decode_")]
        # (2) No GPU fresh-row triangle and no LSE tuple in bank-whole prefill.
        self.assertFalse([e for e in prefill if e["name"] == "gpu_local_score"])
        self.assertFalse([e for e in prefill
                          if e["name"] == "gpu_partial_lse_to_pim"])
        assemblies = [e for e in prefill if e["name"] == "die_score_assembly"]
        self.assertTrue(assemblies)
        # Landing order: every bank-whole prefill scan transitively follows a
        # dram_store of this layer via its die_query_position_transform deps.
        scans = [e for e in prefill
                 if e["name"] == "pim_kv_scan_score_softmax_pv"]
        self.assertTrue(scans)

        def transitively_depends_on_store(event, depth=0):
            if depth > 6:
                return False
            for dep in event["depends_on"]:
                parent = by_id[dep]
                if parent["name"].startswith("dram_store"):
                    return True
                if transitively_depends_on_store(parent, depth + 1):
                    return True
            return False

        self.assertTrue(all(transitively_depends_on_store(e) for e in scans))
        # Scans cover the fresh rows too (rows > reused visible rows alone):
        # the relay workers reuse 500 rows and compute 200, so a full-range
        # scan reads more than the reused set.
        worker_scans = [e for e in scans if e["request"].startswith("t1w")]
        self.assertTrue(worker_scans)
        self.assertTrue(all(e["rows"] > 500 for e in worker_scans))
        # (1) D_i bitmap: one load per (request, partial layer with
        # corrections) for CacheBlend, before that layer's masked scans.
        bitmap_loads = [e for e in prefill if e["name"] == "die_load_di_bitmap"]
        self.assertTrue(bitmap_loads)
        self.assertGreater(report["di_bitmap_bytes"], 0)
        partial_with_corrections = {(e["request"], e["transformer_layer"])
                                    for e in prefill
                                    if e["name"] == "die_load_di_bitmap"}
        self.assertTrue(all(layer == 2 for _, layer in partial_with_corrections))


class AgenticHistoryTests(unittest.TestCase):
    """Agentic multi-turn history KV: attended everywhere, recomputed nowhere.

    ``history_len`` rows are the agent's own KV from earlier turns.  They are
    already resident in PIM memory when the run starts, so every prefill and
    decode attention pass must scan them, while QKV, the GPU-PIM link, and
    the DRAM stores must not grow by a single row.
    """

    def _system(self, ndec=2):
        class GPU:
            def get_time_and_energy(self, layer):
                return .001, [1, 0, 0, 0, 0, 0]

        class PIM:
            peak_memory_bandwidth = 10**12
            softmax_peak_bandwidth = 10**12
            energy_table = {"mem": 1, "sram": 1}

            def get_time_and_energy(self, layer):
                return .002, [2, 0, 0, 0, 0, 0]

            def get_time_and_energy_runs(self, layer):
                # One measurement per physical run, so every run keeps its
                # own scan event (rows are then observable per extent).
                return [(.002, [2, 0, 0, 0, 0, 0])
                        for _ in getattr(layer, "pim_kv_runs", ((),))]

        class System:
            hetero_name = DeviceType.PIM
            devices = {"GPU": GPU(), "Acc": PIM()}
            model = Transformer({"name": "toy", "ndec": ndec, "num_heads": 4,
                                 "hdim": 16, "ff_scale": 4,
                                 "dtype": DataType.W16A16}, tensor_parallel=1)

        return System()

    def _with_history(self, workload, history_len):
        return Workload(workload.kind, tuple(
            replace(request, history_len=history_len)
            for request in workload.requests), workload.raw)

    def test_history_len_is_parsed_from_both_json_kinds(self):
        rag = [{"sample": 0, "seg_lens": [1, 2, 1],
                "seg_sha": ["s", "d", "q"],
                "seg_role": ["sys", "doc", "query"], "L": 4,
                "lout": 2, "history_len": 7}]
        path = ROOT / "tests/.history_workload.json"
        path.write_text(json.dumps(rag), encoding="utf-8")
        self.addCleanup(path.unlink)
        workload = load_workload(path)
        self.assertEqual([request.history_len for request in workload.requests], [7])
        # History is resident context, not input: it joins no segment sum.
        self.assertEqual(workload.requests[0].total_length, 4)
        self.assertEqual(workload_summary(workload)["total_history_tokens"], 7)

        supervisor = {"agents": [
            {"id": "sup", "tier": 0, "parent": None, "lout": 2,
             "segs": [{"role": "sys", "sha": "s", "len": 2}],
             "history_len": 5},
            {"id": "w0", "tier": 1, "parent": "sup", "lout": 1,
             "segs": [{"role": "parent_out", "sha": "o", "len": 2}]},
        ]}
        path.write_text(json.dumps(supervisor), encoding="utf-8")
        workload = load_workload(path)
        self.assertEqual({request.request_id: request.history_len
                          for request in workload.requests},
                         {"sup": 5, "w0": 0})
        self.assertEqual(workload_summary(workload)["total_history_tokens"], 5)

        rag[0]["history_len"] = -1
        path.write_text(json.dumps(rag), encoding="utf-8")
        with self.assertRaises(WorkloadValidationError):
            load_workload(path)

    def test_history_extends_split_prefill_and_decode_scans(self):
        workload = Workload("rag", (
            Request("r", 0, None, 2, (Segment("sys", "s", 4),), 4),), {})
        base = run_reuse_prefill(self._system(), workload,
                                 build_reuse_plan(workload, "epic"), pipe=True)
        hist_workload = self._with_history(workload, 3)
        hist = run_reuse_prefill(self._system(), hist_workload,
                                 build_reuse_plan(hist_workload, "epic"),
                                 pipe=True)
        self.assertEqual(base["history_rows"], 0)
        self.assertEqual(hist["history_rows"], 3)

        def events(report, name):
            return [event for event in report["events"] if event["name"] == name]

        # Nothing without reuse or history: a plain GPU prefill, no PIM scan.
        self.assertFalse(events(base, "pim_kv_scan_score_softmax_pv"))
        # With history the layer takes the split path: the GPU still attends
        # this turn's 4 fresh rows to each other while the PIM scans exactly
        # the 3 resident history rows -- QKV, KV link, and store stay at 4.
        prefill_scans = events(hist, "pim_kv_scan_score_softmax_pv")
        self.assertTrue(prefill_scans)
        self.assertTrue(all(event["rows"] == 3 for event in prefill_scans))
        self.assertTrue(all(event["rows"] == 4
                            for event in events(hist, "gpu_local_score")))
        for report in (base, hist):
            self.assertTrue(all(event["rows"] == 4
                                for event in events(report, "qkv")))
            self.assertTrue(all(event["rows"] == 4
                                for event in events(report, "kv_gpu_to_pim")))
        # Decode's first token scans prefill KV plus history: 4 + 3 rows,
        # summed over that query position's per-pool-run scan events.
        def first_token_scan_rows(report):
            scans = events(report, "decode_pim_kv_scan_score_softmax_pv")
            first = min(event["query_positions"][0] for event in scans)
            return sum(event["rows"] for event in scans
                       if event["query_positions"] == [first] and
                       event["transformer_layer"] == 0)

        self.assertEqual(first_token_scan_rows(base), 4)
        self.assertEqual(first_token_scan_rows(hist), 7)
        # One resident master-pool extent per request and layer.
        history_blocks = [block for block in hist["tlb"]["blocks"]
                          if block["fingerprint"].endswith("::history")]
        self.assertEqual(len(history_blocks), self._system().model.ndec)
        self.assertTrue(all(block["kind"] == "master" and
                            block["vector_count"] == 3
                            for block in history_blocks))
        self.assertGreater(hist["makespan_s"], base["makespan_s"])

    def test_history_is_scanned_even_in_full_recompute_layers(self):
        workload = Workload("rag", (
            Request("r0", 0, None, 2, (Segment("sys", "shared", 4),
                                       Segment("query", "q0", 2)), 6),
            Request("r1", 0, None, 2, (Segment("sys", "shared", 4),
                                       Segment("query", "q1", 2)), 6),), {})
        hist_workload = self._with_history(workload, 2)
        plan = build_reuse_plan(hist_workload, "cacheblend", .5, 7, (0,), (1,))
        report = run_reuse_prefill(self._system(), hist_workload, plan, pipe=True)
        full_layer_scans = [
            event for event in report["events"]
            if event["name"] == "pim_kv_scan_score_softmax_pv" and
            event["transformer_layer"] == 0]
        # The full-recompute layer rebuilds every segment row on the GPU but
        # still scans each agent's 2 resident history rows on the PIM.
        self.assertTrue(full_layer_scans)
        self.assertTrue(all(event["rows"] == 2 for event in full_layer_scans))
        self.assertTrue(all(
            event["rows"] == 6 for event in report["events"]
            if event["name"] == "qkv" and event["transformer_layer"] == 0))

    def test_history_widens_the_physical_no_reuse_scan(self):
        class GPU:
            def get_time_and_energy(self, layer):
                return .001, [1, 0, 0, 0, 0, 0]

        class PIM:
            peak_memory_bandwidth = 10**12
            softmax_peak_bandwidth = 10**12
            energy_table = {"mem": 1, "sram": 1}

            def __init__(self):
                self.prefill_shapes = []

            def get_time_and_energy(self, layer):
                if getattr(layer, "pim_kv_runs", None) is not None and layer.m > 1:
                    self.prefill_shapes.append((layer.m, layer.n,
                                                layer.pim_kv_runs))
                return .002, [2, 0, 0, 0, 0, 0]

        pim = PIM()

        class System:
            hetero_name = DeviceType.PIM
            devices = {"GPU": GPU(), "Acc": pim}
            model = Transformer({"name": "toy", "ndec": 2, "num_heads": 4,
                                 "hdim": 16, "ff_scale": 4,
                                 "dtype": DataType.W16A16}, tensor_parallel=1)

        workload = self._with_history(Workload("rag", (
            Request("r", 0, None, 1, (Segment("sys", "s", 4),), 4),), {}), 2)
        report = run_reuse_prefill(System(), workload,
                                   build_reuse_plan(workload, "no-reuse"),
                                   pipe=True)
        self.assertEqual(report["policy"], "no-reuse-physical")
        self.assertEqual(report["history_rows"], 2)
        # 4 queries against 4 fresh + 2 resident rows; the resident extent is
        # a second physical run, and the KV link/store still carry 4 rows.
        self.assertTrue(pim.prefill_shapes)
        self.assertTrue(all(shape == (4, 6, shape[2]) and len(shape[2]) == 2
                            for shape in pim.prefill_shapes))
        for name, rows in (("kv_gpu_to_pim", 4), ("dram_store_master", 4),
                           ("pim_kv_scan_score_softmax_pv", 6)):
            matching = [event for event in report["events"]
                        if event["name"] == name and
                        not event["name"].startswith("decode_")]
            self.assertTrue(matching)
            self.assertTrue(all(event["rows"] == rows for event in matching))

    def test_history_is_rejected_by_the_legacy_analytic_model(self):
        class FakeSystem:
            def simulate(self, batch, lin, lout, **kwargs):
                perf_ms = [0.0] * 20
                perf_ms[0] = float(lin)
                perf_ms[7] = float(batch)
                kwargs["perfs"].append([[], [], perf_ms, [2.0]])

        workload = self._with_history(Workload("rag", (
            Request("r", 0, None, 2, (Segment("sys", "s", 4),), 4),), {}), 8)
        with self.assertRaisesRegex(WorkloadValidationError, "history_len"):
            run_no_reuse_report(FakeSystem(), workload, pipe=False,
                                parallel_ff=False, power_constraint=False)

    def test_ablation_report_charges_history_in_time_and_memory(self):
        toy = {"name": "toy", "ndec": 4, "num_heads": 4, "hdim": 16,
               "dhead": 4, "ff_scale": 4, "gqa_size": 1,
               "dtype": DataType.W16A16}
        gpu = make_xpu_config(GPUType.A100a, num_gpu=1)["GPU"]
        workload = Workload("rag", (
            Request("r0", 0, None, 3, (
                Segment("sys", "shared-sys", 4), Segment("doc", "own-0", 4),
                Segment("query", "q0", 2)), 10),
            Request("r1", 0, None, 3, (
                Segment("sys", "shared-sys", 4), Segment("doc", "own-1", 4),
                Segment("query", "q1", 2)), 10),), {})

        def stub_runs(op):
            return [(1e-9 * run[2] * op.numOp, [1.0] * 6)
                    for run in op.pim_kv_runs]

        def run(wl):
            plan = build_reuse_plan(wl, "cacheblend", 0.5, 7, (0,), (1, 2, 3), 1)
            system = System(gpu, toy)
            system.hetero_name = DeviceType.PIM
            system.devices["Acc"].pim_type = SimpleNamespace(name="bank")
            system.devices["Acc"].get_time_and_energy_runs = stub_runs
            return run_ablation_report(
                system, wl, plan,
                resolve_config("A6", None, None, None, policy="cacheblend"),
                pipe=True, parallel_ff=True, power_constraint=False)

        base = run(workload)
        hist = run(self._with_history(workload, 16))
        self.assertEqual(base["memory"]["history_rows"], 0)
        self.assertEqual(hist["memory"]["history_rows"], 32)
        self.assertEqual(hist["tiers"][0]["history_rows_per_request"], 16)
        self.assertGreater(hist["prefill_s"], base["prefill_s"])
        self.assertGreater(hist["decode_s"], base["decode_s"])
        self.assertGreater(hist["memory"]["kv_gib"], base["memory"]["kv_gib"])


if __name__ == "__main__":
    unittest.main()
