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
        workload = load_workload(ROOT / "tests/fixtures/workload_2wikimqa_first8.json")
        self.assertEqual(workload.kind, "rag")
        self.assertTrue(all(r.total_length == sum(s.length for s in r.segments)
                            for r in workload.requests))
        self.assertEqual(workload_summary(workload)["tiers"].keys(), {"0"})

    def test_existing_supervisor_workload_is_valid(self):
        workload = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")
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
        rag = load_workload(ROOT / "tests/fixtures/workload_2wikimqa_first8.json")
        supervisor = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")
        self.assertGreater(build_reuse_plan(rag, "epic").reused_tokens, 0)
        plan = build_reuse_plan(supervisor, "cacheblend", 0.5, 7)
        self.assertGreater(plan.reused_tokens, 0)
        self.assertEqual(plan, build_reuse_plan(supervisor, "cacheblend", 0.5, 7))
        self.assertEqual(build_reuse_plan(rag, "no-reuse").reused_tokens, 0)

    def test_parent_output_owner_is_the_declared_parent(self):
        workload = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")
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

        workload = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")
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
        workload = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")
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
        workload = load_workload(ROOT / "tests/fixtures/workload_2wikimqa_first8.json")
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

        config = resolve_config(None, "gpu", "pim", "naive", policy="cacheblend",
                                channel_placement="single")
        profile = _batch_scan_profile(workload.requests, plan, 1, 10, 2, 32, config)
        # Tracked-channel naive layout (2026-08-25): every populated channel
        # is its own serialized single-channel pool; blocks that collide on
        # one channel queue there instead of streaming at full pool width.
        self.assertGreater(len(profile.pools), 1)
        self.assertTrue(all(run[4] == 1
                            for pool in profile.pools for run in pool))
        self.assertEqual(sum(run[2] for pool in profile.pools for run in pool),
                         12)

        config = resolve_config(None, "gpu", "pim", "master-diff", policy="cacheblend")
        profile = _batch_scan_profile(workload.requests, plan, 1, 10, 2, 32, config)
        self.assertEqual(len(profile.pools), 2)
        self.assertEqual({run[3] for run in profile.pools[0]}, {0})
        self.assertEqual({run[3] for run in profile.pools[1]}, {15})
        self.assertEqual(sum(run[2] for run in profile.pools[0]), 12)

    def test_ladder_couples_batching_and_abolishes_split(self):
        """A1-A6 of 2026-08-24: batching follows the rung, split is gone.

        Attention batching (the MQ command) is coupled to prefill-on-PIM:
        A1-A4 resolve to the legacy replicate command, A5/A6 to mq, and an
        explicit --pim-batch-command still overrides the rung.  The former
        GPU/PIM "split" prefill hybrid is no longer a mode at all.
        """
        a4 = resolve_config(None, "gpu", "pim", "master-diff", policy="cacheblend")
        self.assertEqual((a4.prefill_attn, a4.pim_batch_command),
                         ("gpu", "replicate"))
        a5 = resolve_config("A5", None, None, None, policy="cacheblend")
        self.assertEqual((a5.prefill_attn, a5.pim_batch_command),
                         ("pim", "mq"))
        # A5/A6 carry the provisional balance-point microarchitecture and
        # the prefill sweep follows the buffer's resident-Q capacity.
        # Ruling chenyi9 2026-08-26: MQ max n_cap = 8 -> 512 B buffer,
        # balance-point PE clock f* = 1/tCK = 1.3004 GHz (PC energy clamp
        # pins the n=8 interval at 8 tCK; ruling chenyi9 2026-08-27).
        self.assertEqual((a5.pim_pe_freq_ghz, a5.gemv_buffer_bytes,
                          a5.pim_prefill_query_batch), (1.3004, 512, 8))
        stock = resolve_config(None, "gpu", "pim", "master-diff", policy="cacheblend")
        self.assertEqual((stock.pim_pe_freq_ghz, stock.gemv_buffer_bytes),
                         (0.666, 512))
        a6 = resolve_config("A6", None, None, None, policy="cacheblend")
        self.assertEqual((a6.prefill_attn, a6.pim_batch_command),
                         ("dynamic", "mq"))
        override = resolve_config("A5", None, None, None, policy="cacheblend",
                                  pim_batch_command="replicate")
        self.assertEqual(override.pim_batch_command, "replicate")
        with self.assertRaises(WorkloadValidationError):
            resolve_config(None, "split", "pim", "master-diff",
                           policy="cacheblend")

    def test_dynamic_prefill_places_each_class_on_the_cheaper_side(self):
        """A6 = A5 + the Fugue placement rule, priced per layer class.

        The rule charges each reuse-carrying class min(bank path, xPU path),
        so A6 can never price prefill above A5 (the forced-bank rung).  With
        an absurdly slow PIM stub the rule commits the classes to the GPU
        (readback + local attention block), so A6 strictly beats A5 and the
        breakdown shows the GPU pieces.
        """
        toy, gpu, workload = self._ablation_toy()
        plan = build_reuse_plan(workload, "cacheblend", 0.5, 7, (0,),
                                (1, 2, 3), 1)

        def run(preset, scan_seconds_per_row):
            def stub_runs(op):
                return [(scan_seconds_per_row * run[2] * op.numOp, [1.0] * 6)
                        for run in op.pim_kv_runs]
            system = System(gpu, toy)
            system.hetero_name = DeviceType.PIM
            system.devices["Acc"].pim_type = SimpleNamespace(name="bank")
            system.devices["Acc"].get_time_and_energy_runs = stub_runs
            return run_ablation_report(
                system, workload, plan,
                resolve_config(preset, None, None, None, policy="cacheblend"),
                pipe=True, parallel_ff=True, power_constraint=False)

        fast_a5, fast_a6 = run("A5", 1e-9), run("A6", 1e-9)
        self.assertLessEqual(fast_a6["prefill_s"], fast_a5["prefill_s"])

        slow_a5, slow_a6 = run("A5", 1.0), run("A6", 1.0)
        self.assertLess(slow_a6["prefill_s"], slow_a5["prefill_s"])
        slow_breakdown = slow_a6["tiers"][0]["prefill_breakdown_s"]
        self.assertIn("gpu_dynamic_score", slow_breakdown)
        self.assertIn("link_kv_pim_to_gpu", slow_breakdown)

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

    def test_memory_report_stores_shared_chunks_exactly_once(self):
        """Capacity accounting: the owner's copy must not be counted twice.

        Two agents share one 100-token chunk under promptcache (zero diff
        rows, so the arithmetic is exact): the store holds one copy of the
        chunk plus both agents' private tails and outputs -- the consumer's
        avoided copy is the whole saving (audit 2026-08-25: adding the
        stored-once shared rows on top of the private rows double-counted
        the owner copy and understated the saving).
        """
        toy = {"name": "toy", "ndec": 2, "num_heads": 4, "hdim": 16,
               "dhead": 4, "ff_scale": 4, "gqa_size": 1,
               "dtype": DataType.W16A16}
        gpu = make_xpu_config(GPUType.A100a, num_gpu=1)["GPU"]
        workload = Workload("rag", (
            Request("r0", 0, None, 4, (Segment("sys", "shared", 100),
                                       Segment("query", "q0", 20)), 120),
            Request("r1", 0, None, 4, (Segment("sys", "shared", 100),
                                       Segment("query", "q1", 20)), 120),), {})
        plan = build_reuse_plan(workload, "promptcache")
        system = System(gpu, toy)
        system.hetero_name = DeviceType.PIM
        from src.ablation import _memory_report
        report = _memory_report(system, workload, plan,
                                resolve_config(None, "gpu", "pim", "master-diff",
                                               policy="promptcache"))
        # stored = 240 total - 100 avoided consumer copy + 8 outputs.
        self.assertEqual(round(report["kv_rows"]), 148)
        self.assertAlmostEqual(report["kv_bytes_vs_no_reuse"],
                               148 / 248, places=9)

    def test_software_upstream_policy_family_enrichment(self):
        """promptcache / cachecraft / cachetune extend the two anchor policies.

        promptcache reuses chunks verbatim (zero recompute rows); cachecraft
        sizes a per-chunk boundary prefix from the context overlap between
        consumer and owner; cachetune is CacheBlend-shaped ratio recompute
        whose rows are selected offline, so it must not carry full-recompute
        selection layers.
        """
        workload = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")

        zero = build_reuse_plan(workload, "promptcache")
        self.assertTrue(zero.reusable)
        self.assertTrue(all(not decision.epic_prefix_rows
                            for decision in zero.reusable))

        craft = build_reuse_plan(workload, "cachecraft", cachecraft_alpha=0.25)
        shifted_prefixes = [len(decision.epic_prefix_rows)
                            for decision in craft.reusable
                            if decision.epic_prefix_rows]
        self.assertTrue(shifted_prefixes)
        validate_reuse_plan(workload, craft, model_layers=3)
        # A larger alpha never shrinks any chunk's recompute prefix.
        craft_hi = build_reuse_plan(workload, "cachecraft", cachecraft_alpha=1.0)
        by_key = {(d.request_id, d.segment_index): len(d.epic_prefix_rows)
                  for d in craft.reusable}
        for decision in craft_hi.reusable:
            key = (decision.request_id, decision.segment_index)
            self.assertGreaterEqual(len(decision.epic_prefix_rows), by_key[key])

        tune = build_reuse_plan(workload, "cachetune", 0.2, 7, (), (0, 1, 2))
        validate_reuse_plan(workload, tune, model_layers=3)
        self.assertEqual(set(tune.cacheblend_partial_rows), {0, 1, 2})
        with self.assertRaisesRegex(WorkloadValidationError, "offline"):
            build_reuse_plan(workload, "cachetune", 0.2, 7, (0,), (1, 2))

    def test_reuse_structure_checker_covers_layers_and_rows(self):
        workload = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")
        cacheblend = build_reuse_plan(
            workload, "cacheblend", .25, 7, (0, 1), (2,))
        validate_reuse_plan(workload, cacheblend, model_layers=3)
        with self.assertRaisesRegex(WorkloadValidationError, "cover the model"):
            validate_reuse_plan(workload, cacheblend, model_layers=4)
        epic = build_reuse_plan(workload, "epic", epic_prefix_recompute_tokens=2)
        validate_reuse_plan(workload, epic, model_layers=3)

    def test_prefill_placement_menu_emits_matching_events(self):
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

        workload = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")
        plan = build_reuse_plan(workload, "epic", epic_prefix_recompute_tokens=1)

        def prefill_names(report):
            return {event["name"] for event in report["events"]
                    if not event["name"].startswith("decode_")}

        # "pim" (bank-whole, the A5 rung): everything scans in the banks.
        pim = run_reuse_prefill(System(), workload, plan, pipe=True,
                                pim_prefill_mode="pim")
        self.assertEqual(pim["policy"], "epic")
        self.assertGreater(pim["link_bytes"], 0)
        names = prefill_names(pim)
        for name in ("q_gpu_to_pim", "ctx_pim_to_gpu", "kv_gpu_to_pim",
                     "dram_store_diff_and_live", "die_score_assembly"):
            self.assertIn(name, names)
        self.assertNotIn("gpu_prefill_score", names)
        self.assertNotIn("kv_pim_to_gpu", names)

        # "gpu" (software prefill, the A1-A4 rung): readback + local block,
        # no PIM prefill scan; the fresh K/V still lands for PIM decode.
        gpu = run_reuse_prefill(System(), workload, plan, pipe=True,
                                pim_prefill_mode="gpu")
        names = prefill_names(gpu)
        for name in ("kv_pim_to_gpu", "gpu_prefill_score",
                     "gpu_prefill_softmax", "gpu_prefill_context",
                     "kv_gpu_to_pim", "dram_store_diff_and_live"):
            self.assertIn(name, names)
        self.assertNotIn("pim_kv_scan_score_softmax_pv", names)
        self.assertNotIn("die_score_assembly", names)

        # "dynamic" (default, Fugue/A6): a side is chosen and reported per
        # request, and the emitted events belong to the chosen side.
        dyn = run_reuse_prefill(System(), workload, plan, pipe=True)
        self.assertEqual(dyn["pim_prefill_mode"], "dynamic")
        sides = dyn["pim_prefill_sides"]
        self.assertTrue(sides)
        self.assertTrue(set(sides.values()) <= {"gpu", "pim"})
        for request_id, side in sides.items():
            request_names = {event["name"] for event in dyn["events"]
                             if event["request"] == request_id and
                             not event["name"].startswith("decode_")}
            if side == "pim":
                self.assertIn("die_score_assembly", request_names)
                self.assertNotIn("gpu_prefill_score", request_names)
            else:
                self.assertIn("gpu_prefill_score", request_names)
                self.assertNotIn("die_score_assembly", request_names)

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

        workload = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")
        plan = build_reuse_plan(workload, "cacheblend", .1, 7, (0, 1), (2,))
        report = run_reuse_prefill(System(), workload, plan, pipe=True,
                                   pim_prefill_mode="pim")
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
        self.assertNotIn("die_query_position_transform", names)
        self.assertLess(names.index("q_gpu_to_pim"),
                        names.index("tlb_lookup_and_bank_plan"))
        self.assertLess(names.index("tlb_lookup_and_bank_plan"),
                        names.index("pim_kv_scan_score_softmax_pv"))
        self.assertLess(names.index("pim_kv_scan_score_softmax_pv"),
                        names.index("die_score_assembly"))
        self.assertLess(names.index("die_score_assembly"),
                        names.index("ctx_pim_to_gpu"))
        self.assertEqual(report["tlb"]["channel_sets"],
                         {"master": list(range(15)), "diff": [15]})
        blocks = {block["id"]: block for block in report["tlb"]["blocks"]}
        self.assertTrue(blocks)
        for block in blocks.values():
            self.assertIn(block["kind"], ("master", "diff"))
            expected_channels = (range(15) if block["kind"] == "master"
                                 else range(15, 16))
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
        # Per-channel placement (chenyi9 2026-08-29): the Ramulator run is now
        # a fixed per-chunk probe, so a multi-block scan retains its extents by
        # SPREADING over more than one channel event rather than one coalesced
        # multi-run trace.
        scan_events = [event for event in report["events"]
                       if event["name"] == "pim_kv_scan_score_softmax_pv"
                       and event["request"] == "t1w0"]
        self.assertTrue(scan_events)
        self.assertGreater(
            len({event["device"] for event in scan_events}), 1,
            "a multi-block CacheBlend scan must spread over multiple channels")
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
        # The same DAG (same prefill side) serialized: a "dynamic" serial run
        # would pick the GPU for the fresh supervisor and no longer compare.
        serial = run_reuse_prefill(System(), workload, plan, pipe=False,
                                   pim_prefill_mode="pim")
        # pipe=False is serial between macro events, but the per-channel
        # lanes of one PIM KV scan are a parallel phase inside one.
        self.assertEqual(
            serial["overlap_validation"]["contract"],
            "AttAcc serial macro-events with parallel PIM channel phases")
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

        workload = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")
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
        with self.assertRaisesRegex(WorkloadValidationError, "DIE rotation"):
            run_reuse_prefill(System(), workload, plan, pipe=True,
                             cacheblend_batch_size=2, cacheblend_rotate_mode="die")
        bank_report = run_reuse_prefill(System(), workload, plan, pipe=True,
                                        cacheblend_batch_size=2,
                                        cacheblend_rotate_mode="bank")
        self.assertEqual(bank_report["cacheblend_rotate_mode"], "bank")
        self.assertFalse(any("die_rotate" in event["name"] or
                             "die_query_position_transform" in event["name"]
                             for event in report["events"]))
        self.assertTrue(any(event["name"] == "decode_bank_rotate_q_local"
                            for event in bank_report["events"]))
        self.assertGreater(report["link_bytes"], bank_report["link_bytes"])

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
            report = run_reuse_prefill(System(), workload, plan, pipe=True,
                                       pim_prefill_mode="pim")
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

        def channel_of(device):                    # "PIM:pool{c}-{c}" -> c
            return int(device.split("pool")[1].split("-")[0])

        def split_pools(events):
            master = [event for event in events if event["device"].startswith(
                "PIM:pool") and channel_of(event["device"]) < 15]
            diff = [event for event in events
                    if event["device"] == "PIM:pool15-15"]
            return master, diff

        plan, report, corrected, scans = decode_scans(0.25)
        self.assertGreater(corrected, 0)
        master, diff = split_pools(scans)
        # The whole 24-row context is streamed from the master channels
        # (shadowed rows included); only corrected rows come from the diff
        # channel (chenyi9 2026-08-29: master rows now spread over channels
        # 0..14, corrections on channel 15).
        # rows/masked_rows on a PIM pool event are HEAD-FOLDED since
        # 2026-09-03: the event is one CHANNEL's scan and that channel
        # holds every head it serves, which is what its time_s has always
        # measured.  heads_per_hbm is 4 in this fixture, so each figure
        # below is 4x the per-head count it used to be.
        self.assertEqual(sum(event["rows"] for event in master), 4 * 24)
        self.assertEqual(sum(event["masked_rows"] for event in master),
                         4 * corrected)
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff[0]["rows"], 4 * corrected)
        self.assertEqual(diff[0]["masked_rows"], 0)
        # Fragmentation-free: exactly as many master channels as with no
        # correction (corrections go to the diff channel, never the master).
        _, _, _, baseline = decode_scans(0.0)
        base_master, _ = split_pools(baseline)
        self.assertEqual(len(master), len(base_master))
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
                self.assertEqual(location["channel_base"], 15)
            else:
                self.assertEqual(location["key_address"], owner["key_address"])
        # Bank-whole prefill scans of a partial layer stream the shadowed
        # master rows read-masked plus the freshly landed rows (corrected
        # rows land in the diff pool, so diff-pool runs may appear too).
        prefill = [event for event in report["events"]
                   if event["request"] == "r1" and event["transformer_layer"] == 1
                   and event["name"] == "pim_kv_scan_score_softmax_pv"]
        self.assertTrue(prefill)
        master_prefill, _ = split_pools(prefill)
        self.assertTrue(master_prefill)
        self.assertTrue(any(event["masked_rows"] > 0 for event in master_prefill))

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
        self.assertTrue(all(b["channel_base"] == 0 and b["channel_count"] == 15
                            for b in master))
        self.assertEqual({b["channel_offset"] for b in master}, {0, 1})
        spilled = [b for b in master if b["channel_offset"] == 1]
        self.assertEqual(len(spilled), 6)
        self.assertEqual(int(spilled[0]["key_base"], 0), 1 * _HBM_CHANNEL_BYTES)
        diff = [b for b in blocks if b["kind"] == "diff"]
        self.assertEqual(int(diff[0]["key_base"], 0) // _HBM_CHANNEL_BYTES, 15)


class MQBatchCommandTests(unittest.TestCase):
    """MQ-MAC batch command (PLAN_mq_command.md): trace shape and timing knobs."""

    def test_interval_is_preset_floor_or_pe_bound(self):
        from src.ramulator_wrapper import (MQ_POWER_BUDGET_W,
                                           mq_interval_cycles, mq_pe_power_w,
                                           mq_query_capacity)
        # 2026-08-27 model revision (R19, RTL-backed): under PC the floor is
        # a per-window ENERGY budget -- interval = max(floor, PE term,
        # ceil(6*(E_col+n*E_op)/E6)).  n=1 keeps the plain 6-tCK floor.
        self.assertEqual(mq_interval_cycles(1, True, 1.3), 6)
        # n=4 already overruns the window budget by ~10% -> 7 tCK.
        self.assertEqual(mq_interval_cycles(4, True, 1.3), 7)
        # Flat 1.3 GHz misses one MAC/tCK by 0.03% -> PE term 9; the
        # balance-point preset 1.3004 GHz (= 1/tCK) lands the clamp's 8.
        self.assertEqual(mq_interval_cycles(8, True, 1.3), 9)
        self.assertEqual(mq_interval_cycles(8, True, 1.3004), 8)
        # The clamp is frequency-independent: a 3 GHz PE stays at 8 tCK.
        self.assertEqual(mq_interval_cycles(8, True, 3.0), 8)
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
        # ramulator2/trace_gen/ exists once set_pim_ramulator.sh has copied it
        # from pim_ramulator_src/; before provisioning, use the source copy.
        generator = ROOT / "ramulator2" / "trace_gen" / "gen_trace_attacc_bank.py"
        if not generator.exists():
            generator = ROOT / "pim_ramulator_src" / "trace_gen" / "gen_trace_attacc_bank.py"
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
        # ramulator2/trace_gen/ exists once set_pim_ramulator.sh has copied it
        # from pim_ramulator_src/; before provisioning, use the source copy.
        generator = ROOT / "ramulator2" / "trace_gen" / "gen_trace_attacc_bank.py"
        if not generator.exists():
            generator = ROOT / "pim_ramulator_src" / "trace_gen" / "gen_trace_attacc_bank.py"
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

        workload = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")
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

        workload = load_workload(ROOT / "tests/fixtures/workload_relay_s400w4t1.json")
        plan = build_reuse_plan(workload, "cacheblend", .1, 7, (0, 1), (2,))
        report = run_reuse_prefill(System(), workload, plan, pipe=True,
                                   cacheblend_batch_size=4,
                                   pim_batch_command="mq",
                                   pim_prefill_mode="pim")
        self.assertEqual(report["pim_prefill_mode"], "pim")
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
        # dram_store of this layer via its address-plan dependencies.
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
        # Per-channel placement (chenyi9 2026-08-29): a worker's >500-row
        # bank-whole context is the SUM over its scan's channel events at one
        # query position, not one coalesced run.
        totals = {}
        for event in worker_scans:
            key = (event["request"], tuple(event["query_positions"]))
            totals[key] = totals.get(key, 0) + event["rows"]
        self.assertTrue(totals)
        self.assertTrue(all(total > 500 for total in totals.values()))
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

    def test_history_extends_prefill_and_decode_scans(self):
        workload = Workload("rag", (
            Request("r", 0, None, 2, (Segment("sys", "s", 4),), 4),), {})
        base = run_reuse_prefill(self._system(), workload,
                                 build_reuse_plan(workload, "epic"), pipe=True)
        hist_workload = self._with_history(workload, 3)
        hist = run_reuse_prefill(self._system(), hist_workload,
                                 build_reuse_plan(hist_workload, "epic"),
                                 pipe=True, pim_prefill_mode="pim")
        self.assertEqual(base["history_rows"], 0)
        self.assertEqual(hist["history_rows"], 3)

        def events(report, name):
            return [event for event in report["events"] if event["name"] == name]

        # Nothing without reuse or history: a plain GPU prefill, no PIM scan.
        self.assertFalse(events(base, "pim_kv_scan_score_softmax_pv"))
        # With history the layer is reusable and, on the "pim" rung, the
        # bank-whole scan covers the 3 resident history rows PLUS this
        # turn's 4 freshly landed rows -- QKV, KV link, and store stay at 4.
        prefill_scans = events(hist, "pim_kv_scan_score_softmax_pv")
        self.assertTrue(prefill_scans)
        # A sweep's scan is split into one event per contiguous physical run
        # (the resident history extent and the freshly landed extent), so the
        # 7-row coverage shows up as a per-sweep row SUM.
        sweeps = {}
        for event in prefill_scans:
            key = (event["transformer_layer"], tuple(event["query_positions"]))
            sweeps[key] = sweeps.get(key, 0) + event["rows"]
        self.assertTrue(sweeps)
        # rows/masked_rows on a PIM pool event are HEAD-FOLDED since
        # 2026-09-03: the event is one CHANNEL's scan and that channel
        # holds every head it serves, which is what its time_s has always
        # measured.  heads_per_hbm is 4 in this fixture, so each figure
        # below is 4x the per-head count it used to be.
        self.assertTrue(all(total == 4 * 7 for total in sweeps.values()))
        self.assertFalse(events(hist, "gpu_prefill_score"))
        # On the "gpu" rung the same history comes back over the link and
        # the GPU runs the block itself -- no PIM prefill scan at all.
        hist_gpu = run_reuse_prefill(self._system(), hist_workload,
                                     build_reuse_plan(hist_workload, "epic"),
                                     pipe=True, pim_prefill_mode="gpu")
        readbacks = events(hist_gpu, "kv_pim_to_gpu")
        self.assertTrue(readbacks)
        self.assertTrue(all(event["rows"] == 3 for event in readbacks))
        self.assertTrue(all(event["rows"] == 4
                            for event in events(hist_gpu, "gpu_prefill_score")))
        self.assertFalse(events(hist_gpu, "pim_kv_scan_score_softmax_pv"))
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

        # Head-folded since 2026-09-03 (see the note above): 4 x the per-head
        # 4 and 7 rows.  The qkv / kv_gpu_to_pim events above are NOT pool
        # events and keep their per-head counts.
        self.assertEqual(first_token_scan_rows(base), 4 * 4)
        self.assertEqual(first_token_scan_rows(hist), 4 * 7)
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
        report = run_reuse_prefill(self._system(), hist_workload, plan,
                                   pipe=True, pim_prefill_mode="pim")
        full_layer_scans = [
            event for event in report["events"]
            if event["name"] == "pim_kv_scan_score_softmax_pv" and
            event["transformer_layer"] == 0]
        # Even a full-recompute layer keeps the agent's 2 resident history
        # rows; on the "pim" rung its rebuilt 6 rows land first and the
        # bank-whole scan covers history + fresh = 8 rows.
        self.assertTrue(full_layer_scans)
        sweeps = {}
        for event in full_layer_scans:
            key = (event["request"], tuple(event["query_positions"]))
            sweeps[key] = sweeps.get(key, 0) + event["rows"]
        # rows/masked_rows on a PIM pool event are HEAD-FOLDED since
        # 2026-09-03: the event is one CHANNEL's scan and that channel
        # holds every head it serves, which is what its time_s has always
        # measured.  heads_per_hbm is 4 in this fixture, so each figure
        # below is 4x the per-head count it used to be.
        self.assertTrue(all(total == 4 * 8 for total in sweeps.values()))
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

    def _toy_system(self):
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
            model = Transformer({"name": "toy", "ndec": 2, "num_heads": 4,
                                 "hdim": 16, "ff_scale": 4,
                                 "dtype": DataType.W16A16}, tensor_parallel=1)

        return System()

    @staticmethod
    def _prefill_names(report, request_id=None):
        return {event["name"] for event in report["events"]
                if not event["name"].startswith("decode_") and
                (request_id is None or event["request"] == request_id)}

    def test_fresh_prefill_follows_the_rung_prefill_side(self):
        """F04 (2026-09-05): a request that reuses nothing used to be sent
        to the GPU whatever the rung, so A5 never put a fresh prefill in the
        banks and A6's chooser never saw one."""
        workload = Workload("rag", (
            Request("r", 0, None, 2, (Segment("sys", "s", 4),), 4),), {})
        plan = build_reuse_plan(workload, "epic", epic_prefix_recompute_tokens=1)

        gpu = run_reuse_prefill(self._toy_system(), workload, plan, pipe=True,
                                pim_prefill_mode="gpu")
        names = self._prefill_names(gpu)
        self.assertIn("gpu_prefill_score", names)
        self.assertNotIn("pim_kv_scan_score_softmax_pv", names)
        self.assertEqual(gpu["prefill_attention_rows"]["pim"], 0)

        pim = run_reuse_prefill(self._toy_system(), workload, plan, pipe=True,
                                pim_prefill_mode="pim")
        names = self._prefill_names(pim)
        self.assertIn("pim_kv_scan_score_softmax_pv", names)
        self.assertIn("die_score_assembly", names)
        self.assertNotIn("gpu_prefill_score", names)
        self.assertNotIn("kv_pim_to_gpu", names)          # nothing resident
        self.assertEqual(pim["prefill_attention_rows"]["gpu"], 0)

        dyn = run_reuse_prefill(self._toy_system(), workload, plan, pipe=True)
        self.assertEqual(dyn["pim_prefill_mode"], "dynamic")
        self.assertIn("r", dyn["pim_prefill_sides"])
        side = dyn["pim_prefill_sides"]["r"]
        names = self._prefill_names(dyn)
        self.assertEqual("die_score_assembly" in names, side == "pim")
        self.assertEqual("gpu_prefill_score" in names, side == "gpu")

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
