"""AttAcc accounting: shared metadata has dependencies, no extra hardware cost."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.ablation import resolve_config
from src.cpp_eventcore import new_core
from src.model import Transformer
from src.type import DataType, DeviceType
from src.workload import Request, Segment, Workload, build_reuse_plan
import src.workload_runner as wr


class AttaccMetadataTests(unittest.TestCase):
    def _graph(self):
        events = []
        def add(device, duration, deps=()):
            return wr._cacheblend_event(events, layer=0, tier=0, request="r",
                name="accounting", device=device, rows=1,
                time_s=duration, energy=(7.,), deps=deps)
        gpu = add("GPU", 10.)
        link = add("LINK", 1.)
        add("DIE", 9., (gpu,))
        early = add("DIE", 9., (link,))
        plan = add("TLB", 9., (early,))
        add("PIM", 2., (plan,))
        return events

    def test_metadata_keeps_dependencies_without_serializing_other_requests(self):
        with patch.object(wr, "_EC", None):
            events = self._graph()
            for pipe in (False, True):
                with self.subTest(pipe=pipe):
                    scheduled = wr._schedule_cacheblend(events, pipe=pipe)
                    self.assertEqual(scheduled[2].end_s, 10.)
                    self.assertEqual(scheduled[3].end_s, 1. if pipe else 11.)
                    self.assertEqual(scheduled[4].end_s, 1. if pipe else 11.)
                    self.assertEqual(scheduled[5].end_s, 3. if pipe else 13.)
                    self.assertTrue(all(e.time_s == e.energy_nj == 0.
                        for e in scheduled if e.device in ("DIE", "TLB")))
                    self.assertEqual(scheduled[5].energy_nj, .007)
                    wr.validate_cacheblend_attacc_overlap_contract(scheduled, pipe=pipe)
                    finish, available = wr._schedule_cacheblend_incremental(
                        events, pipe=pipe, start_index=0, finish={}, availability={})
                    self.assertEqual(finish, {e.event_id: e.end_s for e in scheduled})
                    self.assertNotIn("DIE", available)
                    self.assertNotIn("TLB", available)

    def test_native_metadata_matches_python_without_reserving_resources(self):
        for pipe in (False, True):
            core = new_core(pipe)
            if core is None:
                self.skipTest("native event core unavailable")
            try:
                with patch.object(wr, "_EC", core):
                    events = self._graph()
                    native = wr._schedule_cacheblend(events, pipe=pipe)
                with patch.object(wr, "_EC", None):
                    python = wr._schedule_cacheblend(events, pipe=pipe)
                self.assertEqual([(e.start_s, e.end_s) for e in native],
                                 [(e.start_s, e.end_s) for e in python])
            finally:
                core.close()

    def test_all_hardware_rungs_exclude_metadata_and_keep_pim_work(self):
        class Device:
            peak_memory_bandwidth = 10**12
            num_hbm = 1
            # Deliberately no DIE bandwidth/SRAM-energy calibration.
            energy_table = {"mem": 1.}
            ramulator = SimpleNamespace(workers=2)

            def get_time_and_energy(self, op):
                return 2e-6, (4.,)

            def get_time_and_energy_runs(self, op):
                return [(2e-6, (4.,)) for _ in op.pim_kv_runs]

        workload = Workload("supervisor", (
            Request("a", 0, None, 2, (Segment("doc", "shared", 8),), 8),
            Request("b", 0, None, 2, (Segment("user", "private", 2),
                Segment("doc", "shared", 8, position_delta=2)), 10)), {})
        for rung in ("A1", "A3b", "A4c", "A4e", "A5", "A6"):
            for warm in (False, True):
                with self.subTest(rung=rung, warm=warm):
                    policy = "no-reuse" if rung == "A1" else "epic"
                    plan = build_reuse_plan(workload, policy,
                                           epic_prefix_recompute_tokens=2)
                    cfg = resolve_config(rung, None, None, None, policy=policy)
                    system = SimpleNamespace(hetero_name=DeviceType.PIM,
                        devices={"GPU": Device(), "Acc": Device()},
                        model=Transformer(dict(name="toy", ndec=1, num_heads=4,
                            hdim=512, ff_scale=4, dtype=DataType.W16A16), 1))
                    try:
                        report = wr.run_reuse_prefill(system, workload, plan,
                            pipe=True, warm=warm, cacheblend_batch_size=2,
                            pim_prefill_mode=cfg.prefill_attn,
                            pim_batch_command=cfg.pim_batch_command,
                            pim_pe_freq_ghz=cfg.pim_pe_freq_ghz,
                            kv_mapping=cfg.kv_mapping,
                            channel_placement=cfg.channel_placement)
                        metadata = [e for e in report["events"]
                                    if e["device"] in ("DIE", "TLB")]
                        self.assertTrue(metadata)
                        self.assertTrue(all(e["time_s"] == e["energy_nj"] == 0.
                                            for e in metadata))
                        self.assertFalse(any("die_rotate" in e["name"] or
                            "die_query_position_transform" in e["name"]
                            for e in report["events"]))
                        scans = [e for e in report["events"] if "pim_kv_scan" in e["name"]]
                        self.assertTrue(scans)
                        self.assertTrue(all(e["time_s"] > 0. and e["energy_nj"] > 0.
                                            for e in scans))
                        self.assertEqual(report["die_time_s_unoverlapped"], 0.)
                        self.assertEqual(report["energy_breakdown_nj"]["by_class"].get("TLB", 0.), 0.)
                    finally:
                        if wr._EC is not None:
                            wr._EC.close()
                            wr._EC = None


if __name__ == "__main__":
    unittest.main()
