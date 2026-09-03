"""Unit tests for the head-aware channel-placement load model (A3..A4b).

These pin the per-physical-channel load each placement policy produces for one
decode scan on one HBM.  ``max(load)`` is proportional to the scan time; the
ladder is single (A3) -> slice (A3b) -> master-diff-slice (A4) ->
master-diff-table (A4b).
"""
import unittest

from types import SimpleNamespace

from src.workload import WorkloadValidationError
from src.workload_runner import (_layout_channel_loads, _layout_scan_max_load,
                                  _HBM_CHANNELS, _MASTER_CHANNELS_DEFAULT,
                                  _STRIPE_UNIT_ROWS, _layout_policy,
                                  _hbm_stacks_local, _heads_per_hbm,
                                  _read_extents, _GEN_ROW_BYTES,
                                  _GEN_BYTES_PER_TOKEN,
                                  _striped_append_channel_extents,
                                  _stream_unit_rows,
                                  _striped_append_channel_rows,
                                  placement_degeneracy_warning)


class SingleLayoutTest(unittest.TestCase):
    """A3: head h -> one channel (h % 16); every chunk of the head piles there."""

    def test_one_head_all_chunks_on_one_channel(self):
        loads = _layout_channel_loads("single", master_chunks=10, diff_chunks=2,
                                      heads_per_hbm=1)
        self.assertEqual(len(loads), _HBM_CHANNELS)
        self.assertEqual(loads[0], 12)                 # 10 master + 2 diff
        self.assertEqual(sum(loads[1:]), 0)            # 15 channels idle
        self.assertEqual(_layout_scan_max_load("single", 10, 2, 1), 12)

    def test_heads_take_distinct_channels(self):
        loads = _layout_channel_loads("single", 5, 0, heads_per_hbm=8)
        self.assertEqual([loads[c] for c in range(8)], [5] * 8)
        self.assertEqual(sum(loads[8:]), 0)
        self.assertEqual(max(loads), 5)                # heads run concurrently

    def test_more_heads_than_channels_collide(self):
        loads = _layout_channel_loads("single", 1, 0, heads_per_hbm=20)
        # heads 0..19 -> channel h % 16; channels 0..3 hold two heads.
        self.assertEqual([loads[c] for c in range(4)], [2, 2, 2, 2])
        self.assertEqual([loads[c] for c in range(4, 16)], [1] * 12)


class SliceLayoutTest(unittest.TestCase):
    """A3b: head h -> a slice of 16//H channels; the head's chunks round-robin."""

    def test_one_head_spreads_over_all_channels(self):
        self.assertEqual(_layout_scan_max_load("slice", 16, 0, heads_per_hbm=1), 1)
        self.assertEqual(_layout_scan_max_load("slice", 17, 0, heads_per_hbm=1), 2)

    def test_two_heads_eight_channels_each(self):
        loads = _layout_channel_loads("slice", 16, 0, heads_per_hbm=2)
        self.assertEqual([loads[c] for c in range(8)], [2] * 8)     # head 0: ch0-7
        self.assertEqual([loads[c] for c in range(8, 16)], [2] * 8)  # head 1: ch8-15
        self.assertEqual(max(loads), 2)

    def test_eight_heads_two_channels_each(self):
        # scenario 3: 8 heads on one HBM -> 2 channels per head.
        loads = _layout_channel_loads("slice", 4, 0, heads_per_hbm=8)
        self.assertEqual(max(loads), 2)                # 4 chunks over 2 channels
        for h in range(8):
            self.assertEqual(loads[2 * h] + loads[2 * h + 1], 4)


class MasterDiffSliceLayoutTest(unittest.TestCase):
    """A4: master pool ch0..14 head-sliced; corrections on the diff channel 15."""

    def test_one_head_master_over_15_diff_on_ch15(self):
        loads = _layout_channel_loads("master-diff-slice", 15, 1, heads_per_hbm=1)
        self.assertEqual([loads[c] for c in range(15)], [1] * 15)
        self.assertEqual(loads[_MASTER_CHANNELS_DEFAULT], 1)        # ch15 = diff
        self.assertEqual(max(loads), 1)

    def test_diff_channel_can_dominate(self):
        # few master chunks but many corrections all land on the one diff channel.
        self.assertEqual(
            _layout_scan_max_load("master-diff-slice", 15, 10, heads_per_hbm=1), 10)

    def test_two_heads_fixed_slices_leave_slack(self):
        # 15 master channels / 2 heads -> 7 each, ch14 unused; 15 chunks over 7.
        loads = _layout_channel_loads("master-diff-slice", 15, 0, heads_per_hbm=2)
        self.assertEqual(max(loads), 3)                # ceil(15/7) on the busy ch


class MasterDiffTableLayoutTest(unittest.TestCase):
    """A4b: master/diff split + global co-read table (balanced over 15 master)."""

    def test_one_head_matches_slice(self):
        # with a single head the table and the slice coincide (all 15 channels).
        self.assertEqual(
            _layout_scan_max_load("master-diff-table", 15, 1, heads_per_hbm=1), 1)

    def test_table_beats_fixed_slice_for_two_heads(self):
        # 2 heads x 15 master chunks: fixed slices give 3, the global table gives 2.
        slice_load = _layout_scan_max_load("master-diff-slice", 15, 0, heads_per_hbm=2)
        table_load = _layout_scan_max_load("master-diff-table", 15, 0, heads_per_hbm=2)
        self.assertEqual(slice_load, 3)
        self.assertEqual(table_load, 2)                # 30 chunks / 15 channels
        self.assertLess(table_load, slice_load)

    def test_diff_scales_with_heads(self):
        loads = _layout_channel_loads("master-diff-table", 15, 3, heads_per_hbm=2)
        self.assertEqual(loads[_MASTER_CHANNELS_DEFAULT], 6)       # 2 heads x 3 diff
        self.assertEqual(max(loads[:15]), 2)                       # 30 master / 15


class LadderMonotonicityTest(unittest.TestCase):
    """Each rung should be no worse than the previous on the shared workload."""

    def test_single_worst_then_improving(self):
        C, H = 30, 2
        single = _layout_scan_max_load("single", C, 0, H)
        sl = _layout_scan_max_load("slice", C, 0, H)
        md_slice = _layout_scan_max_load("master-diff-slice", C, 0, H)
        md_table = _layout_scan_max_load("master-diff-table", C, 0, H)
        self.assertGreaterEqual(single, sl)
        self.assertGreaterEqual(sl, md_table)
        self.assertGreaterEqual(md_slice, md_table)

    def test_unknown_policy_raises(self):
        with self.assertRaises(WorkloadValidationError):
            _layout_channel_loads("bogus", 1, 0, 1)




class HeadsPerHBMTest(unittest.TestCase):
    """--num-hbm is the WHOLE system's stack count (fixed 2026-09-03).

    The old code divided the KV heads by tensor parallelism but handed every
    GPU the full --num-hbm, so GPT-175B read as 12 local heads over 40 stacks
    = 1 per stack with 28 stacks never touched.  Heads and stacks must be
    split the same way, and the result rounds UP because the scan time is the
    busiest stack's.
    """

    # (model, KV heads, ngpu, --num-hbm, heads on the busiest stack)
    SWEEP = [("LLAMA-7B", 32, 1, 1, 32), ("LLAMA3-8B", 8, 1, 1, 8),
             ("GPT-13B", 40, 2, 10, 4), ("LLAMA-33B", 52, 2, 10, 6),
             ("LLAMA-65B", 64, 8, 40, 2), ("GPT-175B", 96, 8, 40, 3)]

    def test_sweep_models_match_the_configured_intent(self):
        for name, kv_total, ngpu, num_hbm, expected in self.SWEEP:
            kv_local = max(1, kv_total // ngpu)
            self.assertEqual(_heads_per_hbm(kv_local, num_hbm, ngpu), expected,
                             name)

    def test_equals_the_global_ratio(self):
        # splitting per GPU and dividing globally are the same thing whenever
        # ngpu divides num_hbm -- which every sweep config does.
        for name, kv_total, ngpu, num_hbm, _ in self.SWEEP:
            kv_local = max(1, kv_total // ngpu)
            self.assertEqual(_heads_per_hbm(kv_local, num_hbm, ngpu),
                             -(-kv_total // num_hbm), name)

    def test_stacks_split_per_gpu_and_floor(self):
        self.assertEqual(_hbm_stacks_local(40, 8), 5)
        self.assertEqual(_hbm_stacks_local(10, 2), 5)
        self.assertEqual(_hbm_stacks_local(1, 1), 1)
        # a GPU never gets more stacks than the machine has, and a
        # non-dividing split floors -- the leftovers do not lighten anyone.
        self.assertEqual(_hbm_stacks_local(10, 4), 2)
        self.assertEqual(_hbm_stacks_local(3, 8), 1)

    def test_rounds_up_to_the_busiest_stack(self):
        # 96 heads over 40 stacks: some hold 3, some 2 -- price 3.
        self.assertEqual(_heads_per_hbm(12, 40, 8), 3)
        # exact division stays exact
        self.assertEqual(_heads_per_hbm(20, 10, 2), 4)

    def test_gpt175b_no_longer_leaves_stacks_idle(self):
        kv_local, num_hbm, ngpu = 12, 40, 8
        h = _heads_per_hbm(kv_local, num_hbm, ngpu)
        stacks = _hbm_stacks_local(num_hbm, ngpu)
        used = -(-kv_local // h)                  # num_hbm_used in the scan
        self.assertEqual((h, stacks, used), (3, 5, 4))
        self.assertLessEqual(used, stacks)
        # the old reading: 1 head per stack, 12 of 40 used, 28 idle.
        self.assertEqual(-(-kv_local // _heads_per_hbm(kv_local, num_hbm, 1)), 12)

    def test_single_gpu_models_are_unchanged_by_the_fix(self):
        for kv_local, num_hbm in ((32, 1), (8, 1)):
            self.assertEqual(_heads_per_hbm(kv_local, num_hbm, 1),
                             _heads_per_hbm(kv_local, num_hbm))



class RealExtentTest(unittest.TestCase):
    """A scan reaches Ramulator as the extents it really touches (2026-09-03).

    AttAcc's MAC_AB names 16 partitions x 4 banks at once, so a token costs
    4 B of address space and one 1024-B DRAM row holds exactly 256 tokens.  A
    k-row repair therefore burns a whole row for k tokens, and every further
    repair burns another -- unless the diff pool packs them together.  These
    pin that the model now says so.
    """

    @staticmethod
    def _loc(owner, fingerprint, kind):
        return SimpleNamespace(owner=owner, fingerprint=fingerprint, kind=kind)

    def _reads(self, segments, repairs, repair_rows=8, shadow=False):
        reads = []
        for index in range(segments):
            name = "seg{}".format(index)
            if index < repairs:
                reads += [self._loc("consumer", name, "diff")] * repair_rows
                reads += [self._loc("owner", name, "master")] * (
                    256 if shadow else 256 - repair_rows)
            else:
                reads += [self._loc("owner", name, "master")] * 256
        return reads

    def test_row_geometry(self):
        # one DRAM row = one stripe unit = 256 tokens
        self.assertEqual(_GEN_BYTES_PER_TOKEN, 4)
        self.assertEqual(_GEN_ROW_BYTES // _GEN_BYTES_PER_TOKEN,
                         _STRIPE_UNIT_ROWS)

    def test_extents_split_a_patch_from_its_chunk(self):
        reads = self._reads(segments=2, repairs=1)
        # the recomputed rows are a separate allocation, not part of the chunk
        self.assertEqual(_read_extents(reads),
                         [("diff", 8), ("master", 248), ("master", 256)])

    def test_each_repair_burns_its_own_row_without_a_diff_pool(self):
        reads = self._reads(segments=4, repairs=4)
        groups = _striped_append_channel_extents(
            reads, policy="slice-append", heads_per_hbm=16)   # stripe = 1
        self.assertEqual(len(groups), 16)                     # one head each
        _channel, _count, placed = groups[0]
        repair_extents = [rows for _k, _v, rows in placed if rows == 8]
        self.assertEqual(len(repair_extents), 4)              # 4 separate rows
        addresses = [key for key, _v, _rows in placed]
        self.assertEqual(len(set(addresses)), len(addresses))
        for key, _value, _rows in placed:                     # every row-aligned
            self.assertEqual(key % _GEN_ROW_BYTES, 0)

    def test_diff_pool_packs_the_repairs_into_one_extent(self):
        reads = self._reads(segments=4, repairs=4, shadow=True)
        groups = _striped_append_channel_extents(
            reads, policy="master-diff-table-append", heads_per_hbm=1)
        diff = [placed for channel, _c, placed in groups
                if channel == _MASTER_CHANNELS_DEFAULT]
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff[0], [(diff[0][0][0], diff[0][0][1], 4 * 8)])

    def test_the_split_is_what_changes_the_activation_count(self):
        """The same 64 recomputed rows: 8 activations scattered, 1 pooled."""
        def act_rows(placed):
            return sum(-(-rows * _GEN_BYTES_PER_TOKEN // _GEN_ROW_BYTES)
                       for _k, _v, rows in placed)

        # heads_per_hbm 16 clamps the stripe to one channel per head, so the
        # arithmetic below is per head.
        scattered = _striped_append_channel_extents(
            self._reads(segments=8, repairs=8),
            policy="slice-append", heads_per_hbm=16)
        head_channel = {channel: placed
                        for channel, _count, placed in scattered}[0]
        repairs = [extent for extent in head_channel if extent[2] == 8]
        self.assertEqual(len(repairs), 8)          # 8 separate allocations
        self.assertEqual(act_rows(repairs), 8)     # 64 rows -> 8 activations

        pooled = _striped_append_channel_extents(
            self._reads(segments=8, repairs=8, shadow=True),
            policy="master-diff-table-append", heads_per_hbm=16)
        diff = {channel: placed
                for channel, _count, placed in pooled}[_MASTER_CHANNELS_DEFAULT]
        self.assertEqual(len(diff), 16)            # one packed extent per head
        self.assertEqual(act_rows(diff), 16)       # 16 x 64 rows -> 16
        # Per head: eight activations scattered against one pooled.
        self.assertEqual(act_rows(repairs), 8 * (act_rows(diff) // 16))

    def test_every_row_is_placed_once(self):
        for policy, shadow in (("slice-append", False),
                               ("master-diff-slice-append", True),
                               ("master-diff-table-append", True)):
            for heads in (1, 3, 4, 16):
                reads = self._reads(segments=6, repairs=6, shadow=shadow)
                groups = _striped_append_channel_extents(
                    reads, policy=policy, heads_per_hbm=heads)
                total = sum(rows for _c, _n, placed in groups
                            for _k, _v, rows in placed)
                self.assertEqual(total, heads * len(reads),
                                 "{} heads={}".format(policy, heads))


class StripedAppendLayoutTest(unittest.TestCase):
    """A3b..A6 (ruling chenyi9 2026-09-03): one head's KV is ONE contiguous
    append stream cut into 256-token stripe units; units round-robin the
    head's channels and are PACKED, so a short correction costs its real rows
    instead of a padded 256-token chunk."""

    def test_policy_routing_leaves_a1_a3_a3a_alone(self):
        # A1 (private extent) and A3/A3a (single) keep the chunk-count model.
        self.assertEqual(_layout_policy("private", "slice"), "slice")
        self.assertEqual(_layout_policy("naive", "single"), "single")
        self.assertEqual(_layout_policy("naive-mask", "single"), "single")
        # A3b and every master-diff rung move to striped append.
        self.assertEqual(_layout_policy("naive", "slice"), "slice-append")
        self.assertEqual(_layout_policy("master-diff", "slice"),
                         "master-diff-slice-append")
        self.assertEqual(_layout_policy("master-diff", "table"),
                         "master-diff-table-append")

    def test_stream_is_cut_at_unit_boundaries(self):
        # every unit full except the last -- that is what makes a segment
        # spill onto the next channel instead of starting a padded page.
        self.assertEqual(_stream_unit_rows(0), [])
        self.assertEqual(_stream_unit_rows(256), [256])
        self.assertEqual(_stream_unit_rows(1040), [256, 256, 256, 256, 16])

    def test_worked_example_a_b_c_d_e_f(self):
        """The 2026-09-03 example: head owns ch0..ch3, segments
        a=256 b=256 c=8 d=8 e=256 f=256 appended in that order.

        a -> ch0 unit0, b -> ch1 unit0, c and d TOGETHER on ch2 unit0,
        e -> ch2 unit0 (240 rows) then ch3 unit0 (16), f -> ch3 unit0 (240)
        then ch0 unit1 (16).  So ch0 carries 256+16 rows and ch1..ch3 carry
        256 each.
        """
        loads = _striped_append_channel_rows(
            master_rows=256 * 4, diff_rows=8 + 8,          # a,b,e,f | c,d
            policy="slice-append", heads_per_hbm=4)
        self.assertEqual([loads[c] for c in range(4)], [272, 256, 256, 256])
        # all four heads on this HBM are laid out, each on its own stripe.
        self.assertEqual([loads[c] for c in range(4, 8)], [272, 256, 256, 256])
        self.assertEqual(sum(loads), 4 * 1040)             # every row placed once

    def test_short_correction_is_not_padded_to_a_chunk(self):
        # 16 real diff rows.  The chunk model charged a whole 256-row chunk;
        # the append model charges 16.
        chunked = _layout_channel_loads("slice", master_chunks=0,
                                        diff_chunks=1, heads_per_hbm=1)
        self.assertEqual(sum(chunked) * _STRIPE_UNIT_ROWS, 256)
        appended = _striped_append_channel_rows(
            master_rows=0, diff_rows=16, policy="slice-append",
            heads_per_hbm=1)
        self.assertEqual(sum(appended), 16)

    def test_rows_are_conserved_by_every_append_policy(self):
        for policy in ("slice-append", "master-diff-slice-append",
                       "master-diff-table-append"):
            for heads in (1, 2, 3, 4, 8, 16, 32):
                loads = _striped_append_channel_rows(
                    master_rows=7936, diff_rows=256, policy=policy,
                    heads_per_hbm=heads)
                self.assertEqual(len(loads), _HBM_CHANNELS)
                self.assertEqual(sum(loads), heads * (7936 + 256),
                                 "{} heads={}".format(policy, heads))

    def test_corrections_land_on_the_diff_channel(self):
        loads = _striped_append_channel_rows(
            master_rows=15 * 256, diff_rows=40,
            policy="master-diff-table-append", heads_per_hbm=2)
        self.assertEqual(loads[_MASTER_CHANNELS_DEFAULT], 2 * 40)
        self.assertEqual(sum(loads[:_MASTER_CHANNELS_DEFAULT]), 2 * 15 * 256)

    def test_heads_sharing_a_channel_interleave(self):
        # 32 heads, stripe clamps to 1 -> two heads per channel, their units
        # interleave and pack, so the channel carries both streams' rows.
        loads = _striped_append_channel_rows(
            master_rows=512, diff_rows=0, policy="slice-append",
            heads_per_hbm=32)
        self.assertEqual(loads, [2 * 512] * _HBM_CHANNELS)

    def test_unknown_append_policy_rejected(self):
        with self.assertRaises(WorkloadValidationError):
            _striped_append_channel_rows(1, 0, policy="bogus", heads_per_hbm=1)


class PlacementScanIntegrationTest(unittest.TestCase):
    """`_append_placement_pim_scan` turns the load model into DAG scan events:
    the busiest channel sets the time, energy is placement-independent."""

    UNIT = 1e-9                                     # stub Ramulator per-row time
    ROW = 256                                       # rows per 256-token chunk

    def _scan(self, policy, chunks, heads_per_hbm=1, kv_heads=8, diff_chunks=0):
        from types import SimpleNamespace
        from src import workload_runner as wr
        # REAL-run stub: Ramulator prices each channel run by its actual rows,
        # so time/energy scale with the per-channel load (not a probe).
        def runs_cost(op):
            return [(run[2] * self.UNIT,
                     [run[2] * 2.0, 0.0, 0.0, 0.0, run[2] * 1.0, 0.0])
                    for run in op.pim_kv_runs]
        acc = SimpleNamespace(get_time_and_energy_runs=runs_cost)
        system = SimpleNamespace(devices={"Acc": acc})
        op = SimpleNamespace(numOp=kv_heads)
        reads = ([SimpleNamespace(kind="master", shadow=None,
                                  key_address=2 * i, value_address=2 * i + 1)
                  for i in range((chunks + diff_chunks) * self.ROW)])
        for i in range(diff_chunks * self.ROW):       # tail rows are corrections
            reads[chunks * self.ROW + i].kind = "diff"
        events = []
        wr._append_placement_pim_scan(
            system, events, op=op, layer=0, tier=0, request="r", name="scan",
            reads=reads, deps=(), positions=(0,), policy=policy,
            heads_per_hbm=heads_per_hbm)
        times = [event.time_s for event in events]
        energy = sum(event.energy_nj for event in events)      # nJ (pJ/1000)
        return max(times), energy, len(events)

    def test_single_piles_on_one_channel(self):
        t, _, n = self._scan("single", chunks=32)
        # all 32 chunks (32*256 rows) on one channel -> one long Ramulator run
        self.assertAlmostEqual(t, 32 * self.ROW * self.UNIT)
        self.assertEqual(n, 1)                                 # one active channel

    def test_slice_spreads_and_beats_single(self):
        t_single, _, _ = self._scan("single", chunks=32)
        t_slice, _, n = self._scan("slice", chunks=32)
        # 32 chunks over 16 channels -> 2 chunks (2*256 rows) on the busiest
        self.assertAlmostEqual(t_slice, 2 * self.ROW * self.UNIT)
        self.assertEqual(n, 16)                                # 16 active channels
        self.assertGreater(t_single, t_slice)                 # single is 16x slower

    def test_energy_is_placement_independent(self):
        # Ramulator prices every row; the SUM of rows is the same however the
        # placement distributes them, so total energy is placement-independent.
        _, e_single, _ = self._scan("single", chunks=32)
        _, e_slice, _ = self._scan("slice", chunks=32)
        _, e_table, _ = self._scan("master-diff-table", chunks=32)
        self.assertAlmostEqual(e_single, e_slice)
        self.assertAlmostEqual(e_single, e_table)

    def test_master_diff_isolates_corrections(self):
        # corrections land on the single diff channel; with many of them that
        # channel (10 chunks) outweighs any one master channel (1 chunk).
        t, _, _ = self._scan("master-diff-slice", chunks=15, diff_chunks=10)
        self.assertAlmostEqual(t, 10 * self.ROW * self.UNIT)  # diff channel busiest


class EnergyByLayerTest(unittest.TestCase):
    """`_energy_breakdown` reports per-layer energy with a 0 for every layer."""

    def _breakdown(self, num_layers):
        from types import SimpleNamespace
        from src.workload_runner import _energy_breakdown
        events = [
            SimpleNamespace(device="GPU", name="qkv", energy_nj=5.0,
                            transformer_layer=0),
            SimpleNamespace(device="PIM:pool0-0", name="scan", energy_nj=3.0,
                            transformer_layer=0),
            SimpleNamespace(device="GPU", name="qkv", energy_nj=7.0,
                            transformer_layer=2),
        ]
        return _energy_breakdown(events, num_layers)

    def test_every_layer_present_zero_filled(self):
        by_layer = self._breakdown(4)["by_layer"]
        self.assertEqual(by_layer, {"0": 8.0, "1": 0.0, "2": 7.0, "3": 0.0})

    def test_pim_pool_devices_fold_into_pim_class(self):
        by_class = self._breakdown(3)["by_class"]
        self.assertEqual(by_class.get("PIM"), 3.0)     # PIM:pool0-0 -> PIM
        self.assertEqual(by_class.get("GPU"), 12.0)


class SliceDegeneracyWarningTest(unittest.TestCase):
    """A3b run on too few HBM stacks IS A3, and has to say so.

    2026-09-02: every --num-hbm 1 A3-vs-A3b comparison this project had made
    was A3 against itself -- LLAMA-7B puts 32 KV heads on one stack's sixteen
    channels, the stripe clamps to 1, and the two rungs return the same load
    vector.  These pin the warning that makes that impossible to miss again.
    """

    @staticmethod
    def _system(q_heads, num_hbm, gqa=1, tp=1):
        return SimpleNamespace(
            model=SimpleNamespace(num_heads=q_heads, tp=tp, gqa_size=gqa),
            devices={"Acc": SimpleNamespace(num_hbm=num_hbm)})

    def test_slice_equals_single_exactly_when_it_warns(self):
        # The warning must fire on precisely the configurations where the two
        # policies are indistinguishable, so it is checked against the loads.
        for heads_per_hbm in range(1, 33):
            collapsed = (_layout_channel_loads("slice", 16, 1, heads_per_hbm) ==
                         _layout_channel_loads("single", 16, 1, heads_per_hbm))
            system = self._system(heads_per_hbm, 1)
            warned = placement_degeneracy_warning(
                system, "naive", "slice") is not None
            self.assertEqual(warned, collapsed,
                             "heads_per_hbm={}".format(heads_per_hbm))

    def test_llama7b_one_hbm_warns_and_names_the_fix(self):
        warning = placement_degeneracy_warning(
            self._system(32, 1), "naive", "slice")
        self.assertIsNotNone(warning)
        self.assertIn("same one-per-head pile-up as A3", warning)
        self.assertIn("--num-hbm to at least 4", warning)

    def test_llama7b_four_hbm_is_a_real_a3b(self):
        self.assertIsNone(placement_degeneracy_warning(
            self._system(32, 4), "naive", "slice"))

    def test_a3_itself_is_never_warned_about(self):
        # 'single' is not pretending to slice, so it has nothing to collapse.
        self.assertIsNone(placement_degeneracy_warning(
            self._system(32, 1), "naive", "single"))

    def test_master_diff_collapses_one_stack_later(self):
        # A4 slices within 15 master channels, so 8 heads/stack already clamps.
        self.assertIsNotNone(placement_degeneracy_warning(
            self._system(32, 4), "master-diff", "slice"))
        self.assertIsNone(placement_degeneracy_warning(
            self._system(32, 8), "master-diff", "slice"))

    def test_gqa_counts_kv_heads_not_query_heads(self):
        # LLAMA3-8B: 32 Q heads, GQA group 4 -> 8 KV heads -> 2 channels each.
        self.assertIsNone(placement_degeneracy_warning(
            self._system(32, 1, gqa=4), "naive", "slice"))


if __name__ == "__main__":
    unittest.main()
