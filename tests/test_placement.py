"""Unit tests for the head-aware channel-placement load model (A3..A4b).

These pin the per-physical-channel load each placement policy produces for one
decode scan on one HBM.  ``max(load)`` is proportional to the scan time; the
ladder is single (A3) -> slice (A3b) -> master-diff-slice (A4) ->
master-diff-table (A4b).
"""
import unittest

from types import SimpleNamespace

from src.ramulator_wrapper import hbm_replicas
from src.workload import WorkloadValidationError
from src.workload_runner import (_layout_channel_loads, _layout_scan_max_load,
                                  _pool_reads, _placement_channel_runs,
                                  _APPEND_POLICIES,
                                  _HBM_CHANNELS, _MASTER_CHANNELS_DEFAULT,
                                  _STRIPE_UNIT_ROWS, _layout_policy,
                                  _hbm_stacks_local, _heads_per_hbm,
                                  _read_extents, _GEN_ROW_BYTES,
                                  _GEN_BYTES_PER_TOKEN,
                                  _striped_append_channel_extents,
                                  _stream_unit_rows,
                                  _striped_append_channel_rows,
                                  placement_degeneracy_warning)


# --- real allocators, because since 2026-09-04 the extents come from
# physical adjacency and only an allocator knows it.  Two layouts, one
# variable: where a repair is written.
def build_allocator(segments, repair_rows, pooled, block=256):
    """Build a TLB holding `segments` chunks, each patched by one repair.

    ``pooled`` routes the repairs to the diff channel; otherwise they are
    written inline, between the consumer's own fresh blocks -- so the
    repairs of different rounds do NOT end up adjacent.  Returns
    (tlb, reads).
    """
    from src.workload_runner import CacheBlendTLB
    from dataclasses import replace

    class _Inline(CacheBlendTLB):
        _kv_mapping = "master-diff"

        def reserve(self, layer, owner, fp, row, kind):
            return CacheBlendTLB.reserve(self, layer, owner, fp, row,
                                         "master" if kind == "diff" else kind)

        def locate(self, layer, owner, fp, row, kind):
            loc = CacheBlendTLB.locate(self, layer, owner, fp, row,
                                       "master" if kind == "diff" else kind)
            return replace(loc, kind="diff") if kind == "diff" else loc

    tlb = (CacheBlendTLB if pooled else _Inline)(256, "table")
    for index in range(segments):
        for row in range(block):
            tlb.reserve(0, "owner", "seg%d" % index, row, "master")
        for row in range(repair_rows):
            tlb.reserve(0, "consumer", "seg%d" % index, row, "diff")
        # the consumer's OWN fresh KV for this round, written between the
        # repair of this round and the repair of the next
        for row in range(block):
            tlb.reserve(0, "consumer", "own%d" % index, row, "master")
    tlb.finalize()
    reads = []
    for index in range(segments):
        for row in range(block):
            reads.append(tlb.locate(0, "owner", "seg%d" % index, row,
                                    "master"))
        for row in range(repair_rows):
            reads.append(tlb.locate(0, "consumer", "seg%d" % index, row,
                                    "diff"))
    return tlb, reads


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


    def test_the_pool_merges_repairs_the_inline_layout_leaves_apart(self):
        """The master/diff split, with one variable and a real allocator.

        Four rounds, each patching a chunk with 8 rows.  Inline, each round's
        repair is separated from the next by the KV that round generated, so
        the four stay four extents.  Pooled, the diff channel sees only
        repairs, so consecutive repairs are consecutive however much traffic
        separated them in time, and they merge into one.
        """
        for pooled, expect in ((False, 4), (True, 1)):
            tlb, reads = build_allocator(segments=4, repair_rows=8,
                                         pooled=pooled)
            groups = _striped_append_channel_extents(
                reads, policy="master-diff-table-append", heads_per_hbm=1,
                tlb=tlb)
            diff = [placed for channel, _c, placed in groups
                    if channel == _MASTER_CHANNELS_DEFAULT]
            self.assertEqual(len(diff), 1)
            self.assertEqual(len(diff[0]), expect,
                             "pooled={} should give {} repair extent(s)"
                             .format(pooled, expect))
            self.assertEqual(sum(entry[2] for entry in diff[0]), 4 * 8)

    def test_the_split_is_what_changes_the_activation_count(self):
        """... and the activations follow, folded over the heads."""
        heads = 4
        acts = {}
        for pooled in (False, True):
            tlb, reads = build_allocator(segments=4, repair_rows=8,
                                         pooled=pooled)
            groups = _striped_append_channel_extents(
                reads, policy="master-diff-table-append", heads_per_hbm=heads,
                tlb=tlb)
            acts[pooled] = sum(
                -(-rows * _GEN_BYTES_PER_TOKEN // _GEN_ROW_BYTES)
                for channel, _c, placed in groups
                if channel == _MASTER_CHANNELS_DEFAULT
                for _k, _v, rows in placed)
        # inline: four separate repair extents, one row each, per head
        # pooled: 4 x 4 x 8 = 128 tokens in one extent -> a single row
        self.assertEqual(acts[False], 4)
        self.assertEqual(acts[True], 1)
        self.assertLess(acts[True], acts[False])

    def test_extents_are_not_packed_across_owners(self):
        """Two owners' chunks are two allocations and never share a row.

        Summing them and re-cutting at 256 was the third collapse found on
        2026-09-04: it let one owner's tail be packed against the next
        owner's head.
        """
        tlb, reads = build_allocator(segments=2, repair_rows=8, pooled=True)
        master = [location for location in reads if location.kind != "diff"]
        runs = tlb.scan_runs(master)
        groups = _striped_append_channel_extents(
            reads, policy="master-diff-table-append", heads_per_hbm=1, tlb=tlb)
        placed = [rows for channel, _c, entries in groups
                  if channel != _MASTER_CHANNELS_DEFAULT
                  for _k, _v, rows in entries]
        self.assertEqual(sum(placed), sum(run[2] for run in runs))
        self.assertGreaterEqual(len(placed), len(runs))


class ConflictAwareSlotTableTest(unittest.TestCase):
    """Fugue sec. 4: a chunk goes to a channel not used by the chunks read
    alongside it.  Two properties, both of which the per-scan rotation that
    A4c/A4d used until 2026-09-04 could not have."""

    def test_naive_rotation_collides_by_luck_and_the_table_does_not(self):
        from src.workload_runner import _chunk_slot_table
        # doc1 and doc5 are written four apart -- exactly the stripe -- so the
        # write-order rotation lands them on the SAME slot although one sweep
        # reads them together.  The table sees that sweep and keeps them apart.
        order = ["doc%d" % i for i in range(1, 7)]
        coread = [frozenset({"doc1", "doc5"}), frozenset({"doc2", "doc6"})]
        naive = _chunk_slot_table(order, coread, 4, "append")
        table = _chunk_slot_table(order, coread, 4, "table")
        self.assertEqual(naive["doc1"], naive["doc5"])          # the collision
        self.assertNotEqual(table["doc1"], table["doc5"])       # removed
        self.assertNotEqual(table["doc2"], table["doc6"])

    def test_slot_is_persistent_across_scans(self):
        """The same chunk lands on the same channel in every scan, however
        it is ordered within any one scan's read list."""
        from src.workload_runner import TableLocalDiffKVLayout
        tlb = TableLocalDiffKVLayout(256, "slice")
        tlb.chunk_order = ["a", "b", "c", "d", "e"]
        tlb.chunk_coread = [frozenset({"a", "e"}), frozenset({"b", "c", "d", "e"})]
        first = {f: tlb.chunk_slot(f, 4, "table") for f in tlb.chunk_order}
        again = {f: tlb.chunk_slot(f, 4, "table") for f in reversed(tlb.chunk_order)}
        self.assertEqual(first, again)
        self.assertNotEqual(first["a"], first["e"])

    def test_a_chunk_the_record_never_saw_is_appended(self):
        from src.workload_runner import TableLocalDiffKVLayout
        tlb = TableLocalDiffKVLayout(256, "slice")
        tlb.chunk_order, tlb.chunk_coread = ["a"], [frozenset({"a"})]
        self.assertEqual(tlb.chunk_slot("a", 4, "table"), 0)
        self.assertEqual(tlb.chunk_slot("never-reserved", 4, "table"), 1)


class PresetRoutesToItsOwnPolicyTest(unittest.TestCase):
    """The rung you ask for is the rung you get.

    A preset names a kv_mapping; the runner picks a TLB class for it; that
    class's ``_kv_mapping`` is what becomes ``layout_policy`` and drives the
    placement.  Three hops, and on 2026-09-04 the third silently rerouted A4d
    onto A4c's placement because the two shared a class.  This pins all three
    hops against each other for every PIM preset.
    """

    def test_every_pim_preset_places_under_its_own_policy(self):
        from src.ablation import PRESETS
        from src.workload_runner import (CacheBlendTLB, NaiveKVLayout,
                                         NaiveMaskKVLayout, LocalDiffKVLayout,
                                         TableLocalDiffKVLayout)
        classes = {"master-diff": CacheBlendTLB, "naive": NaiveKVLayout,
                   "naive-mask": NaiveMaskKVLayout,
                   "master-diff-local": LocalDiffKVLayout,
                   "master-diff-table-local": TableLocalDiffKVLayout}
        seen = set()
        for rung, preset in PRESETS.items():
            mapping = preset["kv_mapping"]
            if mapping not in classes:
                continue                     # A1 (private) / A2 (none)
            placement = preset.get("channel_placement", "slice")
            expected = _layout_policy(mapping, placement)
            tlb = classes[mapping](256, placement)
            self.assertEqual(tlb.layout_policy, expected,
                             "{}: TLB class places as {}, preset asks for {}"
                             .format(rung, tlb.layout_policy, expected))
            seen.add(expected)
        # and the placements that are supposed to differ do differ
        self.assertIn("master-diff-local-append", seen)
        self.assertIn("master-diff-table-local-append", seen)


class ShadowRowsAreActivatedTest(unittest.TestCase):
    """A repaired row's DRAM row is opened whether or not it is scored.

    ``shadow_reads`` is a DIE-side flag: does the stale master copy get
    masked out of the score, or must the scan skip it?  It used to decide
    something else as well -- whether that master row appeared in ``reads``
    at all -- and A3b (``shadow_reads = False``) therefore lost the k
    recomputed rows from its master stream.  ``_striped_append_channel_
    extents`` then repacked what survived, so the master stream SHRANK by
    exactly the repair length and A3b's cost came out independent of k.

    Geometry that makes the skip free: 4 B of address space per token, so a
    1024-B row holds 256 tokens.  Skipping k of them neither shortens the row
    nor saves its activation; the repair is an ADDITION (fix 2026-09-04).
    """

    @staticmethod
    def _tlb(shadow_reads, layout_policy):
        return SimpleNamespace(shadow_reads=shadow_reads,
                               layout_policy=layout_policy)

    @staticmethod
    def _loc(owner, fingerprint, kind, address, shadow=None):
        return SimpleNamespace(owner=owner, fingerprint=fingerprint, kind=kind,
                               key_address=address, value_address=address + 1,
                               shadow=shadow)

    def _visible(self, chunks, k, chunk_rows=256):
        """Consumer-visible KV: ``chunks`` chunks, k repaired tokens each."""
        visible = []
        address = 0
        for index in range(chunks):
            name = "seg{}".format(index)
            for position in range(chunk_rows):
                address += 8
                master = self._loc("owner", name, "master", address)
                if position < k:
                    visible.append(self._loc("consumer", name, "diff",
                                             address + 4, shadow=master))
                else:
                    visible.append(master)
        return visible

    @staticmethod
    def _acts(groups):
        """Activations Ramulator sees: each placed extent opens its own row."""
        return sum(-(-rows * _GEN_BYTES_PER_TOKEN // _GEN_ROW_BYTES)
                   for _channel, _count, placed in groups
                   for _key, _value, rows in placed)

    def test_a3b_gets_the_shadow_rows_despite_having_no_mask_gate(self):
        visible = self._visible(chunks=4, k=8)
        reads, masked, plan_reads = _pool_reads(
            self._tlb(False, "slice-append"), visible)
        # what the DRAM streams: 4 x 256 visible + the 32 shadowed master rows
        self.assertEqual(len(reads), 4 * 256 + 32)
        # what the die drops: NOTHING.  A3b has no mask gate, so it may not
        # claim masked rows -- the first version of this fix did, which
        # handed A3b a capability the rung is defined not to have.
        self.assertEqual(masked, set())
        # what the TLB plans: the UNEXPANDED stream, so the master run still
        # splits at every skipped row and the descriptor count still rises.
        self.assertEqual(len(plan_reads), 4 * 256)
        self.assertIsNot(plan_reads, reads)

    def test_a_maskless_layout_still_splits_its_master_run(self):
        """The defining property of NaiveKVLayout, pinned.

        "a corrected row's master copy is skipped, SPLITTING the master run".
        The first version of the 2026-09-04 fix put the shadow rows into the
        list `tlb.scan_runs` coalesces, so the holes closed, the runs merged
        10 -> 2, the TLB descriptor cost collapsed and A3b came out 5.5%
        faster than it should be.  `plan_reads` is what keeps them apart.
        """
        from src.workload_runner import NaiveKVLayout
        from dataclasses import replace as _replace
        corrected = (20, 132, 155, 197, 207, 215, 244, 248)   # real t1n0 rows
        tlb = NaiveKVLayout(256, "slice")
        for row in range(256):
            tlb.reserve(0, "owner", "fp0", row, "master")
        for row in corrected:
            tlb.reserve(0, "consumer", "fp0", row, "diff")
        tlb.finalize()
        visible = []
        for row in range(256):
            master = tlb.locate(0, "owner", "fp0", row, "master")
            if row in corrected:
                diff = tlb.locate(0, "consumer", "fp0", row, "diff")
                visible.append(_replace(diff, shadow=master))
            else:
                visible.append(master)
        reads, masked, plan_reads = _pool_reads(tlb, visible)
        self.assertEqual(masked, set())                 # no mask gate
        self.assertEqual(len(reads), 256 + len(corrected))   # rows opened
        # eight skipped rows cut the 256-row stream into nine pieces, and the
        # corrected rows are a tenth run of their own
        self.assertEqual(len(tlb.scan_runs(plan_reads)), 10)
        # ... which is exactly what merging them away would have destroyed
        self.assertLess(len(tlb.scan_runs(reads)), 10)

    def test_a3_keeps_the_legacy_behaviour(self):
        """A1/A3/A3a stay on the chunk-count model, so the A3-vs-A3a contrast
        (which IS the mask gate) survives the fix."""
        visible = self._visible(chunks=4, k=8)
        for policy in ("single", "slice"):
            self.assertNotIn(policy, _APPEND_POLICIES)
            reads, masked, plan_reads = _pool_reads(
                self._tlb(False, policy), visible)
            self.assertEqual(len(reads), 4 * 256)
            self.assertEqual(masked, set())
            self.assertEqual(len(plan_reads), 4 * 256)

    def test_a3b_master_stream_no_longer_shrinks_with_k(self):
        """The regression itself: the master side is the owner's chunks and
        owes nothing to k."""
        for k in (0, 1, 8, 32, 64, 128):
            reads, _masked, _plan = _pool_reads(
                self._tlb(False, "slice-append"), self._visible(chunks=4, k=k))
            master = sum(rows for kind, rows in _read_extents(reads)
                         if kind != "diff")
            self.assertEqual(master, 4 * 256,
                             "master stream moved with k={}".format(k))

    def test_a3b_cost_is_monotone_in_k(self):
        """Nothing pinned this before, which is why the bug survived.

        More recompute may never come out cheaper.  Checked on the inline
        layout, where the repairs of different rounds stay apart.
        """
        previous = -1
        for k in (1, 8, 16, 32, 64, 128):
            tlb, reads = build_allocator(segments=4, repair_rows=k,
                                         pooled=False)
            loads, _active, _runs, groups = _placement_channel_runs(
                reads, policy="slice-append", heads_per_hbm=1, tlb=tlb)
            acts = sum(-(-rows * _GEN_BYTES_PER_TOKEN // _GEN_ROW_BYTES)
                       for _c, _n, placed in groups for _k, _v, rows in placed)
            self.assertGreaterEqual(acts, previous,
                                    "cost fell going to k={}".format(k))
            previous = acts
    def test_a3b_chunk_keeps_its_channel_across_scans(self):
        """F02 (2026-09-05): A3b places by the persistent write-order slot.

        The old branch re-rotated by the unit index of the CURRENT scan, so
        a scan that read only seg2 and seg3 put them on ch0 and ch1 -- the
        same chunks sat on ch2 and ch3 in the full scan.  No store can move
        a chunk between scans; and A3b must share A4c's master placement so
        that the A3b -> A4c step is the diff gather alone.
        """
        tlb, reads = build_allocator(segments=4, repair_rows=8, pooled=False)
        full_loads, _a, _r, _g = _placement_channel_runs(
            reads, policy="slice-append", heads_per_hbm=1, tlb=tlb)
        partial = [r for r in reads if r.fingerprint in ("seg2", "seg3")]
        loads, _a, _r, _g = _placement_channel_runs(
            partial, policy="slice-append", heads_per_hbm=1, tlb=tlb)
        self.assertGreater(full_loads[2], 0)
        self.assertGreater(full_loads[3], 0)
        self.assertGreater(loads[2], 0)
        self.assertGreater(loads[3], 0)
        self.assertEqual((loads[0], loads[1]), (0.0, 0.0))
        # ... and the same table A4c consults gives the same master slots
        a4c, _a, _r, _g = _placement_channel_runs(
            partial, policy="master-diff-local-append", heads_per_hbm=1, tlb=tlb)
        self.assertGreater(a4c[2], 0)
        self.assertGreater(a4c[3], 0)

    def test_activations_step_at_the_row_boundary_not_with_k(self):
        """Ruling example (chenyi9 2026-09-04): four 256-token chunks, pooled.

        The diff pool holds 4k tokens at 4 B each.  k=8 fills 4 columns of one
        row, k=32 fills 16 columns of the SAME row -- four times the data, the
        same single activation.  Only crossing 256 tokens buys another ACT.

        Pooled on purpose: the claim is about repairs that are PHYSICALLY
        ADJACENT, which since 2026-09-04 is the allocator's business, not the
        placement rule's.  Written inline they are four extents and four rows
        however small k is, which is the point of the split.
        """
        for k, diff_acts in ((8, 1), (32, 1), (64, 1), (65, 2)):
            tlb, reads = build_allocator(segments=4, repair_rows=k,
                                         pooled=True)
            groups = _striped_append_channel_extents(
                reads, policy="master-diff-table-append", heads_per_hbm=1,
                tlb=tlb)
            diff = sum(-(-rows * _GEN_BYTES_PER_TOKEN // _GEN_ROW_BYTES)
                       for channel, _c, placed in groups
                       if channel == _MASTER_CHANNELS_DEFAULT
                       for _k, _v, rows in placed)
            self.assertEqual(diff, diff_acts,
                             "k={} should cost {} diff ACT".format(k, diff_acts))

    def test_inline_repairs_do_not_get_that_step(self):
        """Same k sweep, written inline: one row per round, always."""
        for k in (8, 32, 64):
            tlb, reads = build_allocator(segments=4, repair_rows=k,
                                         pooled=False)
            groups = _striped_append_channel_extents(
                reads, policy="master-diff-table-append", heads_per_hbm=1,
                tlb=tlb)
            diff = sum(-(-rows * _GEN_BYTES_PER_TOKEN // _GEN_ROW_BYTES)
                       for channel, _c, placed in groups
                       if channel == _MASTER_CHANNELS_DEFAULT
                       for _k, _v, rows in placed)
            self.assertEqual(diff, 4, "k={}".format(k))


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


class HbmReplicaTest(unittest.TestCase):
    """One trace = one stack of ``num_ops_per_hbm`` heads; scale by the heads
    it stands for, never by ``num_hbm`` (audit 2026-09-03)."""

    @staticmethod
    def _per_hbm(num_ops, num_hbm):
        return -(-num_ops // num_hbm)             # what Ramulator.run computes

    def test_head_folded_scan_stands_for_one_stack(self):
        # The placement path forces numOp = 1 (heads are folded into the row
        # count) and applies the real stack count itself as num_hbm_used.  The
        # trace is one head, so its counters stand for exactly one stack --
        # whatever --num-hbm says.  This is the double-count that made PIM scan
        # energy num_hbm times too high.
        for num_hbm in (1, 4, 5, 10, 16, 40):
            self.assertEqual(
                hbm_replicas(1, self._per_hbm(1, num_hbm)), 1.0,
                "--num-hbm {} invented {} phantom heads".format(
                    num_hbm, num_hbm - 1))

    def test_replicas_recover_the_real_head_count(self):
        # The invariant: replicas * heads-in-the-trace == heads on the AttAcc.
        for num_hbm in (1, 2, 5, 10, 40):
            for num_ops in (1, 4, 8, 12, 26, 96, 384):
                per_hbm = self._per_hbm(num_ops, num_hbm)
                self.assertAlmostEqual(
                    hbm_replicas(num_ops, per_hbm) * per_hbm, num_ops,
                    msg="numOp={} num_hbm={}".format(num_ops, num_hbm))

    def test_agrees_with_num_hbm_when_it_divides_the_heads(self):
        # Where the old expression was right it stays right: the legacy AttAcc
        # batches (numOp = kv_heads x batch) divide evenly.
        for num_hbm in (5, 10, 40):
            for per_hbm in (1, 2, 3, 16, 64):
                num_ops = num_hbm * per_hbm
                self.assertEqual(hbm_replicas(num_ops, per_hbm), num_hbm)

    def test_never_exceeds_the_old_value(self):
        # ceil() can only round the per-stack head count UP, so the corrected
        # multiplier can only be <= num_hbm: the fix never inflates energy.
        for num_hbm in (1, 5, 10, 40):
            for num_ops in range(1, 200):
                self.assertLessEqual(
                    hbm_replicas(num_ops, self._per_hbm(num_ops, num_hbm)),
                    num_hbm)


if __name__ == "__main__":
    unittest.main()
