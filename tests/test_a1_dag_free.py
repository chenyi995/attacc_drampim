"""Checks for the DAG-free A1 input enumerator.

The important tests here are the CROSS-VALIDATIONS: the enumerator is compared
against the multiset of PIM invocations the real event DAG asked for, recorded
with ``ATTACC_RECORD_PIM_SIGNATURES`` and stored under tests/fixtures.  The
hand-computed unit tests below are kept, but they cannot be the whole story:
their expected values come from the same understanding the code encodes, so
they only catch a typo, never a shared misreading of A1's placement contract.

Regenerating a fixture (after an intentional change to the DAG or the layout):

    ATTACC_RECORD_PIM_SIGNATURES=tests/fixtures/dag_free/<name>.json \\
    python3 main.py --system dgx-attacc --model <model> --workload <workload> \\
        --reuse no-reuse --ablation A1 --engine dag --powerlimit --no-warm \\
        --workload-report-events none --workload-report /dev/null
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/analytic_a1_0902"))
from src import a1_dag_free  # noqa: E402
from src.a1_dag_free import enumerate_a1_pim_inputs, input_summary  # noqa: E402
from src.config import make_model_config  # noqa: E402
from src.model import Transformer  # noqa: E402
from src.type import DataType  # noqa: E402
from src.workload import load_workload  # noqa: E402
import validate_dag_free  # noqa: E402

FIXTURES = ROOT / "tests/fixtures/dag_free"
# Cases where the enumerator must reproduce the DAG's multiset EXACTLY.
CASES = (
    ("LLAMA-7B", "workload_llama7b_small.json",
     "a1_dag_signatures_small_llama7b.json"),
    # LLAMA3-8B exercises the GQA path (gqa_size=4), which changes the query
    # batch.  It does NOT change heads_per_hbm -- see HEAD_FOLDING_CASES.
    ("LLAMA3-8B", "workload_llama7b_medium.json",
     "a1_dag_signatures_medium_llama3-8b.json"),
    # total_length % cap_rows == 1: the one-row tail sweep, which infers
    # shared_kv differently from the full sweeps if it is derived from
    # shared_queries > 1 instead of being carried explicitly.
    ("LLAMA-7B", "workload_tailsweep.json",
     "a1_dag_signatures_tailsweep_llama7b.json"),
    ("LLAMA-65B", "workload_tailsweep.json",
     "a1_dag_signatures_tailsweep_llama65b.json"),
)

# heads_per_hbm > 1 -- the head-folding half of _slice_channel_rows, which no
# LLAMA-7B/LLAMA3-8B case reaches (both give heads_per_hbm == 1 at tp=8).
# LLAMA-65B gives 2, GPT-175B gives 3.  The DECODE runs match exactly; the
# prefill sweeps differ by the allocator defect of RESULTS.md 2.2, so these
# cases assert the decode half only.
HEAD_FOLDING_CASES = (
    ("LLAMA-65B", "workload_llama7b_small.json",
     "a1_dag_signatures_headfold_llama65b.json", 2),
    ("GPT-175B", "workload_llama7b_small.json",
     "a1_dag_signatures_headfold_gpt175b.json", 3),
)


class EnumeratorMatchesTheDag(unittest.TestCase):
    """Ground truth is the DAG, not the enumerator's own assumptions."""

    def _compare(self, model_name, workload_file, signature_file):
        workload = load_workload(str(FIXTURES / workload_file))
        model = Transformer(make_model_config(model_name, DataType.W16A16),
                            tensor_parallel=8)
        mine = validate_dag_free.enumerator_multiset(
            workload, model, dbyte=2, num_hbm=5, power_constraint=True)
        theirs = validate_dag_free.dag_multiset(FIXTURES / signature_file)
        return mine, theirs, model

    def test_invocation_multiset_is_identical(self):
        """Layer A2: same runs, same multiplicities -- not merely the same total."""
        for model_name, workload_file, signature_file in CASES:
            with self.subTest(model=model_name, workload=workload_file):
                mine, theirs, _ = self._compare(model_name, workload_file,
                                                signature_file)
                self.assertEqual(dict(mine), dict(theirs))

    def test_aggregate_pim_work_is_identical(self):
        """Layer A1: the command totals a makespan would be built from."""
        for model_name, workload_file, signature_file in CASES:
            with self.subTest(model=model_name, workload=workload_file):
                mine, theirs, model = self._compare(model_name, workload_file,
                                                    signature_file)
                self.assertEqual(
                    dict(validate_dag_free._work(mine, model.dhead)),
                    dict(validate_dag_free._work(theirs, model.dhead)))

    def test_fixtures_are_not_trivial(self):
        """A fixture with one signature would make the comparison vacuous."""
        for _, _, signature_file in CASES:
            with self.subTest(fixture=signature_file):
                theirs = validate_dag_free.dag_multiset(FIXTURES / signature_file)
                self.assertGreaterEqual(len(theirs), 5)
                self.assertGreater(sum(theirs.values()), 1000)


class HeadFoldingMatchesTheDag(unittest.TestCase):
    """The head-folded decode runs, at heads_per_hbm = 2 and 3.

    Every earlier comparison had heads_per_hbm == 1, where the slice rule
    degenerates to plain round-robin -- so the head-folding loop, the only
    non-trivial half of _slice_channel_rows, was never checked against the
    DAG at all.  These two cases check it.
    """

    def test_decode_channel_runs_match(self):
        for model_name, workload_file, signature_file, expected in HEAD_FOLDING_CASES:
            with self.subTest(model=model_name):
                workload = load_workload(str(FIXTURES / workload_file))
                model = Transformer(make_model_config(model_name, DataType.W16A16),
                                    tensor_parallel=8)
                q_heads = max(1, model.num_heads // model.tp)
                gqa = max(1, int(getattr(model, "gqa_size", 1) or 1))
                self.assertEqual(
                    a1_dag_free._ceil(max(1, q_heads // gqa), 5), expected,
                    "this case no longer exercises head folding")
                mine = validate_dag_free.enumerator_multiset(
                    workload, model, dbyte=2, num_hbm=5, power_constraint=True)
                theirs = validate_dag_free.dag_multiset(FIXTURES / signature_file)
                channel_index = list(validate_dag_free.Ramulator.PRICING_FIELDS
                                     ).index("channel_count")
                decode_mine = {k: v for k, v in mine.items() if k[channel_index] == 1}
                decode_theirs = {k: v for k, v in theirs.items() if k[channel_index] == 1}
                self.assertTrue(decode_theirs, "no decode runs in the fixture")
                self.assertEqual(decode_mine, decode_theirs)


class ClosedFormEqualsTheLoop(unittest.TestCase):
    """The decode enumeration is closed-form; keep the loop it replaced.

    The loop version built one frozen dataclass per PHYSICAL RUN, which is
    O(requests x layers x output_tokens x channels): 107 s on wl_N64 and 352 s
    on wl_N64/GPT-175B.  The closed form walks 256-row chunk bands instead.
    That is only safe while the two agree exactly, so the loop lives here as
    the reference and this test is what makes the rewrite legitimate.
    """

    @staticmethod
    def _loop_slice(context_rows, heads_per_hbm):
        chunks = a1_dag_free._ceil(context_rows, a1_dag_free._CHUNK_ROWS)
        heads = max(1, int(heads_per_hbm))
        stripe = max(1, a1_dag_free._CHANNELS // heads)
        loads = [0] * a1_dag_free._CHANNELS
        for head in range(heads):
            base = (head * stripe) % a1_dag_free._CHANNELS
            for chunk in range(chunks):
                loads[(base + chunk % stripe) % a1_dag_free._CHANNELS] += 1
        return [(channel, load * a1_dag_free._CHUNK_ROWS)
                for channel, load in enumerate(loads) if load]

    def test_slice_rule_matches_the_loop_exhaustively(self):
        for heads in range(1, 33):
            for chunks in range(0, 40):
                rows = chunks * a1_dag_free._CHUNK_ROWS
                self.assertEqual(
                    list(a1_dag_free._slice_channel_rows(rows, heads)),
                    self._loop_slice(rows, heads),
                    "heads={} chunks={}".format(heads, chunks))

    def test_enumeration_matches_the_loop_on_real_workloads(self):
        for model_name, workload_file, _ in CASES:
            with self.subTest(model=model_name, workload=workload_file):
                workload = load_workload(str(FIXTURES / workload_file))
                model = Transformer(make_model_config(model_name, DataType.W16A16),
                                    tensor_parallel=8)
                self.assertEqual(
                    enumerate_a1_pim_inputs(workload, model, dbyte=2, num_hbm=5),
                    _loop_enumerate(workload, model, dbyte=2, num_hbm=5))


def _loop_enumerate(workload, model, *, dbyte, num_hbm=5):
    """The pre-closed-form implementation, kept as the reference."""
    from collections import Counter
    from src.a1_dag_free import (A1PimInput, _CHANNELS, _HBM_CHANNEL_BYTES,
                                 _KV_GAP_BYTES, _ceil)
    from src.ramulator_wrapper import (MQ_DEFAULT_GEMV_BUFFER_BYTES,
                                       mq_query_capacity)
    q_heads = max(1, int(model.num_heads) // max(1, int(model.tp)))
    gqa = max(1, int(getattr(model, "gqa_size", 1) or 1))
    kv_heads = max(1, q_heads // gqa)
    heads_per_hbm = _ceil(kv_heads, max(1, int(num_hbm)))
    hbm_used = max(1, min(int(num_hbm), _ceil(kv_heads, heads_per_hbm)))
    cap_rows = max(1, mq_query_capacity(MQ_DEFAULT_GEMV_BUFFER_BYTES) // gqa)
    result = Counter()
    for request in workload.requests:
        prefill_rows = int(request.total_length)
        resident_rows = prefill_rows + int(request.history_len)
        full, tail = divmod(prefill_rows, cap_rows)
        ordinal = sum(1 for other in workload.requests
                      if other.request_id < request.request_id)
        for layer in range(int(model.ndec)):
            base = (layer * max(1, len(workload.requests)) + ordinal) * (2 * _KV_GAP_BYTES)
            if full:
                result[A1PimInput("full", resident_rows, heads_per_hbm, dbyte,
                                  model.dhead, _CHANNELS, cap_rows * gqa, 0,
                                  base, base + _KV_GAP_BYTES, queries=cap_rows,
                                  kv_heads=kv_heads, energy_replicas=1,
                                  shared_kv=True)] += full
            if tail:
                result[A1PimInput("full", resident_rows, heads_per_hbm, dbyte,
                                  model.dhead, _CHANNELS, tail * gqa, 0,
                                  base, base + _KV_GAP_BYTES, queries=tail,
                                  kv_heads=kv_heads, energy_replicas=1,
                                  shared_kv=True)] += 1
            for output_row in range(int(request.lout)):
                context = resident_rows + output_row
                for channel, folded in ClosedFormEqualsTheLoop._loop_slice(
                        context, heads_per_hbm):
                    key = channel * _HBM_CHANNEL_BYTES
                    result[A1PimInput("full", folded, 1, dbyte, model.dhead, 1,
                                      gqa, channel, key, key + _KV_GAP_BYTES,
                                      queries=1, kv_heads=1,
                                      energy_replicas=hbm_used,
                                      shared_kv=gqa > 1)] += 1
    return dict(result)


class HandDerivedShapes(unittest.TestCase):
    """Small closed-form checks; see the module docstring for their limits."""

    def test_prefill_sweeps_are_capacity_based_not_event_based(self):
        model = SimpleNamespace(ndec=2, dhead=128, num_heads=32, tp=1, gqa_size=1)
        request = SimpleNamespace(request_id="r0", total_length=17, history_len=3,
                                  lout=0)
        inputs = enumerate_a1_pim_inputs(SimpleNamespace(requests=(request,)),
                                         model, dbyte=2)
        summary = input_summary(inputs)
        # cap=8: two q=8 scans and one q=1 tail per layer, over 20 resident rows.
        self.assertEqual(summary["prefill_sweeps"], 6)
        self.assertEqual(summary["decode_channel_runs"], 0)
        by_q = {}
        for item, count in inputs.items():
            by_q[item.shared_queries] = by_q.get(item.shared_queries, 0) + count
        self.assertEqual(by_q, {1: 2, 8: 4})

    def test_decode_uses_head_folded_channel_runs(self):
        model = SimpleNamespace(ndec=1, dhead=128, num_heads=32, tp=1, gqa_size=1)
        request = SimpleNamespace(request_id="r0", total_length=8, history_len=0,
                                  lout=1)
        inputs = enumerate_a1_pim_inputs(SimpleNamespace(requests=(request,)),
                                         model, dbyte=2)
        summary = input_summary(inputs)
        # ceil(32 / 5)=7 KV heads per HBM; with one 256-row chunk per head,
        # the slice rule activates one channel for each head.
        self.assertEqual(summary["decode_channel_runs"], 7)
        self.assertEqual(summary["prefill_sweeps"], 1)


if __name__ == "__main__":
    unittest.main()
