"""collect_dag_ladder tier rows use one metric definition for every rung."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TierRowsTest(unittest.TestCase):
    def test_identical_summaries_give_identical_tier_rows(self):
        # C6 (2026-09-05): the same request summary must yield the same
        # ttft / decode / total whether or not the rung ran PIM batches
        summary = {"requests": {"r": {"tier": 0, "prefill_end_s": 2.0,
                                      "first_token_s": 3.0, "end_s": 5.0}}}
        with tempfile.TemporaryDirectory() as out:
            json.dump({"makespan_s": 5.0, "energy_nj": 1.0, "summary": summary,
                       "batches": [], "energy_breakdown_nj": {"by_class": {}}},
                      open(os.path.join(out, "dag_A2.json"), "w"))
            json.dump({"makespan_s": 5.0, "energy_nj": 1.0, "summary": summary,
                       "batches": [{"tier": 0, "q_arrival_s": 2.2, "attention_start_s": 2.3},
                                   {"tier": 0, "q_arrival_s": 4.2, "attention_start_s": 4.3}],
                       "energy_breakdown_nj": {"by_class": {}}},
                      open(os.path.join(out, "dag_A3b.json"), "w"))
            wl = os.path.join(out, "wl.json")
            open(wl, "w").write("{}")
            subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "collect_dag_ladder.py"),
                            out, wl, "TOY"], check=True, capture_output=True)
            rows = {r["ablation"]: r for r in csv.DictReader(open(os.path.join(out, "dag_ladder_tiers.csv")))}
        for column in ("ttft_s", "decode_s", "tier_total_s", "cum_end_s"):
            self.assertEqual(rows["A2"][column], rows["A3b"][column], column)
        self.assertEqual(float(rows["A2"]["ttft_s"]), 3.0)
        self.assertEqual(float(rows["A2"]["decode_s"]), 2.0)
        self.assertEqual(float(rows["A2"]["tier_total_s"]), 5.0)
        self.assertEqual(rows["A2"]["decode_batches"], "0")
        self.assertEqual(rows["A3b"]["decode_batches"], "2")


if __name__ == "__main__":
    unittest.main()
