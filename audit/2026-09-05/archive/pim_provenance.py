"""Audit timing provenance, not performance. Run with PYTHONPATH=.

The wrapper probe injects exactly 1000 cycles at the external simulator boundary.
The DAG probe uses the existing device stub. Trace command counts come from the
real checked-in generator. No Ramulator binary is run and no source is changed.
"""
from collections import Counter
from pathlib import Path
import importlib.util
import json
import re
import subprocess
import sys
import tempfile

from src.ablation import resolve_config
from src.model import Layer
from src.ramulator_wrapper import Ramulator
from src.type import DataType, LayerType, PIMType
from src.workload import Request, Segment, Workload


def main():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "audit_probes", Path(__file__).with_name("reproduce.py"))
    probes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probes)
    result = {
        "scope": "Boundary-injection, formula and trace-generation probes; no real Ramulator cycles or speedups measured.",
        "injected_raw_cycles": 1000,
        "wrapper_results": {},
    }
    with tempfile.TemporaryDirectory(prefix="fugue-pim-provenance-") as scratch:
        wrapper = Ramulator({"num_heads": 8, "dhead": 128}, scratch,
                            num_hbm=1, signature_cache=False)
        captured = []

        def inject_cycles(pim_type, length, num_ops, dbyte, yaml_path,
                          file_name, key_addr=None, value_addr=None,
                          channel_count=16, shared_kv=False, shared_queries=1,
                          channel_base=None, **kwargs):
            yaml_text = Path(yaml_path).read_text()
            override = re.search(r"nCCDAB:\s*(\d+)", yaml_text)
            captured.append({
                "simulated_length": length,
                "channel_count": channel_count,
                "channel_base": channel_base,
                "shared_queries": shared_queries,
                "mq_command": kwargs.get("mq_command", False),
                "nCCDAB_override": int(override[1]) if override else None,
                "has_extent_list": bool(kwargs.get("kv_extents")),
            })
            return [1000, 8, 1, 1, 1, 1]

        wrapper.run_ramulator = inject_cycles
        for rung in ("A1", "A3b", "A4c", "A4e", "A5", "A6"):
            policy = "no-reuse" if rung == "A1" else "epic"
            cfg = resolve_config(rung, None, None, None, policy=policy)
            op = Layer("gen", "score", LayerType.MATMUL, False,
                       DataType.W16A16, 8, 257, 128, 1)
            op.pim_shared_kv = True
            op.pim_shared_queries = 8
            op.pim_batch_command = cfg.pim_batch_command
            op.pim_pe_freq_ghz = cfg.pim_pe_freq_ghz
            if rung == "A1":
                op.pim_kv_runs = ((0, 8388608, 257),)
            else:
                op.pim_kv_runs = ((0, 8388608, 257, 0, 1),)
                op.pim_kv_extent_groups = ((0, 1, ((0, 8388608, 257),)),)
            timed = wrapper.output_runs(PIMType.BA, op, True)
            result["wrapper_results"][rung] = {
                "boundary_arguments": captured[-1],
                "returned_time_s": timed[0][0],
                "ratio_to_raw_cycles_times_tCK": timed[0][0] / (1000 * .769e-9),
            }

        result["real_generator_commands"] = {}
        for scheme in ("replicate", "mq"):
            trace = Path(scratch) / (scheme + ".trace")
            command = [sys.executable,
                str(root / "pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py"),
                "--dhead", "128", "--nhead", "1", "--seqlen", "256",
                "--dbyte", "2", "--channels", "1", "--pool-base", "0",
                "--head-hbm-stripe", "--key-addr", "0", "--value-addr", "0x800000",
                "--shared-kv", "--shared-queries", "8", "--output", str(trace)]
            if scheme == "mq":
                command.append("--mq")
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            result["real_generator_commands"][scheme] = dict(Counter(
                line.split()[0] for line in trace.read_text().splitlines() if line))

        # Separate extents in one row versus different rows can have identical
        # cache address components. This checks the key, not resulting cycles.
        same_row = ((0, 8388608, 8), (64, 8388672, 8))
        different_rows = ((0, 8388608, 8), (1088, 8389696, 8))
        def extent_key(extents):
            return tuple((wrapper._address_mapping_signature(k),
                          wrapper._address_mapping_signature(v), n)
                         for k, v, n in extents)
        result["cache_drops_relative_row_identity"] = {
            "same_row_extents": same_row,
            "different_row_extents": different_rows,
            "extent_signature_equal": extent_key(same_row) == extent_key(different_rows),
        }

    workload = Workload("supervisor", (Request("r", 0, None, 1,
        (Segment("doc", "doc", 256),), 256),), {})
    result["zero_diff_master_store_formula"] = {}
    for rung in ("A1", "A2", "A3b", "A4c", "A4e", "A5", "A6"):
        report, _ = probes.run(workload, rung)
        result["zero_diff_master_store_formula"][rung] = [
            {key: event[key] for key in ("name", "device", "rows", "time_s")}
            for event in report["events"] if event["name"] == "dram_store_master"]
    stores = result["zero_diff_master_store_formula"]
    result["A3b_over_A4c_master_store_duration"] = (
        stores["A3b"][0]["time_s"] / stores["A4c"][0]["time_s"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
