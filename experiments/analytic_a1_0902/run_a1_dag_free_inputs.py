#!/usr/bin/env python3
"""Run the DAG-free A1 input enumerator and calibrated PIM model."""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.a1_dag_free import (enumerate_a1_pim_inputs, input_summary,
                             pim_energy_nj)
from src.analytic_pim import estimate, validation_report as analytic_validation_report
from src.config import make_model_config, make_pim_config
from src.model import Transformer
from src.model import Layer
from src.ramulator_wrapper import Ramulator
from src.type import DataType, InterfaceType, LayerType, PIMType
from src.workload import load_workload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True)
    p.add_argument("--model", default="LLAMA-7B")
    p.add_argument("--timing-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-hbm", type=int, default=5)
    p.add_argument("--tensor-parallel", type=int, default=8)
    # main.py's --powerlimit is ON in every A1 run we compare against, so the
    # default here matches it.  It was OFF before, which silently priced the
    # two entry points with different nCCDAB.
    p.add_argument("--no-power-constraint", dest="power_constraint",
                   action="store_false", default=True)
    p.add_argument("--power-constraint", dest="power_constraint",
                   action="store_true")
    p.add_argument("--ramulator-workers", type=int, default=1)
    p.add_argument("--mode", choices=("analytic", "ramulator"), default="analytic")
    a = p.parse_args()
    started = time.perf_counter()
    workload = load_workload(a.workload)
    info = make_model_config(a.model, DataType.W16A16)
    model = Transformer(info, tensor_parallel=a.tensor_parallel)
    t0 = time.perf_counter()
    inputs = enumerate_a1_pim_inputs(workload, model, dbyte=2, num_hbm=a.num_hbm)
    enumerate_s = time.perf_counter() - t0
    models = json.loads(Path(a.timing_model).read_text())
    if models.get("version") != 2:
        raise SystemExit(
            "{}: expected a version-2 (three-layer, held-out validated) timing "
            "model; recalibrate with calibrate.py".format(a.timing_model))
    # Every estimate is accounted for; an uncalibrated or extrapolated one is
    # reported next to the number it produced.
    diagnostics = {}
    ramulator = None
    if a.mode == "ramulator":
        ram_dir = os.environ.get("ATTACC_RAMULATOR_DIR", "ramulator2")
        # output_log='' disables the pandas CSV mirror.  The DAG path never
        # writes it, and rewriting a growing CSV once per priced input would
        # put host I/O into a wall-clock comparison that is meant to measure
        # simulation cost.
        ramulator = Ramulator(info, ram_dir, output_log="",
                              workers=a.ramulator_workers, num_hbm=a.num_hbm)
    pim_config = make_pim_config(PIMType.BA, list(InterfaceType)[0],
                                 num_hbm=a.num_hbm,
                                 power_constraint=a.power_constraint)
    energy_table = pim_config["ENERGY_TABLE"]
    num_attacc = pim_config["NUM_ATTACC"]
    t0 = time.perf_counter()
    total_cycles = 0
    total_energy_nj = 0.0
    commands = Counter()
    for item, count in inputs.items():
        if a.mode == "analytic":
            cycle, mac, sfm, mvgb, mvsb, wrgb = estimate(
                pim_type="BA", run_length=item.run_length,
                num_ops_per_hbm=item.num_ops_per_hbm, dbyte=item.dbyte,
                power_constraint=a.power_constraint, dhead=item.dhead,
                num_hbm=a.num_hbm, channel_count=item.channel_count,
                shared_kv=item.shared_kv,
                shared_queries=item.shared_queries, channel_base=item.channel_base,
                mq_command=False, key_addr=item.key_addr, value_addr=item.value_addr,
                phase=item.phase, trace_revision="chunkstripe1", timing_models=models,
                diagnostics=diagnostics)
        else:
            layer = Layer("dag-free", "score", LayerType.MATMUL, False,
                          DataType.W16A16, 1, item.run_length, item.dhead,
                          item.num_ops_per_hbm * a.num_hbm)
            layer.pim_kv_runs = ((item.key_addr, item.value_addr, item.run_length,
                                  item.channel_base, item.channel_count),)
            layer.pim_shared_queries = item.shared_queries
            # ``run`` returns seconds + traffic.  Convert its timing back to
            # cycles; command counters remain analytic audit counters here.
            layer.pim_shared_kv = item.shared_kv
            seconds, _traffic = ramulator.run(PIMType.BA, layer, a.power_constraint)
            cycle = round(seconds * 1e9 / ramulator.tCK)
            _zero, mac, sfm, mvgb, mvsb, wrgb = estimate(
                pim_type="BA", run_length=item.run_length,
                num_ops_per_hbm=item.num_ops_per_hbm, dbyte=item.dbyte,
                power_constraint=a.power_constraint, dhead=item.dhead,
                num_hbm=a.num_hbm, channel_count=item.channel_count,
                shared_kv=item.shared_kv, shared_queries=item.shared_queries,
                channel_base=item.channel_base, mq_command=False, key_addr=item.key_addr,
                value_addr=item.value_addr, phase=item.phase,
                trace_revision="chunkstripe1", timing_models=models,
                diagnostics=diagnostics)
        total_cycles += count * cycle
        total_energy_nj += count * pim_energy_nj(
            item, (mac, sfm, mvgb, mvsb, wrgb), num_hbm=a.num_hbm,
            num_attacc=num_attacc, energy_table=energy_table)
        for key, value in zip(("mac", "sfm", "mvgb", "mvsb", "wrgb"),
                              (mac, sfm, mvgb, mvsb, wrgb)):
            commands[key] += count * value
    eval_s = time.perf_counter() - t0
    pricing = (ramulator.cache_report()["host_pricing_seconds"]
               if ramulator is not None else None)
    report = {
        "engine": "a1-dag-free-input-enumerator",
        "pim_pricer": a.mode,
        "latency_model": "not-scheduled-yet",
        "dag_build_s": 0.0,
        "input_enumeration_s": enumerate_s,
        "pim_model_eval_s": eval_s,
        "macro_schedule_s": 0.0,
        "wall_s": time.perf_counter() - started,
        "pim_cycles_unordered": total_cycles,
        "pim_time_s_unordered": total_cycles * 0.769e-9,
        # Comparable to the DAG's energy_breakdown_nj.by_event
        # pim_kv_scan_score_softmax_pv + decode_pim_kv_scan_score_softmax_pv.
        # NOT to by_class PIM, which also holds bandwidth-priced KV stores.
        "pim_scan_energy_nj": total_energy_nj,
        "power_constraint": a.power_constraint,
        "tensor_parallel": a.tensor_parallel,
        "pim_command_counts": dict(commands),
        "inputs": input_summary(inputs),
        "workload": a.workload,
        "model": a.model,
        "pim_model_diagnostics": diagnostics,
        "host_pricing_seconds": pricing,
        "pim_model_validation": analytic_validation_report(models),
        # The enumerator produces WORK, not a schedule.  Saying so in the
        # report keeps anyone from reading pim_cycles_unordered as a makespan.
        "caveat": "pim_cycles_unordered is unscheduled work; it is not a makespan",
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
