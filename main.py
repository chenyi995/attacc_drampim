from __future__ import annotations

import argparse
import csv
import json
import os
from src.workload import (WorkloadValidationError, build_reuse_plan,
                          load_workload, workload_summary)

RAMULATOR = False


def parse_layer_indices(value):
    """Parse a comma-separated, non-negative layer-index list."""
    if value is None or not value.strip():
        return ()
    try:
        indices = []
        for raw_part in value.split(','):
            part = raw_part.strip()
            if '-' in part:
                start, end = (int(item.strip()) for item in part.split('-', 1))
                if end < start:
                    raise ValueError
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
        indices = tuple(indices)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layer list must be comma-separated non-negative integers") from exc
    if any(index < 0 for index in indices) or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError(
            "layer list must have unique non-negative indices")
    return indices


def write_csv(logfile, perfs):
    if logfile is not None:
        firstrow = False
        if not os.path.exists(logfile):
            firstrow = True

        f = open(logfile, 'a')
        wrt = csv.writer(f)
        if firstrow:
            col_name = [
                'model', 'dtype', 'xpu', 'cap', 'bw', 'sys_opb', 'hw', 'cores',
                'pipe_level', 'is parallel', 'power constraint', 'gqa_size',
                'Lin', 'Lout', 'bs', 'required_cap', 's_flops',
                'g_flops', 's_time', 's_matmul', 's_fc', 's_comm', 's_softmax',
                's_act', 's_lnorm', 'g_time (ms)', 'g_matmul', 'g_fc', 'g_comm',
                'g_etc', 'g_qkv_time', 'g_prj_time', 'g_ff_time', 'g2g_comm',
                'c2g_comm', 'g_softmax', 'g_act', 'g_lnorm', 'g_energy (nJ)',
                'g_dram_energy', 'g_l2_energy', 'g_l1_energy', 'g_reg_energy',
                'g_alu_energy', 'g_fc_mem_energy', 'g_fc_comp_energy',
                'g_attn_mem_energy', 'g_attn_comp_energy', 'g_etc_mem_energy',
                'g_etc_comp_energy', 'g_comm_energy'
            ]
            wrt.writerow(col_name)

        for perf in perfs:
            tag, config, time, energy = perf
            info = tag + config + time + energy
            wrt.writerow(info)
        f.close()


def run(system: System,
        batch,
        lin,
        lout,
        power_constraint=False,
        pipe=0,
        parallel=False,
        output_file=None):
    print("---Run simple mode Batch {} Lin {} Lout {} pipe {} parall {}---".
          format(batch, lin, lout, pipe, parallel))
    assert system.model_set, "Need to SetModel"
    perfs = []
    system.simulate(batch,
                    lin,
                    lout,
                    perfs=perfs,
                    pipe=pipe,
                    parallel_ff=parallel,
                    power_constraint=power_constraint)
    if output_file is not None:
        write_csv(output_file, perfs)


def main():
    parser = argparse.ArgumentParser(
        description="Model configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## set system configuration
    parser.add_argument(
        "--system",
        type=str,
        default="dgx",
        help="dgx (each GPU has 80GB HBM), \
              dgx-cpu (In dgx, offloading the attention layer to cpu), \
              dgx-attacc (dgx + attacc)")
    parser.add_argument(
        "--gpu",
        type=str,
        default='A100a',
        help="GPU type (A100a and H100), A100a is A100 with HBM3")
    parser.add_argument("--ngpu",
                        type=int,
                        default=8,
                        help="number of GPUs in DGX system. default=8")
    parser.add_argument("--gmemcap",
                        type=int,
                        default=80,
                        help="memory capacity per GPU (GB). default=80")



    ## set attacc configuration
    parser.add_argument("--pim",
                        type=str,
                        default='bank',
                        help="pim mode. list: bank, bg, buffer")
    parser.add_argument("--powerlimit",
                        action='store_true',
                        help="power constraint for PIM ")
    parser.add_argument("--ffopt",
                        action='store_true',
                        help="apply feedforward parallel optimization")
    parser.add_argument("--pipeopt",
                        action='store_true',
                        help="apply pipeline optimization ")

    ## set model and service environment
    parser.add_argument(
        "--model",
        type=str,
        default='GPT-175B',
        help="model list: GPT-175B, LLAMA-65B, MT-530B, OPT-66B")
    parser.add_argument("--word",
                        type=int,
                        default='2',
                        help="word size (precision): 1(INT8), 2(FP16)")
    parser.add_argument("--lin",
                        type=int,
                        default=2048,
                        help="input sequence length")
    parser.add_argument("--lout",
                        type=int,
                        default=128,
                        help="number of generated tokens")
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=
        "batch size, default = 1"
    )
    parser.add_argument(
        "--workload",
        type=str,
        help="path to a RAG legacy-list or supervisor v2-dag workload JSON")
    parser.add_argument(
        "--reuse",
        choices=("no-reuse", "cacheblend", "epic"),
        default="no-reuse",
        help="KV reuse policy; independent from workload kind")
    parser.add_argument(
        "--no-reuse-latency-model",
        choices=("legacy", "physical"),
        default="legacy",
        help="legacy: original AttAcc aggregate model; physical: contiguous private-KV DAG/Ramulator model")
    parser.add_argument(
        "--cacheblend-recompute-ratio",
        type=float,
        default=0.0,
        help="fraction of reusable tokens recomputed in each CacheBlend partial layer")
    parser.add_argument(
        "--cacheblend-full-layers",
        type=parse_layer_indices,
        default=(),
        help="comma-separated CacheBlend full-recompute layer indices (e.g. 0,1)")
    parser.add_argument(
        "--cacheblend-partial-layers",
        type=parse_layer_indices,
        default=(),
        help="comma-separated CacheBlend partial-recompute layer indices (e.g. 2,3)")
    parser.add_argument(
        "--cacheblend-batch-size",
        type=int,
        default=1,
        help="CacheBlend PIM-ready decode batch size (1 preserves the unbatched DAG)")
    parser.add_argument(
        "--cacheblend-latency-model",
        choices=("physical", "analytic"),
        default="physical",
        help="physical: TLB/Ramulator-DAG reference; analytic: legacy-model comparable estimate")
    parser.add_argument(
        "--cacheblend-rotate-mode",
        choices=("gpu", "die", "bank"),
        default="die",
        help="where CacheBlend emits position-shifted Q variants")
    parser.add_argument(
        "--ramulator-workers",
        type=int,
        default=1,
        help="host CPU workers for independent Ramulator trace jobs; does not alter simulated hardware")
    parser.add_argument(
        "--epic-prefix-recompute-tokens",
        type=int,
        default=1,
        help="number of leading tokens to recompute in each reused segment for EPIC")
    parser.add_argument(
        "--reuse-seed",
        type=int,
        default=0,
        help="seed for CacheBlend's randomly distributed recompute tokens")
    parser.add_argument(
        "--validate-workload",
        action="store_true",
        help="validate --workload and print its tier/reuse plan without running hardware simulation")
    parser.add_argument(
        "--workload-plan",
        type=str,
        help="optional path to write the validated workload/reuse plan as JSON")
    parser.add_argument(
        "--workload-report",
        type=str,
        default="workload_output.json",
        help="JSON report path for reuse execution; no-reuse still writes output.csv")

    args = parser.parse_args()
    if args.ramulator_workers < 1:
        parser.error("--ramulator-workers must be at least 1")

    workload = None
    reuse_plan = None
    if args.workload or args.validate_workload or args.workload_plan:
        if not args.workload:
            parser.error("--validate-workload and --workload-plan require --workload")
        try:
            workload = load_workload(args.workload)
            reuse_plan = build_reuse_plan(
                workload, args.reuse, args.cacheblend_recompute_ratio,
                args.reuse_seed, args.cacheblend_full_layers,
                args.cacheblend_partial_layers,
                args.epic_prefix_recompute_tokens)
        except WorkloadValidationError as exc:
            parser.error(str(exc))
        summary = workload_summary(workload, reuse_plan)
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.workload_plan:
            with open(args.workload_plan, "w") as plan_file:
                json.dump(summary, plan_file, indent=2, sort_keys=True)
                plan_file.write("\n")
        if args.validate_workload:
            return

    global RAMULATOR
    if RAMULATOR:
        print("The Ramulator {}".format(RAMULATOR))

    # Validation-only workload commands do not need pandas/Ramulator.  Keep
    # the existing heavy simulator imports on the actual execution path.
    from src.system import System
    from src.type import (DataType, DeviceType, GPUType, InterfaceType,
                          PIMType)
    from src.config import make_model_config, make_pim_config, make_xpu_config

    if args.gpu == 'H100':
        gpu_device = GPUType.H100
    elif args.gpu == 'A100a':
        gpu_device = GPUType.A100a
    else:
        assert 0

    if args.system == 'dgx-attacc':
        print("{}: ({} x {}), PIM:{}, [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu, args.pim,
            [args.lin, args.lout, args.batch]))
    else:
        print("{}: ({} x {}), [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu,
            [args.lin, args.lout, args.batch]))
    num_gpu = args.ngpu
    gmem_cap = args.gmemcap * 1024 * 1024 * 1024
    output_path = "output.csv"
    if os.path.exists(output_path):
        os.system("rm " + output_path)

    # set system
    dtype = DataType.W16A16 if args.word == 2 else DataType.W8A8
    modelinfos = make_model_config(args.model, dtype)
    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=gmem_cap)
    system = System(xpu_config['GPU'], modelinfos)
    if args.system in ['dgx-attacc']:
        if args.pim == "bg":
            pim_type = PIMType.BG
        elif args.pim == "buffer":
            pim_type = PIMType.BUFFER
        else:
            pim_type = PIMType.BA
        pim_config = make_pim_config(pim_type,
                                     InterfaceType.NVLINK3,
                                     power_constraint=args.powerlimit)
        system.set_accelerator(modelinfos, DeviceType.PIM, pim_config,
                               ramulator_workers=args.ramulator_workers)

    elif args.system in ['dgx-cpu']:
        xpu_config = make_xpu_config(gpu_device)
        system.set_xpu(xpu_config['GPU'])
        system.set_accelerator(modelinfos, DeviceType.CPU, xpu_config['CPU'])

    if workload is not None:
        from src.workload_runner import (run_cacheblend_analytic_report,
                                         run_no_reuse_report, run_reuse_prefill)
        if args.reuse == "no-reuse" and args.no_reuse_latency_model == "legacy":
            perfs, report = run_no_reuse_report(
                system, workload, pipe=args.pipeopt, parallel_ff=args.ffopt,
                power_constraint=args.powerlimit)
            write_csv(output_path, perfs)
            report["workload"] = workload_summary(workload, reuse_plan)
            with open(args.workload_report, "w") as report_file:
                json.dump(report, report_file, indent=2, sort_keys=True)
                report_file.write("\n")
            print("Wrote no-reuse execution report to {}".format(
                args.workload_report))
        else:
            try:
                if args.reuse == "no-reuse":
                    report = run_reuse_prefill(
                        system, workload, reuse_plan, pipe=args.pipeopt,
                        cacheblend_batch_size=args.cacheblend_batch_size,
                        cacheblend_rotate_mode=args.cacheblend_rotate_mode)
                elif args.reuse == "cacheblend" and args.cacheblend_latency_model == "analytic":
                    report = run_cacheblend_analytic_report(
                        system, workload, reuse_plan, pipe=args.pipeopt,
                        parallel_ff=args.ffopt,
                        power_constraint=args.powerlimit)
                else:
                    report = run_reuse_prefill(
                        system, workload, reuse_plan, pipe=args.pipeopt,
                        cacheblend_batch_size=args.cacheblend_batch_size,
                        cacheblend_rotate_mode=args.cacheblend_rotate_mode)
            except WorkloadValidationError as exc:
                parser.error(str(exc))
            report["workload"] = workload_summary(workload, reuse_plan)
            with open(args.workload_report, "w") as report_file:
                json.dump(report, report_file, indent=2, sort_keys=True)
                report_file.write("\n")
            print("Wrote reuse execution report to {}".format(args.workload_report))
        return

    run(system,
        args.batch,
        args.lin,
        args.lout,
        pipe=args.pipeopt,
        parallel=args.ffopt,
        output_file=output_path,
        power_constraint=args.powerlimit)


if __name__ == "__main__":
    main()
