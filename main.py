from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from src.ablation import (DECODE_ATTN_MODES, KV_MAPPINGS,
                          PREFILL_ATTN_MODES, PRESETS)
from src.workload import (Workload, WorkloadValidationError, build_reuse_plan,
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
    parser.add_argument(
        "--pim-link",
        choices=("nvlink3", "nvlink4", "pcie5", "pcie4"),
        default="nvlink3",
        help="GPU <-> AttAcc link (600 / 900 / 128 / 64 GB/s bidirectional); "
             "the GPU <-> GPU all-reduce fabric is unaffected")
    parser.add_argument(
        "--attn-splitk",
        action="store_true",
        help="flash GPU model only: let short-Q prefill attention split its "
             "key range across CTAs (flash-attn num_splits heuristic)")
    parser.add_argument(
        "--gpu-model",
        choices=("legacy", "refined", "flash"),
        default="legacy",
        help="GPU performance model. legacy: original AttAcc xPU formulas; "
             "refined: projection and attention GEMMs at the measured cuBLAS "
             "efficiency for their size, NVLink latency + far-HBM streaming "
             "on GPU<->AttAcc transfers, everything else legacy; flash: "
             "refined plus attention as a fused FlashAttention-2 kernel "
             "(128-row Q blocks, softmax on-chip, flash-decoding for m = 1)")
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
        "--history-len",
        type=int,
        default=None,
        help="agentic multi-turn: pre-existing own-KV rows every agent already "
             "holds from earlier turns (overrides per-request 'history_len' in "
             "the workload JSON); attended during prefill/decode, never recomputed")
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
        help="for --reuse cacheblend/epic. physical: TLB/Ramulator-DAG reference; "
             "analytic: legacy AttAcc model with prefill scaled by recomputed work")
    parser.add_argument(
        "--tier-batch-size",
        type=int,
        default=0,
        help="legacy/analytic paths: serve each dependency tier as serial padded "
             "batches of at most this many requests (0 = one batch per tier)")
    parser.add_argument(
        "--cacheblend-rotate-mode",
        choices=("gpu", "die", "bank"),
        default="gpu",
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
        "--ablation",
        choices=tuple(sorted(PRESETS)),
        help="named placement configuration of the KV-reuse study: "
             "A1 pure AttAcc no reuse, A2 pure GPU software reuse, "
             "A3 software prefill + PIM decode with naive KV mapping, "
             "A4 same with master/diff pools, A5 prefill attention also on PIM, "
             "A6 GPU/PIM split prefill.  Individual switches below override it")
    parser.add_argument(
        "--prefill-attn",
        choices=PREFILL_ATTN_MODES,
        help="where prefill attention runs: gpu (pure software), split (GPU "
             "attends fresh rows, PIM scans reused KV), pim (all on PIM)")
    parser.add_argument(
        "--decode-attn",
        choices=DECODE_ATTN_MODES,
        help="where decode attention runs and therefore where the KV cache lives")
    parser.add_argument(
        "--kv-mapping",
        choices=KV_MAPPINGS,
        help="physical KV layout in the PIM: private (no reuse), naive (one "
             "shared channel pool, software chunk order, no PIM-aware remap), "
             "master-diff (immutable rows and recomputed rows in disjoint "
             "channel pools), none (KV stays in GPU memory)")
    parser.add_argument(
        "--pim-prefill-query-batch",
        type=int,
        default=4,
        help="queries sharing one PIM K/V stream when prefill attention runs on "
             "the PIM (--prefill-attn pim/split)")
    parser.add_argument(
        "--pim-prefill-mode",
        choices=("split", "bank-whole"),
        default="split",
        help="reuse-prefill attention placement in the physical DAG.  split "
             "(default): GPU attends fresh rows to each other, PIM scans the "
             "reused KV, DIE merges.  bank-whole: the batch's own K/V lands "
             "first, every query scans the full landed range in the banks and "
             "the DIE drops non-causal positions (no GPU triangle, no LSE)")
    parser.add_argument(
        "--pim-batch-command",
        choices=("mq", "replicate"),
        default="mq",
        help="how a multi-query PIM sweep issues its MACs.  mq (default, the "
             "Fugue design): one MAC_AB per column serves every resident Q; "
             "the bank PE multiplies internally and the command interval "
             "carries the n-fold PE time plus the IDD7 power stretch.  "
             "replicate: legacy one MAC_AB per (column, query)")
    parser.add_argument(
        "--pe-freq-ghz",
        type=float,
        default=0.666,
        help="bank GEMV-unit (PE) clock in GHz for the mq batch command; "
             "0.666 is AttAcc's synthesized point (tCCDS-matched)")
    parser.add_argument(
        "--gemv-buffer-bytes",
        type=int,
        default=512,
        help="per-bank GEMV input-vector buffer size; one query slice is "
             "64 B, so this caps the queries resident in one sweep (the "
             "sweep splits beyond it).  512 = AttAcc's 16 x 256-bit buffer")
    parser.add_argument(
        "--kv-pool-split",
        type=int,
        default=8,
        help="channels given to the master pool under --kv-mapping master-diff; "
             "the remaining 16 - N channels hold the diff pool")
    parser.add_argument(
        "--master-shadow",
        choices=("read-mask", "skip"),
        default="read-mask",
        help="under --kv-mapping master-diff, what the master pool does with a row "
             "a correction has overwritten: read-mask streams it and drops it from "
             "the score (contiguous run); skip leaves it out of the address stream "
             "(breaks the run at every correction)")
    parser.add_argument(
        "--split-attn",
        choices=("overlap", "serial"),
        default="overlap",
        help="under --prefill-attn split, whether the GPU's fresh-row attention "
             "and the PIM's reused-KV scan run concurrently (overlap: the layer "
             "costs max of the two branches, matching KVpim-sim's trace) or are "
             "charged one after the other (serial)")
    parser.add_argument(
        "--no-prefill-kv-readback",
        action="store_true",
        help="assume software prefill already has the reused KV in GPU memory "
             "instead of reading it back from the PIM over the link")
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
    parser.add_argument(
        "--workload-report-events",
        choices=("full", "none"),
        default="full",
        help="full: include every DAG event and TLB entry in the JSON report; "
             "none: keep only the summary (per-request/tier times, energy, blocks)")

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
            if args.history_len is not None:
                if args.history_len < 0:
                    parser.error("--history-len must be non-negative")
                workload = Workload(workload.kind, tuple(
                    replace(request, history_len=args.history_len)
                    for request in workload.requests), workload.raw)
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
    pim_link = {"nvlink3": InterfaceType.NVLINK3, "nvlink4": InterfaceType.NVLINK4,
                "pcie5": InterfaceType.PCIE5, "pcie4": InterfaceType.PCIE4}[args.pim_link]
    pim_link_bw = {"nvlink3": 600, "nvlink4": 900, "pcie5": 128, "pcie4": 64}[args.pim_link] * 1000 * 1000 * 1000
    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=gmem_cap,
                                 gpu_model=args.gpu_model, pim_link_bw=pim_link_bw,
                                 attn_splitk=args.attn_splitk)
    system = System(xpu_config['GPU'], modelinfos)
    if args.system in ['dgx-attacc']:
        if args.pim == "bg":
            pim_type = PIMType.BG
        elif args.pim == "buffer":
            pim_type = PIMType.BUFFER
        else:
            pim_type = PIMType.BA
        pim_config = make_pim_config(pim_type,
                                     pim_link,
                                     power_constraint=args.powerlimit)
        system.set_accelerator(modelinfos, DeviceType.PIM, pim_config,
                               ramulator_workers=args.ramulator_workers)

    elif args.system in ['dgx-cpu']:
        xpu_config = make_xpu_config(gpu_device, gpu_model=args.gpu_model)
        system.set_xpu(xpu_config['GPU'])
        system.set_accelerator(modelinfos, DeviceType.CPU, xpu_config['CPU'])

    if workload is not None:
        from src.workload_runner import (run_cacheblend_analytic_report,
                                         run_no_reuse_report, run_reuse_prefill)
        if args.ablation or args.prefill_attn or args.decode_attn or args.kv_mapping:
            from src.ablation import resolve_config, run_ablation_report
            try:
                ablation = resolve_config(
                    args.ablation, args.prefill_attn, args.decode_attn,
                    args.kv_mapping, policy=args.reuse,
                    pim_prefill_query_batch=args.pim_prefill_query_batch,
                    master_pool_channels=args.kv_pool_split,
                    prefill_kv_readback=not args.no_prefill_kv_readback,
                    master_shadow=args.master_shadow,
                    split_overlap=(args.split_attn == "overlap"),
                    pim_batch_command=args.pim_batch_command,
                    pim_pe_freq_ghz=args.pe_freq_ghz,
                    gemv_buffer_bytes=args.gemv_buffer_bytes)
                report = run_ablation_report(
                    system, workload, reuse_plan, ablation, pipe=args.pipeopt,
                    parallel_ff=args.ffopt, power_constraint=args.powerlimit,
                    batch_size=args.tier_batch_size or None)
            except WorkloadValidationError as exc:
                parser.error(str(exc))
            report["workload"] = workload_summary(workload, reuse_plan)
            with open(args.workload_report, "w") as report_file:
                json.dump(report, report_file, indent=2, sort_keys=True)
                report_file.write("\n")
            print("Wrote ablation report to {}".format(args.workload_report))
            headline = {key: report.get(key) for key in
                        ("policy", "makespan_s", "prefill_s", "decode_s",
                         "energy_nj", "prefill_energy_nj", "decode_energy_nj")}
            headline["ablation"] = report["ablation"]
            headline["kv_gib"] = report["memory"]["kv_gib"]
            headline["kv_bytes_vs_no_reuse"] = report["memory"]["kv_bytes_vs_no_reuse"]
            print("REPORT_SUMMARY " + json.dumps(headline, sort_keys=True))
            return
        if args.reuse == "no-reuse" and args.no_reuse_latency_model == "legacy":
            perfs, report = run_no_reuse_report(
                system, workload, pipe=args.pipeopt, parallel_ff=args.ffopt,
                power_constraint=args.powerlimit,
                batch_size=args.tier_batch_size or None)
            write_csv(output_path, perfs)
            report["workload"] = workload_summary(workload, reuse_plan)
            with open(args.workload_report, "w") as report_file:
                json.dump(report, report_file, indent=2, sort_keys=True)
                report_file.write("\n")
            print("Wrote no-reuse execution report to {}".format(
                args.workload_report))
            print("REPORT_SUMMARY " + json.dumps(
                {key: report.get(key) for key in
                 ("policy", "batch_size", "makespan_s", "energy_nj",
                  "prefill_energy_nj", "decode_energy_nj", "tiers")},
                sort_keys=True))
        else:
            try:
                if args.reuse == "no-reuse":
                    report = run_reuse_prefill(
                        system, workload, reuse_plan, pipe=args.pipeopt,
                        cacheblend_batch_size=args.cacheblend_batch_size,
                        cacheblend_rotate_mode=args.cacheblend_rotate_mode,
                        include_events=(args.workload_report_events == "full"),
                        pim_prefill_mode=args.pim_prefill_mode,
                        pim_batch_command=args.pim_batch_command,
                        pim_pe_freq_ghz=args.pe_freq_ghz,
                        gemv_buffer_bytes=args.gemv_buffer_bytes)
                elif (args.reuse in ("cacheblend", "epic") and
                      args.cacheblend_latency_model == "analytic"):
                    report = run_cacheblend_analytic_report(
                        system, workload, reuse_plan, pipe=args.pipeopt,
                        parallel_ff=args.ffopt,
                        power_constraint=args.powerlimit,
                        batch_size=args.tier_batch_size or None)
                else:
                    report = run_reuse_prefill(
                        system, workload, reuse_plan, pipe=args.pipeopt,
                        cacheblend_batch_size=args.cacheblend_batch_size,
                        cacheblend_rotate_mode=args.cacheblend_rotate_mode,
                        include_events=(args.workload_report_events == "full"),
                        pim_prefill_mode=args.pim_prefill_mode,
                        pim_batch_command=args.pim_batch_command,
                        pim_pe_freq_ghz=args.pe_freq_ghz,
                        gemv_buffer_bytes=args.gemv_buffer_bytes)
            except WorkloadValidationError as exc:
                parser.error(str(exc))
            report["workload"] = workload_summary(workload, reuse_plan)
            with open(args.workload_report, "w") as report_file:
                json.dump(report, report_file, indent=2, sort_keys=True)
                report_file.write("\n")
            print("Wrote reuse execution report to {}".format(args.workload_report))
            headline = {key: report.get(key) for key in
                        ("policy", "makespan_s", "energy_nj", "energy_unit", "link_bytes",
                         "gpu_time_s_unoverlapped", "pim_pool_time_s_unoverlapped",
                         "die_time_s_unoverlapped", "event_count")}
            headline["summary"] = report.get("summary")
            print("REPORT_SUMMARY " + json.dumps(headline, sort_keys=True))
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
