"""Small physical CacheBlend reuse smoke tests derived from both input JSONs.

The full workloads are intentionally not shortened silently by the normal
runner.  This script makes its reduced scope explicit: one decoder layer and
one generated token, retaining a real RAG fingerprint reuse edge and a real
supervisor -> worker parent_out edge.  It still uses the production GPU/PIM
devices, concrete TLB addresses and Ramulator.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import make_model_config, make_pim_config, make_xpu_config
from src.system import System
from src.type import DataType, DeviceType, GPUType, InterfaceType, PIMType
from src.workload import Request, Workload, build_reuse_plan, load_workload
from src.workload_runner import run_reuse_prefill


OUT = Path(__file__).resolve().parent


def make_system():
    modelinfo = make_model_config("GPT-175B", DataType.W16A16)
    xpu = make_xpu_config(GPUType.A100a, num_gpu=8)
    system = System(xpu["GPU"], modelinfo)
    pim = make_pim_config(PIMType.BA, InterfaceType.NVLINK3)
    system.set_accelerator(modelinfo, DeviceType.PIM, pim)
    # Deliberately a one-layer smoke, while retaining GPT-175B width/head
    # dimensions and the production bank-level Ramulator path.
    system.model.ndec = 1
    return system


def rag_smoke():
    source = load_workload(ROOT / "workload/workload_2wikimqa_first8.json")
    owner, consumer = source.requests[:2]
    # Both samples retain their real shared system fingerprint.  The original
    # 41-token segment is widened only for this smoke because AttAcc's GPU
    # tiler has no valid tile below its minimum matrix extent.
    requests = tuple(Request(request_id=request.request_id, tier=0,
                             parent_id=None, lout=1,
                             segments=(replace(request.segments[0], length=512),
                                       replace(request.segments[-1], length=64)),
                             total_length=576)
                     for request in (owner, consumer))
    return Workload("rag", requests, {"smoke_source": "2wikimqa_first8"})


def relay_smoke():
    source = load_workload(ROOT / "workload/workload_relay_s400w4t1.json")
    sup = source.requests[0]
    workers = source.requests[1:]
    # Retain shared sys0, the parent_out dependency, and a private instruction
    # byte.  parent_out is shortened consistently with the supervisor lout.
    sup_segments = (replace(sup.segments[0], length=512),
                    replace(sup.segments[1], length=64))
    reduced = [Request(sup.request_id, 0, None, 1, sup_segments, 576)]
    for worker in workers:
        segments = (replace(worker.segments[0], length=512),
                    replace(worker.segments[1], length=1),
                    replace(worker.segments[2], length=64))
        reduced.append(Request(worker.request_id, 1, sup.request_id, 1,
                               segments, 577))
    return Workload("supervisor", tuple(reduced),
                    {"smoke_source": "relay_s400w4t1"})


def run(name, workload, batch_size):
    plan = build_reuse_plan(workload, "cacheblend", 0.0, 7, (), (0,))
    report = run_reuse_prefill(make_system(), workload, plan, pipe=True,
                               cacheblend_batch_size=batch_size,
                               cacheblend_rotate_mode="die")
    scans = [event for event in report["events"]
             if "pim_kv_scan" in event["name"]]
    assert scans, "smoke must exercise a physical PIM KV scan"
    assert report["tlb"]["entries"], "smoke must populate the TLB"
    report["smoke"] = {
        "scope": "one decoder layer, one generated token",
        "source_workload": name,
        "pim_scan_events": len(scans),
        "batch_size": batch_size,
    }
    with open(OUT / (name + "_cacheblend_smoke.json"), "w") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    run("rag", rag_smoke(), batch_size=2)
    run("relay", relay_smoke(), batch_size=2)
