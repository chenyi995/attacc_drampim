# GPU+PIM cooperative prefill vs pure-GPU prefill (replay-pair micro-benchmark)

Headline result: `RESULTS.md` (two tables: CacheBlend r limit and EPIC recompute-token limit per model and link); details in `RESULTS_details.md`.

User-designed input (2026-08-21): a workload with exactly two requests.
Request 0 = sys(32) + doc(L, shared) + query(n_new) is prefilled entirely on the
GPU (populates the AttAcc KV).  Request 1 = sys(n_off=16, unique) + the same
doc + query(n_new): the shared doc sits n_off positions later, so the GPU has
to absorb a position offset (Q rotation; not priced by the analytic ablation
model -- an O(n_q d) elementwise op).  Generator: `workload/gen_replay_pair.py`
(copies of the generated JSONs in `workloads/`); run with `--tier-batch-size 1`
so request 1 is the second batch of tier 0 and reuses request 0's KV.
Measured quantity: `tiers[1]["prefill_s"]` of A4 (pure GPU, reused K/V read
back over the link) vs A6 (GPU attends fresh rows, PIM scans the reused K/V)
vs A5.

Settings: LLAMA-7B / LLAMA-65B / GPT-175B (run_one.sh honours MODEL=...), 8 x A100a + 8 AttAcc (bank PIM), `--gpu-model flash`,
CacheBlend full layers 0-1 / partial 2-31, seed 7, `--pim-prefill-query-batch 4`,
links NVLink3 (600 GB/s) and PCIe4 (64 GB/s), L = 8192.

`run_one.sh <dir> <workload-name> <A4|A5|A6> <nvlink3|pcie4> <cacheblend|epic> "<extra flags>"`
(`<dir>/wl/<workload-name>.json` in, `<dir>/<tag>.json` out).  Results JSON in
`results/`, table in `RESULTS.md`.

## Turning-point grids (2026-08-21, later the same day)

* `results/grid/`        – flash GPU, n_off 16, n_new ∈ {4,8,16,24,32,48,64,96,128,192,256,384}, CacheBlend r=0 and EPIC p=1, NVLink3 + PCIe4, A4/A6, 3 models (288 runs, `run_one.sh`).
* `results/gridsk/`      – same with `--attn-splitk` (flash-attn num_splits for short-Q prefill; `run_one_splitk.sh`).
* `results/gridsk_small/`– split-K, n_off 1, n_new ∈ {1,2,4,8,12,16,24}, NVLink3 only, to bracket the small turning points.
* `analyze_grid.py <grid>` → `results/turning_points_<grid>.md`; `RESULTS.md` is assembled from the three.
* Earlier coarse sweep (n_new × r × model): `RESULTS_old_coarse_sweep.md`, JSONs at the top of `results/`.

## Reproduce

```bash
# 1. build Ramulator once (see top-level README), then from the repo root:
python workload/gen_replay_pair.py --L 8192 --n-new 8 --n-off 16 --out /tmp/out/wl/replay_L8192_q8.json
# 2. one cell: <outdir> <workload name> <A4|A5|A6> <nvlink3|pcie4> <cacheblend|epic> "<extra flags>"
#    (<outdir>/wl/<name>.json in, <outdir>/<tag>.json out; MODEL=LLAMA-7B|LLAMA-65B|GPT-175B)
MODEL=LLAMA-7B experiments/GPU_PIM_vs_GPU_prefill/run_one.sh /tmp/out replay_L8192_q8 A4 nvlink3 epic "--epic-prefix-recompute-tokens 1"
MODEL=LLAMA-7B experiments/GPU_PIM_vs_GPU_prefill/run_one.sh /tmp/out replay_L8192_q8 A6 nvlink3 epic "--epic-prefix-recompute-tokens 1"
# 3. turning-point tables from a grid directory laid out like results/grid:
python experiments/GPU_PIM_vs_GPU_prefill/analyze_grid.py grid      # -> results/turning_points_grid.md
```
The quantity compared is `tiers[1]["prefill_s"]` of the two JSONs (request 1 = the
reuse prefill).  `run_one.sh` passes `--gpu-model flash --tier-batch-size 1
--pipeopt --ffopt --ramulator-workers 2`; `run_one_splitk.sh` adds `--attn-splitk`.
