> 历史单模型实验文档，保留用于旧结果复现；当前多模型实验见 [KVChime 实验说明](KVChime-multi-model.md)。以下数值与方案定义不属于新版最终结果。

# Reproducing Fugue Experiments 1–5

The supported entry point is `python3 -m fugue`, from this checkout. Historical one-off scripts are archived locally; no step depends on running them or locating old output directories.

## Installation and dependencies

1. Install Python 3.10+, CMake, a C++20 compiler, `make`, and `patch` using your system's package manager. The validated build uses GCC 14; select a suitable compiler with `CXX` if `g++` is older.
2. Create a virtual environment and install `requirements-fugue.txt`. These exact Python package versions reproduce the publication plotting/numeric environment. A package index connection is needed for this installation unless the packages are already available locally.
3. Run `python3 -m fugue doctor`. It checks source/input hashes, token/recomputation consistency, Python packages and the selected compiler.
4. Run `python3 -m fugue all --jobs 8`. There are no runtime downloads; CMake uses `FETCHCONTENT_FULLY_DISCONNECTED=ON` and the four bundled source archives. A source copy receives only this checkout's original overlay/patch recipe and the `-include cstdint` compiler compatibility option.

The original Git submodule is not needed for Fugue. The build does not require `.git` or a particular clone directory name; an exported source tarball containing the listed artifact files works too. It does not import modules from any old checkout.

## Experiments and expected work

| Experiment | Input and method | Fresh run group |
| --- | --- | --- |
| 1 | Final 168 `(Q,C)` samples, with dense integer crossings; ordinary PIM, MQ PIM and GPU | `experiment12` |
| 2 | Final 18×8 query/cache grid; 13 link-bandwidth prices of the same native scan costs | `experiment12` |
| 3 | Two complete CacheBlend demo inputs, native-tokenizer lengths 3420/3458, 10-token output cap; F1–F4 | `experiment3` |
| 4 | EPIC 4K/8K/16K source contexts, `kvlink-16`, 16-token output cap; F1–F4 | `experiment45` |
| 5 | EPIC 8K shared documents, 1/2/4 synchronous readers with actual batch/head dimensions; F1–F4 | `experiment45` |

The union sweep has 276 distinct `(Q,C)` inputs and 519 native signatures (including the published transition evidence). CacheBlend has 26 signatures, and EPIC has 94: **639 fresh profile records** for the complete portable run. This is smaller than the historical sequence of repeated supplementary runs, but covers the same final tables. Main-table verification checks 168+144+8+24 rows.

The observed execution time and verification records are in the [independent reproduction report](Fugue-asplos-reproduction-checks.md). Host speed varies. Default scheduling uses 8 cores and 8 subprocesses; each process has a 20 GB address-space cap. The CLI accepts at most 24 jobs, keeping its parent plus child address-space caps at or below 500 GB and its affinity below 96 cores. For lower-memory hosts, use `--jobs 1` or `--jobs 2`. It does not build an agent DAG, and does not multiply or divide a small request's latency to emulate a larger request batch.

## Running one group and retrying

```bash
python3 -m fugue build --output output/Fugue-asplos-example
python3 -m fugue run --experiments 3 --output output/Fugue-asplos-example
python3 -m fugue plot --experiments 3 --output output/Fugue-asplos-example
python3 -m fugue verify --experiments 3 --output output/Fugue-asplos-example
```

Use the same `--output` for build/run/plot/verify. `--resume` skips successful stages with identical inputs and code. If a stage failed, retain its log and choose a new output directory. Plot/verify may be repeated because they do not rerun the simulator or overwrite checked-in results. A full clean reproduction starts with an empty output directory and does not use `--from-paper`.

## Reading the evidence

Each stage records argv, cwd, input/source hashes, start/finish times and exit status. Each native profile records the exact generator command, head count, trace, YAML, return code, cycles and channel command timestamps. Workload runs include per-layer decisions, GPU operators, ordered events, transfers, capacity snapshots, dynamic energies and native decode parity checks.

`verify` reads reference numbers only after the fresh run. It compares every numeric field of the four main tables with relative tolerance `1e-8` and absolute tolerance `1e-9`, and compares winning-device labels exactly. Small arithmetic representation differences are allowed; this is not permission to shift a crossover. All emitted native-profile commands must succeed and use the new run's executable. Native GPU/DRAM source hashes must match the locked publication model.

The [legacy paper archive](../artifact/legacy-paper/README.md) retains one final PDF/PNG pair per historical figure and all final sampled CSV data, including samples outside the displayed window. Historical `plot` commands read this archive; `Fugue-paper` now contains the KVChime multi-model results. Earlier three-panel/full-range images and one-off script snapshots remain in local `output` archives, outside runtime dependencies.

## Extending the code

- Change experiment grids in `artifact/inputs/sweep.json` for a new sweep. Keep the published grid and references on a separate branch if the new run is intended to compare with the publication.
- `fugue/attention.py` implements native trace generation, MQ expansion/timing and overlap-window observation.
- `fugue/cacheblend.py` and `fugue/epic.py` implement the final workload execution/capacity/energy ledgers using native `src` operators.
- `fugue/plot.py` derives figures and tables; `fugue/validate.py` checks the publication inputs and results.
- Changing `src/` or `pim_ramulator_src/` changes the base simulator. Explicitly update `artifact/native-source-lock.json`, rebuild and establish new reference data for that new model revision; do not relabel changed-model results as reproduced publication results.

Input and reference manifests deliberately detect edits. For a new experiment revision, update them along with a documented source/input change. They are not used to supply timing values to the execution model.
