# Fugue: reproducible KV reuse experiments on AttAcc

This repository reproduces Experiments 1–5 using the original AttAcc GPU/DRAM cost models plus Fugue's MQ, partial-recomputation, KV-sharing accounting and prefill execution choices. The final figures and tables are in [Fugue-paper](Fugue-paper/README.md).

**The repository is self-contained for the simulation:** pinned Ramulator/dependency sources, fixed tokenized workloads, model settings and executable experiment code are included. No sibling repository, old `output`, downloaded LLM weights, CUDA device, tokenizer download or external vLLM installation is needed. Python packages and a C++ toolchain must be installed first.

## Quick start

Requirements: Linux, Python 3.10+, CMake 3.16+, a C++20 compiler (tested with GCC 14; upstream also documents GCC 12), `make`, and `patch`. Default parallelism is 8 CPU cores. Run from the repository root; the directory may have any name.

```bash
git clone https://github.com/chenyi995/attacc_drampim.git attacc-fugue
cd attacc-fugue
git switch chenyi-0906
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-fugue.txt
# If the default g++ is too old, set CXX to an installed C++20 compiler.
python3 -m fugue doctor
python3 -m fugue all --jobs 8
```

If you receive a source archive instead, extract it, enter its root directory and start with the virtual-environment step above.

`all` builds Ramulator from bundled source, runs all five experiments, regenerates the figures, and compares every numeric field of the main result tables with the checked-in final references. It does not copy old timing results into the simulator. Default output: `output/Fugue-asplos-reproduce/`.

For a compiler outside PATH, use `CXX=/path/to/g++ python3 -m fugue all --jobs 8`. On this project's validation host the default GCC 8 is too old; GCC 14 was selected explicitly. CPU simulation is sufficient—A100a is a modeled device, not a hardware requirement.

## Commands and output

```bash
python3 -m fugue build --jobs 8
python3 -m fugue run --experiments 1,2 --jobs 8
python3 -m fugue run --experiments 3 --jobs 8
python3 -m fugue run --experiments 4,5 --jobs 8
python3 -m fugue plot
python3 -m fugue verify
```

Experiments 1/2 share the final shape sweep; 4/5 share the EPIC driver. Asking for one member runs the pair, avoiding duplicate profiles. To redraw only the checked-in final tables:

```bash
python3 -m fugue plot --from-paper --output output/Fugue-asplos-redraw
```

This last command is explicitly a redraw; it does not claim to rerun simulation. Fresh simulation and redraw outputs are separate from the checked-in `Fugue-paper` results.

| Path | Contents |
| --- | --- |
| `fugue/` | Portable CLI, source build, MQ profiles, workload replay, plotting and verification |
| `src/`, `pim_ramulator_src/` | Original AttAcc models and trace mapping; native source lock is checked |
| `vendor/` | Pinned Ramulator2, argparse, spdlog, yaml-cpp source archives and licenses |
| `artifact/inputs/` | Fixed token IDs, recomputation indices, text and provenance; no latency inputs |
| `artifact/reference/` | Final numeric references, read only by the comparison command |
| `Fugue-paper/` | One final version per experiment, including full final numerical data |
| `output/Fugue-asplos-reproduce/logs/` | Stage commands, stdout/stderr and exit status |
| `output/Fugue-asplos-reproduce/experiment*/` | Fresh trace/YAML, Ramulator command logs, timing, operator/event/energy tables |
| `output/Fugue-asplos-reproduce/paper/` | Recreated final figures and derived tables |
| `output/Fugue-asplos-reproduce/Fugue-asplos-verification.json` | Numeric comparison and integrity results |

Existing successful stages can be skipped with `--resume` only when their code/inputs match. Failed or incomplete runs are retained; use a new `--output output/Fugue-asplos-rerun` for a clean retry. Do not delete data to make an assertion disappear. Generated outputs, build products, virtual environments and local archives are Git-ignored.

The [independent source-package check](docs/Fugue-asplos-reproduction-checks.md) reproduced all 50 CSV tables and all 13 PNGs. A complete fresh run took about 128 s on the validation host with 8 workers.

See [full reproduction guide](docs/Fugue-asplos-reproduction.md), [modeling scope](docs/Fugue-asplos-methodology.md), and [input provenance](artifact/inputs/README.md). The original upstream installation and `main.py` usage are preserved in [AttAcc upstream README](docs/AttAcc-upstream.md).

## What is modeled

All final experiments use the native LLAMA-7B/A100a/FP16 configuration, 32 layers and heads, five HBM stacks, bank PIM and the native power constraint. Default link is NVLink 3, 300 GB/s one way. MQ has eight resident queries and eight tCK per full group at approximately 1.3 GHz. The original mapping and read/write timing parameters remain unchanged.

These are hardware cost-model replays of fixed CacheBlend/EPIC workload inputs. They do not reproduce numerical LLM answers or software accuracy experiments. Shared tensor capacity/transfer accounting does not claim a new physical allocator or shared-address hotspot model. Complete service latency, including exposed transfers, determines F4's choice. The detailed per-experiment READMEs retain unfavorable findings, warmup cost and F2 export-overlap sensitivity.
