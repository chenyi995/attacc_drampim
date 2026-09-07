# KVChime: shared KV execution on AttAcc

KVChime combines query-side RoPE alignment, shared KV with private replacements, and Multi-Query PIM execution with a simple prefill device selector. This repository supplies reproducible CPU simulations using the original AttAcc GPU cost operators and bundled Ramulator sources.

The current results are in [Fugue-paper](Fugue-paper/README.md). Experiment 1 uses the agreed six-panel (2×3) layout for each model. Later performance and capacity figures each have one response axis. Every figure includes raw absolute values and a standalone plotting script. Energy remains in the CSVs with a brief README note. Hardware area tables and frequency evidence belong to the separate `kvpim-rtl` repository.

The [completion and verification record](docs/KVChime-reproduction-checks.md) documents the completed TP4 run: four six-panel Experiment 1 figures and ten later single-axis figures, 238,384 numeric fields preserved in publication, and 12 passing model/shared-view tests.

## Reproduce the current experiments

Requirements: Linux, Python 3.10+, CMake 3.16+, C++20 compiler, make, patch. GCC 14 was used on the validation host. No physical GPU, model weights, tokenizer download, sibling repository, or pre-existing output is required. Pinned source archives, licenses, tokenized workload inputs, and provenance are included.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-fugue.txt
# Set CXX=/path/to/g++ if the default compiler lacks C++20.
python3 -m fugue doctor
python3 -m fugue all-models --jobs 8 --output output/KVChime-fresh
```

Use a fresh output directory. The command builds the bundled simulator, checks MHA/GQA geometry and shared-view invariants, runs both the shape sweep and complete workloads, verifies event/capacity accounting, compares available checked-in multi-model numeric references, and generates `output/KVChime-fresh/paper/`. It never reads publication latency tables to produce simulation timings. The default is eight workers; CPU affinity, process memory and BLAS threads are bounded.

| System experiment | Question | Final figures |
|---|---|---|
| 1: Prefill boundary | GPU vs ordinary PIM vs MQ PIM as Q grows | One 2×3 six-panel figure per model, retaining the agreed a–f configuration order |
| 2: Link/cache sensitivity | How link and cache change the MQ advantage | Separate long grouped bars for link and cache |
| 3: Software reuse F0–F4 | Full recomputation, software reuse, materialized PIM, shared views, MQ/selection | Separate TTFT, TBT, E2E, scan and capacity charts |
| 4: Shared-query MQ | Simultaneously ready agents share KV operands | Separate scan and TBT charts |
| 5: Device selection | Fixed GPU, fixed MQ PIM, estimated selector and oracle | Attention service normalized to oracle |

All five use **LLAMA-7B, GPT-13B, LLAMA-65B and LLAMA3.1-8B**. The first three inherit native AttAcc geometry; the last is an explicit GQA extension (32 Q / 8 KV heads). LLAMA-65B uses TP4; other models use TP1. Hardware defaults are native A100a, FP16, five PIM HBM stacks per GPU and NVLink 3 at 300 GB/s one way. MQ has up to eight resident queries, about 1.3 GHz and an eight-tCK full-group interval. Native physical mapping is unchanged.

These are fixed CacheBlend/EPIC **hardware-shape replays**, not numerical LLM inference or per-model retokenization. All requests and Transformer layers are counted; there is no request extrapolation. Read [workloads, costs, metrics and selector](docs/KVChime-multi-model.md) and [RoPE/shared-view equivalence](docs/KVChime-correctness.md) before interpreting the results. Negative gains and selection errors are retained.

## Redraw without simulation

```bash
# Redraw all figures from a completed current run:
python3 -m fugue all-models-plot --output output/KVChime-fresh
python3 -m fugue all-models-verify --output output/KVChime-fresh
# Or copy any final paper/<figure>/ folder elsewhere and run:
python3 plot.py --output-dir redraw
```

Every final `paper/<figure>/` folder contains `raw-data.csv`, `raw-profiles.csv`, `plot-config.json`, `models.json`, `plot.py`, PDF/PNG and provenance. It can be copied outside this repository and redrawn with Python, numpy and matplotlib. Full raw trace/YAML/command logs and stage commands/source hashes remain under the chosen output directory.

## Repository map

| Path | Purpose |
|---|---|
| `src/`, `pim_ramulator_src/` | Native AttAcc operators and physical mapping, protected by source hashes |
| `fugue/kvchime*.py` | Current model adapters, shared traces, sweep, workloads, selector and plotting |
| `vendor/` | Pinned Ramulator and dependency source archives plus licenses |
| `artifact/inputs/` | Frozen token IDs, source text, chunk identities and recomputation indices |
| `Fugue-paper/` | Latest paper outline and current final experiment packages |
| `artifact/legacy-paper/` | Superseded single-model figures/tables retained for historical reproduction |
| `output/` | Fresh or incomplete runs, build products, full command evidence |

The historical `all`, `run`, `plot`, and `verify` commands reproduce the earlier single-model suite; their documentation is explicitly marked legacy. The current command is `all-models` (`kvchime` is an alias). Failed or interrupted attempts are retained; `--resume` skips only successful stages with identical code and inputs. Current simulation and historical references must not be mixed. Original upstream usage remains in [AttAcc upstream README](docs/AttAcc-upstream.md).
