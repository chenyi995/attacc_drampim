# KVChime artifact evaluation

This branch reproduces the experiments included in the final KVChime paper.
It contains the exact native AttAcc source and final paper timing-model corrections,
frozen workload shapes, the calibrated placement policy, and reproducible Python figures.
It runs on a CPU host using a freshly built Ramulator; no GPU, model weights, or
access to the authors' filesystem is required.

The earlier experiments and development history remain on
[`chenyi-0906`](https://github.com/chenyi995/attacc_drampim/tree/chenyi-0906),
saved at `612c56f`. The paper snapshot used for this artifact is
[`f481b77`](https://github.com/chenyi995/KVChime-ASPLOS-2027/tree/f481b77).

## Run everything

Requirements: Linux, Python 3.10 or newer with `venv` and `pip`, CMake 3.16 or newer,
Make, `patch`, and a C++20 compiler. GCC 14 was used for validation. Allow 8 CPU cores,
64 GB RAM and 10 GB free disk space for the default command; these are provisioning
budgets, not measured peak use. Python packages are downloaded on the first run.
All C++ dependencies are bundled at pinned commits, so the build itself needs no network.

```bash
git clone --depth 1 --branch KVChime --single-branch https://github.com/chenyi995/attacc_drampim.git
cd attacc_drampim
./reproduce.sh --jobs 8
```

The script creates an isolated Python environment, installs pinned requirements,
builds Ramulator from source, runs the final experiments, checks the results against
paper reference values, and writes CSV/PDF/PNG figures. It exits with an error if a
required result is missing or differs from the reference. It does not substitute
reference CSVs for simulation results.

To choose the compiler or output location:

```bash
./reproduce.sh --jobs 16 --cxx /path/to/g++ --output /path/to/new-run
```

`--output` must name a **new directory**. With no argument, each run receives a new
UTC timestamp under `output/`. Raw traces, command timing logs, per-request results,
build logs and check records remain in that directory. No old result is deleted or
overwritten. Generated output and the virtual environment are ignored by Git.

If dependencies are already installed, the equivalent entry point is:

```bash
python3 reproduce.py --jobs 8
```

## Final experiment scope

| Paper result | What runs | Output within the run directory |
| --- | --- | --- |
| Figure 1(a) | Llama 7B, Readers 4, prefill memory/compute decomposition | `figure1a/` |
| Figure 5 | Five models × five workloads; E2E throughput | `request-figures/request-e2e/` |
| Figure 6 | The same requests; TTFT and TBT speedups | `request-figures/request-phase-breakdown/` |
| Figure 7 | The same requests; simultaneous peak KV capacity | `request-figures/request-capacity/` |
| Figure 8 | Single-query PIM and MQ PIM crossover against the same GPU curve | `crossover/` |
| Figure 9 | Source-derived frequency/column-interval analysis | `frequency/` |
| Area results in the text | Recalculate die-area accounting from archived synthesis measurements | `area/` |

Figures 5–7 reuse one set of **25 request simulations**, each producing the four
configurations below. There is no additional F3 run, GQA model sweep, workload search,
or historical sensitivity experiment in the entry point. Mechanism illustrations
are not experiments and are not regenerated here.

| Internal ID | Paper name | Execution |
| --- | --- | --- |
| F0 | Full GPU | Full prefill and GPU decode. |
| F1 | GPU reuse | Selective recomputation, remote shared-KV readback, GPU prefill and decode. |
| F2 | Materialized PIM | GPU prefill with readback and complete private-cache export; single-query PIM decode. |
| F4 | KVChime | Shared/private KV view, query-side position alignment, MQ execution, and cost-based GPU/PIM prefill placement. |

TTFT includes the first output token. The configured decode horizon is
`output_tokens - 1`; `E2E = TTFT + (output_tokens - 1) × mean TBT`.
The timing scope covers decoder projections, attention, FFNs and tensor-parallel
communication. It excludes frontend queueing, sampling, the LM head and application
logic. Capacity is the maximum simultaneous GPU-plus-remote KV allocation.

E2E bars are `F0 E2E / method E2E`. Phase bars use `F2 TTFT / F4 TTFT` and
`F1 TBT / F4 TBT`. Capacity bars use `F2 peak KV / F4 peak KV`.
`request-figures/paper-benefits.csv` additionally reports the paper's four main
per-model/per-workload comparisons, including E2E relative to GPU reuse.

## Models and hardware

| Model | Layers | Query / KV heads | Head dimension | A100 GPUs | Remote PIM HBM stacks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama 7B | 32 | 32 / 32 | 128 | 1 | 5 |
| GPT 13B | 40 | 40 / 40 | 128 | 1 | 5 |
| Llama 65B | 80 | 64 / 64 | 128 | 4 | 20 |
| OPT 66B | 64 | 72 / 72 | 128 | 3 | 15 |
| MT 76B | 60 | 80 / 80 | 128 | 4 | 20 |

These are the final five MHA configurations. Each GPU is the modeled 80 GiB A100
with five remote HBM-PIM stacks and the paper's 300 GB/s link. GPU baselines retain
the final capacity-driven staging policy and an unbounded remote capacity budget.
The request runner uses corrected head geometry, synchronized query/probability
resident capacity, the final F2 overlap rule and the frozen calibrated selector.
It does not use the older standalone branch's two-point selector.

The crossover microbenchmark uses **one** Llama 7B GPU, a 1,024-token retained
context and the same 300 GB/s link in both panels. It has its own microbenchmark
service scope; its speedup is not substituted for request TTFT or TBT.
Frequency analysis uses the final fixed-energy-per-operation model. Area accounting
uses archived Genus measurements; the command recomputes their published aggregation,
not a new licensed RTL synthesis run.

## Workloads and original software

The final request experiment uses the same five frozen inputs for every model:

| Workload | Prompt tokens | Selected tokens | Readers | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| Document 8k | 8,026 | 58 | 1 | 16 |
| Readers 4 | 8,028 | 60 | 4 | 16 |
| MuSiQue source request | 8,082 | 1,341 | 1 | 32 |
| Long context 32k | 32,034 | 188 | 2 | 16 |
| LTM recall | 15,832 | 144 | 1 | 16 |

Counts are per reader. Output lengths are fixed timing horizons; TTFT includes
the first token and decode covers the remaining `output_tokens - 1` steps.
This is a selected showcase rather than a dataset average.

Original software: [CacheBlend](https://github.com/YaoJiayi/CacheBlend), commit
`55ad02675939f783a38d579393527d218a7fd581`, and
[EPIC](https://github.com/DerekHJH/epic), commit
`3204410f1723ed0f39575b4eba8c17578a1ee1a1`. LTM uses the
[LongMemEval formatter](https://github.com/xiaowu0162/LongMemEval), commit
`9e0b455f4ef0e2ab8f2e582289761153549043fc`, and its
[cleaned oracle data](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned),
revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`.

Prepare the simulator inputs with one offline command:

```bash
python3 artifact/inputs/prepare_inputs.py --output output/inputs
```

The hardware simulator consumes frozen object lengths and logical positions.
It does not run neural inference in CacheBlend/EPIC/vLLM, generate answers,
or measure QA accuracy. The common LLAMA tokenizer adaptation is frozen across
models. EPIC selects the first 16 tokens of each document plus the query suffix;
its 2,048-token starting chunk size extends to sentence boundaries. MuSiQue
preserves the saved 16% selected count and exact evenly spaced positions,
without claiming measured numerical top-k selection. Its first layer is full,
second layer has full QKV updates with selective attention, and later layers
are selective. Document 8k and Readers 4 share documents but have different
query suffix lengths, so their comparison does not change only reader count.

Warmup is excluded from the timed request. The frozen non-LTM controls include
query objects in the prewarmed pool while still recomputing their positions;
LTM prewarms only past history. LTM recall is one request over four past sessions
with retrieval assumed complete, not a stateful multi-round session. Its 80
fresh tokens remain private and are fully computed; the policy also recomputes
16 tokens per historical session. Output horizons are configured caps, not
observed EOS lengths.

See [input provenance and boundaries](artifact/inputs/README.md) for exact
objects, source hashes, licensing scope, and all preparation details. Optional
source inspection uses a separate, pinned 18.6 MiB download; original text,
token dumps, and raw runs are not bundled:

```bash
python3 artifact/inputs/download_sources.py --output output/upstream-sources
```

## Inspect or run a smaller part

Verify packaged files without simulation:

```bash
python3 reproduce.py --check-only
```

The full command records the freshly built executable and library in
`RUN/build/runtime/`. After that build, an individual final-scope request can be
run in another new directory:

```bash
python3 -m ae.request --simulator simulator --inputs artifact/inputs \
  --runtime-built-dir RUN/build/runtime --output output/one-model \
  --model LLAMA-7B --jobs 8
```

Each output figure folder is self-contained: its `data.csv`, `plot.py`,
`figure_style.py` and `template.json` redraw the figure without running experiments.
For Figure 8 and Figure 9, plotting files are inside the output's `plot/` subdirectory.
The tracked compact reference CSVs support comparison and redraw; they contain
published points only, not historical raw trace archives.

## Code and provenance

| Directory | Contents |
| --- | --- |
| `ae/` | Final request execution, portable adapters and result validation. |
| `simulator/` | Frozen native AttAcc/PIM source and pinned Ramulator build dependencies. |
| `artifact/inputs/` | Five token-free workload descriptions and original software/data links. |
| `artifact/request/` | Exact final model declarations, calibration and placement policy. |
| `artifact/reference/` | Compact published request values for automatic comparison. |
| `microbench/` | Final crossover, frequency, area and Figure 1(a) accounting. |
| `plots/` | Current paper styles and request-figure plotting scripts. |

The native source lock, final wrapper source receipts and `artifact/files.sha256.json`
record source identity. Portable adapters replace author-specific file discovery and
restrict execution to the current paper scope; they retain the final timing equations.
Original source identifiers such as `fugue`, F labels and original dataset names are
kept where needed to trace a result. Calibration is loaded before request replay. The predictor uses only frozen shape
features and coefficients. The offline experiment also replays candidate PIM commands
to measure service costs and selection regret; those measured costs are not inputs to
the predictor.

Original AttAcc: [attacc_simulator](https://github.com/scale-snu/attacc_simulator).
Ramulator: [CMU-SAFARI/ramulator2](https://github.com/CMU-SAFARI/ramulator2), commit
`b7c70275f04126c647edb989270cc429776955d1`. Upstream licenses accompany the bundled
source archives under `simulator/vendor/`; the original AttAcc license is retained.

## Validation record

The complete entry point passed in a fresh Python environment with 16 workers in 13.8 minutes. All 400 published request-metric comparisons passed, along with 21 exact Fig. 1(a) values, all 620 crossover service operands, all 626 frequency intervals, and the area aggregation. See [`artifact/validation.json`](artifact/validation.json) for machine-readable scope, source identities and checks.
