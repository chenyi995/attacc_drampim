# Simulator for AttAcc
This repository includes Python-based simulator designed to analyze the transformer-based generation model (TbGM) inference in a heterogeneous system consisting of an xPU and an Attention Accelerator (AttAcc). 
AttAcc is an accelerator for the attention layer of TbGM, which consists of an HBM-based processing-in-memory (PIM) structure.
In simulating an xPU and AttAcc system, the simulator outputs the performance and energy usage of the xPU, while the behavior of AttAcc is simulated using a properly modified [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2).
We set the memory device of AttAcc in Ramulator2 to HBM3 and implemented AttAcc\_bank, AttAcc\_BG, and AttAcc\_buffer, which represent AttAcc deploying processing units per bank, per bank group, or per pseudo-channel (on the buffer die), respectively.
For more details of AttAcc, please check the [paper](https://dl.acm.org/doi/10.1145/3620665.3640422) **AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference** published at [ASPLOS 2024](https://www.asplos-conference.org/asplos2024).

 
## Prerequisites
- Python
- cmake, g++, and clang++ (for building Ramulator2)

AttAcc simulator is tested under the following system.

* OS: Ubuntu 22.04.3 LTS (Kernel 6.1.45)
* Compiler: g++ version 12.3.0
* python 3.8.8

We use a similar build system (CMake) as original Ramulator 2.0, which automatically downloads following external libraries.
- [argparse](https://github.com/p-ranav/argparse)
- [spdlog](https://github.com/gabime/spdlog)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)


## How to install
1. Clone the Github repository

```bash
$ git clone https://github.com/scale-snu/attacc_simulator.git
$ cd attacc_simulator
$ git submodule update --init --recursive
``` 

2. Build Ramulator2
```bash
$ bash set_pim_ramulator.sh 
$ cd ramulator2
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ cp ramulator2 ../ramulator2
$ cd ../../
```

## How to run

### Run GPU simulator 
```bash
$ export PYTHONPATH=$PYTHONPATH:$PWD
$ python main.py --system {} --gpu {} --ngpu {} --model {} --lin {} --lout {} --batch {} --pim {} --powerlimit --ffopt --pipeopt

$ python main.py --help

    ## set system configuration
    parser.add_argument("--system",  type=str, default="dgx",
            help="dgx(each GPU has 80GB HBM), \
                  dgx-cpu (In dgx-base, offloading the attention layer to cpu), \
                  dgx-attacc (dgx-base + attacc")
    parser.add_argument("--gpu", type=str, default='A100a', 
            help="GPU type (A100a, A100, and H100), A100a is A100 with HBM3")
    parser.add_argument("--ngpu", type=int, default=8, 
            help="number of GPUs")
    parser.add_argument("--gmemcap",
                        type=int,
                        default=80,
                        help="memory capacity per GPU (GB).  default=80")



    ## set attacc configuration
    parser.add_argument("--pim", type=str, default='bank',
            help="pim mode. list: bank, bg, buffer")
    parser.add_argument("--powerlimit",  action='store_true', 
            help="power constraint for PIM ")
    parser.add_argument("--ffopt",  action='store_true', 
            help="apply feedforward parallel optimization ")
    parser.add_argument("--pipeopt",  action='store_true', 
            help="apply pipeline optimization ")


    ## set model and service environment
    parser.add_argument("--model", type=str, default='GPT-175B', 
            help="model list: GPT-175B, LLAMA-65B, MT-530B, OPT-66B")
    parser.add_argument("--word", type=int, default='2', 
            help="word size (precision): 1(INT8), 2(FP16)")
    parser.add_argument("--lin",  type=int, default=2048,
            help="input sequence length")
    parser.add_argument("--lout",  type=int, default=128,
            help="number of generated tokens")
    parser.add_argument("--batch", type=int, default=1,
            help="batch size, default = 1")
```

### Examples
```bash 
# dgx (A100 with HBM3) example 
$ python main.py --system dgx --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1

# 2xdgx (A100 with HBM3) example 
$ python main.py --system dgx --gpu A100a --ngpu 16 --model GPT-175B --lin 2048 --lout 128 --batch 1

# dgx-attacc (based HBM3) example 
 ## bank level PIM
$ python main.py --system dgx-attacc --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1 --pim bank --powerlimit --ffopt --pipeopt

 ## bank group level PIM
$ python main.py --system dgx-attacc --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1 --pim bg --powerlimit --ffopt --pipeopt

 ## buffer level PIM
$ python main.py --system dgx-attacc --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1 --pim buffer --powerlimit --ffopt --pipeopt 

```

### Run JSON workloads (RAG, supervisor, and KV reuse)

The legacy command line above is unchanged.  A workload is an additional input
path: a legacy RAG list (`sample`, `seg_lens`, `seg_sha`, `seg_role`, `L`,
`lout`) or a supervisor `v2-dag` object with `agents`.  The parser retains the
original JSON and validates segment lengths, output lengths, parent/tier
ordering, and `parent_out` length against the parent output.

First validate and inspect a workload without requiring the simulator stack:

```bash
$ python3 main.py --workload workload/workload_relay_s400w4t1.json \
    --reuse epic --epic-prefix-recompute-tokens 1 --validate-workload
```

`no-reuse` executes each dependency tier through the unmodified
`System.simulate` interface.  A tier is the batch unit; heterogeneous requests
in a tier are padded to the tier's maximum input/output lengths.

```bash
$ python3 main.py --system dgx-attacc --model GPT-175B \
    --workload workload/workload_2wikimqa_first8.json --reuse no-reuse
```

For CacheBlend, provide a complete partition of model layers.  Layer lists
accept comma-separated indices and inclusive ranges.  The ratio selects
`ceil(ratio * reusable_tokens)` rows independently, uniformly and without
replacement for each partial layer and request; the seed makes those rows
reproducible.  For GPT-175B (96 decoder layers), the following follows the
two full layers / later partial layers convention from the supplied trace.

```bash
$ python3 main.py --system dgx-attacc --model GPT-175B \
    --workload workload/workload_relay_s400w4t1.json --reuse cacheblend \
    --cacheblend-full-layers 0-1 --cacheblend-partial-layers 2-95 \
    --cacheblend-recompute-ratio 0.15 --reuse-seed 7 \
    --cacheblend-batch-size 4 \
    --workload-report cacheblend_output.json
```

`--cacheblend-batch-size` is the same-tier PIM-attention admission threshold.
`1` preserves the original per-agent CacheBlend DAG.  For values such as `2`
through `8`, GPU QKV first runs from input readiness and emits every Q transfer;
the attention/O/FFN batch then takes the earliest completed GPU-to-PIM Q links
from a global ready queue.  The report records both the admitted members and
their upstream QKV members.  A short final group is flushed at its actual size.

EPIC uses a static correction: it recomputes the leading configured number of
tokens in each shifted/relaid reused segment. Position-stable prefix segments
are reused without correction, matching the supplied trace's treatment of the
system prompt. Its execution shares CacheBlend's address-resolved
GPU/PIM/DIE/TLB event DAG and physical master/diff KV layout; only the
correction-row selection differs.

```bash
$ python3 main.py --system dgx-attacc --model GPT-175B \
    --workload workload/workload_relay_s400w4t1.json --reuse epic \
    --epic-prefix-recompute-tokens 1 --workload-report epic_output.json
```

Reuse reports contain a GPU/PIM split-prefill event stream: weight-bearing
operations run on GPU; PIM scans old KV and performs attention; query, new KV,
and context traffic are recorded on the GPU-PIM link.  PIM score timing uses
AttAcc's Ramulator path.  Before execution, the structural validator checks
layer coverage, event shapes, and each policy's recompute-row rule.

For CacheBlend, the report's `tlb` object is also the source of truth for
Ramulator placement.  Under the current explicit default policy, one reusable
KV block is a 32-byte-aligned contiguous allocation:
`K[0..N-1] | V[0..N-1]`.  Every TLB entry records its `block_id` and
`token_offset`, along with the derived concrete K/V byte addresses.  A PIM
scan coalesces only adjacent vectors from the same physical extent; separate
reused blocks are emitted as separate Ramulator runs and merged by the DIE
softmax event.  Thus no scan silently treats later KV blocks as a continuation
of the first block.  This is the current placement assumption; changing the
layout policy must update both the TLB block table and the trace generator.

## Placement ablation and the GPU+PIM-vs-GPU prefill study (2026-08)

`--ablation A1..A6` selects where prefill attention, decode attention and the
KV cache live (see the docstring of `src/ablation.py`):

| key | prefill attn | decode attn | KV mapping |
|---|---|---|---|
| A1 | gpu | pim | private (original AttAcc, `--reuse no-reuse`) |
| A2 | gpu | gpu | none (pure GPU running CacheBlend/EPIC) |
| A3 | gpu | pim | naive |
| A4 | gpu | pim | master-diff |
| A5 | pim | pim | master-diff |
| A6 | split (GPU fresh rows, PIM reused rows) | pim | master-diff |

Related switches: `--tier-batch-size N` (serve each dependency tier as padded
batches of <= N requests), `--pim-prefill-query-batch q` (queries sharing one PIM
K/V scan), `--kv-pool-split`, `--master-shadow`, `--split-attn {overlap,serial}`.

GPU performance model `--gpu-model {legacy,refined,flash}`: `legacy` = the
original AttAcc xPU formulas (default, bit-identical to the published model);
`refined` = projection and attention GEMMs priced by a measured cuBLAS A100
efficiency table (`src/gemm_table.py`) plus NVLink latency / far-HBM streaming
on GPU<->AttAcc transfers; `flash` = refined plus attention as a fused
FlashAttention-2 kernel (128-row Q blocks, softmax on-chip, flash-decoding for
decode).  `--attn-splitk` (flash only) lets short-Q prefill attention split its
key range across CTAs.  `--pim-link {nvlink3,nvlink4,pcie5,pcie4}` sets the
GPU<->AttAcc link bandwidth used by K/V, Q and context transfers.

```
# one ablation cell (LLAMA-7B, CacheBlend r=0.15, split prefill, flash GPU):
python main.py --system dgx-attacc --model LLAMA-7B \
    --workload workload/workload_rag_shared_p24_s8.json --tier-batch-size 4 \
    --ramulator-workers 4 --pipeopt --ffopt --ablation A6 --reuse cacheblend \
    --cacheblend-full-layers 0-1 --cacheblend-partial-layers 2-31 \
    --cacheblend-recompute-ratio 0.15 --reuse-seed 7 --gpu-model flash \
    --workload-report out.json
```

`experiments/GPU_PIM_vs_GPU_prefill/RESULTS.md` holds the replay-pair study
(when does GPU+PIM cooperative prefill beat pure-GPU prefill) with the command
lines to reproduce it; workload generator `workload/gen_replay_pair.py`, raw
JSONs under `results/`.  When several simulations run concurrently give each
its own Ramulator working directory via `ATTACC_RAMULATOR_DIR` /
`ATTACC_RAMULATOR_LOG` (the Ramulator shape cache `ramulator.out` is rewritten
whole).

## Details of the Ramulator for AttAcc
### How to Run
1. Generate PIM command traces for the Transformer-based Generative Model.
```bash
$ cd ramulator2
$ cd trace_gen
$ python gen_trace_attacc_bank.py
$ python gen_trace_attacc_bg.py
$ python gen_trace_attacc_buffer.py
```

This produces `attacc_bank.trace`, `attacc_bg.trace`, and `attacc_buffer.trace` which are GPT-175B traces of attention layer in a single decoder for AttAcc\_bank, AttAcc\_BG, AttAcc\_buffer, respectively.


You can change the model, batch, and request configuration by setting arguments as below.
```python
  parser.add_argument("-dh", "--dhead", type=int, default=128, 
                      help="dhead, default= 128")
  parser.add_argument("-nh", "--nhead", type=int, default=1, 
                      help="Number of heads, default=1")
  parser.add_argument("-l", "--seqlen", type=int, default=2048,
                      help="Sequence length L, default= 2048")
  parser.add_argument("-maxl", "--maxlen", type=int, default=4096, 
                      help="maximum L, default= 4096")
  parser.add_argument("-db", "--dbyte", type=int, default=2, 
                      help="data type (B), default= 2")
  parser.add_argument("-o", "--output", type=str, default="attacc_bank.trace", 
                      help="output path")
```

2. Run Ramulator-AttAcc
```bash
$ ./ramulator2 -f attacc_bank.yaml
$ ./ramulator2 -f attacc_bg.yaml
$ ./ramulator2 -f attacc_buffer.yaml
```

This will print the total number of DRAM/PIM request and total elapsed memory cycles (`memory_system_cycles`).

The command log will be generated in `log` directory.


### Modeling AttAcc with a Power Contraint
We reflect the DRAM power constraint to AttAcc by increasing the delay between consecutive MAC commands (`nCCDAB`, `nCCDSB`).

We calculate these delay with the activation and read energy.

To evaulate AttAcc with no power constraint (NPC), uncomment `preset: HBM3_5.2Gbps_NPC` and comment out `preset: HBM3_5.2Gbps` in yaml config files.




## Contact
Jaehyun Park jhpark@scale.snu.ac.kr

Jaewan Choi jwchoi@scale.snu.ac.kr
