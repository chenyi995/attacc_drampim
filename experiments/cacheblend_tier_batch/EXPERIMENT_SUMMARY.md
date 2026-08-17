# CacheBlend experiment summary

## Shared-KV batching

Temporary PIM-side experiment: 32 Q arrivals, 4096-B Q shards, serialized
300-GB/s GPU-to-PIM link, HBM tCK = 1.3 ns. A batch starts when its B-th Q is
ready and PIM is idle. These are not full-model inference latencies.

| batch threshold | PIM makespan |
|---:|---:|
| 1 | 27.761 us |
| 2 | **24.717 us** |
| 4 | 24.827 us |
| 8 | 25.059 us |

B=2 is best for this arrival pattern. B=4 is close; B=8 loses to
batch-formation waiting.

## Reused-Q rotation distribution

RoPE rotation is modeled as block-diagonal/pairwise, not as dense matrix
multiplication. The timing-only experiment uses deltas `[0, -3, 5]`, one
8192-B FP16 tensor-parallel Q shard, a 300-GB/s directional link, and zero
rotation-compute charge.

| policy | Q variants | GPU-to-PIM bytes | external-link time | local delay | distribution makespan |
|---|---:|---:|---:|---:|---:|
| GPU rotate | 3 | 24,576 B | 81.92 ns | 0 | 81.92 ns |
| PIM-die rotate | 3 | 8,192 B | 27.31 ns | 2.00 ns | 29.31 ns |
| bank-level rotate | 3 | 8,192 B | 27.31 ns | 0 | 27.31 ns |

Die and bank send one raw Q across the external link. Die serializes two
shifted variants on one rotate unit, adding two 1-ns cycles. Thus die is 2 ns
slower than the bank-local timing-only assumption. With no position changes,
all policies send one 8192-B Q and take 27.31 ns.

## Current latency model

The original AttAcc model uses padded rectangular dependency tiers and rescales
prefill as `fresh tokens + reused tokens x recompute fraction`. It does not
model physical TLB addresses, master/diff placement, Q-ready admission,
Q-variant traffic, link timing, or per-extent Ramulator timing; decode remains
at no-reuse cost. It is therefore not a CacheBlend latency model.

The physical CacheBlend path builds a GPU/PIM/DIE/TLB/link DAG, assigns
physical K/V addresses, uses disjoint master/diff channel pools, times each
contiguous physical scan with Ramulator, and merges partial softmax on the DIE.
It is currently for causality/timing validation, not calibrated throughput:
cross-extent controller queueing, GPU/link/DIE arbitration, and die/bank
rotation hardware overhead remain uncalibrated.

## Reproduction artifacts

- `run_step1_ramulator.py`: direct Ramulator batching and arrival sweep.
- `run_step2_rotate.py`: rotation policies and stable-position check.
- Archived temporary results: `step1_current_layout/arrival_threshold.csv` and
  `step1_current_layout/summary.md` in
  `backup/attacc-before-overlap-arrival-rotate-20260814.zip`.
