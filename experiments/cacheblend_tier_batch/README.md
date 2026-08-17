# CacheBlend tier batching and Q-rotate experiments — 2026-08-14

The source snapshot before these experiments is
[`attacc-0814.zip`](../../../backup/attacc-0814.zip).

## Step 1: shared-KV scan batching

The CacheBlend event DAG keeps a tier barrier: every request in tier `t+1`
waits for the previous tier's final decode post-processing and KV store.  The
TLB is retained across those tiers, so a child resolves its `parent_out` to
the producer's PIM-resident K/V block.  Within a tier, requests have no DAG
edge and may be submitted as a batch.

The direct Ramulator experiment isolates the PIM part of that batch decision:
one 128-token resident KV segment, 64-wide head, and 16 query heads per
request.  `incremental` concatenates one scan trace for each arriving query;
`large` creates one joint trace for the same total query-head work.  ACT is
measured with Ramulator's `CommandCounter` plugin over all 16 HBM channels;
it counts issued `ACT`, `ACTAB`, `ACTSB`, and `ACTPB` commands.

| batch | incremental cycles | large cycles | incremental ACT | large ACT |
|---:|---:|---:|---:|---:|
| 1 | 137 | 137 | 32 | 32 |
| 2 | 285 | 205 | 64 | 32 |
| 4 | 581 | 417 | 128 | 64 |
| 8 | 1173 | 841 | 256 | 128 |
| 16 | 2357 | 1689 | 512 | 256 |
| 32 | 4725 | 3385 | 1024 | 512 |

The K/V base addresses are fixed for **every** query batch (`--shared-kv`),
and the trace audit confirms that the large and incremental cases use the
same K/V MAC address set.  The experiment has 16 channels and 16 query heads
per request.  In the incremental schedule, each request closes the command
group, so every channel issues two `ACTAB` commands (one K and one V group):
`16 channels × 2 × B = 32B` total ACTs.  In the joint trace, the generator
interleaves query pairs while those K/V rows remain open.  For even B it
observes `B` `ACTAB`s per channel, or `16B` total.  This exactly explains the
measured 2× ACT reduction: it is row reuse, not an assumed ACT estimate.

The per-request large-batch time converges to about 106 cycles (B=2–32),
versus 148 cycles for incremental dispatch.  It therefore shows no column
command or ACT saturation in this particular layout.  This is not a general
claim: the trace has only one 128-token KV segment and uses the generator's
pairwise interleaving; a different K/V placement, sequence length, or PIM
buffer capacity can change the result.

Raw trace files, per-channel command counts, CSV, and the reproducible runner
are in [`results/step1`](results/step1) and
[`run_step1_ramulator.py`](run_step1_ramulator.py).

## Step 2: Q rotation distribution

RoPE reuse deltas are `[0, -3, 5]`; a tensor-parallel FP16 Q shard is 8192 B.
Rotation is treated as its true block-diagonal/pairwise operation with zero
compute cost in this timing-only experiment—not as a dense matrix multiply.
The GPU-PIM directional link rate is 300 GB/s, following AttAcc's half-duplex
interface model.

| policy | Q variants | GPU→PIM bytes | external-link time | shifted-Q die delay |
|---|---:|---:|---:|---:|
| GPU rotate | 3 | 24,576 | 81.92 ns | 0 |
| PIM-die rotate | 3 | 8,192 | 27.31 ns | 1 cycle per shifted variant |
| bank-level rotate | 3 | 8,192 | 27.31 ns | 0 (timing-only assumption) |

The all-position-stable check sets every delta to zero.  It passed: all three
policies send one 8192-B Q and have the same 27.31-ns external-link time.

`run_step2_rotate.py --mode {gpu_rotate,die_rotate,bank_rotate}` selects one
policy.  Die and bank modes have the same external distribution transfer: one
raw Q.  In die mode a single rotate unit issues master variants first, then
diff variants, at one cycle per shifted variant.  Thus the first diff Q is at
least one cycle later than the preceding master Q; with several variants the
delay accumulates by the number of queued rotations.
Raw JSON and the reproducible runner are in [`results/step2`](results/step2)
and [`run_step2_rotate.py`](run_step2_rotate.py).
