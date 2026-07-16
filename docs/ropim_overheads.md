# RoPIM-style RoPE overhead estimate

This note matches the current `attacc_drampim` multi-agent RoPE model. `Q_rotate` is modeled as already computed on the GPU, so `Q` never enters AttAcc DRAM. `V` is not rotated. The PIM pre-pass only rotates `K`, using a row-level mapping that places the master `K` section and agents' `Sk` in one row.

## Assumptions

| Item | Value |
| --- | ---: |
| `dhead` | 128 |
| data type | FP16, `dbyte = 2` B |
| `maxlen` | 4096 |
| example runtime `L` | 2048 |
| HBM row columns, `n_col` | 32 |
| prefetch / column granularity | 32 B |
| row size | 1024 B |
| HBM capacity | 16 GiB / HBM |
| HBM count per AttAcc device | 5 |
| GPU-to-AttAcc interface bandwidth | 600 GB/s |
| Ramulator clock | `tCK = 0.769 ns` |

`Sk` is the precomputed RoPE factor table containing `cos` and `sin`.

## Multi-Agent Shared-KV Row Layout

The row buffer cannot hold arbitrary numbers of agents. The trace generator now bounds the number of agents per row and replicates the master `K` section across agent groups.

```text
agents_per_row = floor((n_col - 1(K)) / 2(cos,sin))
               = floor((32 - 1) / 2) = 15
agent_groups = ceil(num_agent / agents_per_row)
```

Each agent group row is laid out as:

```text
K | cos_a0 | sin_a0 | ... | cos_a14 | sin_a14
```

For `num_agent > 15`, the same master `K` chunks are replicated into more row groups:

```text
Group 0: K | Sk agents 0..14
Group 1: K | Sk agents 15..29
...
```

For a full 15-agent group:

```text
cols_per_chunk = 1 + 2 * 15 = 31
chunks_per_row = floor(32 / 31) = 1
vector_chunks = ceil(128 / 16) = 8
row_tiles_per_group = 8
```

So large-agent cases are row-buffer limited: they need one row tile per vector chunk and replicate `K` once per 15-agent group.

## 1. DRAM Capacity Overhead

### `Sk` storage only

```text
Sk_bytes = num_agent * maxlen * dhead * 2(cos,sin) * dbyte
```

| `num_agent` | `Sk` / HBM at `maxlen=4096` | overhead vs 16 GiB/HBM | `Sk` / 5-HBM AttAcc |
| ---: | ---: | ---: | ---: |
| 64 | 128 MiB | 0.7813% | 640 MiB |
| 256 | 512 MiB | 3.1250% | 2560 MiB |

For runtime `L = 2048`, these numbers are halved.

### Row-packed footprint including replicated `K`

The physical row-packed footprint is larger than raw `Sk`, because every agent group row also stores the master `K` column and may include unused columns in the last group.

```text
row_packed_bytes = agent_groups * maxlen * row_tiles_per_group * row_size
                 = ceil(num_agent / 15) * 4096 * 8 * 1024 B
```

| `num_agent` | groups | row-packed / HBM | overhead vs 16 GiB/HBM | row-packed / 5-HBM AttAcc |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 5 | 160 MiB | 0.9766% | 800 MiB |
| 256 | 18 | 576 MiB | 3.5156% | 2880 MiB |

This row-packed number is the better capacity estimate for shared-K multi-agent row locality.

## 2. Latency Estimate: One Multiply-Add Becomes Two

RoPE computes:

```text
x_rot = x * cos + rotate_half(x) * sin
```

Each rotated vector element needs two multiply/add contributions. The pre-pass handles only `K`; `Q_rotate` is already produced by the GPU and `V` is not rotated.

For each decoded token and active head/agent:

```text
extra_mac_per_head = vector_chunks * 1(K) * 4(operand reads)
extra_sync_per_head = row_tiles_per_group * 2(MV_SB,BARRIER)
```

For `nhead_per_hbm = 20`, `L = 2048`, `dhead = 128`, and `n_mac = 16`:

```text
vector_chunks = 8
row_tiles_per_group = 8

extra_mac_cmds = 20 * 8 * 1 * 4 = 640
extra_sync_cmds = 20 * 8 * 2 = 320
extra_total_cmds = 960
```

Sanity-check trace generation with `nhead_per_hbm = 20` produced the same delta for `num_agent = 64` and `num_agent = 256`:

```text
baseline bank trace commands = 16,916
multi-agent K-only RoPE bank trace commands = 17,876
delta = 960
```

A rough serialized lower-bound for bank-level MAC timing uses the trace generator timing note, `PIM_MAC_AB = 8 tCK`:

```text
extra_mac_time ~= 640 * 8 * 0.769 ns = 0.0039 ms per HBM trace
```

The extra `MV_SB` commands are:

```text
extra_mv_sb_cmds = 20 * 8 = 160
extra_mv_sb_time ~= 160 * 4 * 0.769 ns = 0.00049 ms per HBM trace
```

So a simple MAC + MVSB serialized estimate is:

```text
extra_time ~= 0.0044 ms per HBM trace
```

For `num_agent = 64` or `256`, the per-active-head RoPE command count is unchanged for a fixed `nhead_per_hbm`: each active head maps to one agent and rotates one K vector. Total command latency scales with active heads, while the row-packed capacity and `Sk` transfer scale with total `num_agent`.

This is only a command/tCK estimate, not a full Ramulator scheduling result. Full timing depends on row-buffer hits, overlap, barriers, and power-constraint timing (`nCCDAB`, `nCCDSB`, `nCCDPB`).

## 3. Area Overhead

The local RoPIM paper reports the full bank-level RoPIM accelerator as:

```text
area = 1.44 mm^2
normalized overhead = 8.8% of a DDR4 DRAM die
power = 10.96 mW
```

For AttAcc integration, this is a conservative upper bound if adding the full RoPIM engine. If the existing AttAcc MAC datapath is reused, incremental area should be lower because the extra logic is mainly permutation/negation plus control. Supporting 64 or 256 agents mostly increases mapping/control state and storage footprint; this trace model does not instantiate one independent MAC datapath per agent.

Using the paper number as an upper bound:

```text
area_overhead_per_DRAM_die ~= 8.8%
equivalent added area for 5 dies = 5 * 1.44 = 7.20 mm^2
```

If the HBM stack instantiates the RoPE engine on more DRAM dies, scale `1.44 mm^2` by that die count.

## 4. GPU-Computed `Sk` Transfer Latency

For transferring `Sk = {cos, sin}` from GPU to AttAcc DRAM:

```text
Sk_transfer_bytes = num_agent * L * dhead * 2(cos,sin) * dbyte
```

At `L = 2048`:

| `num_agent` | transfer / HBM | latency / HBM at 600 GB/s | transfer / 5-HBM AttAcc | serialized latency at 600 GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 64 MiB | 111.85 us | 320 MiB | 559.24 us |
| 256 | 256 MiB | 447.39 us | 1280 MiB | 2.24 ms |

For `maxlen = 4096`, double those values.

The transfer volume scales linearly with `num_agent`; the row-packed DRAM footprint scales with `ceil(num_agent / 15)` because the master `K` section is replicated per row-buffer-sized agent group.
