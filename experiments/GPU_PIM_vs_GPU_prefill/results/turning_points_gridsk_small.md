
# Link: NVLink3 (600 GB/s)

## CacheBlend (r = 0 here, so n_q = n_new + n_off; for r > 0 the same n_q* applies with n_q = n_new + n_off + r·L)

| model | n_q* (simulated, A6/A4 = 1) | recompute ratio at n_q* (per partial layer / all-layer avg) | reuse ratio at n_q* | bracketing grid points (n_q: A6/A4) | theory n_q* = 4·(t_link + t_gpu_attn)/t_pass | per-layer terms at smallest n_q: t_pass / t_link / t_gpu_attn [us] |
|---|---:|---|---:|---|---:|---|
| LLAMA-7B | **5** | 0.07 % / 6.31 % | 99.93 % | 5: 0.99 -> 9: 1.06 | 17 | 16.6 / 62.0 / 9.9 |
| LLAMA-65B | **15** | 0.18 % / 2.68 % | 99.82 % | 13: 0.98 -> 17: 1.03 | 33 | 16.6 / 117.9 / 19.9 |
| GPT-175B | **19** | 0.23 % / 2.31 % | 99.77 % | 17: 0.98 -> 25: 1.05 | 47 | 16.6 / 173.8 / 19.9 |

A6/A4 over the grid (n_q: ratio):

| model \ n_q | 2 | 3 | 5 | 9 | 13 | 17 | 25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| recompute ratio per partial layer | 0.02% | 0.04% | 0.06% | 0.11% | 0.16% | 0.21% | 0.30% |
| LLAMA-7B | 0.90 | 0.92 | 0.99 | 1.06 | 1.12 | 1.18 | 1.30 |
| LLAMA-65B | 0.82 | 0.83 | 0.89 | 0.93 | 0.98 | 1.03 | 1.11 |
| GPT-175B | 0.83 | 0.84 | 0.88 | 0.91 | 0.95 | 0.98 | 1.05 |

## EPIC (prefix 1 token/segment, n_q = n_new + n_off + 1)

| model | n_q* (simulated, A6/A4 = 1) | recompute ratio at n_q* (per partial layer / all-layer avg) | reuse ratio at n_q* | bracketing grid points (n_q: A6/A4) | theory n_q* = 4·(t_link + t_gpu_attn)/t_pass | per-layer terms at smallest n_q: t_pass / t_link / t_gpu_attn [us] |
|---|---:|---|---:|---|---:|---|
| LLAMA-7B | **6** | 0.07 % / 0.07 % | 99.93 % | 4: 0.87 -> 6: 1.02 | 12 | 24.4 / 62.0 / 9.9 |
| LLAMA-65B | **13** | 0.16 % / 0.16 % | 99.84 % | 10: 0.92 -> 14: 1.01 | 23 | 24.4 / 117.9 / 19.9 |
| GPT-175B | **20** | 0.25 % / 0.25 % | 99.75 % | 18: 0.96 -> 26: 1.09 | 32 | 24.4 / 173.8 / 19.9 |

A6/A4 over the grid (n_q: ratio):

| model \ n_q | 3 | 4 | 6 | 10 | 14 | 18 | 26 |
|---|---:|---:|---:|---:|---:|---:|---:|
| recompute ratio per partial layer | 0.04% | 0.05% | 0.07% | 0.12% | 0.17% | 0.22% | 0.32% |
| LLAMA-7B | 0.83 | 0.87 | 1.02 | 1.18 | 1.33 | 1.49 | 1.79 |
| LLAMA-65B | 0.70 | 0.72 | 0.82 | 0.92 | 1.01 | 1.11 | 1.30 |
| GPT-175B | 0.68 | 0.70 | 0.76 | 0.83 | 0.90 | 0.96 | 1.09 |

# Link: PCIe4 (64 GB/s)

## CacheBlend (r = 0 here, so n_q = n_new + n_off; for r > 0 the same n_q* applies with n_q = n_new + n_off + r·L)

| model | n_q* (simulated, A6/A4 = 1) | recompute ratio at n_q* (per partial layer / all-layer avg) | reuse ratio at n_q* | bracketing grid points (n_q: A6/A4) | theory n_q* = 4·(t_link + t_gpu_attn)/t_pass | per-layer terms at smallest n_q: t_pass / t_link / t_gpu_attn [us] |
|---|---:|---|---:|---|---:|---|
| LLAMA-7B | (no data yet) | | | |
| LLAMA-65B | (no data yet) | | | |
| GPT-175B | (no data yet) | | | |

A6/A4 over the grid (n_q: ratio):

| model \ n_q | 2 | 3 | 5 | 9 | 13 | 17 | 25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| recompute ratio per partial layer | 0.02% | 0.04% | 0.06% | 0.11% | 0.16% | 0.21% | 0.30% |
| LLAMA-7B | · | · | · | · | · | · | · |
| LLAMA-65B | · | · | · | · | · | · | · |
| GPT-175B | · | · | · | · | · | · | · |

## EPIC (prefix 1 token/segment, n_q = n_new + n_off + 1)

| model | n_q* (simulated, A6/A4 = 1) | recompute ratio at n_q* (per partial layer / all-layer avg) | reuse ratio at n_q* | bracketing grid points (n_q: A6/A4) | theory n_q* = 4·(t_link + t_gpu_attn)/t_pass | per-layer terms at smallest n_q: t_pass / t_link / t_gpu_attn [us] |
|---|---:|---|---:|---|---:|---|
| LLAMA-7B | (no data yet) | | | |
| LLAMA-65B | (no data yet) | | | |
| GPT-175B | (no data yet) | | | |

A6/A4 over the grid (n_q: ratio):

| model \ n_q | 3 | 4 | 6 | 10 | 14 | 18 | 26 |
|---|---:|---:|---:|---:|---:|---:|---:|
| recompute ratio per partial layer | 0.04% | 0.05% | 0.07% | 0.12% | 0.17% | 0.22% | 0.32% |
| LLAMA-7B | · | · | · | · | · | · | · |
| LLAMA-65B | · | · | · | · | · | · | · |
| GPT-175B | · | · | · | · | · | · | · |
