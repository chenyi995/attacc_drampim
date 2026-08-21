
# Link: NVLink3 (600 GB/s)

## CacheBlend (r = 0 here, so n_q = n_new + n_off; for r > 0 the same n_q* applies with n_q = n_new + n_off + r·L)

| model | n_q* (simulated, A6/A4 = 1) | recompute ratio at n_q* (per partial layer / all-layer avg) | reuse ratio at n_q* | bracketing grid points (n_q: A6/A4) | theory n_q* = 4·(t_link + t_gpu_attn)/t_pass | per-layer terms at smallest n_q: t_pass / t_link / t_gpu_attn [us] |
|---|---:|---|---:|---|---:|---|
| LLAMA-7B | < 20 (A6/A4 1.18 already > 1) | - | - | - | 9 | 32.2 / 62.0 / 10.0 |
| LLAMA-65B | < 20 (A6/A4 1.02 already > 1) | - | - | - | 21 | 26.9 / 117.9 / 20.0 |
| GPT-175B | **22** | 0.27 % / 2.35 % | 99.73 % | 20: 0.99 -> 24: 1.01 | 29 | 26.9 / 173.8 / 20.0 |

A6/A4 over the grid (n_q: ratio):

| model \ n_q | 20 | 24 | 32 | 40 | 48 | 64 | 80 | 112 | 144 | 208 | 272 | 400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute ratio per partial layer | 0.24% | 0.29% | 0.39% | 0.49% | 0.58% | 0.78% | 0.97% | 1.35% | 1.73% | 2.48% | 3.21% | 4.66% |
| LLAMA-7B | 1.18 | 1.24 | 1.36 | 1.47 | 1.59 | 1.81 | 2.00 | 2.40 | 2.83 | 3.56 | 4.01 | 5.03 |
| LLAMA-65B | 1.02 | 1.07 | 1.16 | 1.24 | 1.32 | 1.55 | 1.70 | 1.98 | 2.17 | 2.57 | 2.80 | 3.35 |
| GPT-175B | 0.99 | 1.01 | 1.08 | 1.13 | 1.19 | 1.30 | 1.40 | 1.57 | 1.72 | 1.96 | 2.18 | 2.42 |

## EPIC (prefix 1 token/segment, n_q = n_new + n_off + 1)

| model | n_q* (simulated, A6/A4 = 1) | recompute ratio at n_q* (per partial layer / all-layer avg) | reuse ratio at n_q* | bracketing grid points (n_q: A6/A4) | theory n_q* = 4·(t_link + t_gpu_attn)/t_pass | per-layer terms at smallest n_q: t_pass / t_link / t_gpu_attn [us] |
|---|---:|---|---:|---|---:|---|
| LLAMA-7B | < 21 (A6/A4 1.64 already > 1) | - | - | - | 9 | 32.2 / 62.0 / 10.0 |
| LLAMA-65B | < 21 (A6/A4 1.20 already > 1) | - | - | - | 17 | 32.2 / 117.9 / 20.0 |
| GPT-175B | < 21 (A6/A4 1.03 already > 1) | - | - | - | 24 | 32.2 / 173.8 / 20.0 |

A6/A4 over the grid (n_q: ratio):

| model \ n_q | 21 | 25 | 33 | 41 | 49 | 65 | 81 | 113 | 145 | 209 | 273 | 401 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute ratio per partial layer | 0.26% | 0.30% | 0.40% | 0.50% | 0.59% | 0.79% | 0.98% | 1.36% | 1.74% | 2.49% | 3.23% | 4.67% |
| LLAMA-7B | 1.64 | 1.79 | 2.09 | 2.38 | 2.61 | 3.08 | 3.48 | 4.15 | 4.96 | 5.98 | 7.35 | 8.16 |
| LLAMA-65B | 1.20 | 1.30 | 1.48 | 1.66 | 1.82 | 2.06 | 2.25 | 2.54 | 2.83 | 3.30 | 3.97 | 4.32 |
| GPT-175B | 1.03 | 1.09 | 1.23 | 1.36 | 1.46 | 1.67 | 1.81 | 2.00 | 2.20 | 2.48 | 2.70 | 2.98 |

# Link: PCIe4 (64 GB/s)

## CacheBlend (r = 0 here, so n_q = n_new + n_off; for r > 0 the same n_q* applies with n_q = n_new + n_off + r·L)

| model | n_q* (simulated, A6/A4 = 1) | recompute ratio at n_q* (per partial layer / all-layer avg) | reuse ratio at n_q* | bracketing grid points (n_q: A6/A4) | theory n_q* = 4·(t_link + t_gpu_attn)/t_pass | per-layer terms at smallest n_q: t_pass / t_link / t_gpu_attn [us] |
|---|---:|---|---:|---|---:|---|
| LLAMA-7B | **70** | 0.85 % / 7.05 % | 99.15 % | 64: 0.95 -> 80: 1.07 | 67 | 32.2 / 530.3 / 10.0 |
| LLAMA-65B | **134** | 1.61 % / 4.07 % | 98.39 % | 112: 0.90 -> 144: 1.04 | 160 | 26.9 / 1054.6 / 20.0 |
| GPT-175B | **201** | 2.40 % / 4.43 % | 97.60 % | 144: 0.84 -> 208: 1.02 | 238 | 26.9 / 1578.9 / 20.0 |

A6/A4 over the grid (n_q: ratio):

| model \ n_q | 20 | 24 | 32 | 40 | 48 | 64 | 80 | 112 | 144 | 208 | 272 | 400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute ratio per partial layer | 0.24% | 0.29% | 0.39% | 0.49% | 0.58% | 0.78% | 0.97% | 1.35% | 1.73% | 2.48% | 3.21% | 4.66% |
| LLAMA-7B | 0.61 | 0.65 | 0.71 | 0.77 | 0.83 | 0.95 | 1.07 | 1.30 | 1.53 | 1.96 | 2.29 | 2.99 |
| LLAMA-65B | 0.44 | 0.46 | 0.50 | 0.54 | 0.58 | 0.67 | 0.75 | 0.90 | 1.04 | 1.29 | 1.51 | 1.92 |
| GPT-175B | 0.41 | 0.43 | 0.46 | 0.49 | 0.52 | 0.58 | 0.63 | 0.74 | 0.84 | 1.02 | 1.19 | 1.44 |

## EPIC (prefix 1 token/segment, n_q = n_new + n_off + 1)

| model | n_q* (simulated, A6/A4 = 1) | recompute ratio at n_q* (per partial layer / all-layer avg) | reuse ratio at n_q* | bracketing grid points (n_q: A6/A4) | theory n_q* = 4·(t_link + t_gpu_attn)/t_pass | per-layer terms at smallest n_q: t_pass / t_link / t_gpu_attn [us] |
|---|---:|---|---:|---|---:|---|
| LLAMA-7B | **63** | 0.76 % / 0.76 % | 99.24 % | 49: 0.84 -> 65: 1.03 | 67 | 32.2 / 530.3 / 10.0 |
| LLAMA-65B | **129** | 1.55 % / 1.55 % | 98.45 % | 113: 0.91 -> 145: 1.09 | 133 | 32.2 / 1054.6 / 20.0 |
| GPT-175B | **193** | 2.30 % / 2.30 % | 97.70 % | 145: 0.83 -> 209: 1.06 | 198 | 32.2 / 1578.9 / 20.0 |

A6/A4 over the grid (n_q: ratio):

| model \ n_q | 21 | 25 | 33 | 41 | 49 | 65 | 81 | 113 | 145 | 209 | 273 | 401 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recompute ratio per partial layer | 0.26% | 0.30% | 0.40% | 0.50% | 0.59% | 0.79% | 0.98% | 1.36% | 1.74% | 2.49% | 3.23% | 4.67% |
| LLAMA-7B | 0.51 | 0.56 | 0.66 | 0.75 | 0.84 | 1.03 | 1.20 | 1.54 | 1.87 | 2.45 | 3.07 | 3.91 |
| LLAMA-65B | 0.32 | 0.35 | 0.41 | 0.46 | 0.51 | 0.62 | 0.72 | 0.91 | 1.09 | 1.39 | 1.71 | 2.15 |
| GPT-175B | 0.27 | 0.30 | 0.33 | 0.36 | 0.41 | 0.48 | 0.56 | 0.71 | 0.83 | 1.06 | 1.25 | 1.57 |
