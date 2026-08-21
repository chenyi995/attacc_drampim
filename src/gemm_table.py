"""Measured-efficiency lookup tables for the refined GPU model.

Two tables drive ``xPU`` (``GPU_MODEL == "refined"`` uses the first, ``"flash"`` both):

* :data:`CUBLAS_A100_FP16_TFLOPS` -- cuBLAS fp16 GEMM throughput on an
  NVIDIA A100-SXM4-80GB, measured by
  https://github.com/harshithkantamneni/triton-vs-cublas-llm-benchmarks
  (``data/gemm_llm_rect_bench_2026-01-13_01-01-18.csv``, ``backend == torch``,
  p50 of 2000 iterations, duplicate runs averaged).  76 rectangular
  (M, N, K) shapes drawn from LLaMA/Mistral projection and FFN dimensions,
  M in {128, 256, 512, 1024, 2048}.  :func:`gemm_efficiency` turns it into a
  fraction of the 312 TFLOPS tensor-core peak that depends on the GEMM
  size, replacing the legacy model's flat ``MAX_COMPUTE_UTIL = 0.8``.

* :data:`FLASH_ATTN_A100_EFFICIENCY` -- fraction of peak reached by a fused
  FlashAttention-2 forward kernel (head dim 128, fp16, no mask) on an A100 as
  a function of the key length it streams.  These are *approximate* values
  read from the FlashAttention-2 A100 forward-throughput plots (Dao, 2023:
  ~130 TFLOPS at 512 tokens rising to ~205 TFLOPS at 16k); they are not a
  measurement made in this repository and are the one soft input of the
  ``flash`` GPU model.  :func:`attention_efficiency` interpolates them.

The GEMM lookup interpolates log-linearly and clamps outside the measured range:
cuBLAS plateaus at M = 2048 (~75-84 % of peak), so larger M reuses that row,
which is slightly conservative for the 30k-row prefill GEMMs of this study.
The table has no K < 4096 shape, so the attention score matmul (K = head dim
= 128) is matched to the nearest measured (N, K) pair.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

A100_FP16_PEAK_TFLOPS = 312.0

# (M, N, K) -> TFLOPS (cuBLAS, fp16, A100-SXM4-80GB)
CUBLAS_A100_FP16_TFLOPS: Dict[Tuple[int, int, int], float] = {
    (128, 4096, 4096): 124.9,
    (256, 4096, 4096): 157.4,
    (512, 4096, 4096): 185.2,
    (1024, 4096, 4096): 167.4,
    (2048, 4096, 4096): 222.9,
    (128, 4096, 11008): 144.7,
    (256, 4096, 11008): 196.1,
    (512, 4096, 11008): 213.3,
    (1024, 4096, 11008): 226.3,
    (2048, 4096, 11008): 239.4,
    (128, 4096, 16384): 155.0,
    (256, 4096, 16384): 207.0,
    (512, 4096, 16384): 225.3,
    (1024, 4096, 16384): 231.0,
    (2048, 4096, 16384): 243.4,
    (128, 5120, 5120): 136.1,
    (256, 5120, 5120): 188.6,
    (512, 5120, 5120): 166.7,
    (1024, 5120, 5120): 213.3,
    (2048, 5120, 5120): 241.9,
    (128, 5120, 13824): 156.5,
    (256, 5120, 13824): 192.4,
    (512, 5120, 13824): 231.3,
    (1024, 5120, 13824): 236.5,
    (2048, 5120, 13824): 258.5,
    (128, 5120, 20480): 164.9,
    (256, 5120, 20480): 218.2,
    (512, 5120, 20480): 238.7,
    (1024, 5120, 20480): 242.2,
    (2048, 5120, 20480): 259.2,
    (128, 8192, 8192): 160.8,
    (256, 8192, 8192): 199.9,
    (512, 8192, 8192): 186.3,
    (1024, 8192, 8192): 230.5,
    (2048, 8192, 8192): 249.9,
    (1024, 8192, 16384): 235.9,
    (128, 8192, 28672): 192.8,
    (256, 8192, 28672): 225.7,
    (512, 8192, 28672): 235.2,
    (1024, 8192, 28672): 233.9,
    (2048, 8192, 28672): 247.5,
    (128, 8192, 32768): 195.5,
    (256, 8192, 32768): 227.0,
    (512, 8192, 32768): 235.6,
    (1024, 8192, 32768): 235.0,
    (2048, 8192, 32768): 246.8,
    (128, 11008, 4096): 136.4,
    (256, 11008, 4096): 201.4,
    (512, 11008, 4096): 206.0,
    (1024, 11008, 4096): 213.6,
    (2048, 11008, 4096): 230.1,
    (128, 13824, 5120): 167.5,
    (256, 13824, 5120): 235.6,
    (512, 13824, 5120): 250.5,
    (1024, 13824, 5120): 242.8,
    (2048, 13824, 5120): 253.7,
    (128, 16384, 4096): 152.5,
    (256, 16384, 4096): 177.9,
    (512, 16384, 4096): 218.5,
    (1024, 16384, 4096): 239.8,
    (2048, 16384, 4096): 236.3,
    (128, 20480, 5120): 176.3,
    (256, 20480, 5120): 204.6,
    (512, 20480, 5120): 245.8,
    (1024, 20480, 5120): 242.2,
    (2048, 20480, 5120): 251.7,
    (128, 28672, 8192): 137.8,
    (256, 28672, 8192): 205.6,
    (512, 28672, 8192): 224.1,
    (1024, 28672, 8192): 239.2,
    (2048, 28672, 8192): 252.5,
    (128, 32768, 8192): 155.9,
    (256, 32768, 8192): 221.6,
    (512, 32768, 8192): 243.6,
    (1024, 32768, 8192): 240.7,
    (2048, 32768, 8192): 256.0,
}

# key length -> fraction of tensor-core peak (FlashAttention-2 fwd, d = 128, A100)
FLASH_ATTN_A100_EFFICIENCY: Dict[int, float] = {
    64: 0.12,
    128: 0.20,
    256: 0.30,
    512: 0.42,
    1024: 0.52,
    2048: 0.58,
    4096: 0.62,
    8192: 0.65,
    16384: 0.66,
}

_M_GRID = sorted({m for m, _, _ in CUBLAS_A100_FP16_TFLOPS})
_NK_PAIRS = sorted({(n, k) for _, n, k in CUBLAS_A100_FP16_TFLOPS})
_M_MIN, _M_MAX = _M_GRID[0], _M_GRID[-1]


def _interp_log(x: float, points) -> float:
    """Log-linear interpolation over sorted (x, y) points; clamps outside."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + t * (y1 - y0)
    return points[-1][1]


def _tflops_for_pair(m: float, n: int, k: int) -> float:
    points = [(mm, CUBLAS_A100_FP16_TFLOPS[(mm, n, k)]) for mm in _M_GRID
              if (mm, n, k) in CUBLAS_A100_FP16_TFLOPS]
    return _interp_log(m, points)


def gemm_tflops(m: float, n: float, k: float) -> float:
    """Interpolated cuBLAS throughput (TFLOPS) for an (m, n, k) fp16 GEMM.

    M is clamped to the measured [128, 2048] range.  (N, K) is matched to the
    measured projection shapes by inverse-distance weighting of the three
    nearest pairs in (log N, log K); an exact pair match is used directly.
    """
    m = min(max(m, _M_MIN), _M_MAX)
    n = max(n, 1)
    k = max(k, 1)
    if (n, k) in _NK_PAIRS:
        return _tflops_for_pair(m, int(n), int(k))
    scored = []
    for pn, pk in _NK_PAIRS:
        d = math.hypot(math.log(n) - math.log(pn), math.log(k) - math.log(pk))
        scored.append((d, pn, pk))
    scored.sort()
    nearest = scored[:3]
    weights = [1.0 / max(d, 1e-9) for d, _, _ in nearest]
    total = sum(weights)
    return sum(w * _tflops_for_pair(m, pn, pk)
               for w, (_, pn, pk) in zip(weights, nearest)) / total


def gemm_efficiency(m: float, n: float, k: float,
                    peak_tflops: float = A100_FP16_PEAK_TFLOPS) -> float:
    """Fraction of ``peak_tflops`` a cuBLAS GEMM of this shape achieves."""
    return min(1.0, gemm_tflops(m, n, k) / peak_tflops)


def attention_efficiency(key_length: float) -> float:
    """Fraction of peak for a fused attention kernel streaming ``key_length`` keys."""
    points = sorted(FLASH_ATTN_A100_EFFICIENCY.items())
    return _interp_log(max(key_length, 1), points)
