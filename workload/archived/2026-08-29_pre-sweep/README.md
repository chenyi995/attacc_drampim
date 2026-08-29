# 归档：pre-sweep 手调 generator（2026-08-29）

**参数化 sweep（`../gen_sweep.py`）之前**的 5 个手调 workload generator 与它们
生成的 workload JSON，归档留存。裁决与原因见 `docs/sessions/2026-08-29.md`。

## 内容
- `gen_star_repair.py` / `gen_pipeline_repair.py` / `gen_debate.py` /
  `gen_mapreduce_sum.py` / `gen_multisource_rag.py`：5 个手调 generator（原始版，
  已 `git checkout` 恢复，未含 8-29 那次失败的"统一到 64 block"改动）；
- `workload_*.json`：它们生成的 workload（star r5w3k47、debate d3r5k49、
  pipeline c5k50、mapreduce m8/m4、RAG n12s96 等）。

## 为什么归档
5 个手调场景的尺寸依赖 workload-specific 的 corpus（47/49/50 chunk，反推到
history extent = 85–86% cap），非整、难在文中一句话讲清。改成参数化 sweep 后：
一个 generator + (topology, N, C, D, k)、全整数、cap 天然不越（≤50%）。
新体系见 `../gen_sweep.py`、`docs/README_sweep_design.md`。

## 复现（如需重跑旧 workload）
`python3 gen_star_repair.py`（等）即在**本目录**重新生成对应 JSON；参数见 `--help`。
