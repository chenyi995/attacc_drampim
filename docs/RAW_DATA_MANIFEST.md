# 原始数据本地清单 (RAW DATA MANIFEST)

**口径（2026-08-29 起）**：所有仿真原始 run 数据（`dag_A*.json` 事件轨迹 +
`.log`）**一律不入 git、只在本机**（体积大、GitHub 拒 >100 MB）。git 里只留
**`output/analysis/`**（分析脚本与 RESULTS 表）。

本机根路径 `/data2/chenyi9/KV-PIM/attacc_drampim_xinyao`。

## 原始 run 数据在哪
- **新参数化 sweep**：`output/sweep_<时间戳>/<config>_k<k>/dag_A*.{json,log}`
  （由 `experiments/run_sweep.sh` 产出；workload 定义 `workload/sweep/`，入库）；
- **归档的旧手调结果**：`output/archived/2026-08-29_pre-unify/`（21 个 run，
  9.6 GB，含旧 workload 备份与 RESULTS 快照；见该目录 README）。

## 需要这些原始数据时
本机直接取；异地复现用 `workload/`（入库的 workload 定义）+
`experiments/run_sweep.sh` / `run_dag_ladder.sh` 重跑即可（RESULTS 表由
`output/analysis/make_results_tables.py` 从 `dag_A*.json` 重算）。
